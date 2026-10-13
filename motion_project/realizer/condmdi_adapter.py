"""Pretrained CondMDI as the AI Motion Realizer (handoff Part V / Phase 7).

Uses the repository's own model construction, checkpoint loading, normalisation statistics, sampler
(GaussianDiffusion.p_sample_loop, or ddim_sample_loop on respaced timesteps) and decoder (recover_from_ric, abs_3d). Nothing in CondMDI is modified.
"""
import hashlib
import os
import random
import sys
import numpy as np
import torch

from utils.parser_util import cond_synt_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils import dist_util
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from sample.conditional_synthesis import load_dataset

from motion_project.realizer import condmdi_mask_adapter as cma


class FusedCpuUnet(torch.nn.Module):
    """CPU only: the UNet traced, frozen and run with oneDNN graph fusion (same math; ~1.5x faster per step on
    8 cores). One traced graph per input shape (batch size); falls back to the eager module if tracing fails."""

    def __init__(self, unet):
        super().__init__()
        self.eager, self.graphs = unet, {}

    def forward(self, x, cond):
        key = (tuple(x.shape), tuple(cond.shape))
        if key not in self.graphs:
            try:
                with torch.no_grad():
                    g = torch.jit.freeze(torch.jit.trace(self.eager, (x, cond), check_trace=False).eval())
                    for _ in range(2):                       # oneDNN graph compiles its fused kernels on the first runs
                        g(x, cond)
                self.graphs[key] = g
            except Exception as e:                           # pragma: no cover - keep working without fusion
                print(f'FusedCpuUnet: tracing failed ({e}); using the eager UNet')
                self.graphs[key] = self.eager
        return self.graphs[key](x, cond)


def _file_fingerprint(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


class CondMDIRealizer:
    def __init__(self, cfg):
        self.cfg = cfg
        argv = sys.argv
        sys.argv = [argv[0], '--model_path', cfg['model_path'], '--edit_mode', 'benchmark_sparse', '--num_samples', '1']
        try:
            args = cond_synt_args()            # the checkpoint's own args.json + repo defaults
        finally:
            sys.argv = argv
        args.batch_size = 1
        self.args = args
        self.max_frames = cfg['max_frames']
        dist_util.setup_dist(args.device)
        self.dev = dist_util.dev()
        data = load_dataset(args, self.max_frames, split='test')   # used only for normalisation statistics
        self.t2m = data.dataset.t2m_dataset
        std, mean = self.t2m.get_std_mean()
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        model, self.diffusion = create_model_and_diffusion(args, data)
        load_saved_model(model, cfg['model_path'])
        self.base_model = model
        self.checkpoint_sha256 = _file_fingerprint(cfg['model_path'])
        self._base_conf = self.diffusion.conf               # full 1000-step schedule (respacing [1000] keeps every beta)
        self._cache_text_encoding(model)
        if self.dev.type == 'cpu':
            model.float()                                    # CLIP weights are stored in fp16, which has no CPU kernels
            torch.set_num_threads(min(16, torch.get_num_threads()))   # more threads span both sockets and get slower
            if cfg.get('cpu_graph_fusion', True):
                torch.jit.enable_onednn_fusion(True)
                model.unet = FusedCpuUnet(model.unet)
        self.set_sampler(cfg.get('sampler', 'ddpm'), cfg.get('sampling_steps', 1000))

    @staticmethod
    def _cache_text_encoding(model):
        """The UNet re-encodes the (constant) text with CLIP at every diffusion step; encode once per prompt list."""
        encode, cache = model.encode_text, {}
        def cached(raw_text):
            key = tuple(raw_text)
            if key not in cache:
                cache.clear()
                cache[key] = encode(raw_text)
            return cache[key]
        model.encode_text = cached

    def set_sampler(self, sampler='ddpm', steps=1000):
        """ddpm = GaussianDiffusion.p_sample_loop, ddim = ddim_sample_loop (eta 0), both on `steps` evenly spaced
        timesteps of the checkpoint's 1000-step schedule (SpacedDiffusion, the repo's own respacing)."""
        if sampler not in ('ddpm', 'ddim'):
            raise ValueError(sampler)
        self.sampler, self.sampling_steps = sampler, int(steps)
        spec = f'ddim{steps}' if sampler == 'ddim' else [int(steps)]
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, spec), conf=self._base_conf)

    def realize(self, features_abs, joint_mask, seed, num_repetitions, text='', imputate=False, feature_mode='pos_rot_vel'):
        """features_abs (T,263) source of observed values; joint_mask (22,T) bool.
        Returns positions (R,T,22,3) and a run record."""
        T = len(features_abs)
        if T > self.max_frames:
            raise ValueError(f'{T} frames > CondMDI limit {self.max_frames}: segmentation is not implemented; '
                             'refusing to truncate silently')
        fmask = cma.feature_mask(joint_mask, self.max_frames, feature_mode)
        obs = cma.observed_features(features_abs, fmask, self.mean, self.std, self.max_frames)
        guidance = 1.0 if text else 0.0                                  # 0 = text dropped (as --no_text)
        # Guidance 0 gives ClassifierFreeSampleModel's unconditional pass exactly (out_uncond + 0 * (out - out_uncond)),
        # so the base model is called once per step with y['uncond'] instead of twice through the wrapper.
        model = self.base_model
        model.to(self.dev)                                               # MDM_UNET._apply returns None: no chaining
        model.eval()
        R = num_repetitions                                              # all repetitions sampled as one batch
        obs_x0 = torch.from_numpy(obs)[None, :, None, :].repeat(R, 1, 1, 1).to(self.dev)
        obs_mask = torch.from_numpy(fmask)[None, :, None, :].repeat(R, 1, 1, 1).to(self.dev)
        frame_mask = torch.zeros((R, 1, 1, self.max_frames), dtype=torch.bool, device=self.dev)
        frame_mask[..., :T] = True
        y = {'mask': frame_mask, 'lengths': torch.full((R,), T, device=self.dev), 'text': [text] * R,
             'diffusion_steps': self.args.diffusion_steps}
        if guidance == 0:
            y['uncond'] = True
        if imputate:
            y.update({'imputate': 1, 'stop_imputation_at': 0, 'replacement_distribution': 'conditional',
                      'inpainted_motion': obs_x0, 'inpainting_mask': obs_mask, 'reconstruction_guidance': False})
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            kw = dict(clip_denoised=False, model_kwargs={'obs_x0': obs_x0, 'obs_mask': obs_mask, 'y': y},
                      skip_timesteps=0, init_image=None, progress=False, dump_steps=None, noise=None)
            if self.sampler == 'ddim':
                sample = self.diffusion.ddim_sample_loop(model, (R, 263, 1, self.max_frames), eta=0.0, **kw)
            else:
                sample = self.diffusion.p_sample_loop(model, (R, 263, 1, self.max_frames), const_noise=False, **kw)
        feat = self.t2m.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        out = list(recover_from_ric(feat, 22, abs_3d=True)[:, 0, :T].numpy().astype(np.float64))
        record = {'checkpoint': os.path.abspath(self.cfg['model_path']), 'checkpoint_sha256': self.checkpoint_sha256,
                  'sampler': {'ddpm': 'GaussianDiffusion.p_sample_loop', 'ddim': 'ddim_sample_loop (eta 0)'}[self.sampler],
                  'sampling_steps': self.sampling_steps, 'diffusion_steps': int(self.args.diffusion_steps),
                  'guidance_param': guidance, 'text': text, 'imputate': imputate, 'feature_mode': feature_mode,
                  'seeds': [seed], 'batched_repetitions': num_repetitions, 'n_frames': T, 'max_frames': self.max_frames,
                  'observed_joint_frames': int(joint_mask.sum()), 'observed_feature_entries': int(fmask.sum()),
                  'normalisation': 'dataset/HumanML3D/Mean_abs_3d.npy, Std_abs_3d.npy (GMD abs-root statistics)'}
        return np.stack(out), record
