"""Pretrained CondMDI as the AI Motion Realizer (handoff Part V / Phase 7).

Uses the repository's own model construction, checkpoint loading, normalisation statistics, sampler
(GaussianDiffusion.p_sample_loop) and decoder (recover_from_ric, abs_3d). Nothing in CondMDI is modified.
"""
import hashlib
import os
import random
import sys
import numpy as np
import torch

from utils.parser_util import cond_synt_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from sample.conditional_synthesis import load_dataset

from motion_project.realizer import condmdi_mask_adapter as cma


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
        model = ClassifierFreeSampleModel(self.base_model) if guidance != 1 else self.base_model
        model.to(self.dev).eval()
        obs_x0 = torch.from_numpy(obs)[None, :, None, :].to(self.dev)
        obs_mask = torch.from_numpy(fmask)[None, :, None, :].to(self.dev)
        frame_mask = torch.zeros((1, 1, 1, self.max_frames), dtype=torch.bool, device=self.dev)
        frame_mask[..., :T] = True
        y = {'mask': frame_mask, 'lengths': torch.tensor([T], device=self.dev), 'text': [text],
             'diffusion_steps': self.args.diffusion_steps}
        if guidance != 1:
            y['text_scale'] = torch.ones(1, device=self.dev) * guidance
        if imputate:
            y.update({'imputate': 1, 'stop_imputation_at': 0, 'replacement_distribution': 'conditional',
                      'inpainted_motion': obs_x0, 'inpainting_mask': obs_mask, 'reconstruction_guidance': False})
        out = []
        for r in range(num_repetitions):
            s = seed + r
            torch.manual_seed(s); np.random.seed(s); random.seed(s); torch.cuda.manual_seed_all(s)
            with torch.no_grad():
                sample = self.diffusion.p_sample_loop(model, (1, 263, 1, self.max_frames), clip_denoised=False,
                                                      model_kwargs={'obs_x0': obs_x0, 'obs_mask': obs_mask, 'y': y},
                                                      skip_timesteps=0, init_image=None, progress=False,
                                                      dump_steps=None, noise=None, const_noise=False)
            feat = self.t2m.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            out.append(recover_from_ric(feat, 22, abs_3d=True)[0, 0, :T].numpy().astype(np.float64))
        record = {'checkpoint': os.path.abspath(self.cfg['model_path']), 'checkpoint_sha256': self.checkpoint_sha256,
                  'sampler': 'GaussianDiffusion.p_sample_loop', 'diffusion_steps': int(self.args.diffusion_steps),
                  'guidance_param': guidance, 'text': text, 'imputate': imputate, 'feature_mode': feature_mode,
                  'seeds': [seed + r for r in range(num_repetitions)], 'n_frames': T, 'max_frames': self.max_frames,
                  'observed_joint_frames': int(joint_mask.sum()), 'observed_feature_entries': int(fmask.sum()),
                  'normalisation': 'dataset/HumanML3D/Mean_abs_3d.npy, Std_abs_3d.npy (GMD abs-root statistics)'}
        return np.stack(out), record
