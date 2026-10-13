"""Speed vs quality of CondMDI samplers on the CMU clips (projector direct_1s by default).

python -m motion_project.experiments.benchmark_sampler                       # GPU sweep
python -m motion_project.experiments.benchmark_sampler --configs ddpm:1000 ddim:50 --clips cmu_120_21

Baseline: the saved 1000-step DDPM outputs (outputs/generated/<clip>__<projector>__B__dense.npz). Each config is
scored with the realism indicators of evaluation/compare_realism.py plus the key error (observed joint-frames vs
input); results -> outputs/reports/sampler_benchmark__<device>.json.
"""
import argparse
import os
import time

import numpy as np
import torch

from motion_project.evaluation.compare_realism import score, mean_scores
from motion_project.experiments.common import CONFIGS, OUT, _json, load_cfg
from motion_project.realizer import condmdi_mask_adapter as cma
from motion_project.realizer import motion_feature_adapter as mfa
from motion_project.realizer.condmdi_adapter import CondMDIRealizer


def quality(gen, rough, obs_mask, ref=None):
    s = mean_scores([score(g) for g in gen])
    s['key_err_cm'] = float(np.mean([100 * np.linalg.norm(g - rough, axis=-1)[obs_mask].mean() for g in gen]))
    # diversity: mean joint distance between the samples (a few-step sampler can collapse them)
    s['diversity_cm'] = float(np.mean([100 * np.linalg.norm(gen[i] - gen[j], axis=-1).mean()
                                       for i in range(len(gen)) for j in range(i + 1, len(gen))])) if len(gen) > 1 else 0.0
    # detail kept: all-frame error vs the input, and movement amplitude relative to the input (1.0 = same)
    s['mpjpe_all_cm'] = float(np.mean([100 * np.linalg.norm(g - rough, axis=-1).mean() for g in gen]))
    sp = lambda x: np.linalg.norm(np.diff(x, axis=0), axis=-1).mean()
    s['speed_ratio'] = float(np.mean([sp(g) / sp(rough) for g in gen]))
    rng = lambda x: (x - x[:, :1]).std(0).mean()                  # root-relative pose variation over time
    s['pose_range_ratio'] = float(np.mean([rng(g) / rng(rough) for g in gen]))
    if ref is not None:                                   # distance to the 1000-step output (same seed and noise draw)
        s['vs_1000step_cm'] = float(np.mean([100 * np.linalg.norm(g - r, axis=-1).mean() for g, r in zip(gen, ref)]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--projector', default='direct_1s')
    ap.add_argument('--clips', nargs='+', default=open(os.path.join(CONFIGS, 'cmu_clips.txt')).read().split())
    ap.add_argument('--configs', nargs='+', default=['ddpm:200', 'ddpm:100', 'ddpm:50', 'ddim:100', 'ddim:50', 'ddim:20'])
    ap.add_argument('--repetitions', type=int, default=3)
    ap.add_argument('--baseline_dir', default=f'{OUT}/generated', help='where the 1000-step DDPM outputs are')
    ap.add_argument('--check_identical', default='cmu_120_21', help='re-run 1000-step DDPM on this clip vs saved output')
    a = ap.parse_args()
    ccfg = load_cfg('condmdi')
    realizer = CondMDIRealizer(ccfg)
    dev = realizer.dev.type
    inputs = {}
    for clip in a.clips:
        r = np.load(f'{OUT}/rough/{clip}__{a.projector}.npz')
        jm, _ = cma.build_joint_mask(r['observation_mask'], r['orientation_mask'], 'dense', 'none')
        inputs[clip] = (mfa.positions_to_abs_features(r['rough_positions']), jm, r['rough_positions'], r['observation_mask'])
    res = {'device': dev, 'projector': a.projector, 'repetitions': a.repetitions, 'configs': {}}

    base, saved = {}, {}
    for clip in a.clips:
        g = np.load(f'{a.baseline_dir}/{clip}__{a.projector}__B__dense.npz')['generated'][:a.repetitions]
        base[clip] = quality(g, inputs[clip][2], inputs[clip][3])
        saved[clip] = g
    res['configs']['ddpm:1000 (saved baseline)'] = {'quality': mean_scores(list(base.values())), 'per_clip': base}

    if a.check_identical and a.check_identical in inputs:
        realizer.set_sampler('ddpm', 1000)
        feat, jm, rough, om = inputs[a.check_identical]
        t0 = time.time()
        g, _ = realizer.realize(feat, jm, ccfg['seed'], a.repetitions, '', False, ccfg['feature_mode'])
        sec = time.time() - t0
        saved = np.load(f'{a.baseline_dir}/{a.check_identical}__{a.projector}__B__dense.npz')['generated'][:a.repetitions]
        res['text_cache_check'] = {'clip': a.check_identical, 'max_abs_diff_m': float(np.abs(g - saved).max()), 'seconds': sec}
        print(f'ddpm:1000 with cached text encoding on {a.check_identical}: max |diff| vs saved = '
              f'{np.abs(g - saved).max():.2e} m, {sec:.1f} s for {a.repetitions} samples', flush=True)

    for c in a.configs:
        sampler, steps = c.split(':')
        realizer.set_sampler(sampler, int(steps))
        per, secs = {}, []
        for clip in a.clips:
            feat, jm, rough, om = inputs[clip]
            if dev == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            g, _ = realizer.realize(feat, jm, ccfg['seed'], a.repetitions, '', False, ccfg['feature_mode'])
            secs.append(time.time() - t0)
            per[clip] = quality(g, rough, om, saved[clip])
        q = mean_scores(list(per.values()))
        res['configs'][c] = {'quality': q, 'per_clip': per, 'seconds_per_call_mean': float(np.mean(secs))}
        print(f'{c:10s} {np.mean(secs):6.1f} s/call ({a.repetitions} samples) | key err {q["key_err_cm"]:5.1f} cm  jerk {q["jerk"]:5.0f}  '
              f'hf {q["hf_energy"]:.3f}  slide {q["foot_slide"]:.2f}  bone {q["bone_std"]:.2f}  below floor {q["below_floor"]:.3f}  '
              f'diversity {q["diversity_cm"]:.1f} cm  vs-1000 {q["vs_1000step_cm"]:.1f} cm  | all-frame err {q["mpjpe_all_cm"]:.2f} cm  '
              f'speed x{q["speed_ratio"]:.3f}  pose range x{q["pose_range_ratio"]:.3f}', flush=True)
    b = res['configs']['ddpm:1000 (saved baseline)']['quality']
    print(f'{"ddpm:1000":10s} {"(saved)":>6s}               | key err {b["key_err_cm"]:5.1f} cm  jerk {b["jerk"]:5.0f}  '
          f'hf {b["hf_energy"]:.3f}  slide {b["foot_slide"]:.2f}  bone {b["bone_std"]:.2f}  below floor {b["below_floor"]:.3f}  '
          f'diversity {b["diversity_cm"]:.1f} cm  | all-frame err {b["mpjpe_all_cm"]:.2f} cm  speed x{b["speed_ratio"]:.3f}  pose range x{b["pose_range_ratio"]:.3f}')
    _json(res, f'{OUT}/reports/sampler_benchmark__{dev}.json')


if __name__ == '__main__':
    main()
