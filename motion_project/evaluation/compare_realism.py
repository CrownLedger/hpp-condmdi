"""Realism of CondMDI output vs its input (no learned evaluator available: T2M evaluators are not downloadable).

python -m motion_project.evaluation.compare_realism --projector direct
-> outputs/reports/realism__<projector>.json, a table on stdout, and outputs/reports/realism_plots/*.png

Physical / kinematic indicators, the same for input and output (lower is more realistic unless noted):
  jerk           mean |d3x/dt3| over joints (m/s^3)            - smoothness
  hf_energy      share of joint-velocity power above 4 Hz        - jitter (human movement is mostly < 4 Hz)
  foot_slide     horizontal foot speed while near the floor (m/s)
  below_floor    fraction of frames with a foot below the floor
  floating       fraction of frames with both feet > 10 cm above the floor
  bone_std       mean per-bone length std (cm)                   - a real body is rigid
  knee_backward  fraction of frames with a knee bent > 15 deg the wrong way (hyperextension)
Real HumanML3D clips (000021, 012314) are scored as a reference for natural motion.
"""
import argparse
import glob
import json
import os

import numpy as np

from motion_project.evaluation import metrics
from motion_project.experiments.common import OUT, _json
from motion_project.hpp import skeleton as sk

FPS = 20.0
LEGS = [(1, 4, 7), (2, 5, 8)]              # hip, knee, ankle (left, right)


def hf_energy(pos, fps=FPS, cutoff=4.0):
    v = np.diff(pos, axis=0) * fps
    v = v - v.mean(0)
    P = np.abs(np.fft.rfft(v, axis=0)) ** 2
    f = np.fft.rfftfreq(len(v), 1 / fps)
    return float(P[f > cutoff].sum() / max(P.sum(), 1e-12))


def knee_backward(pos, min_flex_deg=15.0):
    """Knee flexion axis must point to the body's left (both legs bend the same way); opposite = hyperextension."""
    left = pos[:, 1] - pos[:, 2]
    left /= np.linalg.norm(left, axis=-1, keepdims=True)
    bad = []
    for h, k, a in LEGS:
        th, sh = pos[:, k] - pos[:, h], pos[:, a] - pos[:, k]
        cr = np.cross(th, sh)
        flex = np.degrees(np.arccos(np.clip((th * sh).sum(-1) / (np.linalg.norm(th, axis=-1) * np.linalg.norm(sh, axis=-1)), -1, 1)))
        bad.append((flex > min_flex_deg) & ((cr * left).sum(-1) < 0))
    return float(np.mean(bad))


def score(pos):
    y = pos[:, metrics.FOOT_JOINTS, 1]
    return {'jerk': metrics.jerk(pos, FPS), 'hf_energy': hf_energy(pos),
            'foot_slide': metrics.foot_sliding(pos, FPS)['mean_mps'],
            'below_floor': float((y.min(1) < -0.01).mean()), 'floating': float((y.min(1) > 0.10).mean()),
            'bone_std': 100 * float(sk.bone_lengths(pos)[:, 1:].std(0).mean()), 'knee_backward': knee_backward(pos)}


def mean_scores(list_of_scores):
    return {k: float(np.mean([s[k] for s in list_of_scores])) for k in list_of_scores[0]}


def plots(clip, inp, gen, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(len(inp)) / FPS
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for a, (j, c, name) in zip(ax, [(21, 1, 'right wrist height'), (20, 0, 'left wrist x'), (10, 1, 'left toe height')]):
        a.plot(t, inp[:, j, c], color='#7e57c2', lw=2.2, label='input (CMU)')
        for r, col in zip(gen, ['#29b6f6', '#ff8a65', '#66bb6a']):
            a.plot(t, r[:, j, c], color=col, lw=1, alpha=0.9)
        for k in np.arange(0, len(inp), 10) / FPS:
            a.axvline(k, color='#bbbbbb', lw=0.5, zorder=0)
        a.set_ylabel(f'{name} (m)')
    ax[0].legend(['input (CMU)', 'CondMDI sample 0', 'sample 1', 'sample 2'], loc='upper right', fontsize=8, ncol=4)
    ax[-1].set_xlabel('time (s); grey lines = observed key frames (every 0.5 s)')
    fig.suptitle(f'{clip}: input vs CondMDI output')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/{clip}.png', dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--projector', default='direct')
    ap.add_argument('--experiment', default='B')
    ap.add_argument('--root_policy', default='dense')
    a = ap.parse_args()
    out_dir = f'{OUT}/reports/realism_plots'
    os.makedirs(out_dir, exist_ok=True)
    res = {'reference_real_humanml3d': {}, 'clips': {}}
    for c in ('000021', '012314'):
        p = f'{OUT}/original/{c}.npz'
        if os.path.exists(p):
            res['reference_real_humanml3d'][c] = score(np.load(p)['positions'])
    for g in sorted(glob.glob(f'{OUT}/generated/*__{a.projector}__{a.experiment}__{a.root_policy}.npz')):
        clip = os.path.basename(g).split('__')[0]
        inp = np.load(f'{OUT}/rough/{clip}__{a.projector}.npz')['rough_positions']
        gen = np.load(g)['generated']
        res['clips'][clip] = {'input': score(inp), 'output_mean': mean_scores([score(x) for x in gen]),
                              'output_samples': [score(x) for x in gen]}
        plots(clip, inp, gen, out_dir)
    keys = list(next(iter(res['clips'].values()))['input'])
    res['summary'] = {
        'input_mean': mean_scores([v['input'] for v in res['clips'].values()]),
        'output_mean': mean_scores([v['output_mean'] for v in res['clips'].values()]),
        'reference_mean': mean_scores(list(res['reference_real_humanml3d'].values())) if res['reference_real_humanml3d'] else None,
        'clips_where_output_better': {k: sum(v['output_mean'][k] < v['input'][k] - 1e-9 for v in res['clips'].values()) for k in keys},
        'n_clips': len(res['clips'])}
    _json(res, f'{OUT}/reports/realism__{a.projector}.json')

    fmt = {'jerk': '{:6.0f}', 'hf_energy': '{:6.3f}', 'foot_slide': '{:6.2f}', 'below_floor': '{:6.2f}', 'floating': '{:6.2f}',
           'bone_std': '{:6.2f}', 'knee_backward': '{:6.3f}'}
    print(f"{'':22s}" + ''.join(f'{k:>14s}' for k in keys))
    rows = [(f'{c} input', v['input']) for c, v in res['clips'].items()]
    for c, v in res['clips'].items():
        print(f'{c + " input":22s}' + ''.join(f'{fmt[k].format(v["input"][k]):>14s}' for k in keys))
        print(f'{c + " output":22s}' + ''.join(f'{fmt[k].format(v["output_mean"][k]):>14s}' for k in keys))
    s = res['summary']
    for name, v in (('MEAN input', s['input_mean']), ('MEAN output', s['output_mean']), ('real HumanML3D ref', s['reference_mean'])):
        if v:
            print(f'{name:22s}' + ''.join(f'{fmt[k].format(v[k]):>14s}' for k in keys))
    print(f"{'output better (clips)':22s}" + ''.join(f"{str(s['clips_where_output_better'][k]) + '/' + str(s['n_clips']):>14s}" for k in keys))


if __name__ == '__main__':
    main()
