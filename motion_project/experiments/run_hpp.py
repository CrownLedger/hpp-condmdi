"""Phase 3/4: HPP projection + standalone validation (original vs rough).

python -m motion_project.experiments.run_hpp                          # all clips x configured projectors
python -m motion_project.experiments.run_hpp --clips 000021 --projectors hpp_v02 robotic_slerp
"""
import argparse
from motion_project.experiments.common import load_cfg, run_hpp_stage


def main():
    exp = load_cfg('experiment')
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='+', default=[c['id'] for c in exp['clips']])
    ap.add_argument('--projectors', nargs='+', default=exp['projectors'])
    a = ap.parse_args()
    for clip in a.clips:
        for proj in a.projectors:
            r = run_hpp_stage(clip, proj)
            ee = max(r['end_effector_err_vs_original_m'].values())
            print(f"{clip} {proj:14s} keys/traj {r['keys_per_trajectory']['root']:3d}  observed {r['observed_joint_frames']}/{r['total_joint_frames']}"
                  f"  MPJPE {100*r['mpjpe_vs_original_m']:.1f} cm  max EE {100*ee:.1f} cm  root {100*r['root_vs_original']['mean_m']:.1f} cm"
                  f"  bone dev {r['bone_length']['max_dev_m']:.1e} m  contacts {r['temporal']['contact_events']['recall']:.2f}"
                  f"  -> " + ' '.join(f"{k.split('_')[0]}={'ok' if v is True else ('FAIL' if v is False else v)}" for k, v in r['acceptance'].items()))


if __name__ == '__main__':
    main()
