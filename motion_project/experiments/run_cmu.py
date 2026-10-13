"""CMU robotic mocap -> CondMDI, one clip at a time, without a rougher (projector `direct`).

python -m motion_project.experiments.prepare_cmu_bvh /root/cmu-robotic/*.bvh
python -m motion_project.experiments.run_cmu                     # clips from configs/cmu_clips.txt
python -m motion_project.experiments.run_cmu --clips cmu_120_21
python -m motion_project.experiments.run_cmu --projector direct_1s   # fewer keys (every 1 s)

Per clip: direct projection (rough = input; full body every key_every_s observed), Experiment B with the root
observed on every frame, then the viewer data is re-exported so each result appears in the viewer as soon as it is done.
"""
import argparse
import os
import numpy as np

from motion_project.experiments.common import CONFIGS, load_cfg, run_condition, run_hpp_stage
from motion_project.experiments import export_viewer
from motion_project.realizer.condmdi_adapter import CondMDIRealizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='+', default=open(os.path.join(CONFIGS, 'cmu_clips.txt')).read().split())
    ap.add_argument('--root_policy', default='dense')
    ap.add_argument('--projector', default='direct', help='direct (keys every 0.5 s), direct_1s (1 s) or direct_1p5s (1.5 s)')
    a = ap.parse_args()
    ccfg = load_cfg('condmdi')
    realizer = CondMDIRealizer(ccfg)
    for i, clip in enumerate(a.clips, 1):
        run_hpp_stage(clip, a.projector)
        rep = run_condition(realizer, clip, a.projector, 'B', a.root_policy, ccfg)
        export_viewer.main()
        c = [x['constraint_vs_target'] for x in rep['reps']]
        print(f"[{i}/{len(a.clips)}] {clip:12s} constraint err mean {np.mean([x['mean'] for x in c]) * 100:5.1f} cm  "
              f"p95 {np.mean([x['p95'] for x in c]) * 100:5.1f} cm  jerk {np.mean([x['jerk_mps3'] for x in rep['reps']]):6.1f}  "
              f"({rep['record']['seconds']} s) -> viewer updated", flush=True)


if __name__ == '__main__':
    main()
