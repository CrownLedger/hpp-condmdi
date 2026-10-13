"""Phase 5, Experiment A - HPP key-selection test: ORIGINAL features at HPP-selected entries only.

python -m motion_project.experiments.run_mask_experiment                      # all clips x projectors x root policies
python -m motion_project.experiments.run_mask_experiment --clips 000021 --projectors hpp_v02 --root_policies dense
Requires run_hpp first. --repro_check reruns the first condition and compares outputs (acceptance C8).
"""
import argparse
import numpy as np
from motion_project.experiments.common import load_cfg, run_condition, OUT
from motion_project.realizer.condmdi_adapter import CondMDIRealizer

EXPERIMENT = 'A'


def main():
    exp = load_cfg('experiment')
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='+', default=[c['id'] for c in exp['clips']])
    ap.add_argument('--projectors', nargs='+', default=exp['projectors'])
    ap.add_argument('--root_policies', nargs='+', default=exp['root_policies'])
    ap.add_argument('--repro_check', action='store_true')
    a = ap.parse_args()
    ccfg = load_cfg('condmdi')
    realizer = CondMDIRealizer(ccfg)
    first = None
    for clip in a.clips:
        for proj in a.projectors:
            for rp in a.root_policies:
                rep = run_condition(realizer, clip, proj, EXPERIMENT, rp, ccfg)
                first = first or (clip, proj, rp, rep['name'])
                c = [x['constraint_vs_target'] for x in rep['reps']]
                print(f"{rep['name']:45s} constraint err mean {np.mean([x['mean'] for x in c])*100:5.1f} cm  "
                      f"p95 {np.mean([x['p95'] for x in c])*100:5.1f} cm  ({rep['record']['seconds']} s)")
    if a.repro_check and first:
        clip, proj, rp, name = first
        before = np.load(f'{OUT}/generated/{name}.npz')['generated']
        again = run_condition(realizer, clip, proj, EXPERIMENT, rp, ccfg, tag='__repro')
        after = np.load(f"{OUT}/generated/{again['name']}.npz")['generated']
        print(f'reproducibility ({name}): max |diff| = {np.abs(before - after).max():.2e} m')
        with open(f'{OUT}/reports/repro__{name}.txt', 'w') as f:
            f.write(f'{np.abs(before - after).max()}\n')


if __name__ == '__main__':
    main()
