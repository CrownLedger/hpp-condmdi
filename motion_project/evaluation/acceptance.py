"""Apply the pre-registered tolerances (configs/acceptance.yaml) to every report and write summary tables.

python -m motion_project.evaluation.acceptance
-> outputs/reports/acceptance.json, outputs/reports/acceptance_tables.md
"""
import glob
import json
import os
import numpy as np

from motion_project.experiments.common import OUT, load_cfg, _json


def _cm(x):
    return f'{100 * x:.1f}'


def gen_checks(rep, a, orig_slide, ref_jerk):
    c = {
        'C1_valid': bool(rep['finite'] and rep['temporal']['duration_equal']
                         and rep['bone_length']['max_dev_m'] <= a['C1_bone_length_max_dev_m']),
        'C2_constraints': bool(rep['constraint_vs_target']['mean'] <= a['C2_constraint_mean_err_m']
                               and rep['constraint_vs_target']['p95'] <= a['C2_constraint_p95_err_m']),
        'C3_root': bool(rep['root_vs_target']['mean_m'] <= a['C3_root_mean_err_m']),
        'C4_foot_sliding': bool(rep['foot_sliding']['mean_mps'] <= a['C4_foot_slide_max_ratio'] * orig_slide),
        'C5_penetration': bool(rep['ground_penetration']['min_foot_height_m'] >= a['C5_min_foot_height_m']),
        'C6_contact_timing': bool(rep['temporal']['contact_events']['recall'] >= a['C6_contact_recall_min']),
        'C7_jitter': bool(rep['jerk_mps3'] <= a['C7_jerk_max_ratio'] * ref_jerk) if ref_jerk else None,
    }
    return c


def main():
    acc = load_cfg('acceptance')
    out = {'hpp': {}, 'generated': {}, 'reproducibility': {}, 'tolerances': acc}
    lines = ['## HPP stage (rough vs original)', '',
             '| clip | projector | keys/traj | observed joint-frames | MPJPE cm | max end-eff. cm | root cm | bone dev m | contact recall | H1-H7 |',
             '|---|---|---|---|---|---|---|---|---|---|']
    hpp_reports = {}
    for p in sorted(glob.glob(f'{OUT}/reports/hpp__*.json')):
        r = json.load(open(p))
        _, clip, proj = os.path.basename(p)[:-5].split('__')
        hpp_reports[(clip, proj)] = r
        auto = {k: v for k, v in r['acceptance'].items() if v != 'manual'}
        out['hpp'][f'{clip}__{proj}'] = {'checks': r['acceptance'], 'pass_auto': all(auto.values())}
        lines.append(f"| {clip} | {proj} | {r['keys_per_trajectory']['root']} | {r['observed_joint_frames']}/{r['total_joint_frames']} "
                     f"| {_cm(r['mpjpe_vs_original_m'])} | {_cm(max(r['end_effector_err_vs_original_m'].values()))} "
                     f"| {_cm(r['root_vs_original']['mean_m'])} | {r['bone_length']['max_dev_m']:.0e} "
                     f"| {r['temporal']['contact_events']['recall']:.2f} | "
                     + ('PASS' if all(auto.values()) else 'FAIL: ' + ', '.join(k.split('_')[0] for k, v in auto.items() if not v)) + ' |')

    ref_jerk = {}
    for p in glob.glob(f'{OUT}/reports/reference__*.json'):
        r = json.load(open(p))
        ref_jerk[r['record']['clip']] = float(np.mean(r['jerk_mps3']))
    lines += ['', '## CondMDI stage (3 samples per condition; values are the mean over samples)', '',
              '| condition | constraint err vs target mean / p95 cm | vs original cm (MPJPE) | root vs target cm | foot slide m/s (orig) | min foot y cm | contact recall | jerk (ref) | velocity err vs target m/s | samples passing C1-C7 |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for p in sorted(glob.glob(f'{OUT}/reports/gen__*.json')):
        g = json.load(open(p))
        if g['name'].endswith('__repro'):
            continue
        clip, proj = g['record']['clip'], g['record']['projector']
        h = hpp_reports[(clip, proj)]
        orig_slide = h['original_reference']['foot_sliding']['mean_mps']
        checks = [gen_checks(r, acc['condmdi'], orig_slide, ref_jerk.get(clip)) for r in g['reps']]
        per = {k: sum(c[k] is True for c in checks) for k in checks[0]}
        out['generated'][g['name']] = {'per_sample': checks, 'samples_passing': per,
                                       'all_pass': all(all(v is not False for v in c.values()) for c in checks)}
        m = lambda f: float(np.mean([f(r) for r in g['reps']]))
        lines.append(
            f"| {g['name']} | {_cm(m(lambda r: r['constraint_vs_target']['mean']))} / {_cm(m(lambda r: r['constraint_vs_target']['p95']))} "
            f"| {_cm(m(lambda r: r['mpjpe_vs_original_m']))} | {_cm(m(lambda r: r['root_vs_target']['mean_m']))} "
            f"| {m(lambda r: r['foot_sliding']['mean_mps']):.2f} ({orig_slide:.2f}) | {_cm(m(lambda r: r['ground_penetration']['min_foot_height_m']))} "
            f"| {m(lambda r: r['temporal']['contact_events']['recall']):.2f} | {m(lambda r: r['jerk_mps3']):.0f} ({ref_jerk.get(clip, float('nan')):.0f}) "
            f"| {m(lambda r: r['velocity_vs_target']['err_at_constraints_mps']):.2f} "
            f"| " + ' '.join(f"{k.split('_')[0]}:{v}/{len(checks)}" for k, v in per.items()) + ' |')
    for p in glob.glob(f'{OUT}/reports/repro__*.txt'):
        d = float(open(p).read())
        out['reproducibility'][os.path.basename(p)[7:-4]] = {'max_diff_m': d,
                                                             'C8_pass': d <= acc['condmdi']['C8_reproducible_max_diff_m']}
    lines += ['', '## Reproducibility (C8)', ''] + [f"- {k}: max |diff| {v['max_diff_m']:.2e} m -> {'PASS' if v['C8_pass'] else 'FAIL'}"
                                                   for k, v in out['reproducibility'].items()]
    _json(out, f'{OUT}/reports/acceptance.json')
    with open(f'{OUT}/reports/acceptance_tables.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
