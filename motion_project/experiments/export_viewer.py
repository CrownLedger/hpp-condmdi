"""Export outputs/ to the comparison viewer's data files.

python -m motion_project.experiments.export_viewer
-> outputs/viewer_data/index.json + <clip>__<projector>.json (served as viewer/data/ via a symlink)
"""
import glob
import json
import os
import numpy as np

from motion_project.experiments.common import OUT, load_cfg, _json
from motion_project.evaluation.acceptance import gen_checks

LABEL = {'hpp_v02': 'HPP v0.2-style baseline (0.12 s schedule, Hermite + IK)',
         'hpp_adaptive': 'HPP adaptive keys (extension, Hermite + IK)',
         'robotic_slerp': 'Robotic slerp (EXPERIMENTAL, not HPP)',
         'direct': 'Direct robotic input (CMU mocap, no rougher; full body every 0.5 s)',
         'direct_1s': 'Direct robotic input (CMU mocap, no rougher; full body every 1 s)',
         'direct_1p5s': 'Direct robotic input (CMU mocap, no rougher; full body every 1.5 s)'}
EXP_LABEL = {'A': 'Exp A · real features at HPP entries', 'B': 'Exp B · rough-motion features only'}


def frames(x):
    return np.round(np.asarray(x), 4).tolist()


def per_frame(mask_jt):            # (22,T) -> list of joint lists per frame
    return [np.nonzero(mask_jt[:, t])[0].tolist() for t in range(mask_jt.shape[1])]


def summarize(r):
    return {'constraint_mean': r['constraint_vs_target']['mean'], 'constraint_p95': r['constraint_vs_target']['p95'],
            'mpjpe': r['mpjpe_vs_original_m'], 'root_vs_target': r['root_vs_target']['mean_m'],
            'foot_slide': r['foot_sliding']['mean_mps'], 'min_foot': r['ground_penetration']['min_foot_height_m'],
            'contact_recall': r['temporal']['contact_events']['recall'], 'jerk': r['jerk_mps3'],
            'vel_err': r['velocity_vs_target']['err_at_constraints_mps'],
            'speed_corr': r['velocity_vs_target']['speed_corr_primary'],
            'bone_dev': r['bone_length']['max_dev_m']}


def main():
    acc = load_cfg('acceptance')
    vd = f'{OUT}/viewer_data'
    os.makedirs(vd, exist_ok=True)
    ref_jerk = {}
    for p in glob.glob(f'{OUT}/reports/reference__*.json'):
        r = json.load(open(p)); ref_jerk[r['record']['clip']] = float(np.mean(r['jerk_mps3']))
    index = []
    for p in sorted(glob.glob(f'{OUT}/rough/*.npz')):
        clip, proj = os.path.basename(p)[:-4].split('__')
        o = np.load(f'{OUT}/original/{clip}.npz'); oj = json.load(open(f'{OUT}/original/{clip}.json'))
        r = np.load(p); rj = json.load(open(p[:-4] + '.json'))
        h = json.load(open(f'{OUT}/reports/hpp__{clip}__{proj}.json'))
        orig_slide = h['original_reference']['foot_sliding']['mean_mps']
        item = {'clip': clip, 'projector': proj, 'label': LABEL.get(proj, proj), 'caption': oj['captions'][0] if oj['captions'] else '',
                'source': oj['source'], 'fps': float(o['fps']), 'frames': int(len(o['positions'])),
                'original': frames(o['positions']), 'rough': frames(r['rough_positions']),
                'obs_per_frame': per_frame(r['observation_mask'].T),
                'observations': [{'name': x['name'], 'joint': x['joint'], 'variable': x['variable'], 'frames': x['frames'],
                                  'reasons': x['reasons']} for x in rj['selected_observations']],
                'hpp': {'checks': h['acceptance'], 'mpjpe': h['mpjpe_vs_original_m'], 'observed': h['observed_joint_frames'],
                        'total': h['total_joint_frames'], 'keys': h['keys_per_trajectory'],
                        'rough_summary': {'foot_slide': h['foot_sliding']['mean_mps'], 'min_foot': h['ground_penetration']['min_foot_height_m'],
                                          'contact_recall': h['temporal']['contact_events']['recall'], 'jerk': h['jerk_mps3'],
                                          'mpjpe': h['mpjpe_vs_original_m'], 'bone_dev': h['bone_length']['max_dev_m']},
                        'original_summary': {'foot_slide': orig_slide, 'jerk': h['original_reference']['jerk_mps3'],
                                             'min_foot': h['original_reference']['ground_penetration']['min_foot_height_m']}},
                'ref_jerk': ref_jerk.get(clip), 'conditions': []}
        for g in sorted(glob.glob(f'{OUT}/generated/{clip}__{proj}__*.npz')):
            name = os.path.basename(g)[:-4]
            if name.endswith('__repro'):
                continue
            exp, rp = name.split('__')[2:4]
            d = np.load(g); rep = json.load(open(f'{OUT}/reports/gen__{name}.json'))
            item['conditions'].append({
                'name': name, 'experiment': exp, 'root_policy': rp, 'target': rep['record']['obs_source'],
                'label': f"{EXP_LABEL[exp]} · root {'every frame' if rp == 'dense' else 'on observed frames'}",
                'reps': [frames(x) for x in d['generated']], 'constrained_per_frame': per_frame(d['joint_mask']),
                'metrics': [summarize(x) for x in rep['reps']],
                'checks': [gen_checks(x, acc['condmdi'], orig_slide, ref_jerk.get(clip)) for x in rep['reps']],
                'record': {k: rep['record'][k] for k in ('seeds', 'checkpoint_sha256', 'feature_mode', 'observed_joint_frames', 'mask_counts', 'imputate', 'text')}})
        ref = f'{OUT}/generated/{clip}__reference_fullbody_every5.npz'
        if os.path.exists(ref):
            d = np.load(ref)
            item['conditions'].append({'name': 'reference', 'experiment': 'ref', 'root_policy': '-', 'target': 'original',
                                       'label': 'Reference · CondMDI benchmark: full body of the ORIGINAL every 5 frames',
                                       'reps': [frames(x) for x in d['generated']], 'constrained_per_frame': per_frame(d['joint_mask']),
                                       'metrics': None, 'checks': None, 'record': None})
        fn = f'{clip}__{proj}.json'
        _json(item, f'{vd}/{fn}')
        index.append({'clip': clip, 'projector': proj, 'label': item['label'], 'caption': item['caption'], 'file': fn,
                      'conditions': len(item['conditions'])})
    order = {'hpp_v02': 0, 'hpp_adaptive': 1, 'robotic_slerp': 2}
    index.sort(key=lambda x: (x['clip'], order.get(x['projector'], 9)))
    _json({'items': index, 'tolerances': acc}, f'{vd}/index.json')
    print('exported', [(x['clip'], x['projector'], x['conditions']) for x in index])


if __name__ == '__main__':
    main()
