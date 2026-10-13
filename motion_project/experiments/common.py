"""Shared plumbing for the experiment scripts: config loading, projection, CondMDI runs, scoring, file layout.

Outputs (motion_project/outputs/):
  original/<clip>.npz                         positions, abs features, captions
  rough/<clip>__<projector>.npz / .json       rough motion + masks / selected observations + metadata
  generated/<clip>__<projector>__<exp>__<root_policy>.npz / .json     realized motions / run record
  reports/hpp__<clip>__<projector>.json, gen__....json, reference__<clip>.json, acceptance.json
"""
import json
import os
import time
import numpy as np
import yaml

from motion_project.hpp.humanml3d_loader import load_humanml3d_motion
from motion_project.hpp.hpp_projector import hpp_project
from motion_project.hpp.robotic_slerp import project_robotic_slerp
from motion_project.hpp.direct_input import project_direct
from motion_project.realizer import motion_feature_adapter as mfa
from motion_project.realizer import condmdi_mask_adapter as cma
from motion_project.evaluation import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
CONFIGS = os.path.join(ROOT, 'configs')


def load_cfg(name):
    with open(os.path.join(CONFIGS, f'{name}.yaml')) as f:
        return yaml.safe_load(f)


def _json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    tmp = f'{path}.tmp{os.getpid()}'                # atomic: the viewer may read while results are being exported
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=conv)
    os.replace(tmp, path)


def project(positions, fps, projector):
    cfg = load_cfg(projector)
    if projector == 'robotic_slerp':
        return project_robotic_slerp(positions, fps, cfg)
    if projector.startswith('direct'):
        return project_direct(positions, fps, cfg)
    return hpp_project(positions, fps, cfg)


# ---------------------------------------------------------------------------------------------- HPP stage
def run_hpp_stage(clip, projector):
    acc = load_cfg('acceptance')
    m = load_humanml3d_motion(clip)
    orig = m['positions']
    np.savez(f'{OUT}/original/{clip}.npz', positions=orig, features_abs=m['features_abs'], fps=m['fps'])
    _json({k: m[k] for k in ('motion_id', 'captions', 'source', 'fps', 'skeleton')}, f'{OUT}/original/{clip}.json')

    res = project(orig, m['fps'], projector)
    again = project(orig, m['fps'], projector)
    deterministic = bool(np.array_equal(res['rough_positions'], again['rough_positions']))
    assert np.array_equal(res['original_positions'], orig) and np.array_equal(m['positions'], orig)

    rough, mask = res['rough_positions'], res['observation_mask']
    np.savez(f'{OUT}/rough/{clip}__{projector}.npz', rough_positions=rough, observation_mask=mask,
             observation_mask_xyz=res['observation_mask_xyz'], orientation_mask=res['orientation_mask'])
    _json({k: res[k] for k in ('selected_observations', 'fps', 'joint_names', 'joint_parents', 'coordinate_system',
                               'units', 'metadata')}, f'{OUT}/rough/{clip}__{projector}.json')

    ref_len = np.array(res['metadata']['reference_bone_lengths'])
    rep = metrics.full_report(rough, orig, ref_len, m['fps'], target=orig, mask=mask,
                              floor=acc['floor_height_m'], contact_height=acc['contact_height_m'], tol_s=acc['event_tolerance_s'])
    rep['keys_per_trajectory'] = {o['name']: len(o['frames']) for o in res['selected_observations']}
    rep['observed_joint_frames'] = int(mask.sum()); rep['total_joint_frames'] = int(mask.size)
    rep['deterministic'] = deterministic
    rep['original_reference'] = metrics.full_report(orig, orig, ref_len, m['fps'], floor=acc['floor_height_m'],
                                                    contact_height=acc['contact_height_m'], tol_s=acc['event_tolerance_s'])
    rep['acceptance'] = check_hpp(rep, acc['hpp'])
    rep['method'] = res['metadata']['method']
    _json(rep, f'{OUT}/reports/hpp__{clip}__{projector}.json')
    return rep


def check_hpp(rep, a):
    ee = rep['end_effector_err_vs_original_m']
    c = {
        'H1_duration_equal': rep['temporal']['duration_equal'],
        'H2_connectivity_bone_length': bool(rep['bone_length']['max_dev_m'] <= a['H2_bone_length_max_dev_m']),
        'H3_root_trajectory': bool(rep['root_vs_original']['mean_m'] <= a['H3_root_mean_err_m']
                                   and rep['root_vs_original']['final_disp_err_m'] <= a['H3_root_final_disp_err_m']),
        'H4_end_effectors': bool(max(ee.values()) <= a['H4_end_effector_mean_err_m']),
        'H5_key_reproduction': bool(rep['constraint_vs_target']['mean'] <= a['H5_key_reproduction_mean_err_m']),
        'H6_contact_timing': bool(rep['temporal']['contact_events']['recall'] >= a['H6_contact_recall_min']),
        'H7_deterministic': rep['deterministic'],
        'H8_recognizable': 'manual',
    }
    return c


# ---------------------------------------------------------------------------------------------- CondMDI stage
def load_stage_inputs(clip, projector):
    o = np.load(f'{OUT}/original/{clip}.npz')
    r = np.load(f'{OUT}/rough/{clip}__{projector}.npz')
    meta = json.load(open(f'{OUT}/rough/{clip}__{projector}.json'))
    return o, r, meta


def run_condition(realizer, clip, projector, experiment, root_policy, ccfg, tag=''):
    """Experiment A (obs_source=original) or B (obs_source=rough) for one root policy."""
    acc = load_cfg('acceptance')
    exp_cfg = load_cfg('experiment')['experiments'][experiment]
    o, r, meta = load_stage_inputs(clip, projector)
    original, rough = o['positions'], r['rough_positions']
    if exp_cfg['obs_source'] == 'original':
        source_feat, target = o['features_abs'], original          # Exp A: real values, HPP-selected entries only
    else:
        source_feat, target = mfa.positions_to_abs_features(rough), rough   # Exp B: rough motion only
    joint_mask, prov = cma.build_joint_mask(r['observation_mask'], r['orientation_mask'], root_policy,
                                            ccfg['torso_orientation_as'])
    t0 = time.time()
    gen, record = realizer.realize(source_feat, joint_mask, ccfg['seed'], ccfg['num_repetitions'], ccfg['text'],
                                   ccfg['imputate'], ccfg['feature_mode'])
    record.update({'clip': clip, 'projector': projector, 'experiment': experiment, 'obs_source': exp_cfg['obs_source'],
                   'root_policy': root_policy, 'torso_orientation_as': ccfg['torso_orientation_as'],
                   'seconds': round(time.time() - t0, 1),
                   'mask_counts': {'hpp_position_observations': int((prov == 1).sum()),
                                   'root_added': int((prov == 2).sum()), 'torso_as_joint': int((prov == 3).sum())}})
    name = f'{clip}__{projector}__{experiment}__{root_policy}{tag}'
    np.savez(f'{OUT}/generated/{name}.npz', generated=gen, joint_mask=joint_mask, provenance=prov)
    _json(record, f'{OUT}/generated/{name}.json')

    ref_len = np.array(meta['metadata']['reference_bone_lengths'])
    hpp_mask = (prov == 1).T                                           # HPP-selected position observations
    reps = [metrics.full_report(g, original, ref_len, 20.0, target=target, mask=hpp_mask, floor=acc['floor_height_m'],
                                contact_height=acc['contact_height_m'], tol_s=acc['event_tolerance_s']) for g in gen]
    for rp, g in zip(reps, gen):
        rp['constraint_vs_target_incl_root'] = metrics.constraint_error(g, target, joint_mask.T)
    rep = {'name': name, 'record': record, 'reps': reps}
    _json(rep, f'{OUT}/reports/gen__{name}.json')
    return rep


def run_reference(realizer, clip, ccfg, every=5):
    """CondMDI's own benchmark setting: all joints of the original every `every` frames. Baseline for jerk (C7)."""
    o = np.load(f'{OUT}/original/{clip}.npz')
    T = len(o['positions'])
    jm = np.zeros((22, T), bool)
    jm[:, ::every] = True
    gen, record = realizer.realize(o['features_abs'], jm, ccfg['seed'], ccfg['num_repetitions'], '', False, ccfg['feature_mode'])
    record.update({'clip': clip, 'reference': f'full body every {every} frames from the original'})
    np.savez(f'{OUT}/generated/{clip}__reference_fullbody_every{every}.npz', generated=gen, joint_mask=jm)
    rep = {'record': record, 'jerk_mps3': [metrics.jerk(g) for g in gen],
           'constraint_vs_original': [metrics.constraint_error(g, o['positions'], jm.T) for g in gen]}
    _json(rep, f'{OUT}/reports/reference__{clip}.json')
    return rep
