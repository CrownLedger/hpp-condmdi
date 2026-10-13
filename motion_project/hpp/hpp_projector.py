"""HPP projector R = Pi_D(M) for HumanML3D (handoff sec. 17 / 20).

This is a RECONSTRUCTED v0.2-style baseline: the original HPP v0.2 source was not available, so behaviour follows
the handoff description (sparse hand/foot keys at ~0.12 s, cubic Hermite, canonical skeleton, two-bone IK).
Every assumption beyond that description is marked in configs/hpp_*.yaml and docs/implementation_notes.md.
"""
import hashlib
import json
import numpy as np

from motion_project.hpp import skeleton as sk
from motion_project.hpp.hermite import hermite_reconstruct
from motion_project.hpp import keyframe_selection as ks
from motion_project.hpp import canonical_ik as cik

# Trajectories HPP keys explicitly: (name, joint, semantic body part, uses contact events)
TRAJECTORIES = [('root', 0, 'root', False),
                ('left_hand', 20, 'left_hand', False), ('right_hand', 21, 'right_hand', False),
                ('left_ankle', 7, 'left_foot', True), ('right_ankle', 8, 'right_foot', True),
                ('left_foot', 10, 'left_foot', True), ('right_foot', 11, 'right_foot', True)]


def _select(name, values, fps, cfg, err_fn, eps, events_fn, normalize=None):
    sel = cfg['selection']
    if sel['mode'] == 'v02_schedule':
        return ks.schedule_keys(len(values), fps, sel['interval_s'])
    if sel['mode'] == 'adaptive':
        a = sel['adaptive']
        gap = max(1, round(a['min_key_interval_s'] * fps))
        return ks.adaptive_keys(values, err_fn, eps, events_fn(), gap, normalize)
    raise ValueError(sel['mode'])


def hpp_project(positions, fps, config):
    """positions (T,22,3) world -> contract dict (handoff sec. 20). The input array is never modified."""
    M = np.array(positions, dtype=np.float64, copy=True)
    T = len(M)
    lengths = sk.reference_bone_lengths(M)
    canon = cik.canonical_torso(M, lengths)
    up, across, fwd = cik.torso_frame(M)
    a_cfg = config['selection'].get('adaptive', {})
    pos_err = lambda x, y: np.linalg.norm(x - y, axis=-1)

    observations = []
    dense = {}
    for name, j, part, contacts in TRAJECTORIES:
        traj = M[:, j]
        eps = a_cfg.get('eps_m', {}).get(name, 0.03)
        frames, reasons = _select(name, traj, fps, config, pos_err, eps,
                                  lambda: ks.position_events(traj, eps, fps, a_cfg, contacts=contacts))
        observations.append({'name': name, 'body_part': part, 'joint': j, 'joint_name': sk.JOINT_NAMES[j],
                             'variable': 'position', 'frame': 'WORLD', 'priority': 'SOFT',
                             'frames': frames, 'values': M[frames, j].tolist(), 'reasons': reasons})
        dense[name] = hermite_reconstruct(frames, M[frames, j], T)

    # Torso orientation (up + across axes), compared by angle
    tv = np.concatenate([up, across], -1)
    t_err = lambda x, y: np.maximum(ks._angle_deg(x[:, :3], y[:, :3]), ks._angle_deg(x[:, 3:], y[:, 3:]))
    def torso_events():
        yaw = np.degrees(np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2])))
        lean = ks._angle_deg(up, np.array([0.0, 1.0, 0.0]))
        ext = set()
        for sig in (yaw, -yaw, lean, -lean):
            ext.update(int(x) for x in ks.find_peaks(sig, prominence=a_cfg.get('torso_eps_deg', 8.0))[0])
        return {'start_end': [0, T - 1], 'extremum': sorted(ext)}
    tframes, treasons = _select('torso', tv, fps, config, t_err, a_cfg.get('torso_eps_deg', 8.0), torso_events)
    observations.append({'name': 'torso_orientation', 'body_part': 'torso', 'joint': None, 'variable': 'orientation',
                         'frame': 'WORLD', 'priority': 'SOFT', 'frames': tframes,
                         'values': np.concatenate([up[tframes], across[tframes]], -1).tolist(),
                         'value_layout': 'up_xyz + across_xyz (unit vectors)', 'reasons': treasons})
    up_d = hermite_reconstruct(tframes, up[tframes], T)
    across_d = hermite_reconstruct(tframes, across[tframes], T)

    rough = cik.decode(dense['root'], up_d, across_d,
                       wrist={'left': dense['left_hand'], 'right': dense['right_hand']},
                       ankle={'left': dense['left_ankle'], 'right': dense['right_ankle']},
                       foot={'left': dense['left_foot'], 'right': dense['right_foot']},
                       lengths=lengths, canon=canon, poles=config['decoder']['poles'])
    return build_output(M, rough, observations, fps, config, method='hpp_v02_style:' + config['selection']['mode'],
                        lengths=lengths)


def observation_masks(observations, T):
    """(T,22) joint-position mask, (T,22,3) per-coordinate mask, (T,) torso-orientation mask."""
    pos = np.zeros((T, 22), bool)
    orient = np.zeros(T, bool)
    for o in observations:
        if o['variable'] == 'position':
            pos[o['frames'], o['joint']] = True
        elif o['variable'] == 'orientation':
            orient[o['frames']] = True
    return pos, np.repeat(pos[..., None], 3, -1), orient


def build_output(original, rough, observations, fps, config, method, lengths):
    T = len(original)
    pos_mask, xyz_mask, orient_mask = observation_masks(observations, T)
    return {
        'original_positions': original, 'rough_positions': rough, 'selected_observations': observations,
        'observation_mask': pos_mask, 'observation_mask_xyz': xyz_mask, 'orientation_mask': orient_mask,
        'fps': fps, 'joint_names': sk.JOINT_NAMES, 'joint_parents': sk.JOINT_PARENTS,
        'coordinate_system': sk.COORDINATE_SYSTEM, 'units': 'm',
        'metadata': {'method': method, 'config': config, 'reference_bone_lengths': lengths.tolist(),
                     'config_sha1': hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                     'rough_sha1': hashlib.sha1(np.ascontiguousarray(rough).tobytes()).hexdigest(),
                     'n_frames': T, 'observed_joint_frames': int(pos_mask.sum()),
                     'observed_fraction': float(pos_mask.sum() / pos_mask.size)},
    }
