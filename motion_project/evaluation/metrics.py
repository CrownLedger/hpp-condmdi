"""Evaluation metrics (handoff sec. 33). All positions are (T,22,3) in HumanML3D world metres, 20 fps."""
import numpy as np

from motion_project.hpp import skeleton as sk
from motion_project.hpp import keyframe_selection as ks

FOOT_JOINTS = [7, 8, 10, 11]
END_EFFECTORS = {'left_wrist': 20, 'right_wrist': 21, 'left_ankle': 7, 'right_ankle': 8, 'left_foot': 10, 'right_foot': 11}


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {'mean': None, 'p95': None, 'max': None, 'n': 0}
    return {'mean': float(x.mean()), 'p95': float(np.percentile(x, 95)), 'max': float(x.max()), 'n': int(x.size)}


def constraint_error(gen, target, mask):
    """E_constraint over the actually constrained joint-frames only. mask: (T,22) bool."""
    return _stats(np.linalg.norm(gen - target, axis=-1)[mask])


def root_trajectory_error(gen, ref):
    d = gen[:, 0] - ref[:, 0]
    disp = (gen[-1, 0] - gen[0, 0]) - (ref[-1, 0] - ref[0, 0])
    return {'mean_m': float(np.linalg.norm(d, axis=-1).mean()), 'max_m': float(np.linalg.norm(d, axis=-1).max()),
            'mean_xz_m': float(np.linalg.norm(d[:, [0, 2]], axis=-1).mean()), 'mean_y_m': float(np.abs(d[:, 1]).mean()),
            'final_disp_err_m': float(np.linalg.norm(disp[[0, 2]]))}


def bone_length_error(pos, ref_lengths):
    L = sk.bone_lengths(pos)[:, 1:]
    e = np.abs(L - np.asarray(ref_lengths)[1:])
    return {'max_dev_m': float(e.max()), 'mean_dev_m': float(e.mean()), 'std_per_bone_mean_m': float(L.std(0).mean())}


def foot_sliding(pos, fps=20.0, floor=0.0, height=0.05):
    """Mean / p95 horizontal speed (m/s) of foot joints while within `height` of the floor (height-based contact,
    so sliding feet still count as in contact)."""
    sp = []
    for j in FOOT_JOINTS:
        low = (pos[:-1, j, 1] < floor + height) & (pos[1:, j, 1] < floor + height)
        v = np.linalg.norm(pos[1:, j, [0, 2]] - pos[:-1, j, [0, 2]], axis=-1) * fps
        sp.append(v[low])
    sp = np.concatenate(sp)
    return {'mean_mps': float(sp.mean()) if sp.size else 0.0, 'p95_mps': float(np.percentile(sp, 95)) if sp.size else 0.0,
            'contact_frames': int(sp.size)}


def ground_penetration(pos, floor=0.0):
    y = pos[:, FOOT_JOINTS, 1] - floor
    below = y[y < 0]
    return {'min_foot_height_m': float(y.min()), 'frames_below_floor_frac': float((y.min(1) < 0).mean()),
            'mean_depth_when_below_m': float(-below.mean()) if below.size else 0.0}


def jerk(pos, fps=20.0):
    return float(np.linalg.norm(np.diff(pos, n=3, axis=0) * fps ** 3, axis=-1).mean())


def contact_events(pos, fps=20.0):
    out = {}
    for j in FOOT_JOINTS:
        c = ks.debounce(ks.contact_flags(pos[:, j]), max(1, round(0.10 * fps)))
        out[j] = np.array(ks.transitions(c))
    return out


def direction_events(pos, fps=20.0):
    cfg = {'contact_disp_sq': 0.002, 'min_contact_s': 0.1, 'pause_speed': 0.1, 'min_pause_s': 0.25,
           'turn_window_s': 0.1, 'turn_min_speed': 0.2, 'turn_angle_deg': 45.0, 'extrema_prominence': 1e9}
    return {j: np.array(ks.position_events(pos[:, j], 1.0, fps, cfg)['direction']) for j in (0, 20, 21)}


def _match(ref_events, test_events, tol):
    hit = tot = back = tot_b = 0
    for j in ref_events:
        a, b = ref_events[j], test_events[j]
        tot += len(a); tot_b += len(b)
        hit += sum(bool(len(b)) and np.any(np.abs(b - t) <= tol) for t in a)
        back += sum(bool(len(a)) and np.any(np.abs(a - t) <= tol) for t in b)
    return {'recall': hit / tot if tot else 1.0, 'precision': back / tot_b if tot_b else 1.0,
            'ref_events': tot, 'test_events': tot_b}


def temporal_fidelity(gen, ref, fps=20.0, tol_s=0.10):
    tol = round(tol_s * fps)
    return {'duration_equal': len(gen) == len(ref), 'frames': len(gen), 'ref_frames': len(ref),
            'contact_events': _match(contact_events(ref, fps), contact_events(gen, fps), tol),
            'direction_events': _match(direction_events(ref, fps), direction_events(gen, fps), tol)}


def velocity_agreement(gen, target, mask, fps=20.0):
    """Forward-difference joint velocities (as in the HumanML3D velocity features) vs the target motion."""
    v = lambda p: np.diff(np.concatenate([p, p[-1:]]), axis=0) * fps
    e = np.linalg.norm(v(gen) - v(target), axis=-1)
    prim = [0, 7, 8, 10, 11, 20, 21]
    sg, st = np.linalg.norm(v(gen), axis=-1)[:, prim].ravel(), np.linalg.norm(v(target), axis=-1)[:, prim].ravel()
    return {'err_at_constraints_mps': float(e[mask].mean()) if mask.any() else None, 'err_all_mps': float(e.mean()),
            'speed_corr_primary': float(np.corrcoef(sg, st)[0, 1])}


def full_report(gen, original, ref_lengths, fps=20.0, target=None, mask=None, floor=0.0, contact_height=0.05, tol_s=0.10):
    """Everything for one motion. target/mask: what the motion was asked to satisfy (for constraint metrics)."""
    err = np.linalg.norm(gen - original, axis=-1)
    rep = {
        'mpjpe_vs_original_m': float(err.mean()),
        'end_effector_err_vs_original_m': {n: float(err[:, j].mean()) for n, j in END_EFFECTORS.items()},
        'root_vs_original': root_trajectory_error(gen, original),
        'bone_length': bone_length_error(gen, ref_lengths),
        'foot_sliding': foot_sliding(gen, fps, floor, contact_height),
        'ground_penetration': ground_penetration(gen, floor),
        'temporal': temporal_fidelity(gen, original, fps, tol_s),
        'jerk_mps3': jerk(gen, fps),
        'finite': bool(np.isfinite(gen).all()),
    }
    if target is not None and mask is not None:
        rep['constraint_vs_target'] = constraint_error(gen, target, mask)
        rep['root_vs_target'] = root_trajectory_error(gen, target)
        rep['velocity_vs_target'] = velocity_agreement(gen, target, mask, fps)
        rep['constraint_vs_original'] = constraint_error(gen, original, mask)
    return rep
