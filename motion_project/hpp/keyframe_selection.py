"""Sparse perceptual key selection (handoff sec. 11).

Two selection modes, chosen by config:

* ``v02_schedule`` - BASELINE ("reconstructed v0.2-style"). The only selection rule known from v0.2 is a time-based
  sampling interval of ~0.12 s for sparse hand and foot trajectories. Keys are placed at t_k = k * interval seconds,
  i.e. frame round(k * interval * fps), plus the last frame. At 20 fps, 0.12 s = 2.4 frames, so keys fall 2 or 3
  frames apart (mean spacing exactly 2.4 frames - time-based, not a fixed frame count).
  ASSUMPTION (unverified): the same schedule is used for root and torso orientation.

* ``adaptive`` - EXTENSION, not part of verified v0.2 (handoff sec. 11 lists these as future rules): start/end,
  contact/release events, motion pauses, direction changes and prominent extrema, then error-driven insertion until
  the Hermite reconstruction is within a per-trajectory tolerance, with a minimum key spacing.

Both are deterministic. Every key carries the reason it was selected.
"""
import numpy as np
from scipy.signal import find_peaks

from motion_project.hpp.hermite import hermite_reconstruct


# ------------------------------------------------------------------------------------------- v0.2 schedule
def schedule_keys(n_frames, fps, interval_s):
    t = np.arange(0.0, (n_frames - 1) / fps + 1e-9, interval_s)
    frames = sorted(set(int(round(x * fps)) for x in t) | {n_frames - 1})
    frames = [f for f in frames if 0 <= f < n_frames]
    reasons = {f: ('start_end' if f in (0, n_frames - 1) else 'schedule') for f in frames}
    return frames, reasons


# ------------------------------------------------------------------------------------------- adaptive extension
def _angle_deg(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def debounce(flags, min_len):
    """Remove interior runs shorter than min_len frames (absorbed by the preceding state)."""
    f = flags.copy()
    if min_len <= 1:
        return f
    changed = True
    while changed:
        changed = False
        edges = np.concatenate([[0], np.nonzero(f[1:] != f[:-1])[0] + 1, [len(f)]])
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < min_len and 0 < a and b < len(f):
                f[a:b] = f[a - 1]
                changed = True
                break
    return f


def transitions(flags):
    return [int(t) for t in np.nonzero(flags[1:] != flags[:-1])[0] + 1]


def contact_flags(traj, disp_sq=0.002):
    """HumanML3D foot-contact rule: squared per-frame displacement below threshold."""
    d = ((traj[1:] - traj[:-1]) ** 2).sum(-1) < disp_sq
    return np.concatenate([d, d[-1:]])


def position_events(traj, eps, fps, cfg, contacts=False):
    T = len(traj)
    vel = np.gradient(traj, axis=0) * fps
    speed = np.linalg.norm(vel, axis=-1)
    ev = {'start_end': [0, T - 1]}
    if contacts:
        c = debounce(contact_flags(traj, cfg['contact_disp_sq']), max(1, round(cfg['min_contact_s'] * fps)))
        ev['contact'] = transitions(c)
    still = debounce(speed < cfg['pause_speed'], max(1, round(cfg['min_contact_s'] * fps)))
    edges = [0] + transitions(still) + [T]
    ev['pause'] = [x for a, b in zip(edges[:-1], edges[1:]) if still[a] and b - a >= cfg['min_pause_s'] * fps
                   for x in (a, b - 1)]
    k = max(1, round(cfg['turn_window_s'] * fps))
    ang = np.zeros(T)
    if T > 2 * k:
        a, b = vel[:-2 * k], vel[2 * k:]
        moving = (np.linalg.norm(a, axis=-1) > cfg['turn_min_speed']) & (np.linalg.norm(b, axis=-1) > cfg['turn_min_speed'])
        ang[k:T - k] = np.where(moving, _angle_deg(a, b), 0.0)
    ev['direction'] = [int(t) for t in find_peaks(ang, height=cfg['turn_angle_deg'])[0]]
    ext = set()
    for ax in range(traj.shape[1]):
        for sgn in (1, -1):
            ext.update(int(t) for t in find_peaks(sgn * traj[:, ax], prominence=cfg['extrema_prominence'] * eps)[0])
    ev['extremum'] = sorted(ext)
    return ev


PRIORITY = {'start_end': 0, 'contact': 1, 'extremum': 2, 'direction': 3, 'pause': 4}


def adaptive_keys(values, err_fn, eps, events, gap, normalize=None):
    """Accept events greedily by priority with a minimum spacing, then insert the worst-reconstructed frame until the
    error is <= eps everywhere or no frame is at least `gap` frames from existing keys."""
    T = len(values)
    keys = []
    for _, f in sorted({(PRIORITY[k], int(f)) for k, fs in events.items() for f in fs}):
        if f in (0, T - 1) or all(abs(f - x) >= gap for x in keys):
            if f not in keys:
                keys.append(f)
    keys.sort()
    reasons = {}
    for kind in sorted(events, key=lambda k: -PRIORITY[k]):          # most important reason wins
        for f in events[kind]:
            if f in keys:
                reasons[f] = kind
    while True:
        rec = hermite_reconstruct(keys, values[keys], T)
        if normalize is not None:
            rec = normalize(rec)
        err = err_fn(values, rec)
        far = np.ones(T, bool)
        for x in keys:
            far[max(0, x - gap + 1):x + gap] = False
        bad = np.where(far & (err > eps), err, -1.0)
        if bad.max() < 0:
            return keys, {int(k): reasons.get(k, 'error') for k in keys}
        f = int(np.argmax(bad))
        keys = sorted(keys + [f])
        reasons[f] = 'error'
