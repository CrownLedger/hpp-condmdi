"""EXPERIMENTAL, NOT the HPP baseline: robotic rough motion by per-joint SLERP between key poses.

Added on explicit user request. It contradicts the handoff's statement that HPP is not intended to be robotic
(sec. 6), so it is kept out of the baseline and labelled `robotic_slerp` everywhere.

Local joint rotations are recovered from positions in HumanML3D's FK convention and tracked continuously in time
(root = unwrapped yaw; each bone advanced by the frame-to-frame minimal rotation, so slerp never meets a
180-degree representation flip). All joints share key poses; each joint's rotation is slerped with linear time
(constant angular velocity) and the root position is lerped. FK with fixed bone lengths rebuilds positions.
"""
import numpy as np

from motion_project.hpp import skeleton as sk
from motion_project.hpp.hpp_projector import build_output, TRAJECTORIES
from motion_project.hpp.canonical_ik import normalize as _normalize

RAW_OFFSETS = sk.RAW_OFFSETS
KINEMATIC_CHAINS = sk.KINEMATIC_CHAINS
R_HIP, L_HIP, R_SHOULDER, L_SHOULDER = 2, 1, 17, 16

def q_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def q_inv(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def q_rot(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return q_mul(q_mul(q, qv), q_inv(q))[..., 1:]


def q_between(u, v):
    """Minimal rotation taking unit vector u to unit vector v (w, x, y, z)."""
    w = 1.0 + (u * v).sum(-1, keepdims=True)
    q = np.concatenate([w, np.cross(u, v)], -1)
    # opposite vectors: rotate 180 deg about any axis perpendicular to u
    opp = w[..., 0] < 1e-8
    if np.any(opp):
        alt = np.cross(u[opp], np.array([1.0, 0.0, 0.0]))
        alt[np.linalg.norm(alt, axis=-1) < 1e-6] = np.cross(u[opp][np.linalg.norm(alt, axis=-1) < 1e-6], [0.0, 0.0, 1.0])
        q[opp] = np.concatenate([np.zeros((opp.sum(), 1)), _normalize(alt)], -1)
    return _normalize(q)


def slerp(q0, q1, u):
    """Spherical linear interpolation, shortest arc, constant angular velocity in u in [0,1]."""
    u = np.asarray(u, dtype=np.float64)[..., None]
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-6
    w0 = np.where(small, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(small, 1.0, s))
    w1 = np.where(small, u, np.sin(u * theta) / np.where(small, 1.0, s))
    return _normalize(w0 * q0 + w1 * q1)


def local_rotations(pos):
    """Positions (T,22,3) -> local quaternions (T,22,4) in HumanML3D's FK convention (each kinematic chain
    starts from the root rotation; joint 0 holds the root rotation).

    Rotations are tracked continuously through time so slerp never sees a representation flip: the root is a
    pure yaw from the unwrapped facing angle, and every bone's world rotation is advanced frame-to-frame by the
    minimal rotation between consecutive bone directions (a per-frame shortest-rotation decomposition would
    flip by 180 deg whenever a bone points opposite its rest direction, e.g. when the body turns around)."""
    T = len(pos)
    across = (pos[:, R_HIP] - pos[:, L_HIP]) + (pos[:, R_SHOULDER] - pos[:, L_SHOULDER])
    fwd = np.cross(np.array([0.0, 1.0, 0.0]), across)
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    root_q = np.stack([np.cos(yaw / 2), np.zeros(T), np.sin(yaw / 2), np.zeros(T)], -1)   # rotates +Z to forward
    q = np.zeros((T, 22, 4)); q[..., 0] = 1.0
    q[:, 0] = root_q
    for chain in KINEMATIC_CHAINS:
        R_prev = root_q
        for a, b in zip(chain[:-1], chain[1:]):
            v = _normalize(pos[:, b] - pos[:, a])
            u = RAW_OFFSETS[b]
            R = np.zeros((T, 4))
            R[0] = q_mul(R_prev[0], q_between(u, q_rot(q_inv(R_prev[0]), v[0])))
            for t in range(1, T):
                R[t] = _normalize(q_mul(q_between(v[t - 1], v[t]), R[t - 1]))
            loc = q_mul(q_inv(R_prev), R)
            # keep consecutive quaternions in the same hemisphere
            sign = np.cumprod(np.where((loc[1:] * loc[:-1]).sum(-1) < 0, -1.0, 1.0))
            loc[1:] *= sign[:, None]
            q[:, b] = loc
            R_prev = R
    return q


def forward_kinematics(q, root_pos, lengths):
    T = len(q)
    pos = np.zeros((T, 22, 3))
    pos[:, 0] = root_pos
    for chain in KINEMATIC_CHAINS:
        R = q[:, 0]
        for a, b in zip(chain[:-1], chain[1:]):
            R = q_mul(R, q[:, b])
            pos[:, b] = pos[:, a] + q_rot(R, np.repeat(RAW_OFFSETS[b][None] * lengths[b], T, 0))
    return pos


def slerp_track(keys, values, T):
    """Per-key quaternions (K,...,4) -> (T,...,4) by piecewise slerp with linear time."""
    out = np.zeros((T,) + values.shape[1:])
    keys = list(keys)
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        u = (np.arange(a, b + 1) - a) / (b - a)
        u = u.reshape((-1,) + (1,) * (values.ndim - 2))
        out[a:b + 1] = slerp(values[i][None], values[i + 1][None], np.broadcast_to(u, (len(u),) + values.shape[1:-1]))
    if len(keys) == 1:
        out[:] = values[0]
    return out


def lerp_track(keys, values, T):
    return np.stack([np.interp(np.arange(T), keys, values[:, d]) for d in range(values.shape[1])], -1)


    @property
    def min_gap(self):
        return max(1, int(round(self.min_key_interval_s * self.fps)))


def _adaptive(T, gap, err_of_keys, eps):
    keys = [0, T - 1]
    while True:
        err = err_of_keys(keys)
        far = np.ones(T, bool)
        for k in keys:
            far[max(0, k - gap + 1):k + gap] = False
        bad = np.where(far & (err > eps), err, -1.0)
        if bad.max() < 0:
            return keys, err
        keys = sorted(keys + [int(np.argmax(bad))])


def _q_angle(a, b):
    return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0.0, 1.0)))


def project_robotic_slerp(positions, fps, config):
    M = np.array(positions, dtype=np.float64, copy=True)
    T = len(M)
    c = config['robotic_slerp']
    gap = max(1, round(c['min_key_interval_s'] * fps))
    lengths = sk.reference_bone_lengths(M)
    Q = local_rotations(M)
    root = M[:, 0]

    def recon(keys):
        return forward_kinematics(slerp_track(keys, Q[keys], T), lerp_track(keys, root[keys], T), lengths)
    keys, _ = _adaptive(T, gap, lambda k: np.linalg.norm(recon(k) - M, axis=-1).max(-1), c['eps_pos_m'])
    rough = recon(keys)
    reasons = {int(f): ('start_end' if f in (0, T - 1) else 'key_pose') for f in keys}
    observations = [{'name': name, 'body_part': part, 'joint': j, 'joint_name': sk.JOINT_NAMES[j],
                     'variable': 'position', 'frame': 'WORLD', 'priority': 'SOFT', 'frames': keys,
                     'values': M[keys, j].tolist(), 'reasons': reasons} for name, j, part, _ in TRAJECTORIES]
    observations.append({'name': 'torso_orientation', 'body_part': 'torso', 'joint': None, 'variable': 'orientation',
                         'frame': 'WORLD', 'priority': 'SOFT', 'frames': keys, 'values': [], 'reasons': reasons,
                         'note': 'robotic mode keys full poses (all joint rotations) at these frames'})
    return build_output(M, rough, observations, fps, config, method='robotic_slerp (experimental)', lengths=lengths)
