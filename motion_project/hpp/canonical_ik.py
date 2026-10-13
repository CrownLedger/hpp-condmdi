"""Canonical skeleton reconstruction (handoff sec. 13): rigid canonical torso + two-bone IK.

Connectivity is guaranteed by construction: every joint is placed as parent + direction * reference bone length,
so no subtree can detach (the historical "translate the child subtree after IK" fix is not needed).
"""
import numpy as np

from motion_project.hpp.skeleton import (JOINT_PARENTS, ROOT, TORSO, ARMS, LEGS, J)

NECK = J['neck']
L_HIP, R_HIP, L_SHOULDER, R_SHOULDER = J['left_hip'], J['right_hip'], J['left_shoulder'], J['right_shoulder']


def normalize(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def torso_frame(pos):
    """Body frame per frame: up (pelvis->neck), across (towards the body's right, from hips + shoulders as in
    HumanML3D's facing rule, made orthogonal to up) and forward = up x across."""
    up = normalize(pos[:, NECK] - pos[:, ROOT])
    across = (pos[:, R_HIP] - pos[:, L_HIP]) + (pos[:, R_SHOULDER] - pos[:, L_SHOULDER])
    across = normalize(across - (across * up).sum(-1, keepdims=True) * up)
    return up, across, normalize(np.cross(up, across))


def frame_matrix(up, across):
    across = normalize(across - (across * up).sum(-1, keepdims=True) * up)
    fwd = normalize(np.cross(up, across))
    return np.stack([across, up, fwd], axis=-1), across, fwd   # columns = axes


def canonical_torso(pos, lengths):
    """Median torso shape in the body frame, rebuilt along the tree with exact reference bone lengths."""
    up, across, _ = torso_frame(pos)
    R, _, _ = frame_matrix(up, across)
    local = np.einsum('tik,tji->tjk', R, pos - pos[:, :1])         # R^T (p - pelvis)
    med = np.median(local, axis=0)
    canon = np.zeros((22, 3))
    for j in TORSO[1:]:                                             # parent-first order
        p = JOINT_PARENTS[j]
        canon[j] = canon[p] + normalize(med[j] - med[p]) * lengths[j]
    return canon


def two_bone_ik(a, target, l1, l2, pole):
    """Anchor a (shoulder/hip), end target, segment lengths, canonical bend direction -> (middle joint, end).
    Unreachable targets are clamped along a->target so bone lengths stay exact; the bend is chosen in the plane of
    a->target and the pole (deterministic canonical choice among the circle of valid solutions)."""
    d_vec = target - a
    d = np.clip(np.linalg.norm(d_vec, axis=-1), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    dirn = normalize(d_vec)
    x = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(np.maximum(l1 ** 2 - x ** 2, 0.0))
    n = normalize(pole - (pole * dirn).sum(-1, keepdims=True) * dirn)
    return a + dirn * x[:, None] + n * h[:, None], a + dirn * d[:, None]


def decode(root, up, across, wrist, ankle, foot, lengths, canon, poles):
    """Deterministic decoder: dense root / torso-orientation / end-effector trajectories -> (T,22,3) skeleton.
    wrist/ankle/foot: dicts side -> (T,3) targets. poles: canonical bend weights from the config."""
    T = len(root)
    R, across, fwd = frame_matrix(normalize(up), across)
    out = np.zeros((T, 22, 3))
    out[:, ROOT] = root
    for j in TORSO[1:]:
        out[:, j] = root + np.einsum('tij,j->ti', R, canon[j])
    for side, (s, e, w) in ARMS.items():
        outward = across if side == 'right' else -across
        pa = poles['elbow']
        pole = normalize(pa['forward'] * fwd + pa['up'] * up + pa['outward'] * outward)
        out[:, e], out[:, w] = two_bone_ik(out[:, s], wrist[side], lengths[e], lengths[w], pole)
    for side, (h, k, a, f) in LEGS.items():
        outward = across if side == 'right' else -across
        pk = poles['knee']
        pole = normalize(pk['forward'] * fwd + pk['up'] * up + pk['outward'] * outward)
        out[:, k], out[:, a] = two_bone_ik(out[:, h], ankle[side], lengths[k], lengths[a], pole)
        toe = foot[side] - out[:, a]
        degenerate = np.linalg.norm(toe, axis=-1) < 1e-6
        toe[degenerate] = fwd[degenerate]
        out[:, f] = out[:, a] + normalize(toe) * lengths[f]
    return out
