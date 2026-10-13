"""Joint positions <-> CondMDI's 263-d HumanML3D features (handoff sec. 27, Phase 6).

Feature layout (verified against data_loaders/humanml_utils.py MAT_POS/MAT_ROT/MAT_VEL/MAT_CNT):
    [0]        root yaw angle            (abs_3d: absolute, accumulated; relative format: yaw velocity)
    [1:3]      root x, z                 (abs_3d: absolute world position; relative format: local velocity)
    [3]        root height y
    [4:67]     ric: 21 joints x 3, root-relative positions in the root's heading frame
    [67:193]   rot: 21 joints x 6, continuous-6D local joint rotations
    [193:259]  vel: 22 joints x 3, per-frame joint displacement in the root's heading frame
    [259:263]  foot contacts (L ankle, L foot, R ankle, R foot)

Feature definitions follow dataset/HumanML3D_abs/motion_representation.ipynb (process_file_abs_root), using
CondMDI's own Skeleton / quaternion code. Only positions are decoded by CondMDI (recover_from_ric), so the
round trip positions -> features -> positions is exact by construction for ric + root; the rot/vel/contact blocks
are derived (not fabricated) from the same positions with the dataset's IK and finite differences.

One deliberate deviation from the notebook: the per-frame root rotation is sign-normalised (w >= 0) before the yaw
velocity is read with arcsin, which fixes a heading flip whenever the body turns through +-180 deg.
"""
import numpy as np
import torch

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import (qrot_np, qmul_np, qinv_np, qbetween_np,
                                                    quaternion_to_cont6d_np)
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos, recover_from_ric

FACE_JOINT_IDX = [2, 1, 17, 16]      # r_hip, l_hip, r_shoulder, l_shoulder (notebook)
FID_L, FID_R = [7, 10], [8, 11]
FEET_THRE = 0.002                    # notebook value (squared displacement per frame)


def _relative_features(pos, feet_thre=FEET_THRE):
    """Notebook feature extraction (no normalisation). pos: (N,22,3) -> (N-1, 263) relative-root features."""
    global_positions = pos.copy()
    fl = (((pos[1:, FID_L] - pos[:-1, FID_L]) ** 2).sum(-1) < feet_thre).astype(np.float32)
    fr = (((pos[1:, FID_R] - pos[:-1, FID_R]) ** 2).sum(-1) < feet_thre).astype(np.float32)

    skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    quat_params = skel.inverse_kinematics_np(pos, FACE_JOINT_IDX, smooth_forward=True)
    cont_6d = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()
    velocity = qrot_np(r_rot[1:], (pos[1:, 0] - pos[:-1, 0]).copy())
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    # DEVIATION from the notebook (bug fix): q and -q are the same rotation, but arcsin(q_y) below only gives the
    # per-frame half-angle for w >= 0. When the facing direction crosses +-180 deg the notebook's r_velocity
    # comes out as the -358 deg form and the yaw velocity gets the wrong sign, so decoded positions drift.
    r_velocity = r_velocity * np.where(r_velocity[:, :1] < 0, -1.0, 1.0)

    local = pos.copy()
    local[..., 0] -= local[:, 0:1, 0]
    local[..., 2] -= local[:, 0:1, 2]
    local = qrot_np(np.repeat(r_rot[:, None], local.shape[1], axis=1), local)

    root_data = np.concatenate([np.arcsin(r_velocity[:, 2:3]), velocity[:, [0, 2]], local[:-1, 0, 1:2]], axis=-1)
    rot_data = cont_6d[:, 1:].reshape(len(cont_6d), -1)
    ric_data = local[:, 1:].reshape(len(local), -1)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], pos.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1]).reshape(len(pos) - 1, -1)
    return np.concatenate([root_data, ric_data[:-1], rot_data[:-1], local_vel, fl, fr], axis=-1)


def relative_to_abs_features(feat, root_xz0=(0.0, 0.0)):
    """Relative-root -> absolute-root (abs_3d) features, as in the notebook; root_xz0 anchors the trajectory."""
    _, r_pos, rot_ang = recover_root_rot_pos(torch.from_numpy(np.asarray(feat)).float(), return_rot_ang=True)
    out = np.array(feat, dtype=np.float32, copy=True)
    out[:, 0] = rot_ang.numpy()
    out[:, [1, 2]] = r_pos[:, [0, 2]].numpy() + np.asarray(root_xz0, dtype=np.float32)
    return out


def abs_features_to_positions(feat_abs):
    """CondMDI's own decoder (recover_from_ric, abs_3d=True): (T,263) -> (T,22,3)."""
    return recover_from_ric(torch.from_numpy(np.asarray(feat_abs)).float(), 22, abs_3d=True).numpy().astype(np.float64)


def positions_to_abs_features(positions):
    """(T,22,3) world positions -> (T,263) abs_3d features, without moving the motion.
    The notebook yields N-1 rows for N frames; the last frame is held once (zero final velocity) so that every
    input frame gets a feature row."""
    pos = np.concatenate([positions, positions[-1:]], 0).astype(np.float64)
    return relative_to_abs_features(_relative_features(pos), positions[0, 0, [0, 2]])


# ---------------------------------------------------------------------------------------------
# Dataset preprocessing for raw joint files (HumanML3D ./joints/*.npy, e.g. dataset/000021.npy)
# ---------------------------------------------------------------------------------------------
def _uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0])).numpy()
    tgt_offset = target_offset.numpy()
    l_idx1, l_idx2 = 5, 8
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    tgt_root_pos = positions[:, 0] * (tgt_leg_len / src_leg_len)
    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_IDX)
    src_skel.set_offset(target_offset)
    return src_skel.forward_kinematics_np(quat_params, tgt_root_pos)


def target_offsets(example_raw_joints):
    """HumanML3D's target skeleton = offsets of clip 000021, frame 0 (notebook)."""
    skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    return skel.get_offsets_joints(torch.from_numpy(example_raw_joints[0, :22]))


def process_raw_joints(raw, tgt_offsets):
    """Notebook process_file_abs_root: uniform skeleton, put on floor, XZ at origin, face +Z, features.
    Returns (positions (T-1,22,3), abs features (T-1,263))."""
    positions = _uniform_skeleton(raw[:, :22].astype(np.float64), tgt_offsets)
    positions[:, :, 1] -= positions.min(axis=0).min(axis=0)[1]
    root_pos_init = positions[0]
    positions = positions - root_pos_init[0] * np.array([1, 0, 1])
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_IDX
    across = (root_pos_init[r_hip] - root_pos_init[l_hip]) + (root_pos_init[sdr_r] - root_pos_init[sdr_l])
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., None]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., None]
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * qbetween_np(forward_init, np.array([[0, 0, 1]]))
    positions = qrot_np(root_quat_init, positions)
    rel = _relative_features(positions)
    feat_abs = relative_to_abs_features(rel)
    # notebook saves rec_ric_data (positions decoded from the features) as the joint file
    pos_out = recover_from_ric(torch.from_numpy(rel).unsqueeze(0).float(), 22).squeeze(0).numpy().astype(np.float64)
    return pos_out, feat_abs
