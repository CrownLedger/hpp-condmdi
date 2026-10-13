"""HumanML3D 22-joint skeleton adapter (handoff sec. 18).

Index -> name mapping and topology are taken from (and unit-tested against) CondMDI's own constants:
data_loaders/humanml_utils.py (HML_JOINT_NAMES) and data_loaders/humanml/utils/paramUtil.py
(t2m_kinematic_chain, t2m_raw_offsets).

Coordinate system (verified in docs/environment_report.md): metres, Y up, right-handed; every HumanML3D clip is
canonicalised so that frame 0 has the pelvis at x = z = 0 and the body facing +Z.
"""
import numpy as np

JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
               'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar',
               'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
               'left_wrist', 'right_wrist']
NUM_JOINTS = 22
KINEMATIC_CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
JOINT_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# HumanML3D rest-pose bone directions (paramUtil.t2m_raw_offsets)
RAW_OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0],
                        [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1],
                        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]], dtype=np.float64)

J = {n: i for i, n in enumerate(JOINT_NAMES)}
LEFT_RIGHT_PAIRS = [(J[f'left_{p}'], J[f'right_{p}']) for p in
                    ['hip', 'knee', 'ankle', 'foot', 'collar', 'shoulder', 'elbow', 'wrist']]
# HumanML3D has no hand joints: the wrist is the hand end effector; "foot" is the toe/ball joint.
END_EFFECTORS = {'left_hand': J['left_wrist'], 'right_hand': J['right_wrist'],
                 'left_ankle': J['left_ankle'], 'right_ankle': J['right_ankle'],
                 'left_foot': J['left_foot'], 'right_foot': J['right_foot'], 'head': J['head']}
END_EFFECTOR_INDICES = sorted(END_EFFECTORS.values())
FACE_JOINT_IDX = [J['right_hip'], J['left_hip'], J['right_shoulder'], J['left_shoulder']]  # HumanML3D facing rule
FOOT_JOINTS = {'left': [J['left_ankle'], J['left_foot']], 'right': [J['right_ankle'], J['right_foot']]}

# Semantic groups used by the HPP decoder
ROOT = J['pelvis']
TORSO = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17]     # parent-first order; rebuilt as a rigid canonical torso
ARMS = {'left': (J['left_shoulder'], J['left_elbow'], J['left_wrist']),
        'right': (J['right_shoulder'], J['right_elbow'], J['right_wrist'])}
LEGS = {'left': (J['left_hip'], J['left_knee'], J['left_ankle'], J['left_foot']),
        'right': (J['right_hip'], J['right_knee'], J['right_ankle'], J['right_foot'])}

COORDINATE_SYSTEM = {'up_axis': '+Y', 'forward_axis': '+Z (body facing at frame 0)', 'handedness': 'right-handed',
                     'units': 'metres', 'root_origin': 'pelvis x,z = 0 at frame 0; y = height above floor',
                     'skeleton_scale': 'HumanML3D uniform skeleton (target offsets of clip 000021)'}


def bone_lengths(pos):
    """Per-frame bone lengths (T, 22); entry 0 (root) is 0."""
    L = np.zeros(pos.shape[:2])
    for j, p in enumerate(JOINT_PARENTS):
        if p >= 0:
            L[:, j] = np.linalg.norm(pos[:, j] - pos[:, p], axis=-1)
    return L


def reference_bone_lengths(pos):
    """Reference bone lengths: per-bone median over the clip."""
    return np.median(bone_lengths(pos), axis=0)


def skeleton_metadata(pos):
    return {'joint_names': JOINT_NAMES, 'joint_parents': JOINT_PARENTS,
            'reference_bone_lengths': reference_bone_lengths(pos).tolist(),
            'left_right_pairs': LEFT_RIGHT_PAIRS, 'end_effector_indices': END_EFFECTOR_INDICES,
            'end_effectors': END_EFFECTORS, 'coordinate_system': COORDINATE_SYSTEM}
