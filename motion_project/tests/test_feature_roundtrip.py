import numpy as np

from motion_project.hpp import skeleton as sk
from motion_project.realizer import motion_feature_adapter as mfa
from motion_project.hpp.robotic_slerp import local_rotations


def test_roundtrip_positions(motion):
    p = motion['positions']
    back = mfa.abs_features_to_positions(mfa.positions_to_abs_features(p))
    assert back.shape == p.shape                                   # temporal alignment: same frame count
    assert np.abs(back - p).max() < 1e-5                           # joint positions
    assert np.abs(back[:, 0] - p[:, 0]).max() < 1e-5               # root trajectory
    assert np.abs(sk.bone_lengths(back) - sk.bone_lengths(p)).max() < 1e-5


def test_roundtrip_orientation(motion):
    p = motion['positions']
    back = mfa.abs_features_to_positions(mfa.positions_to_abs_features(p))
    q1, q2 = local_rotations(p), local_rotations(back)
    ang = np.degrees(2 * np.arccos(np.clip(np.abs((q1 * q2).sum(-1)), 0, 1)))
    assert ang.max() < 0.1


def test_extracted_features_match_dataset(motion):
    """Re-extracting features from the dataset positions reproduces the dataset's own feature vectors."""
    f_ds, f = motion['features_abs'], mfa.positions_to_abs_features(motion['positions'])
    n = len(f_ds) - 1                                              # last row: held final frame
    tol = {'root': 0.02, 'ric': 0.02, 'rot': 0.05, 'vel': 0.002}  # dataset rows come from un-normalised raw joints
    for name, (a, b) in {'root': (0, 4), 'ric': (4, 67), 'rot': (67, 193), 'vel': (193, 259)}.items():
        assert np.abs(f[:n, a:b] - f_ds[:n, a:b]).max() < tol[name], name
    assert np.array_equal(f[:n, 259:], f_ds[:n, 259:])            # foot contacts


def test_turning_through_180_deg_does_not_drift():
    """Regression: the notebook's yaw-velocity sign flip when facing crosses +-180 deg."""
    T = 60
    p = np.zeros((T, 22, 3))
    yaw = np.linspace(0.01, 2 * np.pi, T)                         # one full turn in place, never exactly 180 deg
    # slightly perturbed rest pose: an exactly symmetric T-pose makes HumanML3D's own IK (qbetween of opposite
    # vectors) undefined at 180 deg, which real motion never hits exactly
    rest = sk.RAW_OFFSETS * 0.2 + np.random.default_rng(1).normal(scale=0.02, size=(22, 3))
    for t in range(T):
        c, s = np.cos(yaw[t]), np.sin(yaw[t])
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pts = np.zeros((22, 3)); pts[0] = [0, 1, 0]
        for j in range(1, 22):
            pts[j] = pts[sk.JOINT_PARENTS[j]] + R @ rest[j]
        p[t] = pts
    back = mfa.abs_features_to_positions(mfa.positions_to_abs_features(p))
    assert np.abs(back - p).max() < 1e-4
