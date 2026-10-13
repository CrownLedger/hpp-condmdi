import numpy as np
import pytest

from motion_project.hpp import skeleton as sk
from motion_project.hpp.humanml3d_loader import load_humanml3d_motion, validate_positions


def test_joint_names_match_condmdi():
    from data_loaders.humanml_utils import HML_JOINT_NAMES
    assert sk.JOINT_NAMES == HML_JOINT_NAMES


def test_topology_matches_condmdi():
    from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets
    assert sk.KINEMATIC_CHAINS == t2m_kinematic_chain
    assert np.array_equal(sk.RAW_OFFSETS, t2m_raw_offsets)
    parents = [-1] * 22
    for chain in t2m_kinematic_chain:
        for a, b in zip(chain[:-1], chain[1:]):
            parents[b] = a
    assert sk.JOINT_PARENTS == parents


def test_left_right_pairs_are_mirrored_names():
    for l, r in sk.LEFT_RIGHT_PAIRS:
        assert sk.JOINT_NAMES[l].replace('left_', '') == sk.JOINT_NAMES[r].replace('right_', '')


def test_shapes_and_fps(motion):
    p, f = motion['positions'], motion['features_abs']
    assert p.ndim == 3 and p.shape[1:] == (22, 3)
    assert f.shape == (len(p), 263)
    assert motion['fps'] == 20.0
    assert len(p) <= 196, 'test clips must fit CondMDI without segmentation'


def test_coordinates(motion):
    p = motion['positions']
    validate_positions(p)
    assert abs(p[:, [7, 8, 10, 11], 1].min()) < 0.02, 'lowest foot point should be on the floor (y=0)'
    # frame 0 faces +Z: hips/shoulders across vector is perpendicular to Z
    across = (p[0, 2] - p[0, 1]) + (p[0, 17] - p[0, 16])
    fwd = np.cross([0, 1, 0], across)
    assert fwd[2] / np.linalg.norm(fwd) > 0.95


def test_bone_lengths_constant(motion):
    L = sk.bone_lengths(motion['positions'])[:, 1:]
    assert L.std(0).max() < 1e-4, 'HumanML3D positions come from FK of a fixed skeleton'


def test_validate_rejects_bad_input():
    with pytest.raises(AssertionError):
        validate_positions(np.zeros((10, 21, 3)))
    with pytest.raises(AssertionError):
        validate_positions(np.full((10, 22, 3), np.nan))
