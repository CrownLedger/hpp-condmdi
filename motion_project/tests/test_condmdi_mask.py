import numpy as np
import pytest

from data_loaders import humanml_utils as hu
from motion_project.realizer import condmdi_mask_adapter as cma
from motion_project.realizer import motion_feature_adapter as mfa
from motion_project.experiments.common import project


def _obs(T=40):
    m = np.zeros((T, 22), bool)
    m[[0, 10, 20], 20] = True                     # left wrist at three frames
    m[[0, 39], 7] = True                          # left ankle at two frames
    return m, np.zeros(T, bool)


def test_root_policies():
    m, o = _obs()
    jm, prov = cma.build_joint_mask(m, o, 'observed_frames')
    assert jm.shape == (22, 40)
    assert set(np.nonzero(jm[0])[0]) == {0, 10, 20, 39}
    assert (prov == 2).sum() == 4 and (prov == 1).sum() == m.sum()
    jm, prov = cma.build_joint_mask(m, o, 'dense')
    assert jm[0].all()
    assert not jm[[j for j in range(1, 22) if j not in (7, 20)]].any(), 'no other joint may be observed'


def test_feature_mask_matches_condmdi_layout():
    m, o = _obs()
    jm, _ = cma.build_joint_mask(m, o, 'observed_frames')
    fm = cma.feature_mask(jm, 196, 'pos_rot_vel')
    assert fm.shape == (263, 196)
    assert not fm[:, 40:].any(), 'padding frames must stay unobserved'
    t = 10   # root + left wrist observed
    expect = hu.MAT_POS[0] | hu.MAT_VEL[0] | hu.MAT_ROT[0] | hu.MAT_POS[20] | hu.MAT_ROT[20] | hu.MAT_VEL[20]
    assert np.array_equal(fm[:, t], expect)
    t = 39   # root + left ankle -> includes the ankle contact bit
    assert fm[259, t] and not fm[261, t]
    assert not fm[:, 5].any()


def test_no_leakage_unobserved_features_are_zero():
    T = 40
    feat = np.random.default_rng(0).normal(size=(T, 263)).astype(np.float32)
    m, o = _obs()
    jm, _ = cma.build_joint_mask(m, o, 'dense')
    fm = cma.feature_mask(jm, 196)
    x = cma.observed_features(feat, fm, np.zeros(263, np.float32), np.ones(263, np.float32), 196)
    assert not x[~fm].any()
    assert np.allclose(x[fm], feat.T[fm[:, :T]] if False else x[fm])
    assert np.allclose(x[:, :T][fm[:, :T]], feat.T[fm[:, :T]])


def test_experiment_b_uses_only_rough_motion(motion):
    """Exp B obs_x0 must be a function of the rough motion alone: changing the original must not change it."""
    res = project(motion['positions'], motion['fps'], 'hpp_v02')
    jm, _ = cma.build_joint_mask(res['observation_mask'], res['orientation_mask'], 'dense')
    fm = cma.feature_mask(jm, 196)
    mu, sd = np.zeros(263, np.float32), np.ones(263, np.float32)
    x1 = cma.observed_features(mfa.positions_to_abs_features(res['rough_positions']), fm, mu, sd, 196)
    tampered = dict(motion)
    tampered['features_abs'] = motion['features_abs'] + 1.0           # original changed, rough unchanged
    x2 = cma.observed_features(mfa.positions_to_abs_features(res['rough_positions']), fm, mu, sd, 196)
    assert np.array_equal(x1, x2)


def test_torso_mapping_options():
    m, o = _obs()
    o[[5, 6]] = True
    jm, prov = cma.build_joint_mask(m, o, 'observed_frames', 'none')
    assert not jm[15].any()
    jm, prov = cma.build_joint_mask(m, o, 'observed_frames', 'head')
    assert jm[15, [5, 6]].all() and (prov == 3).sum() == 2
    with pytest.raises(ValueError):
        cma.build_joint_mask(m, o, 'sometimes')
