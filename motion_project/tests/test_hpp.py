import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from motion_project.hpp import skeleton as sk
from motion_project.hpp.hermite import hermite_reconstruct, basis
from motion_project.hpp.keyframe_selection import schedule_keys
from motion_project.experiments.common import project, load_cfg

PROJECTORS = ['hpp_v02', 'hpp_adaptive', 'robotic_slerp']


# ------------------------------------------------------------------------------------- Hermite
def test_hermite_basis_partition_and_endpoints():
    u = np.linspace(0, 1, 11)
    h00, h10, h01, h11 = basis(u)
    assert np.allclose(h00 + h01, 1)
    assert (h00[0], h01[-1], h10[0], h11[-1]) == (1, 1, 0, 0)


def test_hermite_matches_reference_pchip_and_passes_keys():
    rng = np.random.default_rng(0)
    keys = np.array([0, 3, 5, 9, 14, 15, 22])
    vals = rng.normal(size=(len(keys), 3))
    ours = hermite_reconstruct(keys, vals, 23)
    ref = PchipInterpolator(keys, vals, axis=0)(np.arange(23))
    assert np.allclose(ours, ref, atol=1e-10)
    assert np.allclose(ours[keys], vals)


def test_hermite_no_overshoot():
    keys = np.array([0, 4, 6, 15, 20])
    vals = np.array([[0.0], [1.0], [1.0], [-2.0], [3.0]])
    out = hermite_reconstruct(keys, vals, 21)
    for a, b, va, vb in zip(keys[:-1], keys[1:], vals[:-1, 0], vals[1:, 0]):
        seg = out[a:b + 1, 0]
        assert seg.min() >= min(va, vb) - 1e-12 and seg.max() <= max(va, vb) + 1e-12


def test_v02_schedule_is_time_based():
    frames, _ = schedule_keys(179, 20.0, 0.12)
    assert frames[0] == 0 and frames[-1] == 178
    d = np.diff(frames[:-1])
    assert set(d) <= {2, 3} and abs(d.mean() - 2.4) < 0.05
    frames120, _ = schedule_keys(1000, 120.0, 0.12)                # same 0.12 s at 120 fps
    assert set(np.diff(frames120[:-1])) <= {14, 15}


# ------------------------------------------------------------------------------------- projector
@pytest.fixture(scope='module', params=PROJECTORS)
def projection(request, motion):
    return request.param, motion, project(motion['positions'], motion['fps'], request.param)


def test_output_contract(projection):
    name, m, res = projection
    T = len(m['positions'])
    for k in ['original_positions', 'rough_positions', 'selected_observations', 'fps', 'joint_names',
              'joint_parents', 'coordinate_system', 'units', 'metadata', 'observation_mask', 'observation_mask_xyz']:
        assert k in res
    assert res['rough_positions'].shape == (T, 22, 3)
    assert res['observation_mask'].shape == (T, 22) and res['observation_mask'].dtype == bool
    assert res['observation_mask_xyz'].shape == (T, 22, 3)
    assert res['fps'] == 20.0 and res['units'] == 'm'
    assert np.isfinite(res['rough_positions']).all()


def test_original_not_modified_and_separate(projection):
    name, m, res = projection
    assert np.array_equal(res['original_positions'], m['positions'])
    assert res['original_positions'] is not m['positions']
    assert not np.shares_memory(res['rough_positions'], m['positions'])


def test_deterministic(projection):
    name, m, res = projection
    again = project(m['positions'], m['fps'], name)
    assert np.array_equal(res['rough_positions'], again['rough_positions'])


def test_connectivity_and_bone_lengths(projection):
    name, m, res = projection
    ref = np.array(res['metadata']['reference_bone_lengths'])
    L = sk.bone_lengths(res['rough_positions'])
    assert np.abs(L[:, 1:] - ref[1:]).max() < 1e-9            # every bone exactly its reference length


def test_duration_and_root_endpoints(projection):
    name, m, res = projection
    o, r = m['positions'], res['rough_positions']
    assert len(r) == len(o)
    assert np.allclose(r[0, 0], o[0, 0]) and np.allclose(r[-1, 0], o[-1, 0])   # root keyed at start and end


def test_observations_are_original_values(projection):
    name, m, res = projection
    for obs in res['selected_observations']:
        if obs['variable'] == 'position':
            assert np.allclose(np.array(obs['values']), m['positions'][obs['frames'], obs['joint']])
            assert obs['frames'][0] == 0 and obs['frames'][-1] == len(m['positions']) - 1


def test_mask_matches_observations(projection):
    name, m, res = projection
    mask = np.zeros_like(res['observation_mask'])
    for obs in res['selected_observations']:
        if obs['variable'] == 'position':
            mask[obs['frames'], obs['joint']] = True
    assert np.array_equal(mask, res['observation_mask'])


def test_hermite_baseline_keeps_root_on_keys(motion):
    res = project(motion['positions'], motion['fps'], 'hpp_v02')
    root = [o for o in res['selected_observations'] if o['name'] == 'root'][0]
    assert np.allclose(res['rough_positions'][root['frames'], 0], motion['positions'][root['frames'], 0])
