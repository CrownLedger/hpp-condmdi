"""HumanML3D motion loader (handoff sec. 17, Phase 2).

Sources, in order:
  1. dataset/HumanML3D/new_joint_vecs/<id>.npy  (+ new_joints/<id>.npy cross-check) - processed HumanML3D clip
  2. dataset/<id>.npy                            - raw HumanML3D joint file (T,52,3), processed with the dataset
                                                   notebook pipeline (motion_feature_adapter.process_raw_joints)
Returns positions (T,22,3) in HumanML3D world coordinates, abs_3d features (T,263), fps = 20, captions, metadata.
"""
import os
import numpy as np

from motion_project.hpp.skeleton import NUM_JOINTS, skeleton_metadata
from motion_project.realizer import motion_feature_adapter as mfa

FPS = 20.0
DATA_ROOT = 'dataset/HumanML3D'
RAW_ROOT = 'dataset'
REFERENCE_SKELETON_CLIP = 'dataset/000021.npy'   # HumanML3D's target skeleton (notebook example_id)


def _captions(motion_id, data_root):
    path = f'{data_root}/texts/{motion_id}.txt'
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.split('#')[0] for line in f if line.strip()]


def load_humanml3d_motion(motion_id, data_root=DATA_ROOT, raw_root=RAW_ROOT):
    vec_path = f'{data_root}/new_joint_vecs/{motion_id}.npy'
    raw_path = f'{raw_root}/{motion_id}.npy'
    if os.path.exists(vec_path):
        rel = np.load(vec_path)
        feat_abs = mfa.relative_to_abs_features(rel)
        positions = mfa.abs_features_to_positions(feat_abs)
        joints_path = f'{data_root}/new_joints/{motion_id}.npy'
        if os.path.exists(joints_path):
            ref = np.load(joints_path)
            n = min(len(ref), len(positions))
            err = float(np.abs(ref[:n] - positions[:n]).max())
            assert err < 1e-4, f'feature-decoded positions disagree with new_joints ({err} m)'
        source = 'humanml3d_processed'
    elif os.path.exists(raw_path):
        raw = np.load(raw_path)
        assert raw.ndim == 3 and raw.shape[2] == 3 and raw.shape[1] >= NUM_JOINTS, raw.shape
        tgt = mfa.target_offsets(np.load(REFERENCE_SKELETON_CLIP))
        positions, feat_abs = mfa.process_raw_joints(raw, tgt)
        source = 'humanml3d_raw_joints_processed_with_notebook_pipeline'
    else:
        raise FileNotFoundError(f'no HumanML3D data for {motion_id}: looked for {vec_path} and {raw_path}')

    validate_positions(positions)
    return {'motion_id': motion_id, 'positions': positions, 'features_abs': feat_abs, 'fps': FPS,
            'captions': _captions(motion_id, data_root), 'source': source,
            'skeleton': skeleton_metadata(positions)}


def validate_positions(positions, fps=FPS):
    """Shape, finiteness, units and coordinate sanity checks; raises AssertionError on failure."""
    assert positions.ndim == 3 and positions.shape[1:] == (NUM_JOINTS, 3), positions.shape
    assert np.isfinite(positions).all(), 'non-finite joint positions'
    assert len(positions) >= 2, 'need at least two frames'
    # metres, Y up: pelvis well above the floor, body ~1-2 m tall
    height = positions[:, 15, 1].max() - positions[:, [7, 8, 10, 11], 1].min()
    assert 0.8 < height < 2.5, f'implausible body height {height:.2f} m (units / up axis?)'
    assert abs(positions[0, 0, 0]) < 0.05 and abs(positions[0, 0, 2]) < 0.05, 'frame 0 root not at x=z=0'
