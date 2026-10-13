"""HPP observations -> CondMDI conditioning tensors (handoff sec. 25 / 28, Phase 5).

Conditioning semantics, verified in the installed code:
* model input: obs_x0, obs_mask of shape [B, 263, 1, max_frames]; the UNet replaces x by obs_x0 where obs_mask
  is set and concatenates the mask as extra channels (model/mdm_unet.py: x = obs_x0*obs_mask + x*~obs_mask).
* masks are per joint and frame, expanded to features by utils/editing_util.joint_to_full_mask:
    joint 0 (root): yaw [0], x/z [1:3], y [3]                      (+ root velocity [193:196] in pos_rot_vel)
    joint j > 0   : ric position, 6D rotation, velocity            (pos_rot_vel)
                    + foot contact bit for joints 7, 8, 10, 11
  i.e. observing a joint observes its whole feature vector; position-only ('pos') is supported by the code but the
  checkpoint was trained with full joint vectors (keyframe_selection_scheme=random_joints) and fails with it.
* ric/rot/vel of joint j are expressed relative to the root's position and heading at that frame, so the root is
  always observed on frames where any joint is observed (as in random_joints training).
"""
import numpy as np
import torch

from utils.editing_util import joint_to_full_mask

HEAD = 15


def build_joint_mask(observation_mask, orientation_mask, root_policy='observed_frames', torso_orientation_as='none'):
    """observation_mask (T,22), orientation_mask (T,) -> joint mask (22,T) and provenance labels.
    provenance: 1 = HPP position observation, 2 = added root (representation requirement / dense policy),
    3 = torso orientation mapped to a joint."""
    T = observation_mask.shape[0]
    m = observation_mask.T.copy()
    prov = np.where(m, 1, 0)
    if torso_orientation_as == 'head':
        add = orientation_mask & ~m[HEAD]
        m[HEAD, orientation_mask] = True
        prov[HEAD, add] = 3
    elif torso_orientation_as != 'none':
        raise ValueError(torso_orientation_as)
    if root_policy == 'observed_frames':
        frames = m.any(0)
    elif root_policy == 'dense':
        frames = np.ones(T, bool)
    else:
        raise ValueError(root_policy)
    prov[0, frames & ~m[0]] = 2
    m[0, frames] = True
    return m, prov


def feature_mask(joint_mask, max_frames, feature_mode='pos_rot_vel'):
    """(22,T) -> (263, max_frames) bool, via CondMDI's own joint_to_full_mask."""
    T = joint_mask.shape[1]
    jm = torch.zeros((1, 22, 1, max_frames), dtype=torch.bool)
    jm[0, :, 0, :T] = torch.from_numpy(joint_mask)
    return joint_to_full_mask(jm, mode=feature_mode)[0, :, 0].numpy()


def observed_features(features_abs, fmask, mean, std, max_frames):
    """Normalised, padded obs_x0 (263, max_frames). Entries outside the mask are ZEROED, so no unobserved value of
    the source motion can reach the model, whichever motion the features come from."""
    T = len(features_abs)
    x = np.zeros((max_frames, 263), np.float32)
    x[:T] = (features_abs - mean) / std
    x = x.T * fmask
    assert not np.any(x[~fmask]), 'leakage: unobserved features are non-zero'
    return x.astype(np.float32)
