"""Direct input: the motion is already robotic (e.g. CMU mocap of robot-style dancing), so no rougher is applied.

rough = input, unchanged. CondMDI observes the full body of the input every `key_every_s` seconds (plus the last
frame); the frames in between are left for CondMDI to fill. Same output contract as hpp_project.
"""
import numpy as np

from motion_project.hpp import skeleton as sk
from motion_project.hpp.hpp_projector import build_output


def project_direct(positions, fps, config):
    M = np.array(positions, dtype=np.float64, copy=True)
    T = len(M)
    step = max(1, int(round(config['direct']['key_every_s'] * fps)))
    frames = sorted(set(range(0, T, step)) | {T - 1})
    obs = [{'name': 'root' if j == 0 else sk.JOINT_NAMES[j], 'joint': j, 'joint_name': sk.JOINT_NAMES[j],
            'variable': 'position', 'frame': 'WORLD', 'priority': 'SOFT', 'frames': frames,
            'values': M[frames, j].tolist(), 'reasons': {f: 'key' for f in frames}} for j in range(22)]
    return build_output(M, M.copy(), obs, fps, config, method='direct input (no rougher)',
                        lengths=sk.reference_bone_lengths(M))
