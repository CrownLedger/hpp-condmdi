"""Add CMU mocap BVH clips (CMU skeleton, 120 fps) as raw HumanML3D clips, used directly as robotic input.

    python -m motion_project.experiments.prepare_cmu_bvh /root/cmu-robotic/*.bvh

Each BVH is evaluated with forward kinematics, mapped onto the 22 HumanML3D (SMPL) joints, converted to metres,
downsampled to 20 fps and cropped to CondMDI's limit (196 frames after processing). It is written as
dataset/cmu_<name>.npy (raw joints, Y up); the existing loader then processes it with the dataset notebook pipeline
(uniform skeleton of 000021, floor, origin, face +Z), exactly like dataset/000021.npy.

Joint mapping (HumanML3D <- CMU): pelvis <- Hips, hips <- Up/Leg, knees <- Leg, ankles <- Foot, feet <- ToeBase,
spine1 <- Spine, spine2 <- midpoint(Spine, Spine1), spine3 <- Spine1, neck <- Neck1, head <- Head,
collars <- 45 % Spine1 -> Arm, shoulders <- Arm, elbows <- ForeArm, wrists <- Hand.
"""
import argparse
import os
import re

import numpy as np

CMU_UNIT_M = 0.056444          # CMU BVH length unit in metres ((1/0.45) inch)
SRC_FPS = 120
TARGET_FPS = 20
MAX_RAW_FRAMES = 197           # process_raw_joints drops one frame -> 196 = CondMDI max_frames
CLIPS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'cmu_clips.txt')

# name, or (a, b, w) = a + w * (b - a). CMU's Shoulder joints coincide with Spine1 (zero-length spine3 -> collar bone,
# undefined for HumanML3D's IK), so collars are placed 45 % of the way from Spine1 to the shoulder.
MAPPING = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg', ('Spine', 'Spine1', 0.5), 'LeftFoot',
           'RightFoot', 'Spine1', 'LeftToeBase', 'RightToeBase', 'Neck1', ('Spine1', 'LeftArm', 0.45),
           ('Spine1', 'RightArm', 0.45), 'Head',
           'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand']


def parse_bvh(path):
    """Returns joints [(name, parent_index, offset, channels)], frame_time, motion (F, C)."""
    with open(path) as f:
        text = f.read()
    head, motion = text.split('MOTION')
    joints, stack = [], []
    tokens = head.split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ('ROOT', 'JOINT'):
            joints.append([tokens[i + 1], stack[-1] if stack else -1, None, []])
            pending = len(joints) - 1
            i += 2
        elif t == 'End':
            joints.append([None, stack[-1], None, []])
            pending = len(joints) - 1
            i += 2
        elif t == '{':
            stack.append(pending)
            i += 1
        elif t == '}':
            stack.pop()
            i += 1
        elif t == 'OFFSET':
            joints[pending][2] = np.array([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
        elif t == 'CHANNELS':
            n = int(tokens[i + 1])
            joints[pending][3] = tokens[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    lines = motion.strip().splitlines()
    frame_time = float(re.search(r'Frame Time:\s*([\d.eE+-]+)', motion).group(1))
    data = np.array([[float(x) for x in ln.split()] for ln in lines[2:] if ln.strip()])
    return joints, frame_time, data


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    m = {'X': [[o, z, z], [z, c, -s], [z, s, c]],
         'Y': [[c, z, s], [z, o, z], [-s, z, c]],
         'Z': [[c, -s, z], [s, c, z], [z, z, o]]}[axis]
    return np.moveaxis(np.array(m), [0, 1], [-2, -1])


def forward_kinematics(joints, data):
    """Global joint positions {name: (F,3)} in BVH units."""
    F = len(data)
    pos, rot = [None] * len(joints), [None] * len(joints)
    col = 0
    for j, (name, parent, offset, channels) in enumerate(joints):
        R = np.broadcast_to(np.eye(3), (F, 3, 3)).copy()
        trans = np.zeros((F, 3))
        for ch in channels:
            if ch.endswith('position'):
                trans[:, 'XYZ'.index(ch[0])] = data[:, col]
            else:
                R = R @ _rot(ch[0], data[:, col])
            col += 1
        if parent < 0:
            pos[j], rot[j] = offset + trans, R
        else:
            pos[j] = pos[parent] + np.einsum('fij,j->fi', rot[parent], offset)
            rot[j] = rot[parent] @ R
    return {name: pos[j] for j, (name, *_rest) in enumerate(joints) if name}


def bvh_to_humanml3d_raw(path):
    joints, frame_time, data = parse_bvh(path)
    assert abs(1 / frame_time - SRC_FPS) < 1, f'{path}: {1 / frame_time:.1f} fps, expected {SRC_FPS}'
    P = forward_kinematics(joints, data[1:])              # frame 0 of CMU BVH files is the T-pose
    out = np.stack([P[m[0]] + m[2] * (P[m[1]] - P[m[0]]) if isinstance(m, tuple) else P[m] for m in MAPPING], 1)
    out = out * CMU_UNIT_M
    return out[::SRC_FPS // TARGET_FPS].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bvh', nargs='+')
    ap.add_argument('--out_dir', default='dataset')
    ap.add_argument('--start_s', type=float, default=0.0, help='window start in seconds')
    a = ap.parse_args()
    ids = []
    for path in a.bvh:
        raw = bvh_to_humanml3d_raw(path)
        full = len(raw)
        s = int(round(a.start_s * TARGET_FPS))
        raw = raw[s:s + MAX_RAW_FRAMES]
        cid = 'cmu_' + os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(a.out_dir, f'{cid}.npy'), raw)
        ids.append(cid)
        print(f'{cid:14s} {full:5d} frames @20fps ({full / TARGET_FPS:5.1f} s) -> kept {len(raw)} '
              f'from {a.start_s:.1f} s, height {raw[0, 15, 1] - raw[0, [7, 8], 1].min():.2f} m')
    with open(CLIPS_FILE, 'w') as f:
        f.write('\n'.join(ids) + '\n')
    print(f'{len(ids)} clips -> {a.out_dir}/; list in {CLIPS_FILE}')


if __name__ == '__main__':
    main()
