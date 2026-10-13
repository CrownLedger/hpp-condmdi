"""Convert a HumanML3D relative-root feature file to the absolute-root (abs_3d) format,
following dataset/HumanML3D_abs/motion_representation.ipynb."""
import sys, numpy as np, torch
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos, recover_from_ric
src, dst = sys.argv[1], sys.argv[2]
data = np.load(src)
_, r_pos, rot_ang = recover_root_rot_pos(torch.from_numpy(data).float(), return_rot_ang=True)
new = data.copy()
new[:, 0] = rot_ang.numpy()
new[:, [1, 2]] = r_pos[:, [0, 2]].numpy()
# sanity check: both representations must give the same joint positions
a = recover_from_ric(torch.from_numpy(data).float(), 22)
b = recover_from_ric(torch.from_numpy(new).float(), 22, abs_3d=True)
print('max joint diff', (a - b).abs().max().item())
np.save(dst, new)
