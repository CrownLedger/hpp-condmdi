# Environment report (Phase 1 audit)

Audited 2026-10-12 on the RunPod instance. Nothing in the CondMDI repository was modified; the project is an
integration layer (`motion_project/`) that imports it.

## Hardware / OS
| | |
|---|---|
| GPU | NVIDIA RTX A6000, 49 140 MiB, driver 580.159.03 (CUDA 13.0 driver; PyTorch uses its bundled CUDA 11.7 runtime) |
| OS | Ubuntu 24.04.3 LTS (container), 30 GB overlay disk |
| Pod | RunPod; only TCP 22 exposed. The viewer is published through a Cloudflare quick tunnel. |

## Software
| | |
|---|---|
| Repository | `/root/diffusion-motion-inbetweening`, commit `0c58473` (2024-09-17) |
| Python env | `.env_condmdi/` (uv venv), **Python 3.8.20** (repo targets 3.7; 3.8 is the oldest available) |
| PyTorch | 1.13.1+cu117, cuDNN 8.5 — as pinned by the repo |
| Others | numpy 1.23.5, scipy 1.10.1, spaCy 3.7.5 (3.3.1 has no py3.8 wheels), CLIP (openai, git), pytest |
| Headless | matplotlib needs `MPLBACKEND=Agg` (no Tk) |

## Checkpoint
* `save/condmdi_random_joints/model000750000.pt` (1 901 660 525 bytes; sha256 recorded in every run record).
  This is the README's "conditionally trained on randomly sampled frames and joints (CondMDI)" model.
* Training args (`args.json`): `arch=unet`, `abs_3d=True`, `keyframe_conditioned=True`,
  `keyframe_selection_scheme=random_joints`, `diffusion_steps=1000`, cosine schedule, `predict_xstart=True`,
  `cond_mask_prob=0.1`, `zero_keyframe_loss=False`.

## Data actually available
| Item | State |
|---|---|
| HumanML3D texts | all captions (`dataset/HumanML3D/texts/`, from the HumanML3D repo) |
| HumanML3D motions | **only two clips**: `012314` (processed sample shipped with the HumanML3D repo) and `000021` (raw joints shipped with CondMDI at `dataset/000021.npy`, processed here with the dataset notebook pipeline). The full set needs AMASS (licence-gated, not available). |
| Normalisation | `dataset/HumanML3D/Mean_abs_3d.npy`, `Std_abs_3d.npy` (GMD absolute-root statistics from `dataset/HumanML3D_abs/`) |
| T2M evaluators / GloVe | GloVe downloaded; T2M evaluator files are blocked on Google Drive ("too many accesses"), so FID / R-precision cannot be computed |
| SMPL body model | downloaded (`body_models/smpl`) |

**Workaround that touches dataset files:** CondMDI's sampling scripts always build the test split to obtain
normalisation statistics. `dataset/HumanML3D/test.txt` was replaced by a 2-entry split (`012314`, `012314b`; original
kept as `test.txt.orig`), and `new_joint_vecs_abs_3d/` holds the absolute-root conversion of `012314`. The split is
used only for statistics and model construction; no conditioning value comes from it.

## Verified CondMDI interfaces
* **Representation:** 263-d HumanML3D features, absolute root (`abs_3d`): `[0]` yaw, `[1:3]` root x/z, `[3]` root y,
  `[4:67]` ric positions, `[67:193]` 6D rotations, `[193:259]` velocities, `[259:263]` foot contacts. 20 fps,
  `max_frames = 196`, tensor layout `[B, 263, 1, T]`. Decoding: `recover_from_ric(..., abs_3d=True)` (positions only).
* **Conditioning:** `model_kwargs['obs_x0']`, `model_kwargs['obs_mask']`, both `[B,263,1,196]`. The UNet input is
  `obs_x0*mask + x*~mask` concatenated with the mask (`model/mdm_unet.py`). Joint masks `[B,22,1,T]` are expanded
  with `utils/editing_util.joint_to_full_mask(mode)`; `pos_rot_vel` (the training setting) observes a joint's
  position, rotation, velocity and (feet) contact together.
* **Sampler:** `GaussianDiffusion.p_sample_loop`; text dropped with `ClassifierFreeSampleModel` and guidance 0 (as `--no_text`).
* **Imputation (`--imputate`):** because the model input already overwrites observed entries, imputation only has an
  effect at the last step (`stop_imputation_at=0`), where it pastes the observed values into the output.
* **Inference check:** pretrained inference runs on the GPU; ~37 s per 196-frame sample with text dropped
  (classifier-free wrapper evaluates the model twice per step).

## Known issues found during the audit
1. Dataset notebook bug: the root yaw velocity flips sign when the facing direction crosses ±180°, so features of
   turning motions decode with drift (16 cm on clip 000021). Fixed in `realizer/motion_feature_adapter.py` (documented deviation).
2. HumanML3D's IK (`Skeleton.inverse_kinematics_np`) resets the root rotation of frame 0 and is undefined for a bone
   exactly opposite its rest direction; handled by using the dataset's own convention for features and a
   time-continuous IK for the experimental slerp mode.
3. Position-only conditioning (`feature_mode=pos`) is supported by the code but the checkpoint fails with it
   (30–44 cm errors in a pilot run): it was trained with whole-joint feature masks.
