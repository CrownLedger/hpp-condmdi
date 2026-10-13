# Implementation notes and deviations

Every place where the implementation departs from, or goes beyond, the handoff specification.

## HPP
| # | Topic | What was done | Why |
|---|---|---|---|
| D1 | v0.2 provenance | **Reconstructed v0.2-style baseline, not a port.** The original v0.2 source was not provided. | Handoff sec. 37/55 |
| D2 | v0.2 key selection | `hpp_v02`: keys at t = k·0.12 s → frame round(k·0.12·fps), plus the last frame. At 20 fps keys are 2 or 3 frames apart (mean 2.4). | Only known v0.2 rule (sec. 11); time-based per sec. 19 |
| D3 | Root / torso keys in v0.2 | **Assumption:** same 0.12 s schedule as hands/feet. | v0.2 behaviour unknown |
| D4 | Adaptive selection | Separate config `hpp_adaptive` (contacts, pauses, direction changes, extrema, error-driven insertion). Labelled an extension; never reported as v0.2. | Sec. 11 lists these as future rules |
| D5 | Hermite tangents | Fritsch–Carlson (monotone) tangents, time-scaled; own implementation of the spec's basis, unit-tested against SciPy's PCHIP. Original v0.2 tangent rule unknown. | Sec. 12 requirements (no overshoot, keys exact) |
| D6 | Torso | Rigid canonical torso (median body-frame shape) driven by keyed up/across axes. Spine bending inside the torso is not represented. | Sec. 13 "canonical body configuration" |
| D7 | IK poles | Fixed canonical bend directions in the body frame (elbows back/down/out, knees forward), in `configs/hpp_*.yaml`. | Sec. 13.1/13.2, deterministic |
| D8 | Connectivity | Every joint placed as parent + direction·reference length, so subtrees cannot detach; the historical "translate the child subtree" fix is unnecessary. | Sec. 13.4 |
| D9 | Hands | HumanML3D has no hand joints; the wrist is the hand end effector. "Foot" = ankle (IK target) + toe joint (keyed). | Skeleton limitation |
| D10 | Robotic slerp | Kept as `robotic_slerp` (experimental, user request); contradicts sec. 6, excluded from the baseline. | User request in an earlier session |

## Realizer
| # | Topic | What was done | Why |
|---|---|---|---|
| D11 | Root observation | Root always observed on frames where any joint is observed; `dense` policy observes it on every frame (sec. 28 example). Both evaluated. | Joint features are root-relative; training observed the root at keyframes |
| D12 | Torso orientation | HPP's torso-orientation keys are **not** passed to CondMDI by default (`torso_orientation_as: none`). | Sec. 24 lists torso as a future constraint; mapping it to the head joint would add information HPP did not select |
| D13 | Feature granularity | Observed joints carry position + rotation + velocity (+ contact for feet), i.e. the checkpoint's training mask. Per-coordinate (`observation_mask_xyz`) masks are exported but CondMDI cannot use them; position-only fails (pilot). | Sec. 28 distinction; model limitation |
| D14 | Feature extraction | Notebook feature definitions **without** its normalisation (no floor / origin / facing changes), last frame held to keep T rows, and the yaw-velocity sign fix (see environment report). | Rough motion must stay in place; correct round trip (sec. 27) |
| D15 | Experiment A | Original **dataset** features at HPP-selected entries; all other entries zeroed before reaching the model (unit-tested). | Sec. 26 A, no leakage (sec. 44) |
| D16 | Experiment B | Features encoded from the rough motion only; unit test proves the original cannot influence obs_x0. | Sec. 26 B / 44 |
| D17 | Long sequences | > 196 frames raises an error; no segmentation implemented. Both test clips fit. | Sec. 46: no silent truncation |
| D18 | Seeds | Sample r uses seed (10 + r) for every condition, so conditions share initial noise. | Sec. 29 reproducibility, fair comparison |

## Evaluation
* Acceptance tolerances: `configs/acceptance.yaml`, fixed before the final runs. Pilot numbers on clip 012314 had
  been seen earlier in the session (disclosed in the file).
* Constraint error is measured against the **conditioning target** (original for A, rough for B) over the
  HPP-selected joint-frames; root entries added by the adapter are reported separately.
* Floor = y 0 (HumanML3D normalisation); foot sliding uses height-based contact (< 5 cm) so that sliding feet are
  not excluded by a velocity-based contact test.
* Jitter (C7) is judged relative to a reference CondMDI run on the same clip (full body of the original every 5
  frames), because the model's own output jerk differs a lot between clips (≈155 vs ≈428 m/s³).
* No learned naturalness metric: the T2M evaluators could not be downloaded. Naturalness is judged visually.
