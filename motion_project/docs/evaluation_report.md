# Evaluation report — HPP (reconstructed v0.2-style) × pretrained CondMDI

Run date 2026-10-12/13. All numbers are measured and reproducible with the commands in `README.md`; the full
tables are in `outputs/reports/acceptance_tables.md` and per-sample metrics in `outputs/reports/*.json`.
Tolerances were fixed in `configs/acceptance.yaml` before these runs (pilot numbers on 012314 had been seen earlier).

## Summary
| Question | Answer on the available data |
|---|---|
| Does HPP produce a recognizable, connected, timing-preserving rough motion? | **Yes** for both HPP configurations on both clips (H1–H7 pass, visually recognizable). |
| Do HPP's sparse observations let pretrained CondMDI rebuild the action (Exp A)? | **Walking: yes, within tolerance** (v0.2 schedule + root every frame: all C1–C7 pass, 3/3 samples). **Tennis: no** (constraints 6–8 cm, bone stretch, contact timing). |
| Does the full pipeline (rough motion → features → CondMDI, Exp B) work? | **Walking: nearly** — every check except jitter passes on all samples; one of 3 samples is 5 % over the jitter limit. **Tennis: no**, and worse than Exp A. |
| Which root policy? | Observing the root on **every frame** is consistently better (constraint error 4.5–4.9 vs 5.4–6.9 cm on the walk); with root only on observed frames the walk fails C2. |
| Is the result "natural"? | Visually plausible human motion on both clips, but CondMDI adds high-frequency jitter: jerk ≈ 145–165 m/s³ vs 21 for the original walk (the model's own reference run on the same clip: 155). |

Evidence is limited to **two clips** (one walk, one tennis swing) because the full HumanML3D motion set requires AMASS.

## Experiments

### E1 — HPP projection and standalone validation (Phase 3–4)
* **Tested:** `hpp_v02` (0.12 s schedule), `hpp_adaptive` (extension), `robotic_slerp` (experimental, not HPP).
* **Input:** 000021 (walk in a circle, 179 frames, 8.95 s), 012314 (tennis swing, 170 frames, 8.5 s).
* **Expected:** connected skeleton with exact bone lengths, root and end effectors within 5 cm, contact events within ±0.1 s, deterministic.
* **Observed:**

| clip | projector | keys / trajectory | observed joint-frames | MPJPE | worst end-effector mean | root | contact recall | H1–H7 |
|---|---|---|---|---|---|---|---|---|
| 000021 | hpp_v02 | 75 | 525 / 3938 (13 %) | 0.8 cm | 0.2 cm | 0.1 cm | 1.00 | pass |
| 000021 | hpp_adaptive | 25 | 205 / 3938 (5 %) | 1.7 cm | 1.6 cm | 0.7 cm | 1.00 | pass |
| 012314 | hpp_v02 | 72 | 504 / 3740 (13 %) | 2.7 cm | 0.8 cm | 0.3 cm | 0.95 | pass |
| 012314 | hpp_adaptive | 30 | 254 / 3740 (7 %) | 3.2 cm | 1.5 cm | 0.8 cm | 0.95 | pass |
| 012314 | robotic_slerp | 20 | 140 / 3740 | 4.0 cm | 6.3 cm | 3.2 cm | 0.61 | fail H4, H6 |

  Bone lengths are exact (max deviation 3e-16 m); two runs are bit-identical; the original is never modified.
* **Visual (H8):** both rough motions are clearly recognizable as the source action. The rigid canonical torso and
  fixed elbow/knee bend directions are visible (elbow pose differs from the original during the tennis swing).
* **Failure cases:** the largest rough-motion errors are at elbows and knees (canonical IK poles), as intended.
  At 20 fps the v0.2 0.12 s rule keys every 2–3 frames (13 % of joint-frames), which is not very sparse.

### E2 — Reference CondMDI run (model baseline, not HPP)
Original full body every 5 frames: constraint error 6.2 cm (walk) / 10.9 cm (tennis); jerk 155 / 428 m/s³.
Even with dense, complete observations the model does not reproduce observed joints exactly.

### E3 — Experiment A: HPP key selection, real features at HPP entries (Phase 5)
* **Configuration:** `condmdi.yaml` (no text, pos_rot_vel, seeds 10/11/12), torso orientation not mapped, root policies `observed_frames` and `dense`.
* **Expected:** constraint error ≤ 5 cm mean / 15 cm p95 at HPP-selected joint-frames; other checks C1–C7.
* **Observed** (mean of 3 samples; constraint error vs the original at HPP entries):

| clip | projector | root | constraint mean / p95 | MPJPE | contact recall | jerk (ref) | C1–C7 all samples |
|---|---|---|---|---|---|---|---|
| 000021 | hpp_v02 | dense | **4.7 / 10.8 cm** | 4.7 cm | 1.00 | 145 (155) | **pass 3/3** |
| 000021 | hpp_v02 | observed | 6.5 / 13.4 | 6.4 | 1.00 | 148 | C2 fails |
| 000021 | hpp_adaptive | dense | 4.9 / 11.3 | 4.5 | 1.00 | 150 | C2 fails 1/3 |
| 000021 | hpp_adaptive | observed | 5.5 / 12.7 | 5.1 | 1.00 | 150 | C2 fails |
| 012314 | hpp_v02 | dense | 6.1 / 13.2 | 6.3 | 0.79 | 439 (428) | C1, C2 fail; C4 and C6 fail on 2/3 |
| 012314 | hpp_adaptive | dense | 7.0 / 14.5 | 6.8 | 0.77 | 425 | C1, C2, C6 fail |

* **Reproducibility (C8):** rerunning `000021__hpp_v02__A__observed_frames` with the same seeds gives bit-identical output (max |diff| 0 m).

### E4 — Experiment B: full pipeline from the rough motion (Phase 6–7)
* **Feature adapter validation:** positions → 263-d → positions round trip < 1e-6 m on both clips, orientation
  < 0.1°, re-extracted features match the dataset's own (contacts identical). A notebook bug (yaw sign at ±180°
  facing, 16 cm drift on the walk) was found and fixed.
* **Observed** (constraint error vs the rough motion at HPP entries):

| clip | projector | root | constraint mean / p95 | MPJPE vs original | contact recall | jerk (ref) | C1–C7 |
|---|---|---|---|---|---|---|---|
| 000021 | hpp_v02 | dense | **4.5 / 10.5 cm** | 4.8 cm | 1.00 | 165 (155) | all pass except **C7 on 1/3 samples** (179 vs limit 170) |
| 000021 | hpp_adaptive | dense | 4.8 / 11.5 | 4.5 | 1.00 | 162 | C7 fails 1/3 (170.6) |
| 000021 | hpp_v02 | observed | 6.9 / 14.4 | 7.0 | 1.00 | 164 | C2 fails |
| 012314 | hpp_v02 | dense | 7.7 / 17.6 | 7.9 | 0.76 | 426 | C1, C2, C6 fail |
| 012314 | hpp_adaptive | dense | 9.8 / 21.6 | 9.0 | 0.84 | 411 | C1, C2 fail; min foot −2.0 cm |
| 000021 | robotic_slerp | dense | 6.2 / 12.6 | 5.9 | 0.94 | 157 | C2, C7 (1/3) fail |
| 012314 | robotic_slerp | dense | 7.3 / 17.0 | 7.9 | 0.73 | 412 | C1, C2, C4, C6 fail |

* **Velocity:** at constrained joint-frames the output's velocity differs from the rough motion's by 0.17 m/s (walk)
  and ~0.38 m/s (tennis); primary-joint speed profiles correlate 0.98 (walk) and 0.95 (tennis). Velocities are followed only approximately.

### Visual observations (C9, static frames reviewed in the viewer)
* Walk (A and B): all samples look like natural walking along the circle; feet land on their targets, hands are
  close to theirs. Root markers sit on the pelvis.
* Tennis (A): samples reproduce the crouch-and-swing posture; hands miss their targets by several cm mid-swing.
* Static frames cannot show jitter; the jerk measurements above show CondMDI output is ~7× jerkier than the
  original walk. This should be checked by watching the viewer at speed 1.

## Acceptance against handoff sec. 34
| # | Criterion | Walk (000021) | Tennis (012314) |
|---|---|---|---|
| 1 | HPP recognizable simplified action | pass | pass |
| 2 | HPP skeleton connected | pass | pass |
| 3 | Root / end-effector geometry preserved | pass | pass |
| 4 | CondMDI produces valid motion from custom conditioning | pass | **fail C1** (bones stretch up to 6.8 cm; the reference run also stretches 4.9 cm) |
| 5 | Observations satisfied within tolerance | pass (A and B, root dense) | **fail** (6–10 cm mean) |
| 6 | Perceptually plausible | plausible in stills; jitter above the original | plausible in stills; misses hand targets |
| 7 | Reproducible with fixed seeds | pass (bit-identical) | pass (same mechanism) |

## Remaining integration problems
1. **Data:** only two HumanML3D motions are available; conclusions need many more clips (AMASS required).
2. **Original HPP v0.2 source** is missing; the baseline is a reconstruction (see implementation notes D1–D7).
3. **Constraint residual:** pretrained CondMDI misses sparse observations by 4.5–10 cm; imputation makes them exact
   but adds discontinuities (pilot). Candidates: post-generation constrained IK, stronger guidance, fine-tuning on
   (HPP(motion) → motion) pairs — none attempted yet.
4. **Bone-length drift** in CondMDI output on fast motion (positions are decoded without a skeleton constraint);
   a post-generation projection onto fixed bone lengths would fix C1 but is not implemented.
5. **Jitter:** CondMDI output is much jerkier than the source motion; a temporal filter or a different sampler
   could be evaluated against C7.
6. **Torso orientation** from HPP is not conditioned (no matching CondMDI feature except via joints).
7. **Rough-motion features are partly out of distribution** (rigid torso, canonical elbow poles): Exp B is worse
   than Exp A on the tennis clip.
8. **No learned naturalness metric** (T2M evaluators could not be downloaded).
9. **Sequences > 196 frames** are rejected; segmentation is not implemented.
10. **v0.2 schedule at 20 fps** keeps 13 % of joint-frames; whether that counts as "sparse" for the research
    question needs a decision (the adaptive extension keeps 5–7 % with similar CondMDI results).

## Next corrective actions (proposed, not done)
1. Obtain AMASS and rerun on a set of walking / reaching / sitting clips.
2. Provide the original HPP v0.2 source to replace the reconstruction where they differ.
3. Try a post-generation skeleton projection + constraint IK and re-score C1/C2/C7.
