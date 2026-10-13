# Architecture (current implementation)

```
HumanML3D clip ──► hpp/humanml3d_loader ──► positions [T,22,3] + abs features [T,263]
                                                │
                     ┌──────────────────────────┤ (original kept unchanged, stored separately)
                     ▼                          │
     hpp/hpp_projector  (deterministic)         │
       keyframe_selection ─► selected_observations (per trajectory: frames, values, reasons)
       hermite            ─► dense root / hand / foot / torso-axis curves
       canonical_ik       ─► rough_positions [T,22,3]  (rigid torso + two-bone IK, exact bone lengths)
                     │                          │
                     ▼                          ▼
     realizer/condmdi_mask_adapter      Experiment A: values = original features
       observation_mask [T,22] ─► joint mask (+ root policy) ─► feature mask [263,196]
                     │          Experiment B: values = motion_feature_adapter(rough_positions)
                     ▼
     realizer/condmdi_adapter  (pretrained CondMDI, unmodified sampler) ─► generated [R,T,22,3]
                     │
                     ▼
     evaluation/metrics + acceptance ─► outputs/reports/*.json, acceptance_tables.md
     experiments/export_viewer       ─► evaluation/comparison_viewer.html (original | rough | generated)
```

Separation of roles (handoff sec. 43): HPP never calls the model; the realizer only sees the observation mask and
the values of one chosen source motion; evaluation is the only stage that reads both the original and the outputs.

## Module map
| Module | Role |
|---|---|
| `hpp/skeleton.py` | joint names, parents, chains, left/right pairs, end effectors, coordinate conventions (tested against CondMDI constants) |
| `hpp/humanml3d_loader.py` | load processed clips or raw joint files; validation |
| `hpp/keyframe_selection.py` | v0.2 0.12 s schedule; adaptive extension |
| `hpp/hermite.py` | cubic Hermite with Fritsch–Carlson tangents |
| `hpp/canonical_ik.py` | body frame, canonical torso, two-bone IK, decoder |
| `hpp/hpp_projector.py` | `hpp_project(positions, fps, config)` → output contract |
| `hpp/robotic_slerp.py` | experimental robotic rough motion (not HPP) |
| `realizer/motion_feature_adapter.py` | positions ↔ 263-d features, raw-joint preprocessing |
| `realizer/condmdi_mask_adapter.py` | observation masks → CondMDI masks / obs_x0, leakage guard |
| `realizer/condmdi_adapter.py` | `CondMDIRealizer.realize(...)` + run record |
| `experiments/*.py` | runnable stages; `common.py` holds shared plumbing |
| `evaluation/metrics.py`, `acceptance.py` | metrics and pre-registered acceptance checks |
