# HPP v0.2 × HumanML3D × CondMDI

Integration layer that tests whether a deterministic, simplified motion (HPP) can be turned back into natural
human motion by pretrained CondMDI while keeping its essential geometry and timing. It lives inside the CondMDI
repository and imports it; CondMDI itself is not modified.

* Design and module map: `docs/architecture.md`
* Environment audit: `docs/environment_report.md`
* Deviations / assumptions: `docs/implementation_notes.md`
* Results: `docs/evaluation_report.md` (tables generated into `outputs/reports/acceptance_tables.md`)

## Run (from the repository root, `/root/diffusion-motion-inbetweening`)
```bash
source .env_condmdi/bin/activate
export MPLBACKEND=Agg

python -m pytest -q motion_project/tests                       # unit tests (CPU, ~10 s)

python -m motion_project.experiments.run_hpp --projectors hpp_v02 hpp_adaptive robotic_slerp   # HPP + validation
python -m motion_project.experiments.run_reference              # CondMDI baseline per clip (full body every 5 frames)
python -m motion_project.experiments.run_mask_experiment --repro_check   # Experiment A
python -m motion_project.experiments.run_full_pipeline         # Experiment B
python -m motion_project.experiments.run_full_pipeline --projectors robotic_slerp   # optional, experimental
python -m motion_project.evaluation.acceptance                  # apply configs/acceptance.yaml
python -m motion_project.experiments.export_viewer              # data for the comparison viewer
```
GPU stages take ~110 s per condition (3 samples × 1000 diffusion steps).

The comparison viewer (`evaluation/comparison_viewer.html`) is served from `viewer/` (which holds three.js):
`viewer/hpp.html -> ../motion_project/evaluation/comparison_viewer.html`, `viewer/data -> ../motion_project/outputs/viewer_data`.
```bash
cd viewer && python3 -m http.server 8000      # then open http://localhost:8000/hpp.html
```

## Configuration
`configs/hpp_v02.yaml` (baseline), `hpp_adaptive.yaml` (extension), `robotic_slerp.yaml` (experimental),
`condmdi.yaml` (checkpoint, seeds, feature mode, text), `experiment.yaml` (clips, root policies, experiments A/B),
`acceptance.yaml` (tolerances fixed before evaluation).

## Outputs
`outputs/original/`, `outputs/rough/`, `outputs/generated/` (motions + run records with seeds, checkpoint sha256,
masks), `outputs/reports/` (metrics per stage, `acceptance.json`, `acceptance_tables.md`).
