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
GPU stages took ~110 s per condition with the original sampler (3 samples × 1000 diffusion steps, two model passes
per step). The default is now 10 respaced DDPM steps, one model pass, batched samples (see "Fast sampling" below).

## Robotic mocap input (CMU BVH, no rougher)
Already-robotic motion is fed to CondMDI unchanged (projector `direct`, `hpp/direct_input.py`): rough = input, CondMDI
observes the full body every `key_every_s` (`configs/direct.yaml` 0.5 s, `direct_1s.yaml` 1 s, `direct_1p5s.yaml`
1.5 s) and the root on every frame.
```bash
mkdir -p /root/cmu-robotic && cd /root/cmu-robotic
for f in 120/120_21 054/54_19 020/20_08 074/74_20 132/132_41 105/105_08 104/104_03 104/104_41 120/120_22 091/91_19; do
  wget -q https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/$f.bvh
done
cd -
python -m motion_project.experiments.prepare_cmu_bvh /root/cmu-robotic/*.bvh   # -> dataset/cmu_<id>.npy (<= 9.8 s)
python -m motion_project.experiments.run_cmu --projector direct_1s             # one clip at a time, viewer updated after each
python -m motion_project.evaluation.compare_realism --projector direct_1s      # input vs output realism indicators
```
Viewer: `viewer/cmu.html` (input vs 3 CondMDI samples, key-spacing selector, refreshes while clips are running).
Finding: the CMU input is already real mocap; CondMDI's output follows its timing but adds jitter (jerk ~3x the
input) and shifts poses by 6-12 cm at the keys, so it is not more realistic than the input
(`outputs/reports/realism__direct*.json`).

## Fast sampling
`configs/condmdi.yaml`: `sampler: ddpm`, `sampling_steps: 10`, `cpu_graph_fusion: true`.
* Respaced DDPM (the repo's `SpacedDiffusion`) with 10 steps matches 1000 steps on every quality metric, including
  pose range and sample diversity; below 8 steps motion starts to flatten
  (`python -m motion_project.experiments.benchmark_sampler --baseline_dir motion_project/outputs/generated_ddpm1000_backup`,
  results in `outputs/reports/sampler_benchmark__cuda.json`; 1000-step reference outputs in `outputs/generated_ddpm1000_backup/`).
* Text is dropped with a single unconditional pass (identical to the classifier-free wrapper at guidance 0), the
  CLIP text encoding is computed once per call, and repetitions are sampled as one batch.
* CPU: CLIP runs in fp32, threads are capped at 16, and the UNet is traced + frozen with oneDNN graph fusion
  (same output, ~1.5x faster per step).

One 9.8 s clip (1 s keys): GPU ~0.4 s for 3 batched samples (was ~110 s); CPU on 8 cores ~0.75 s per sample (was 178 s).
Inputs longer than 196 frames (9.8 s) are rejected; segmentation is not implemented.

The comparison viewer (`evaluation/comparison_viewer.html`) is served from `viewer/` (which holds three.js):
`viewer/hpp.html -> ../motion_project/evaluation/comparison_viewer.html`, `viewer/data -> ../motion_project/outputs/viewer_data`.
```bash
cd viewer && python3 -m http.server 8000      # then open http://localhost:8000/hpp.html
```

## Configuration
`configs/hpp_v02.yaml` (baseline), `hpp_adaptive.yaml` (extension), `robotic_slerp.yaml` (experimental),
`condmdi.yaml` (checkpoint, seeds, feature mode, text, sampler / steps / CPU fusion), `direct*.yaml` (CMU key spacing), `experiment.yaml` (clips, root policies, experiments A/B),
`acceptance.yaml` (tolerances fixed before evaluation).

## Outputs
`outputs/original/`, `outputs/rough/`, `outputs/generated/` (motions + run records with seeds, checkpoint sha256,
masks), `outputs/reports/` (metrics per stage, `acceptance.json`, `acceptance_tables.md`).
