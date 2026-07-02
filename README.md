# VoLN-UAV: Vision-only Language-Model-based Navigation for UAVs

VoLN-UAV provides the benchmark construction pipeline, dataset release tools, training code, offline evaluation, and AirSim collection utilities for vision-only UAV navigation. The benchmark removes episode-level language instructions and evaluates agents that navigate from egocentric RGB observations, proprioception, visual goals, and visual-semantic beacons.

## Hugging Face Resources

- env: https://huggingface.co/datasets/Louj/VoLN-UAV-ENV
- dataset: https://huggingface.co/datasets/Louj/VoLN-UAV-Dataset

The environment package and the navigation dataset are intentionally separated. Download `env` for simulator assets and use `dataset` for routes, frames, split manifests, and benchmark metadata.

## Repository Layout

```text
src/voln_uav/
  benchmark/      Benchmark builder, visual goals, beacon augmentation, scene-level split
  data/           PyTorch datasets and dataset-release packaging utilities
  models/         DINO-to-CLIP adapter, semantic bank, planner, LoRA modules
  training/       Adapter/planner training and DAgger-style collection
  evaluation/     Offline closed-loop evaluation and metrics
  simulators/     Route replay and AirSim environment interfaces
  cli/            Command-line entry points
configs/          Toy and real-data configuration templates
examples/         Toy source generator and replay helpers
airsim_plugin/    AirSim/Unreal launcher utilities
```

## Installation

```bash
conda create -n voln-uav python=3.10 -y
conda activate voln-uav
pip install -r requirement.txt
pip install -e .
```

Optional real-backbone dependencies:

```bash
pip install -e .[real]
```

Model-name conventions:

- `hf:<model_name>` loads a Hugging Face `transformers.AutoModel` image backbone.
- `open_clip:<model_name>[:<pretrained_tag>]` loads an OpenCLIP image encoder.

Install the CUDA-specific PyTorch build first if your machine needs a non-default wheel.

## Quick Verification With Toy Data

```bash
python examples/generate_toy_source.py --out_dir data/toy_source
python -m voln_uav.cli.build_benchmark --config configs/benchmark_toy.yaml
python -m voln_uav.cli.train_adapter --config configs/train_adapter_toy.yaml
python -m voln_uav.cli.train_planner --config configs/train_planner_toy.yaml
python -m voln_uav.cli.eval_offline --config configs/eval_toy.yaml
```

The toy pipeline writes outputs to `data/toy_benchmark/` and `work_dirs/`.

## Dataset Release Preparation

To organize the local raw dataset into a Hugging Face-ready release package:

```bash
python -m voln_uav.cli.prepare_dataset_release \
  --source-root /path/to/source_root_a \
  --source-root /path/to/source_root_b \
  --source-root /path/to/source_root_c \
  --out-root D:/VoLN_dataset/VoLN-UAV-Dataset-release \
  --zip-path D:/VoLN_dataset/VoLN-UAV-Dataset-release.zip \
  --asset-mode index
```

`--asset-mode index` creates a compact ZIP containing route metadata, split manifests, checksums, and a Hugging Face dataset card. Use it for a quick metadata-only release check.

For a full upload package that copies selected egocentric RGB frames into the release tree, use:

```bash
python -m voln_uav.cli.prepare_dataset_release \
  --source-root /path/to/source_root_a \
  --source-root /path/to/source_root_b \
  --source-root /path/to/source_root_c \
  --out-root D:/VoLN_dataset/VoLN-UAV-Dataset-release-full \
  --zip-path D:/VoLN_dataset/VoLN-UAV-Dataset-release-full.zip \
  --asset-mode copy
```

The full package can be very large. Keep at least the raw-data size plus ZIP workspace available before running `copy` mode.

The generated release tree contains:

```text
VoLN-UAV-Dataset-release/
  README.md                  Hugging Face dataset card
  manifest.json              Dataset summary and HF links
  checksums.sha256           File checksums for the generated package
  source/
    scenes.jsonl             Scene manifest with split assignment
    preset_routes/           Route JSON files from preset-style trajectories
    custom_routes/           Route JSON files from custom-style trajectories
    frames/                  Copied RGB frames when asset_mode=copy
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
  metadata/                   Episode and source metadata
```

After uploading the ZIP contents to `dataset`, keep the `env` link in the dataset card so users can fetch simulator assets separately.

## Benchmark Construction From Release Metadata

Once a release source tree exists, update `configs/benchmark_dataset_release.yaml` if your dataset path differs:

```yaml
source_root: D:/VoLN_dataset/VoLN-UAV-Dataset-release-full/source
output_root: D:/VoLN_dataset/VoLN-UAV-Dataset-release-full/benchmark
scene_manifest: scenes.jsonl
preset_routes_dir: preset_routes
custom_routes_dir: custom_routes
```

Then run:

```bash
python -m voln_uav.cli.build_benchmark --config configs/benchmark_dataset_release.yaml
```

The builder writes:

- `episodes.jsonl`
- `train.jsonl`, `val.jsonl`, `test.jsonl`
- `records/train.jsonl`, `records/val.jsonl`, `records/test.jsonl`
- `semantic_bank/categories.txt`
- `summary.json`

## Full Dataset Training

For the packaged dataset at `D:/VoLN_dataset/VoLN-UAV-Dataset-release-full`, the ready-to-run configs are:

- `configs/benchmark_dataset_release.yaml`
- `configs/train_adapter_dataset_release.yaml`
- `configs/train_planner_dataset_release.yaml`
- `configs/eval_offline_dataset_release.yaml`
- `configs/eval_airsim_dataset_release.yaml`

Install the real-backbone dependencies before full training:

```bash
pip install -e .[real]
```

Run the real-data pipeline:

```bash
python scripts/run_dataset_release_pipeline.py --device cuda
```

This runs benchmark construction, adapter training, planner training, and offline evaluation. Outputs are written to:

```text
D:/VoLN_dataset/VoLN-UAV-Dataset-release-full/benchmark
D:/VoLN_dataset/VoLN-UAV-runs/adapter_dataset_release
D:/VoLN_dataset/VoLN-UAV-runs/planner_dataset_release
D:/VoLN_dataset/VoLN-UAV-runs/eval_offline_dataset_release
```

Planner training stores reusable image embeddings in the planner run directory and reuses them when training is resumed.

To run only selected stages:

```bash
python scripts/run_dataset_release_pipeline.py --stages build train-adapter --device cuda
python scripts/run_dataset_release_pipeline.py --stages train-planner offline-eval --device cuda
```

On Windows, the helper scripts use the dataset-release configs by default:

```bat
scripts\run_train_adapter.cmd
scripts\run_train_planner.cmd
scripts\run_eval_offline.cmd
```

Set `CONFIG`, `DEVICE`, or `PYTHON` before calling a script to override the defaults.

## Training And Evaluation

```bash
python -m voln_uav.cli.train_adapter --config configs/train_adapter_library_update.yaml
python -m voln_uav.cli.train_planner --config configs/train_planner_library_update.yaml
python -m voln_uav.cli.eval_offline --config configs/eval_library_update.yaml
```

The offline evaluator reports `SR`, `OSR`, `NE`, `nDTW`, `SPL`, `CT`, and `EER`.

## AirSim Environment Evaluation

```bash
python airsim_plugin/AirVoLNSimulatorServerTool.py \
  --root_path /path/to/envs \
  --scene BrushifyUrban \
  --port 41451 \
  --dry_run
```

Provide `--mapping_json` if your local executable names differ from the default scene mapping.

Check the AirSim setup before loading the model:

```bash
python -m voln_uav.cli.eval_airsim \
  --config configs/eval_airsim_dataset_release.yaml \
  --preflight
```

After the AirSim scene is running, evaluate the trained policy in the simulator:

```bash
python -m voln_uav.cli.eval_airsim \
  --config configs/eval_airsim_dataset_release.yaml \
  --device cuda
```

The AirSim evaluator connects to the running simulator, resets each episode to its start pose, captures live RGB observations, executes predicted waypoints with `moveToPositionAsync`, and writes metrics plus executed trajectories to `D:/VoLN_dataset/VoLN-UAV-runs/eval_airsim_dataset_release`.

For trajectory-level sanity checks, run the online reference and random-flight baselines. Both execute waypoints through AirSim movement commands by default:

```bat
set BASELINE=reference
set TRIALS=10
scripts\run_online_baseline.cmd --work-dir D:/VoLN_dataset/VoLN-UAV-runs/online_reference_pose_10

set BASELINE=random
set TRIALS=10
scripts\run_online_baseline.cmd --work-dir D:/VoLN_dataset/VoLN-UAV-runs/online_random_flight_10
```

To summarize a run directory as paper-style metrics:

```bat
scripts\report_metrics.cmd D:/VoLN_dataset/VoLN-UAV-runs/eval_airsim_dataset_release
```

To inspect the test scene before running full evaluation, start the local RGB stream. The stream command resets the selected episode, places route-aware visual beacons, and serves the live AirSim camera:

```bat
scripts\start_airsim_stream.cmd
```

Open `http://127.0.0.1:8765/` in a browser. Useful endpoints are `/status.json`, `/beacons.json`, `/snapshot.jpg`, `/takeoff.json`, `/reference.json?step=18&mode=status`, `/reference.json?step=18&mode=move_to_position`, and `/reference.json?step=18&mode=teleport`.

For a one-episode AirSim smoke test, run:

```bat
scripts\run_airsim_eval_smoke.cmd
```

Stop the stream before running AirSim evaluation so that only one client controls the simulator:

```bat
scripts\stop_airsim_stream.cmd
```

If you want the evaluator to launch scenes automatically, set `env.auto_launch: true` in `configs/eval_airsim_dataset_release.yaml` and update `configs/airsim_scene_mapping_dataset_release.json` to match the executable paths in your `env` package.

## Acknowledgement

We thank the authors of TravelUAV and AirVLN for releasing their codebase and providing useful engineering references for UAV navigation projects.
