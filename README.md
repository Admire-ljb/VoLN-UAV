# VoLN-UAV: Vision-only Language-Model-based Navigation for UAVs

## Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Preparation](#preparation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Acknowledgement](#acknowledgement)

# Introduction

This repository provides the benchmark construction pipeline, training code, offline evaluation, and online data collection tools for VoLN-UAV.

- 🤗 VoLN-UAV-ENV (Environment assets, Hugging Face): https://huggingface.co/datasets/Louj/VoLN-UAV-ENV/tree/main
- 🤗 VoLN-UAV-Dataset (Navigation data): coming soon

Main entry points:

- benchmark construction from source scenes and routes
- adapter training
- planner training
- offline closed-loop evaluation
- optional DAgger-style data collection
- AirSim environment launcher
- toy data and configs for quick verification

# Dependencies

### Create `voln-uav` environment

```bash
conda create -n voln-uav python=3.10 -y
conda activate voln-uav
pip install -r requirement.txt
pip install -e .
```

If you want to use external vision/language backbones, install the optional dependencies:

```bash
pip install -e .[real]
```

Notes for real backbones:

- names beginning with `hf:` use `transformers.AutoModel` (for example `hf:facebook/dinov2-base`)
- names beginning with `open_clip:` use OpenCLIP image encoder (`open_clip:<model_name>[:<pretrained_tag>]`, e.g. `open_clip:ViT-B-32:laion2b_s34b_b79k`)

If you need a specific CUDA build of PyTorch, install the matching PyTorch package first, then run the commands above.

# Preparation

## Data

Prepare the source data in the following structure:

```text
your_source_root/
├── scenes.jsonl
├── preset_routes/
├── custom_routes/
└── frames/
```

Each route JSON should contain:

- `scene_id`
- `trajectory_id`
- `source`
- `goal_category`
- `states`

Each item in `states` should contain at least:

- `position`
- `yaw`
- `image`
- `imu`
- `odometry`

For a quick start, the repository already includes a toy source dataset under `data/toy_source`.

Before building a benchmark, update the config file you want to use:

- `configs/benchmark_toy.yaml` for the included toy example
- `configs/benchmark_library_update.yaml` for your own source data

The main fields to check are:

- `source_root`
- `output_root`
- `scene_manifest`
- `preset_routes_dir`
- `custom_routes_dir`
- `semantic_bank.categories`
- `beacons.task_category_allowlist`

## Simulator environments

If you want to connect the workflow to local AirSim/UE environments, use the launcher in `airsim_plugin/`.

```bash
python airsim_plugin/AirVoLNSimulatorServerTool.py \
  --root_path /path/to/your/envs \
  --scene urban_001 \
  --port 30000 \
  --dry_run
```

To use your own scene-to-executable mapping, provide `--mapping_json`.

# Usage

## 1. Generate toy source data

```bash
python examples/generate_toy_source.py --out_dir data/toy_source
```

## 2. Build benchmark

```bash
bash scripts/build_benchmark.sh configs/benchmark_toy.yaml
```

For your own data, update `configs/benchmark_library_update.yaml` and run:

```bash
bash scripts/build_benchmark.sh configs/benchmark_library_update.yaml
```

For real-data training/evaluation presets aligned with the baseline setting, use:

- `configs/train_adapter_library_update.yaml`
- `configs/train_planner_library_update.yaml`
- `configs/eval_library_update.yaml`

These presets keep the same hyperparameter intent as the original workflow while using updated file naming.

## 3. Train adapter

```bash
bash scripts/train_adapter.sh configs/train_adapter_toy.yaml
```

## 4. Train planner

```bash
bash scripts/train_planner.sh configs/train_planner_toy.yaml
```

## 5. Run offline evaluation

```bash
bash scripts/eval_offline.sh configs/eval_toy.yaml
```

## Optional: LLM-based visual beacon planning from manifest

If you have a manifest-style dataset and want to generate sign placements (`--sign_at`) for replay,
you can use:

```bash
python examples/annotate_visual_beacons.py \
  --manifest /path/to/manifest.jsonl \
  --dataset_root /path/to/dataset_root \
  --start 0 \
  --count 1 \
  --out_root /path/to/output \
  --replay_script examples/replay_fix_trajectory_normal.py
```

Environment variables required by this script:

- `XHANG_API_KEY`
- `XHANG_BASE_URL` (optional; defaults to xhang service URL)
- `XHANG_MODEL` (optional; defaults to `xhang`)

Replay helpers are also provided for difficulty presets:

- `examples/replay_fix_trajectory_easy.py`
- `examples/replay_fix_trajectory_normal.py`
- `examples/replay_fix_trajectory_hard.py`

## 6. Run online collection

```bash
bash scripts/dagger_toy.sh configs/dagger_toy.yaml
```

## 7. Run the full smoke test

```bash
bash scripts/smoke_test.sh
```

The smoke test runs the full toy pipeline in order:

1. toy source generation
2. benchmark construction
3. adapter training
4. planner training
5. offline evaluation
6. one round of online collection

## Python module entry points

You can also run each stage directly with Python:

```bash
python -m voln_uav.cli.build_benchmark --config configs/benchmark_toy.yaml
python -m voln_uav.cli.train_adapter --config configs/train_adapter_toy.yaml
python -m voln_uav.cli.train_planner --config configs/train_planner_toy.yaml
python -m voln_uav.cli.eval_offline --config configs/eval_toy.yaml
python -m voln_uav.cli.run_dagger --config configs/dagger_toy.yaml
```

# Outputs

After running the toy pipeline, the main outputs are written to:

- `data/toy_benchmark/`
  - `episodes.jsonl`
  - `train.jsonl`, `val.jsonl`, `test.jsonl`
  - `records/train.jsonl`, `records/val.jsonl`, `records/test.jsonl`
  - `semantic_bank/categories.txt`
  - `summary.json`
- `work_dirs/adapter_toy/`
  - adapter checkpoints and training metrics
- `work_dirs/planner_toy/`
  - planner checkpoints and training metrics
- `work_dirs/eval_toy/`
  - offline evaluation metrics
- `work_dirs/dagger_toy/`
  - collected rollout records and summary

The offline evaluator writes metrics including `SR`, `OSR`, `NE`, `nDTW`, `SPL`, `CT`, and `EER`.

# Acknowledgement

We thank the authors of TravelUAV and AirVLN for releasing their codebase and providing a clear engineering reference for UAV navigation projects.
