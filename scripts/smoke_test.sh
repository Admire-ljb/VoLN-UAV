#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
cd "$(dirname "$0")/.."
python examples/generate_toy_source.py --out_dir data/toy_source
bash scripts/build_benchmark.sh configs/benchmark_toy.yaml
bash scripts/train_adapter.sh configs/train_adapter_toy.yaml
bash scripts/train_planner.sh configs/train_planner_toy.yaml
bash scripts/eval_offline.sh configs/eval_toy.yaml
bash scripts/dagger_toy.sh configs/dagger_toy.yaml
