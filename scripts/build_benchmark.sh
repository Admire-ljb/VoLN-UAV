#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
CONFIG=${1:-configs/benchmark_dataset_release.yaml}
python -m voln_uav.cli.build_benchmark --config "$CONFIG"
