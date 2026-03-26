#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
CONFIG=${1:-configs/eval_toy.yaml}
python -m voln_uav.cli.eval_offline --config "$CONFIG"
