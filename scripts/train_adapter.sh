#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
CONFIG=${1:-configs/train_adapter_library_update.yaml}
python -m voln_uav.cli.train_adapter --config "$CONFIG"