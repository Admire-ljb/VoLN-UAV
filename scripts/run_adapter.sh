#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
CONFIG=${1:-configs/dagger_library_update.yaml}
python -m voln_uav.cli.run_dagger --config "$CONFIG"