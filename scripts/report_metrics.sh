#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:-runs/eval_offline_dataset_release}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -gt 0 && "$1" != -* ]]; then
  RUN_DIR="$1"
  shift
fi

exec "${PYTHON}" -m voln_uav.cli.report_metrics --run-dir "${RUN_DIR}" "$@"
