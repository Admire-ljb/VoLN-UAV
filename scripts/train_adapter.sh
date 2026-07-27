#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/train_adapter_dataset_release.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi

exec "${PYTHON}" -m voln_uav.cli.train_adapter \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  "$@"
