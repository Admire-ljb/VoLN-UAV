#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/eval_airsim_dataset_release.yaml}"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

connection_args=()
if [[ -n "${AIRSIM_IP:-}" ]]; then
  connection_args+=(--ip "${AIRSIM_IP}")
fi
if [[ -n "${AIRSIM_PORT:-}" ]]; then
  connection_args+=(--port "${AIRSIM_PORT}")
fi

exec "${PYTHON}" -m voln_uav.cli.eval_airsim \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  "${connection_args[@]}" \
  "$@"
