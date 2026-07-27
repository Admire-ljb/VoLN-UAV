#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/eval_airsim_dataset_release.yaml}"
EPISODES_FILE="${EPISODES_FILE:-episodes.jsonl}"
BASELINE="${BASELINE:-random}"
TRIALS="${TRIALS:-10}"
EVAL_MODE="${EVAL_MODE:-normal}"
SCENE="${SCENE:-BrushifyUrban}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

reference_bootstrap_args=()
if [[ "${BASELINE,,}" == "reference" ]]; then
  reference_bootstrap_args=(--reference-bootstrap-steps 2)
fi

case "${EVAL_MODE,,}" in
  normal)
    eval_mode_args=(--control-mode move_to_position)
    if [[ "${BASELINE,,}" == "reference" ]]; then
      eval_mode_args=(--control-mode move_on_path)
    fi
    ;;
  fast)
    eval_mode_args=(
      --control-mode teleport
      --fast-reset
      --settle-sec 0.0
      --max-teleport-step-m 10.0
    )
    ;;
  exact)
    eval_mode_args=(
      --control-mode teleport
      --fast-reset
      --settle-sec 0.0
      --max-teleport-step-m 100.0
      --max-teleport-vertical-step-m 100.0
    )
    ;;
  *)
    printf 'Unsupported EVAL_MODE %q. Use normal, exact, or fast.\n' "${EVAL_MODE}" >&2
    exit 2
    ;;
esac

connection_args=()
if [[ -n "${AIRSIM_IP:-}" ]]; then
  connection_args+=(--ip "${AIRSIM_IP}")
fi
if [[ -n "${AIRSIM_PORT:-}" ]]; then
  connection_args+=(--port "${AIRSIM_PORT}")
fi

exec "${PYTHON}" -m voln_uav.cli.eval_online_baselines \
  --config "${CONFIG}" \
  --episodes-file "${EPISODES_FILE}" \
  --baseline "${BASELINE}" \
  --trials "${TRIALS}" \
  --scene "${SCENE}" \
  "${connection_args[@]}" \
  "${reference_bootstrap_args[@]}" \
  "${eval_mode_args[@]}" \
  "$@"
