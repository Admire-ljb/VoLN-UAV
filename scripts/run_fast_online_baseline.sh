#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export BASELINE="${BASELINE:-random}"
export TRIALS="${TRIALS:-10}"
export EVAL_MODE="fast"

EPISODE_INDEX="${EPISODE_INDEX:-0}"
EPISODE_STRIDE="${EPISODE_STRIDE:-1}"
REFERENCE_STRIDE="${REFERENCE_STRIDE:-1}"
RANDOM_STEPS="${RANDOM_STEPS:-80}"
WORK_DIR="${WORK_DIR:-runs/${BASELINE}_test_${TRIALS}_fast}"

random_args=()
if [[ "${BASELINE,,}" == "random" ]]; then
  random_args=(--random-steps "${RANDOM_STEPS}")
fi

exec "${SCRIPT_DIR}/run_online_baseline.sh" \
  --episode-index "${EPISODE_INDEX}" \
  --episode-stride "${EPISODE_STRIDE}" \
  --reference-stride "${REFERENCE_STRIDE}" \
  "${random_args[@]}" \
  --work-dir "${WORK_DIR}" \
  "$@"
