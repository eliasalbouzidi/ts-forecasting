#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/datasets}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT_DIR}/logs}"
RUN_ARGS="${RUN_ARGS:-}"

DATASET="exchange_ltsf"
read -r -a EXTRA_ARGS <<< "${RUN_ARGS}"

mkdir -p "${RUN_LOG_DIR}" "${LOG_DIR}"

LAST_PID=""

run_loss() {
  local loss="$1"
  local config_path="${ROOT_DIR}/config/test_config/armd_tsmixer/${DATASET}/${loss}.yaml"
  local log_path="${RUN_LOG_DIR}/armd_tsmixer_${DATASET}_${loss}.log"

  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" run.py \
      --config "${config_path}" \
      --data.data_manager.init_args.path "${DATA_DIR}" \
      --trainer.default_root_dir "${LOG_DIR}" \
      "${EXTRA_ARGS[@]}"
  ) > "${log_path}" 2>&1 &

  LAST_PID=$!
}

echo "=== armd_tsmixer / ${DATASET}: l1 + l2 in parallel ==="
run_loss "l1"
pid_l1="${LAST_PID}"
run_loss "l2"
pid_l2="${LAST_PID}"

s1=0
s2=0
wait "${pid_l1}" || s1=$?
wait "${pid_l2}" || s2=$?

echo "Completed armd_tsmixer/${DATASET}: l1=${s1}, l2=${s2}"

if [[ ${s1} -ne 0 || ${s2} -ne 0 ]]; then
  exit 1
fi
