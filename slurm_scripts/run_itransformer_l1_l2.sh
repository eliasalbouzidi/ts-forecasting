#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/datasets}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT_DIR}/logs}"
RUN_ARGS="${RUN_ARGS:-}"

DATASETS=(etth1 exchange_ltsf ettm1 solar_nips)
read -r -a EXTRA_ARGS <<< "${RUN_ARGS}"

mkdir -p "${RUN_LOG_DIR}" "${LOG_DIR}"

LAST_PID=""

run_loss() {
  local dataset="$1"
  local loss="$2"
  local config_path="${ROOT_DIR}/config/test_config/itransformer/${dataset}/${loss}.yaml"
  local log_path="${RUN_LOG_DIR}/itransformer_${dataset}_${loss}.log"

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

status=0

for dataset in "${DATASETS[@]}"; do
  echo "=== itransformer / ${dataset}: l1 + l2 in parallel ==="
  run_loss "${dataset}" "l1"
  pid_l1="${LAST_PID}"
  run_loss "${dataset}" "l2"
  pid_l2="${LAST_PID}"

  s1=0
  s2=0
  wait "${pid_l1}" || s1=$?
  wait "${pid_l2}" || s2=$?

  echo "Completed itransformer/${dataset}: l1=${s1}, l2=${s2}"
  if [[ ${s1} -ne 0 || ${s2} -ne 0 ]]; then
    status=1
  fi
done

exit "${status}"
