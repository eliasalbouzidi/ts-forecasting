#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/datasets}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT_DIR}/logs}"
L1_GPU="${L1_GPU:-}"
L2_GPU="${L2_GPU:-}"
RUN_ARGS="${RUN_ARGS:-}"

read -r -a MODELS <<< "${MODELS:-nlinear itransformer}"
read -r -a DATASETS <<< "${DATASETS:-solar_nips etth1 ettm1 electricity_ltsf exchange_ltsf}"
read -r -a EXTRA_ARGS <<< "${RUN_ARGS}"

mkdir -p "${RUN_LOG_DIR}" "${LOG_DIR}"

run_config() {
  local model="$1"
  local dataset="$2"
  local loss="$3"
  local gpu="$4"
  local config_path="${ROOT_DIR}/config/test_config/${model}/${dataset}/${loss}.yaml"
  local log_path="${RUN_LOG_DIR}/${model}_${dataset}_${loss}.log"

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config: ${config_path}" >&2
    return 1
  fi

  echo "Launching ${model}/${dataset}/${loss}" >&2
  echo "  config: ${config_path}" >&2
  echo "  log:    ${log_path}" >&2

  (
    cd "${ROOT_DIR}"
    if [[ -n "${gpu}" ]]; then
      export CUDA_VISIBLE_DEVICES="${gpu}"
    fi

    "${PYTHON_BIN}" run.py \
      --config "${config_path}" \
      --data.data_manager.init_args.path "${DATA_DIR}" \
      --trainer.default_root_dir "${LOG_DIR}" \
      "${EXTRA_ARGS[@]}"
  ) > "${log_path}" 2>&1 &

  echo $!
}

overall_status=0

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "=== Running ${model} on ${dataset}: l1 + l2 in parallel ==="

    pid_l1="$(run_config "${model}" "${dataset}" "l1" "${L1_GPU}")" || {
      overall_status=1
      continue
    }
    pid_l2="$(run_config "${model}" "${dataset}" "l2" "${L2_GPU}")" || {
      overall_status=1
      wait "${pid_l1}" || true
      continue
    }

    status_l1=0
    status_l2=0

    wait "${pid_l1}" || status_l1=$?
    wait "${pid_l2}" || status_l2=$?

    echo "Completed ${model}/${dataset}: l1=${status_l1}, l2=${status_l2}"

    if [[ ${status_l1} -ne 0 || ${status_l2} -ne 0 ]]; then
      overall_status=1
    fi
  done
done

exit "${overall_status}"
