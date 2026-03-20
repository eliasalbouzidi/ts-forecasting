#!/bin/bash

#SBATCH --job-name=mds_project_test
#SBATCH --partition=gpu_prod_long
#SBATCH --constraint=sh
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=/usr/users/diffusionts/albouz_eli/ts-forecasting/logs/output_%j.log

set -euo pipefail

cd ~/ts-forecasting
mkdir -p logs

CONDA_SH="/usr/users/diffusionts/albouz_eli/miniforge3/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: ${CONDA_SH} not found." >&2
  exit 1
fi
source "${CONDA_SH}"

conda activate py310
export PYTHON_BIN="python"
export DATA_DIR="${DATA_DIR:-$PWD/datasets}"
export LOG_DIR="${LOG_DIR:-$PWD/results}"
export RUN_ARGS="${RUN_ARGS:-}"

# bash slurm_scripts/run_nlinear_l1_l2.sh
# bash slurm_scripts/run_itransformer_l1_l2.sh
# bash slurm_scripts/run_armd_linear_exchange_l1_l2.sh
bash slurm_scripts/run_armd_tsmixer_exchange_l1_l2.sh
# bash slurm_scripts/run_armd_dlinear_exchange_l1_l2.sh
