#!/usr/bin/env bash
set -e

echo "==> Initializing conda"
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> Creating conda environment: py310"
conda create -n py310 python=3.10 -y

echo "==> Activating environment: py310"
conda activate py310

echo "==> Verifying Python version"
python --version

echo "==> Installing project (pip install .)"
pip install .

echo "==> Installing ipykernel"
pip install ipykernel

echo "==> Registering Jupyter kernel: Python (py310)"
python -m ipykernel install --user --name py310 --display-name "Python (py310)"

echo "==> Preparing datasets"
bash scripts/prepare_datasets.sh "./datasets"

echo "==> Setup completed successfully"
