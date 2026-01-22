#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

# --- Clean up old artifacts to prevent Permission Errors
echo "==> Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info __pycache__

# --- Initialize Miniforge ---
echo "==> Initializing Mamba..."
# We explicitly point to Miniforge install to be safe
export MINIFORGE_ROOT="$HOME/miniforge3"
export MAMBA_ROOT_PREFIX="$MINIFORGE_ROOT"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
source "$HOME/miniforge3/etc/profile.d/mamba.sh"

# --- Create Environment (Faster with Mamba) ---
echo "==> Creating environment: py310..."
# --force reinstalls it if it already exists, ensuring a fresh start
mamba create -n py310 python=3.10 -y 

echo "==> Activating environment..."
conda activate py310

echo "==> Verifying Python..."
python --version

# --- Install Project ---
echo "==> Installing project..."
pip install .

echo "==> Installing Kernel..."
pip install ipykernel
python -m ipykernel install --user --name py310 --display-name "Python (py310)"

echo "==> Preparing datasets..."
bash scripts/prepare_datasets.sh "./datasets"

echo "==> Setup completed successfully!"
