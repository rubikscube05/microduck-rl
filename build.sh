#!/bin/bash
#
# build.sh - create a Python environment for the Microduck RL project.
#
# This project currently reuses the environment from ../duck via a .venv
# symlink to avoid downloading PyTorch twice.  If the symlink is missing, this
# script creates a normal, standalone virtual environment instead.
#
set -e

cd "$(dirname "$0")"

if [ -e .venv ]; then
    echo "Using existing .venv"
else
    echo "Creating .venv"
    python3 -m venv .venv
fi

./.venv/bin/pip install --upgrade pip setuptools wheel

# Install the CUDA build of PyTorch first.  If you do not have an NVIDIA GPU,
# use the CPU index instead:
#   ./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install the remaining dependencies.
./.venv/bin/pip install -r requirements.txt

echo
echo "Environment ready."
echo "  View the model : ./.venv/bin/python view.py"
echo "  Train          : ./.venv/bin/python train.py --timesteps 1000000 --n-envs 4"
echo "  Watch          : ./.venv/bin/python watch.py"
