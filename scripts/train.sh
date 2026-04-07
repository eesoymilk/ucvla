#!/bin/bash
# UCVLA proxy training launcher
# Usage: bash scripts/train.sh [extra args]
#
# Ablation matrix (edit lambda_* in configs/dp_ucvla.yaml or pass overrides):
#   Run 1: lambda_triplet=0.0  lambda_ortho=0.0   (MSE only — baseline)
#   Run 2: lambda_triplet=0.1  lambda_ortho=0.0   (MSE + triplet)
#   Run 3: lambda_triplet=0.0  lambda_ortho=0.01  (MSE + ortho)
#   Run 4: lambda_triplet=0.1  lambda_ortho=0.01  (MSE + triplet + ortho)

set -euo pipefail

PYTHONPATH="$(pwd)" uv run accelerate launch \
    --num_processes=1 \
    scripts/train.py \
    --config configs/dp_ucvla.yaml \
    --data_dir data/ \
    --output_dir outputs/dp_ucvla/stage1 \
    "$@"
