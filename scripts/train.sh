#!/bin/bash
# Stage 1: train user_bias + bias_proj on frozen pretrained backbone.
# Usage: bash scripts/train.sh [--backbone path/to/backbone.pt] [extra args]
#
# Run pretrain.sh first to get the backbone checkpoint, then:
#   bash scripts/train.sh --backbone outputs/dp_ucvla/pretrain/step-20000/backbone.pt
#
# Ablation matrix (edit lambda_* in configs/dp_ucvla.yaml):
#   Run 1: lambda_kl=0.0   lambda_sdtw=0.0   (MSE only — baseline)
#   Run 2: lambda_kl=0.001 lambda_sdtw=0.0   (MSE + KL)
#   Run 3: lambda_kl=0.0   lambda_sdtw=0.05  (MSE + sDTW)
#   Run 4: lambda_kl=0.001 lambda_sdtw=0.05  (MSE + KL + sDTW — default)

set -euo pipefail

PYTHONPATH="$(pwd)" uv run accelerate launch \
    --num_processes=1 \
    scripts/train.py \
    --config configs/dp_ucvla.yaml \
    --data_dir data/ \
    --output_dir outputs/dp_ucvla/stage1_10d \
    "$@"
