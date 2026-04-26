#!/bin/bash
# Stage 1: train user_bias + bias_proj on frozen pretrained backbone.
# Usage: bash scripts/train.sh [--backbone path] [--chunk_weights path] [extra args]
#
# Basic run (MSE + KL):
#   bash scripts/train.sh --backbone outputs/dp_ucvla/pretrain/step-20000/backbone.pt
#
# With preference-weighted MSE (run compute_chunk_weights.py first):
#   bash scripts/train.sh --backbone outputs/dp_ucvla/pretrain/step-20000/backbone.pt \
#                         --chunk_weights data/chunk_weights.pt

set -euo pipefail

PYTHONPATH="$(pwd)" uv run accelerate launch \
    --num_processes=1 \
    scripts/train.py \
    --config configs/dp_ucvla.yaml \
    --data_dir data/ \
    --output_dir outputs/dp_ucvla/stage1_10d \
    "$@"
