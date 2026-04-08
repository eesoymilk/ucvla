#!/bin/bash
# Stage 0: pretrain DiT backbone on all users mixed (no bias).
# Run this before train.sh. Pass the output backbone to train.sh via --backbone.

set -euo pipefail

PYTHONPATH="$(pwd)" uv run accelerate launch \
    --num_processes=1 \
    scripts/pretrain.py \
    --config configs/dp_pretrain.yaml \
    --data_dir data/ \
    --output_dir outputs/dp_ucvla/pretrain \
    "$@"
