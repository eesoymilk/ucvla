#!/bin/bash
# Usage: bash scripts/eval_confusion.sh <run_dir> [step]
# Example: bash scripts/eval_confusion.sh mse_kl_sdtw 4000

set -euo pipefail

RUN="${1:-mse_kl_sdtw}"
STEP="${2:-10000}"

PYTHONPATH="$(pwd)" uv run python scripts/eval_confusion.py \
    --weights "outputs/dp_ucvla/${RUN}/step-${STEP}/ucvla_weights.pt" \
    --config  configs/dp_ucvla.yaml \
    --data_dir data/
