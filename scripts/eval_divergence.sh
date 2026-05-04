#!/bin/bash
# Usage: bash scripts/eval_divergence.sh <run_dir> [step]
# Example: bash scripts/eval_divergence.sh stage1_10d 35000

set -euo pipefail

RUN="${1:-stage1_10d}"
STEP="${2:-50000}"

PYTHONPATH="$(pwd)" uv run python scripts/eval_divergence.py \
    --weights      "outputs/dp_ucvla/${RUN}/step-${STEP}/ucvla_weights.pt" \
    --chunk_weights data/chunk_weights.pt \
    --config        configs/dp_ucvla.yaml \
    --data_dir      data/
