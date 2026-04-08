#!/bin/bash
# Usage: bash scripts/eval_confusion.sh [step]
# Defaults to step-10000 if no argument given.

set -euo pipefail

STEP="${1:-10000}"

PYTHONPATH="$(pwd)" uv run python scripts/eval_confusion.py \
    --weights "outputs/dp_ucvla/stage1/step-${STEP}/ucvla_weights.pt" \
    --config  configs/dp_ucvla.yaml \
    --data_dir data/
