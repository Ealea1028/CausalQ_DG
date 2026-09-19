#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source scripts/activate_autodl.sh

MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
VALIDATION_MAX_SAMPLES="${VALIDATION_MAX_SAMPLES:-50}"
RUN_ID="${RUN_ID:-A5_FULL_SMOKE_${MAX_ITERATIONS}_$(git rev-parse --short HEAD)}"

python tools/train.py \
  --config configs/full/gta_dinov3l_full.yaml \
  --run-id "$RUN_ID" \
  --max-iterations "$MAX_ITERATIONS" \
  --validation-max-samples "$VALIDATION_MAX_SAMPLES"
