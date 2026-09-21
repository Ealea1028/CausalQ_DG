#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source scripts/activate_autodl.sh

QUERY_COUNT="${QUERY_COUNT:?Set QUERY_COUNT to 1, 2, or 4}"
case "$QUERY_COUNT" in
  1|2|4) ;;
  *) echo "QUERY_COUNT must be 1, 2, or 4" >&2; exit 2 ;;
esac

MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
VALIDATION_MAX_SAMPLES="${VALIDATION_MAX_SAMPLES:-50}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-Q${QUERY_COUNT}_QUERY_COUNT_SMOKE_${MAX_ITERATIONS}_$(git rev-parse --short HEAD)}"

python tools/train.py \
  --config configs/query_count/gta_dinov3l.yaml \
  --run-id "$RUN_ID" \
  --max-iterations "$MAX_ITERATIONS" \
  --validation-max-samples "$VALIDATION_MAX_SAMPLES" \
  --seed "$SEED" \
  --queries-per-class "$QUERY_COUNT"
