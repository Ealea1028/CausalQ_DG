#!/usr/bin/env bash
set -euo pipefail

# Call from the exact checkout; argument is the full SHA supplied by Codex.
EXPECTED_SHA="${1:?Pass the full Git SHA supplied by Codex}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo invalid_full_sha; exit 1; }
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }

# Standard library only: no environment installation, CUDA or weight loading.
PY="${PY:-python3}"
GTA5_ROOT="${GTA5_ROOT:-/root/autodl-tmp/datasets/gta5}"
OUTPUT_ROOT="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
OUT="$OUTPUT_ROOT/phase17/GTA5_SOURCE_STRUCTURE_${EXPECTED_SHA:0:7}_${STAMP}"
LOG="${OUT}.log"
mkdir -p "$OUTPUT_ROOT/phase17"
test ! -e "$OUT" || { echo existing_output; exit 1; }
test ! -e "$LOG" || { echo existing_log; exit 1; }
export CUDA_VISIBLE_DEVICES=""

set +e
"$PY" -m tools.inspect_gta5_source_structure \
  --gta5-root "$GTA5_ROOT" --output-dir "$OUT" 2>&1 | tee "$LOG"
PIPE_EXITS=("${PIPESTATUS[@]}")
INVENTORY_EXIT=${PIPE_EXITS[0]}
if [[ "${PIPE_EXITS[1]}" != 0 ]]; then
  echo console_log_write_failed
  INVENTORY_EXIT=1
fi
set -e
echo "inventory_exit_code=$INVENTORY_EXIT"
if [[ -f "$OUT/SHA256SUMS" ]]; then
  (cd "$OUT" && sha256sum -c SHA256SUMS)
fi
sha256sum "$LOG"
git rev-parse HEAD
git status --short
echo "Return inventory.json, SHA256SUMS, and console log: $OUT ; $LOG"
echo "STOP: source grouping review is pending. Do not train or read target data."
exit "$INVENTORY_EXIT"
