#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA="${1:?Pass the full Git SHA supplied by Codex}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo invalid_full_sha; exit 1; }
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }

PY="${PY:-python3}"
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
INVENTORY="$BASE/phase17/GTA5_SOURCE_STRUCTURE_f41dbee_20261016T135221_54521/inventory.json"
test -f "$INVENTORY" || { echo "missing_inventory=$INVENTORY"; exit 1; }
OUT="$BASE/phase17/GTA5_SOURCE_SPLIT_${EXPECTED_SHA:0:7}_$(date -u +%Y%m%dT%H%M%S)_$$"
LOG="${OUT}.log"
test ! -e "$OUT" || { echo existing_output; exit 1; }
test ! -e "$LOG" || { echo existing_log; exit 1; }
export CUDA_VISIBLE_DEVICES=""

set +e
"$PY" -m tools.build_gta5_source_split \
  --inventory "$INVENTORY" --source-val-fraction 0.1 \
  --output-dir "$OUT" 2>&1 | tee "$LOG"
PIPE_EXITS=("${PIPESTATUS[@]}")
set -e
SPLIT_EXIT=${PIPE_EXITS[0]}
if [[ "${PIPE_EXITS[1]}" != 0 ]]; then SPLIT_EXIT=1; fi
echo "split_exit_code=$SPLIT_EXIT"
if [[ -f "$OUT/SHA256SUMS" ]]; then (cd "$OUT" && sha256sum -c SHA256SUMS); fi
if [[ -f "$OUT/source_split_export.zip" ]]; then sha256sum "$OUT/source_split_export.zip"; fi
sha256sum "$LOG"
git rev-parse HEAD
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_export; exit 1; }
echo "Return $OUT/source_split_export.zip and $LOG plus the complete console output."
echo "STOP: candidate split is not frozen in Git; no training or target access."
exit "$SPLIT_EXIT"
