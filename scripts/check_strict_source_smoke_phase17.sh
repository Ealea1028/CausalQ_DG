#!/usr/bin/env bash
set -euo pipefail
EXPECTED_SHA="${1:?Pass the full Git SHA supplied by Codex}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo invalid_full_sha; exit 1; }
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
test -x "$PY" || { echo missing_dinov3_runtime; exit 1; }
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
DATA="${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}"
WEIGHTS="${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
OUT="$BASE/phase17/M0_STRICT_SOURCE_SMOKE_${EXPECTED_SHA:0:7}"
LOG="${OUT}.log"
mkdir -p "$BASE/phase17"
test ! -e "$OUT" || { echo existing_output; exit 1; }
test ! -e "$LOG" || { echo existing_log; exit 1; }
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
set +e
"$PY" -m tools.check_strict_source_smoke --data-root "$DATA" \
  --pretrained-root "$WEIGHTS" --output-dir "$OUT" 2>&1 | tee "$LOG"
PIPE_EXITS=("${PIPESTATUS[@]}")
set -e
SMOKE_EXIT=${PIPE_EXITS[0]}
if [[ "${PIPE_EXITS[1]}" != 0 ]]; then SMOKE_EXIT=1; fi
echo "source_smoke_exit_code=$SMOKE_EXIT"
if [[ -f "$OUT/SHA256SUMS" ]]; then (cd "$OUT" && sha256sum -c SHA256SUMS); fi
sha256sum "$LOG"
git rev-parse HEAD
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_smoke; exit 1; }
echo "Return $OUT/smoke_report.json, $OUT/train.jsonl, $OUT/SHA256SUMS and final launcher output"
echo "Keep $OUT/final.pth on AutoDL. STOP: no long training or target access."
exit "$SMOKE_EXIT"
