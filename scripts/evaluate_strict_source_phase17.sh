#!/usr/bin/env bash
set -euo pipefail
EXPECTED_SHA="${1:?Pass full Git SHA from Codex}"
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
SMOKE="$BASE/phase17/M0_STRICT_SOURCE_SMOKE_b16236c"
for NAME in smoke_report.json train.jsonl final.pth; do
  test -f "$SMOKE/$NAME" || { echo "missing_original_smoke_file=$NAME"; exit 1; }
done
OUT="$BASE/phase17/M0_SOURCE_HOLDOUT_${EXPECTED_SHA:0:7}"
LOG="${OUT}.log"
test ! -e "$OUT" || { echo existing_output; exit 1; }
test ! -e "$LOG" || { echo existing_log; exit 1; }
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
set +e
"$PY" -m tools.evaluate_strict_source --data-root "$DATA" --pretrained-root "$WEIGHTS" \
  --smoke-dir "$SMOKE" --output-dir "$OUT" 2>&1 | tee "$LOG"
PIPE_EXITS=("${PIPESTATUS[@]}")
set -e
EVAL_EXIT=${PIPE_EXITS[0]}
if [[ "${PIPE_EXITS[1]}" != 0 ]]; then EVAL_EXIT=1; fi
echo "source_evaluation_exit_code=$EVAL_EXIT"
if [[ -f "$OUT/SHA256SUMS" ]]; then (cd "$OUT" && sha256sum -c SHA256SUMS); fi
sha256sum "$LOG"
git rev-parse HEAD
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_evaluation; exit 1; }
echo "Return source_evaluation.json, SHA256SUMS and final launcher output from $OUT"
echo "Preserve source_eval.jsonl and the original smoke. STOP: no training or target access."
exit "$EVAL_EXIT"
