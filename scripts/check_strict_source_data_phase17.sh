#!/usr/bin/env bash
set -euo pipefail
EXPECTED_SHA="${1:?Pass the full Git SHA supplied by Codex}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo invalid_full_sha; exit 1; }
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
test -x "$PY" || { echo "missing_cpu_data_runtime=$PY"; exit 1; }
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
DATA="${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}"
OUT="$BASE/phase17/STRICT_SOURCE_DATA_${EXPECTED_SHA:0:7}"
LOG="${OUT}.log"
mkdir -p "$BASE/phase17"
test ! -e "$OUT" || { echo existing_output; exit 1; }
test ! -e "$LOG" || { echo existing_log; exit 1; }
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
set +e
"$PY" -m tools.check_strict_source_data --data-root "$DATA" --output-dir "$OUT" 2>&1 | tee "$LOG"
PIPE_EXITS=("${PIPESTATUS[@]}")
set -e
GATE_EXIT=${PIPE_EXITS[0]}
if [[ "${PIPE_EXITS[1]}" != 0 ]]; then GATE_EXIT=1; fi
echo "source_data_exit_code=$GATE_EXIT"
if [[ -f "$OUT/SHA256SUMS" ]]; then (cd "$OUT" && sha256sum -c SHA256SUMS); fi
sha256sum "$LOG"
git rev-parse HEAD
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_gate; exit 1; }
echo "Return $OUT/source_data_report.json, $OUT/SHA256SUMS and $LOG"
echo "STOP: no training, weights, GPU or target access is authorized."
exit "$GATE_EXIT"
