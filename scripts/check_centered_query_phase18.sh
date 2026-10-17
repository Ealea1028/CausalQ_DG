#!/usr/bin/env bash
set -euo pipefail
EXPECTED_SHA="${1:?Pass full Git SHA from Codex}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo invalid_sha; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
OUT="$BASE/phase18/M1_CENTERED_GATE20_${EXPECTED_SHA:0:7}"
test ! -e "$OUT" || { echo existing_output; exit 1; }
test ! -e "${OUT}.log" || { echo existing_log; exit 1; }
mkdir -p "$BASE/phase18"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
set +e
"$PY" -m tools.check_centered_query --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
  --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}" \
  --output-dir "$OUT" 2>&1 | tee "${OUT}.log"
EXITS=("${PIPESTATUS[@]}")
set -e
CODE=${EXITS[0]}
if [[ "${EXITS[1]}" != 0 ]]; then CODE=1; fi
echo "centered_query_exit_code=$CODE"
if [[ -f "$OUT/SHA256SUMS" ]]; then (cd "$OUT" && sha256sum -c SHA256SUMS); fi
sha256sum "${OUT}.log"
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_gate; exit 1; }
echo "Return centered_report.json, SHA256SUMS and full final output from $OUT."
echo "STOP: no full M1 training, new losses or target access."
exit "$CODE"
