#!/usr/bin/env bash
set -euo pipefail
EXPECTED_SHA="${1:?Pass full Git SHA from Codex}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo invalid_sha; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
DATA="${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}"
WEIGHTS="${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
OUT="$BASE/phase17/M0_SOURCE_RUNNER_PILOT500_${EXPECTED_SHA:0:7}"
test ! -e "$OUT" || { echo existing_output; exit 1; }
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
for STAGE in prefix finish summary; do
  set +e
  "$PY" -m tools.train_strict_source --stage "$STAGE" --data-root "$DATA" \
    --pretrained-root "$WEIGHTS" --output-root "$OUT" 2>&1 | tee "$OUT/${STAGE}.log"
  EXITS=("${PIPESTATUS[@]}")
  set -e
  if [[ "${EXITS[0]}" != 0 || "${EXITS[1]}" != 0 ]]; then
    (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    echo "source_runner_pilot_exit_code=1 failed_stage=$STAGE; return worker_report.json and full log; STOP."
    exit 1
  fi
done
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$OUT" && sha256sum -c SHA256SUMS)
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_pilot; exit 1; }
echo "source_runner_pilot_exit_code=0"
echo "Return $OUT/pilot_report.json, SHA256SUMS and final launcher output."
echo "Preserve all checkpoints/traces on AutoDL. STOP: no formal 40k training or target access."
