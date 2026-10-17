#!/usr/bin/env bash
set -euo pipefail
EXPECTED_SHA="${1:?Pass the full Git SHA from Codex}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo invalid_sha; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
DATA="${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}"
WEIGHTS="${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
OUT="$BASE/phase17/M0_SOURCE_RESUME_${EXPECTED_SHA:0:7}"
test ! -e "$OUT" || { echo existing_output; exit 1; }
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
for MODE in continuous prefix resume compare; do
  set +e
  "$PY" -m tools.check_strict_source_resume --mode "$MODE" --data-root "$DATA" \
    --pretrained-root "$WEIGHTS" --output-root "$OUT" 2>&1 | tee "$OUT/${MODE}.log"
  EXITS=("${PIPESTATUS[@]}")
  set -e
  if [[ "${EXITS[0]}" != 0 || "${EXITS[1]}" != 0 ]]; then
    echo "source_resume_exit_code=1 failed_mode=$MODE"
    echo "Return worker_report.json/resume_report.json and logs from $OUT; STOP."
    exit 1
  fi
done
# The compare log is still being written when the report is produced; checksum
# it only after the process and tee have both completed.
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$OUT" && sha256sum -c SHA256SUMS)
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_gate; exit 1; }
echo "source_resume_exit_code=0"
echo "Return $OUT/resume_report.json, SHA256SUMS and final console output."
echo "Preserve checkpoints on AutoDL. STOP: no formal training or target access."
