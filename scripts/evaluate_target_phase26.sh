#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'Run with bash; do not source this launcher.' >&2
  return 2
fi
set -euo pipefail
EXPECTED_SHA="${1:-}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Pass the full40-character Git SHA.' >&2; exit 2; }
MODE="${2:-launch}"
[[ "$MODE" == launch || "$MODE" == --worker ]] || { echo invalid_mode; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
OUT="$BASE/phase26/FINAL_TARGET_SEED0_${EXPECTED_SHA:0:7}"
LOG="${OUT}_launcher.log"
test -x "$PY" || { echo missing_python; exit 1; }
command -v flock >/dev/null || { echo missing_flock; exit 1; }
mkdir -p "$OUT"
export PY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$MODE" == launch ]]; then
  nohup bash scripts/evaluate_target_phase26.sh "$EXPECTED_SHA" --worker >> "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "final_target_worker_pid=$PID"
  echo "launcher_log=$LOG"
  echo "output_root=$OUT"
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    set +e
    wait "$PID"
    CODE=$?
    tail -n 60 "$LOG"
    exit "$CODE"
  fi
  exit 0
fi
trap 'CODE=$?; if [[ "$CODE" != 0 ]]; then echo "final_target_formal_exit_code=$CODE"; echo "Preserve outputs and return the last80 launcher log lines: $LOG"; fi' EXIT
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'Final target worker already holds this run lock.'; exit 3; }
echo "final_target_worker_start_utc=$(date -u +%FT%TZ) git_sha=$EXPECTED_SHA"
for VARIANT in M0 M1 P1 M2 summarize; do
  "$PY" -m tools.evaluate_target --variant "$VARIANT" --output-root "$OUT" \
    --training-output-root "${CAUSALQ_TRAINING_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}" \
    --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
    --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
done
(cd "$OUT" && sha256sum final_comparison.json > REPORT_SHA256SUMS && sha256sum -c REPORT_SHA256SUMS)
echo 'final_target_exit_code=0'
echo "Return $OUT/final_comparison.json and REPORT_SHA256SUMS."
echo 'STOP: final target results are reporting only; no checkpoint selection or recipe changes.'
