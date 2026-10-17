#!/usr/bin/env bash
# Run in a child bash. The caller's interactive terminal is never exited.
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
OUT="$BASE/phase22/M0_M1_FORMAL_SEED0_${EXPECTED_SHA:0:7}"
LOG="${OUT}_launcher.log"
test -x "$PY" || { echo missing_python; exit 1; }
command -v flock >/dev/null || { echo missing_flock; exit 1; }
mkdir -p "$OUT"
export PY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$MODE" == launch ]]; then
  # Nonblocking lock prevents duplicate GPU workers, including racing launches.
  nohup bash scripts/train_formal_source_phase22.sh "$EXPECTED_SHA" --worker >> "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "formal_worker_pid=$PID"
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
trap 'CODE=$?; if [[ "$CODE" != 0 ]]; then echo "formal_pair_exit_code=$CODE"; echo "Preserve outputs and return the last80 launcher log lines: $LOG"; fi' EXIT
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'A formal worker already holds this run lock.'; exit 3; }
echo "formal_worker_start_utc=$(date -u +%FT%TZ) git_sha=$EXPECTED_SHA"
for VARIANT in M0 M1 summarize; do
  "$PY" -m tools.train_formal_source --variant "$VARIANT" --output-root "$OUT" \
    --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
    --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
done
(cd "$OUT" && sha256sum M0/formal_report.json M1/formal_report.json pair_report.json > REPORT_SHA256SUMS && sha256sum -c REPORT_SHA256SUMS)
echo 'formal_pair_exit_code=0'
echo "Return $OUT/pair_report.json and REPORT_SHA256SUMS. Keep all checkpoints/traces on AutoDL."
echo 'STOP: M0/M1 seed0 formal pair finished; target evaluation and new losses remain separate.'
