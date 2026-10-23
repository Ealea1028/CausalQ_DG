#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'Run with bash; do not source this launcher.' >&2
  return 2
fi
set -euo pipefail
EXPECTED_SHA="${1:-}"
MODE="${2:-baseline-diagnostics}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Pass the full40-character Git SHA.' >&2; exit 2; }
[[ "$MODE" == baseline-diagnostics || "$MODE" == --worker || "$MODE" == train-uncentered || "$MODE" == --train-uncentered-worker ]] || { echo 'Use baseline-diagnostics or train-uncentered.' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG/diagnostic_20261020}"
OUT="$BASE/phase28/BASELINES_${EXPECTED_SHA:0:7}"
PHASE=28
WORKER_MODE=--worker
if [[ "$MODE" == train-uncentered || "$MODE" == --train-uncentered-worker ]]; then
  PHASE=29
  OUT="$BASE/phase29/UNCENTERED_SEED0_${EXPECTED_SHA:0:7}"
  WORKER_MODE=--train-uncentered-worker
fi
LOG="$OUT/launcher.log"
test -x "$PY" || { echo missing_python; exit 1; }
command -v flock >/dev/null || { echo missing_flock; exit 1; }
mkdir -p "$OUT"
export PY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$MODE" == baseline-diagnostics || "$MODE" == train-uncentered ]]; then
  nohup bash scripts/run_diagnostic_20261020.sh "$EXPECTED_SHA" "$WORKER_MODE" >> "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "worker_pid=$PID"
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
exec 8>"$BASE/worker.lock"
flock -n 8 || { echo 'Another diagnostic worker is running; preserve its checkout and logs.'; exit 3; }
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'An existing diagnostic worker holds the run lock.'; exit 3; }
echo "$$" > "$OUT/worker.pid"
printf '{"state":"running","phase":%s,"git_sha":"%s"}\n' "$PHASE" "$EXPECTED_SHA" > "$OUT/status.json"
trap 'CODE=$?; printf "{\"exit_code\":%s,\"phase\":%s,\"git_sha\":\"%s\"}\n" "$CODE" "$PHASE" "$EXPECTED_SHA" > "$OUT/status.json"; if [[ "$CODE" != 0 ]]; then echo "phase${PHASE}_exit_code=$CODE; preserve outputs and return last80 log lines: $LOG"; fi' EXIT
if [[ "$PHASE" == 29 ]]; then
  echo "phase29_start_utc=$(date -u +%FT%TZ) git_sha=$EXPECTED_SHA"
  df -hT "$BASE"
  "$PY" -m tools.package_diagnostic_formal --output-root "$OUT" --initialize
  for VARIANT in UP UF UA UR; do
    echo "phase29_variant=$VARIANT"
    "$PY" -m tools.train_formal_source --variant "$VARIANT" --output-root "$OUT" \
      --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
      --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
  done
  "$PY" -m tools.package_diagnostic_formal --output-root "$OUT"
  (cd "$OUT" && sha256sum -c SHA256SUMS && sha256sum phase29_export.zip)
  echo 'phase29_exit_code=0'
  echo "Return $OUT/phase29_export.zip and last80 lines of $LOG."
  echo 'STOP: next Phase30 CA/CR; no target access or repeated baseline diagnostics.'
  exit 0
fi
echo "phase28_start_utc=$(date -u +%FT%TZ) git_sha=$EXPECTED_SHA"
for VARIANT in U0 C0 CP CF package; do
  "$PY" -m tools.run_diagnostic_20261020 --variant "$VARIANT" --output-root "$OUT" \
    --training-output-root "${CAUSALQ_TRAINING_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}" \
    --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
    --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
done
(cd "$OUT" && sha256sum -c SHA256SUMS && sha256sum phase28_export.zip)
echo 'phase28_exit_code=0'
echo "Return $OUT/phase28_export.zip and last80 lines of $LOG."
echo 'STOP: next is formal UP/UF/UA/UR; no raw-stability ranking gate, no target access.'
