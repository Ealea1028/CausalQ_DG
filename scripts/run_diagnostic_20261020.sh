#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'Run with bash; do not source this launcher.' >&2
  return 2
fi
set -euo pipefail
EXPECTED_SHA="${1:-}"
MODE="${2:-baseline-diagnostics}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Pass the full40-character Git SHA.' >&2; exit 2; }
case "$MODE" in
  baseline-diagnostics|--worker|train-uncentered|--train-uncentered-worker|train-centered-controls|--train-centered-worker|final-source-diagnostics|--final-source-worker|target-evaluation|--target-worker) ;;
  *) echo 'Use baseline-diagnostics, train-uncentered, train-centered-controls, final-source-diagnostics or target-evaluation.' >&2; exit 2 ;;
esac
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
if [[ "$MODE" == train-centered-controls || "$MODE" == --train-centered-worker ]]; then
  PHASE=30
  OUT="$BASE/phase30/CENTERED_SEED0_${EXPECTED_SHA:0:7}"
  WORKER_MODE=--train-centered-worker
fi
if [[ "$MODE" == final-source-diagnostics || "$MODE" == --final-source-worker ]]; then
  PHASE=31
  OUT="$BASE/phase31/FINALS_SEED0_${EXPECTED_SHA:0:7}"
  WORKER_MODE=--final-source-worker
fi
if [[ "$MODE" == target-evaluation || "$MODE" == --target-worker ]]; then
  PHASE=32
  OUT="$BASE/phase32/FINAL_TARGET_SEED0_${EXPECTED_SHA:0:7}"
  WORKER_MODE=--target-worker
fi
LOG="$OUT/launcher.log"
test -x "$PY" || { echo missing_python; exit 1; }
command -v flock >/dev/null || { echo missing_flock; exit 1; }
mkdir -p "$OUT"
export PY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$MODE" == baseline-diagnostics || "$MODE" == train-uncentered || "$MODE" == train-centered-controls || "$MODE" == final-source-diagnostics || "$MODE" == target-evaluation ]]; then
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
if [[ "$PHASE" == 32 ]]; then
  echo "phase32_start_utc=$(date -u +%FT%TZ) git_sha=$EXPECTED_SHA"
  for VARIANT in initialize UP UF UA UR CA CR package; do
    "$PY" -m tools.evaluate_diagnostic_target --variant "$VARIANT" --output-root "$OUT" \
      --training-output-root "${CAUSALQ_TRAINING_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}" \
      --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
      --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
  done
  (cd "$OUT" && sha256sum -c SHA256SUMS && sha256sum phase32_export.zip)
  echo 'phase32_exit_code=0'
  echo "Return $OUT/phase32_export.zip and last80 lines of $LOG."
  echo 'STOP: all target results are reporting only; no recipe changes or additional training.'
  exit 0
fi
if [[ "$PHASE" == 31 ]]; then
  echo "phase31_start_utc=$(date -u +%FT%TZ) git_sha=$EXPECTED_SHA"
  for VARIANT in initialize UP UF UA UR CA CR package; do
    "$PY" -m tools.run_final_source_diagnostics --variant "$VARIANT" --output-root "$OUT" \
      --training-output-root "${CAUSALQ_TRAINING_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}" \
      --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
      --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
  done
  (cd "$OUT" && sha256sum -c SHA256SUMS && sha256sum phase31_export.zip)
  echo 'phase31_exit_code=0'
  echo "Return $OUT/phase31_export.zip and last80 lines of $LOG."
  echo 'STOP: next local Phase32 target freeze; no training or target access in Phase31.'
  exit 0
fi
if [[ "$PHASE" == 29 || "$PHASE" == 30 ]]; then
  VARIANTS=(UP UF UA UR)
  if [[ "$PHASE" == 30 ]]; then VARIANTS=(CA CR); fi
  echo "phase${PHASE}_start_utc=$(date -u +%FT%TZ) git_sha=$EXPECTED_SHA"
  df -hT "$BASE"
  "$PY" -m tools.package_diagnostic_formal --output-root "$OUT" --phase "$PHASE" --initialize
  for VARIANT in "${VARIANTS[@]}"; do
    echo "phase${PHASE}_variant=$VARIANT"
    "$PY" -m tools.train_formal_source --variant "$VARIANT" --output-root "$OUT" \
      --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
      --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
  done
  "$PY" -m tools.package_diagnostic_formal --output-root "$OUT" --phase "$PHASE"
  (cd "$OUT" && sha256sum -c SHA256SUMS && sha256sum "phase${PHASE}_export.zip")
  echo "phase${PHASE}_exit_code=0"
  echo "Return $OUT/phase${PHASE}_export.zip and last80 lines of $LOG."
  if [[ "$PHASE" == 29 ]]; then
    echo 'STOP: next Phase30 CA/CR; no target access or repeated baseline diagnostics.'
  else
    echo 'STOP: next Phase31 new-model source diagnostics; no target access or repeated training.'
  fi
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
