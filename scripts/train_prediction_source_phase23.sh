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
OUT="$BASE/phase23/P1_FORMAL_SEED0_${EXPECTED_SHA:0:7}"
LOG="${OUT}_launcher.log"
test -x "$PY" || { echo missing_python; exit 1; }
command -v flock >/dev/null || { echo missing_flock; exit 1; }
mkdir -p "$OUT"
export PY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$MODE" == launch ]]; then
  nohup bash scripts/train_prediction_source_phase23.sh "$EXPECTED_SHA" --worker >> "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "p1_worker_pid=$PID"
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
trap 'CODE=$?; if [[ "$CODE" != 0 ]]; then echo "p1_formal_exit_code=$CODE"; echo "Preserve outputs and return the last80 launcher log lines: $LOG"; fi' EXIT
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'P1 formal worker already holds this run lock.'; exit 3; }
echo "p1_worker_start_utc=$(date -u +%FT%TZ) git_sha=$EXPECTED_SHA"
"$PY" -m tools.train_formal_source --variant P1 --output-root "$OUT" \
  --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
  --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}"
(cd "$OUT" && sha256sum P1/formal_report.json P1/config.yaml P1/metadata.json > REPORT_SHA256SUMS && sha256sum -c REPORT_SHA256SUMS)
echo 'p1_formal_exit_code=0'
echo "Return $OUT/P1/formal_report.json and REPORT_SHA256SUMS; keep checkpoints/traces on AutoDL."
echo 'STOP: P1 seed0 complete; no target access or CQE training in this release.'
