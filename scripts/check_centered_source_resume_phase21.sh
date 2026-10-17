#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'Run with bash, do not source this launcher.' >&2
  return 2
fi
set -euo pipefail
EXPECTED_SHA="${1:-}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'ERROR: pass the full40-character Git SHA.' >&2; exit 2; }
LAUNCH_MODE="${2:-launch}"
[[ "$LAUNCH_MODE" == launch || "$LAUNCH_MODE" == --worker ]] || { echo invalid_mode; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; git status --short; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
OUT="$BASE/phase21/M1_RUNNER_RESUME8_${EXPECTED_SHA:0:7}"
LOG="${OUT}_launcher.log"
test -x "$PY" || { echo missing_python; exit 1; }
test ! -e "$OUT" || { echo "existing_output=$OUT"; exit 3; }
mkdir -p "$BASE/phase21"
export PY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$LAUNCH_MODE" == launch ]]; then
  if [[ -e "$LOG" ]]; then echo "existing_log=$LOG"; tail -n 60 "$LOG"; exit 3; fi
  if command -v pgrep >/dev/null && pgrep -af '[t]ools.check_centered_source_resume'; then
    echo 'STOP: M1 continuation worker is already running.'
    exit 3
  fi
  nohup bash scripts/check_centered_source_resume_phase21.sh "$EXPECTED_SHA" --worker > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "m1_resume_pid=$PID"
  echo "launcher_log=$LOG"
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    set +e
    wait "$PID"
    CODE=$?
    set -e
    tail -n 60 "$LOG"
    exit "$CODE"
  fi
  exit 0
fi
mkdir "$OUT"
for MODE in continuous prefix resume compare; do
  set +e
  "$PY" -m tools.check_centered_source_resume --mode "$MODE" \
    --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
    --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}" \
    --output-root "$OUT" 2>&1 | tee "$OUT/${MODE}.log"
  EXITS=("${PIPESTATUS[@]}")
  set -e
  CODE=${EXITS[0]}
  if [[ "${EXITS[1]}" != 0 ]]; then CODE=1; fi
  if [[ "$CODE" != 0 ]]; then
    echo "m1_runner_resume_exit_code=$CODE failed_mode=$MODE"
    echo "Return $OUT/${MODE}/worker_report.json (or $OUT/resume_report.json for compare) and logs; STOP."
    exit "$CODE"
  fi
done
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum -c SHA256SUMS)
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_gate; exit 1; }
echo 'm1_runner_resume_exit_code=0'
echo "Return $OUT/resume_report.json, SHA256SUMS and final output. Keep checkpoints on AutoDL."
echo 'STOP: no formal training, new loss or target access.'
