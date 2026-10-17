#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'Run with bash, do not source this launcher.' >&2
  return 2
fi
set -euo pipefail
EXPECTED_SHA="${1:-}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'ERROR: pass the full 40-character Git SHA.' >&2; exit 2; }
MODE="${2:-launch}"
[[ "$MODE" == launch || "$MODE" == --worker ]] || { echo invalid_mode; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo wrong_commit; exit 1; }
test -z "$(git status --porcelain)" || { echo dirty_checkout; git status --short; exit 1; }
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
OUT="$BASE/phase20/QUERY_COMPONENT_CENSUS2497_${EXPECTED_SHA:0:7}"
CENTERED="$BASE/phase18/M1_CENTERED_GATE20_9583925"
ANCHOR="$BASE/phase19/ANCHOR_COMPONENT_READONLY_f5ca5b5"
DIAGNOSTICS="$BASE/phase20/QUERY_COMPONENT_DIAGNOSTICS20_9439771"
test -x "$PY" || { echo "missing_python=$PY"; exit 1; }
test -f "$CENTERED/state.pth" || { echo missing_centered_checkpoint; exit 1; }
test -f "$ANCHOR/anchor_report.json" || { echo missing_anchor_evidence; exit 1; }
test ! -e "$OUT" || { echo "existing_output=$OUT"; exit 3; }
test ! -e "${OUT}.log" || { echo "existing_log=${OUT}.log"; exit 3; }
test -f "$DIAGNOSTICS/component_effect_report.json" || { echo missing_diagnostic_evidence; exit 1; }
mkdir -p "$BASE/phase20"
export PY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$MODE" == launch ]]; then
  LAUNCH_LOG="${OUT}_launcher.log"
  if [[ -e "$LAUNCH_LOG" ]]; then
    echo "existing_launcher_log=$LAUNCH_LOG"
    tail -n 60 "$LAUNCH_LOG"
    exit 3
  fi
  if command -v pgrep >/dev/null && pgrep -af '[t]ools.analyze_query_components'; then
    echo 'STOP: component analysis is already running.'
    exit 3
  fi
  "$PY" -c "import matplotlib; print('matplotlib=' + matplotlib.__version__)"
  nohup bash scripts/analyze_query_components_full_phase20.sh "$EXPECTED_SHA" --worker > "$LAUNCH_LOG" 2>&1 < /dev/null &
  PID=$!
  echo "diagnostics_pid=$PID"
  echo "launcher_log=$LAUNCH_LOG"
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    set +e
    wait "$PID"
    CODE=$?
    set -e
    tail -n 60 "$LAUNCH_LOG"
    exit "$CODE"
  fi
  exit 0
fi
set +e
"$PY" -m tools.analyze_query_components \
  --data-root "${CAUSALQ_DATA_ROOT:-/root/autodl-tmp/datasets}" \
  --pretrained-root "${CAUSALQ_PRETRAINED_ROOT:-/root/autodl-tmp/pretrained}" \
  --checkpoint "$CENTERED/state.pth" --source-manifest "$PWD/manifests/gta5_source_val.txt" \
  --anchor-dir "$ANCHOR" --output-dir "$OUT" \
  --full-source --diagnostic-dir "$DIAGNOSTICS" 2>&1 | tee "${OUT}.log"
EXITS=("${PIPESTATUS[@]}")
set -e
CODE=${EXITS[0]}
if [[ "${EXITS[1]}" != 0 ]]; then CODE=1; fi
echo "component_census_exit_code=$CODE"
if [[ -f "$OUT/SHA256SUMS" ]]; then (cd "$OUT" && sha256sum -c SHA256SUMS); fi
sha256sum "${OUT}.log"
test -z "$(git status --porcelain)" || { echo dirty_checkout_after_analysis; exit 1; }
echo "Return component_effect_report.json, SHA256SUMS, analysis_export.zip and final output from $OUT."
echo 'STOP: no training, new loss or target access.'
exit "$CODE"
