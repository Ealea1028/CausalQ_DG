#!/usr/bin/env bash
# Execute with bash; all failures belong to this child, never the user's shell.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo 'Run with bash, do not source this launcher.' >&2
  return 2
fi
set -euo pipefail
EXPECTED_SHA="${1:-}"
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'ERROR: pass the full 40-character Git SHA.' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || { echo 'ERROR: wrong checkout.' >&2; exit 1; }
test -z "$(git status --porcelain)" || { echo 'ERROR: dirty checkout; preserve and inspect changes.' >&2; git status --short; exit 1; }
BASE="${CAUSALQ_OUTPUT_ROOT:-/root/autodl-tmp/outputs/CausalQ_DG}"
OUT="$BASE/phase19/ANCHOR_COMPONENT_READONLY_${EXPECTED_SHA:0:7}"
PY="${PY:-/root/autodl-tmp/envs/causalq-dg/bin/python}"
test -x "$PY" || { echo "ERROR: Python executable missing: $PY" >&2; exit 1; }
test -d "$BASE/phase18/M1_CENTERED_GATE20_9583925" || { echo 'ERROR: accepted Phase18 run directory missing.' >&2; exit 1; }
LAUNCH_LOG="${OUT}_launcher.log"
OLD="$BASE/phase19/ANCHOR_COMPONENT_READONLY_4347352"
if command -v pgrep >/dev/null && pgrep -af '[t]ools.check_anchor_query_effect'; then
  echo 'STOP: anchor inference is already running; inspect its log.'
  exit 3
fi
for artifact in "$OUT" "${OUT}.log" "$LAUNCH_LOG" "$OLD" "${OLD}.log" "${OLD}_launcher.log"; do
  if [[ -e "$artifact" ]]; then
    echo "STOP: existing evidence preserved: $artifact"
    if [[ -f "$LAUNCH_LOG" ]]; then tail -n 60 "$LAUNCH_LOG"; fi
    if [[ "$artifact" == "$OLD"* && -f "${OLD}_launcher.log" ]]; then tail -n 60 "${OLD}_launcher.log"; fi
    echo 'Return the existing log/report before requesting another run.'
    exit 3
  fi
done
mkdir -p "$BASE/phase19"
export PY
nohup bash scripts/check_anchor_query_effect_phase19.sh "$EXPECTED_SHA" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
PID=$!
echo "anchor_component_pid=$PID"
echo "launcher_log=$LAUNCH_LOG"
sleep 1
if kill -0 "$PID" 2>/dev/null; then
  echo 'Background process started. Read the launcher log for progress.'
else
  set +e
  wait "$PID"
  CODE=$?
  set -e
  tail -n 60 "$LAUNCH_LOG"
  echo "Background process finished during startup; exit_code=$CODE"
  exit "$CODE"
fi
