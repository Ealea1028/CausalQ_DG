#!/usr/bin/env bash
set -Eeuo pipefail
SHA=${1:?full published SHA required}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:?set the existing Python interpreter}
test -x "$PY"
BASE=/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114
OUT="$BASE/phase45_source_${SHA:0:7}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf '%s\n' "$OUT" > "$BASE/phase45_source_latest.txt"
STATUS=0
# Forward a tmux stop to the worker and then export its completed records' summaries.
trap 'if [[ -n ${worker_pid:-} ]]; then kill -TERM "$worker_pid" 2>/dev/null || true; fi' INT TERM
"$PY" -u -m tools.run_lora_phase45_source --expected-sha "$SHA" --output "$OUT" \
  >>"$OUT/launcher.log" 2>&1 &
worker_pid=$!
while true; do
  wait "$worker_pid" && STATUS=0 || STATUS=$?
  if ! kill -0 "$worker_pid" 2>/dev/null; then break; fi
done
trap - INT TERM
printf 'diagnostic_exit_code=%s\n' "$STATUS" | tee "$OUT/exit_code.txt"
ARCHIVE="${OUT}_$(date -u +%Y%m%dT%H%M%SZ)_$$_return.tar.gz"
"$PY" -m tools.run_lora_phase45_source --expected-sha "$SHA" --output "$OUT" --archive "$ARCHIVE"
sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
printf 'Return both files: %s and %s.sha256\n' "$ARCHIVE" "$ARCHIVE"
exit "$STATUS"
