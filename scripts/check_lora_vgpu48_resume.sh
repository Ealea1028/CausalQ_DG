#!/usr/bin/env bash
set -Eeuo pipefail
SHA=${1:?full published SHA required}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:?set existing Python interpreter}
test -x "$PY"
BASE=/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114
SESSION=$(mktemp -d "$BASE/phase42_vgpu48_gate_${SHA:0:7}_XXXXXX")
printf '%s\n' "$SESSION" | tee "$BASE/phase42_vgpu48_gate_latest.txt"
printf '%s\n' "$SHA" > "$SESSION/git_sha.txt"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STATUS=0
"$PY" -u -m tools.check_lora_vgpu48_resume --output "$SESSION/probe" --expected-sha "$SHA" \
  2>&1 | tee "$SESSION/launcher.log" || STATUS=$?
printf 'gate_exit_code=%s\n' "$STATUS" | tee "$SESSION/exit_code.txt"
if [[ -d "$SESSION/probe" ]]; then
  "$PY" -m tools.check_lora_vgpu48_resume --output "$SESSION/probe" --expected-sha "$SHA" \
    --package --archive "$SESSION/evidence.tar.gz"
fi
ARCHIVE="${SESSION}_return.tar.gz"
tar --exclude='./probe' -czf "$ARCHIVE" -C "$SESSION" .
sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
ls -lh "$ARCHIVE" "${ARCHIVE}.sha256"
exit "$STATUS"
