#!/usr/bin/env bash
set -Eeuo pipefail
SHA=${1:?full published SHA required}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:?set the existing Python interpreter}
test -x "$PY"
test -s /root/autodl-tmp/pretrained/dinov2_vitl14/dinov2_vitl14_hpi_p16.pth
BASE=/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114
OUT="$BASE/phase44_b3_${SHA:0:7}"
mkdir -p "$OUT/launches"
LAUNCH="$OUT/launches/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir "$LAUNCH"
printf '%s\n' "$LAUNCH" > "$OUT/latest_launch.txt"
printf '%s\n' "$SHA" > "$LAUNCH/git_sha.txt"
printf '%s\n' "$OUT" > "$BASE/phase44_b3_latest.txt"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STATUS=0
"$PY" -u -m tools.train_lora_phase44_b3 --output "$OUT" --expected-sha "$SHA" \
  2>&1 | tee -a "$OUT/launcher.log" || STATUS=$?
printf 'controller_exit_code=%s\n' "$STATUS" | tee "$LAUNCH/exit_code.txt"
ARCHIVE="${OUT}_$(date -u +%Y%m%dT%H%M%SZ)_$$_return.tar.gz"
"$PY" -m tools.train_lora_phase44_b3 --output "$OUT" --expected-sha "$SHA" \
  --package --archive "$ARCHIVE"
sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
printf 'Return both files: %s and %s.sha256\n' "$ARCHIVE" "$ARCHIVE"
exit "$STATUS"
