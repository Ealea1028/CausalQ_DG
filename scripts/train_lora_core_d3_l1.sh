#!/usr/bin/env bash
# Foreground launcher: run inside tmux. Repeating this exact SHA resumes commits.
set -euo pipefail
SHA=${1:?full published SHA required}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:?set the existing Python interpreter}
test -x "$PY"
test -s /root/autodl-tmp/pretrained/dinov3_vitl16/config.json
test -d /root/autodl-tmp/datasets/gta5/images/images
test -d /root/autodl-tmp/datasets/gta5/labels_trainIds
OUT="/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114/phase42_core_d3_l1_${SHA:0:7}"
mkdir -p "$OUT/launches"
LAUNCH="$OUT/launches/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir "$LAUNCH"
printf '%s\n' "$LAUNCH" > "$OUT/latest_launch.txt"
printf '%s\n' "$SHA" > "$LAUNCH/git_sha.txt"
printf '%s\n' "$PY" > "$LAUNCH/python.txt"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STATUS=0
"$PY" -u -m tools.train_lora_core_d3_l1 --output "$OUT" --expected-sha "$SHA" \
  2>&1 | tee -a "$OUT/launcher.log" || STATUS=$?
printf 'controller_exit_code=%s\n' "$STATUS" | tee "$LAUNCH/exit_code.txt"
exit "$STATUS"
