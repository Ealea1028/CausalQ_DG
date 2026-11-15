#!/usr/bin/env bash
# First real-weight integration gate, not Phase40 prepare or formal training.
set -euo pipefail
EXPECTED_SHA=${1:?expected full SHA required}
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:?set the Python interpreter}
OUT="/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114/phase39_${EXPECTED_SHA:0:7}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -f "$OUT/report.json" ]]; then
  echo "Existing report: $OUT/report.json. Return it before starting another attempt."
  exit 0
fi
nohup flock -n "$OUT/run.lock" "$PY" -u -m tools.check_lora_backbones \
  --pretrained-root /root/autodl-tmp/pretrained --output "$OUT" \
  >"$OUT/launcher.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid"
echo "PID=$(cat "$OUT/launcher.pid") OUT=$OUT"
