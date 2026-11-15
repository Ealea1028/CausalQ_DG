#!/usr/bin/env bash
set -euo pipefail
SHA=${1:?full SHA required}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:?set Python interpreter}
test -x "$PY" || { echo "Missing Python: $PY"; exit 2; }
for P in /root/autodl-tmp/datasets/gta5/images/images /root/autodl-tmp/datasets/gta5/labels_trainIds; do
  test -d "$P" || { echo "Missing source data directory: $P"; exit 2; }
done
OUT="/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114/phase40_resume_${SHA:0:7}"
mkdir -p "$OUT"
if [[ -f "$OUT/report.json" ]]; then echo "Return existing $OUT/report.json before retry"; exit 0; fi
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup flock -n "$OUT/run.lock" "$PY" -u -m tools.check_lora_resume --output "$OUT" \
  >"$OUT/launcher.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid"
echo "PID=$(cat "$OUT/launcher.pid") OUT=$OUT"
