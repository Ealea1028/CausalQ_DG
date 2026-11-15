#!/usr/bin/env bash
set -euo pipefail
SHA=${1:?full SHA required}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:?set Python interpreter}
test -x "$PY" || { echo "Missing Python: $PY"; exit 2; }
for P in /root/autodl-tmp/datasets/gta5/images/images /root/autodl-tmp/datasets/gta5/labels_trainIds; do
  test -d "$P" || { echo "Missing source data: $P"; exit 2; }
done
OUT="/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114/phase41_core_${SHA:0:7}"
mkdir -p "$OUT"
if [[ -f "$OUT/launcher.pid" ]] && kill -0 "$(cat "$OUT/launcher.pid")" 2>/dev/null; then
  echo "Already running: PID=$(cat "$OUT/launcher.pid") OUT=$OUT"; exit 0
fi
if [[ -f "$OUT/report.json" ]]; then
  if "$PY" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("ok") else 1)' "$OUT/report.json"; then
    echo "Already complete: $OUT/report.json"; exit 0
  fi
fi
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup flock -n -E 75 "$OUT/../gpu0.lock" flock -n -E 75 "$OUT/run.lock" "$PY" -u -m tools.train_lora_core_l0 --output "$OUT" \
  >>"$OUT/launcher.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid"
echo "PID=$(cat "$OUT/launcher.pid") OUT=$OUT"
