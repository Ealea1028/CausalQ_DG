#!/usr/bin/env bash
# Four matched seed-0 source-only jobs. Rerun at the same SHA to resume.
set -Eeuo pipefail
SHA=${1:?usage: run_scs_p4.sh FULL_GIT_SHA}
[[ "$SHA" =~ ^[a-f0-9]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
BASE=/root/autodl-tmp
PY="$BASE/outputs/scs_p0_setup_2eb18c2_nP68O8/env/bin/python"
REIN="$BASE/external/rein-phase16-dc063429"
OUT="$BASE/outputs/CausalQ_DG/scs_p4_${SHA:0:7}"
test -x "$PY"
mkdir -p "$OUT"
echo "OUT=$OUT"
exec 9>>"$BASE/outputs/CausalQ_DG/lora_causalq_1114/gpu0.lock"
flock -n 9 || { echo 'GPU is busy; retry when idle'; exit 1; }
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$$
"$PY" -m pip list --format=freeze > "$OUT/packages_$STAMP.txt"
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv --loop=5 > "$OUT/gpu_$STAMP.csv" 9>&- &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT

files=("packages_$STAMP.txt" "gpu_$STAMP.csv")
for backbone in dinov2_vitl14_hpi_p16 dinov3_vitl16; do
  for arm in A0 A4; do
    job="${backbone}_${arm}_s0"
    mkdir -p "$OUT/$job"
    "$PY" -u -m tools.train_scs_p4 --expected-sha "$SHA" --rein-root "$REIN" \
      --gta-root "$BASE/datasets/gta5" --output "$OUT/$job" --backbone "$backbone" --arm "$arm" \
      2>&1 | tee -a "$OUT/$job/train.log"
    files+=("$job/summary.json" "$job/run.json" "$job/train.log")
  done
done
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
trap - EXIT
(
  cd "$OUT"
  tar -czf "p4_return_$STAMP.tar.gz" -- "${files[@]}"
  sha256sum "p4_return_$STAMP.tar.gz" | tee "p4_return_$STAMP.tar.gz.sha256"
)
echo "Return $OUT/p4_return_$STAMP.tar.gz and its .sha256"
