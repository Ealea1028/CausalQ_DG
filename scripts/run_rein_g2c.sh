#!/usr/bin/env bash
set -Eeuo pipefail

STAGE=${1:?usage: run_rein_g2c.sh prepare|smoke|train|resume|test FULL_SHA RUN_ID}
SHA=${2:?full CausalQ release SHA required}
RUN_ID=${3:?stable run ID required}
[[ "$SHA" =~ ^[0-9a-f]{40}$ && "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
ROOT=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
test "$(git -C "$ROOT" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)"

BASE=/root/autodl-tmp
REIN="$BASE/external/rein-phase16-dc063429"
REIN_SHA=dc063429c4dadc0da9c6252b3db22fc55a9882ab
PY="$BASE/envs/rein-g2c-py310-cu118/bin/python"
BACKBONE="$BASE/pretrained/rein-g2c/dinov2_vitl14_rein_patch16_512.pth"
SETUP="$BASE/outputs/rein-g2c-setup"
GPU_LOCK="$BASE/outputs/CausalQ_DG/lora_causalq_1114/gpu0.lock"
OUT="$BASE/outputs/REIN_GTAV_CITYS/$RUN_ID"
CONFIG="$OUT/gta_cityscapes.py"
FINAL="$OUT/train/iter_40000.pth"
test -x "$PY"
test "$(git -C "$REIN" rev-parse HEAD)" = "$REIN_SHA"
test -z "$(git -C "$REIN" status --porcelain --untracked-files=no)"
grep -qx 'REIN_SETUP_OK' "$SETUP/setup.log"
test -s "$BACKBONE"
mkdir -p "$OUT"
exec 9>>"$OUT/run.lock"
flock -n 9 || { echo "Run already active: $RUN_ID" >&2; exit 1; }
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH

if [ "$STAGE" = prepare ]; then
  test ! -e "$OUT/labels" || { echo "Run already prepared or incomplete; use a fresh run ID" >&2; exit 1; }
  test "$(wc -l < "$SETUP/backbone.sha256")" -eq 1
  sha256sum -c "$SETUP/backbone.sha256"
  test "$(awk '{print $2}' "$SETUP/backbone.sha256")" = "$BACKBONE"
  printf '%s\n' "$SHA" > "$OUT/project_sha.txt"
  printf '%s\n' "$REIN_SHA" > "$OUT/rein_sha.txt"
  "$PY" -m pip check
  "$PY" -m pip freeze > "$OUT/pip_freeze.txt"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES= PYTHONPATH="$REIN" "$PY" -m tools.prepare_rein_g2c \
    --rein "$REIN" --data "$BASE/datasets" --city-manifest "$ROOT/manifests/cityscapes_val500.txt" \
    --backbone "$BACKBONE" --out "$OUT" 2>&1 | tee "$OUT/prepare.log"
  sha256sum "$CONFIG" "$BACKBONE" "$OUT/pairs.tsv" "$OUT/data_report.json" \
    "$OUT/pip_freeze.txt" > "$OUT/inputs.sha256"
  echo "Data report: $OUT/data_report.json"
  exit 0
fi

test "$(cat "$OUT/project_sha.txt")" = "$SHA"
test "$(cat "$OUT/rein_sha.txt")" = "$REIN_SHA"
sha256sum -c "$OUT/inputs.sha256"
"$PY" -m pip freeze | cmp -s "$OUT/pip_freeze.txt" - || { echo 'Rein environment changed' >&2; exit 1; }
test -f "$GPU_LOCK"
exec 8>>"$GPU_LOCK"
flock -n 8 || { echo 'GPU lock is busy; try when the active job releases it' >&2; exit 1; }
cd "$REIN"
export CUDA_VISIBLE_DEVICES=0
case "$STAGE" in
  smoke)
    test ! -e "$OUT/smoke" || { echo 'Smoke directory exists; inspect it or prepare a fresh run ID' >&2; exit 1; }
    "$PY" tools/train.py "$CONFIG" --work-dir "$OUT/smoke" \
      --cfg-options train_cfg.max_iters=20 train_cfg.val_begin=21 \
      default_hooks.logger.interval=5 default_hooks.checkpoint.interval=20 2>&1 | tee "$OUT/smoke.log"
    test -s "$OUT/smoke/iter_20.pth"
    grep -q 'loss:' "$OUT/smoke.log"
    if grep -Eiq 'loss[^:]*:[[:space:]]*(nan|[+-]?inf)([[:space:]]|$)' "$OUT/smoke.log"; then
      echo 'Smoke log contains non-finite loss' >&2; exit 1
    fi
    sha256sum "$OUT/smoke/iter_20.pth" > "$OUT/smoke.sha256"
    echo 'REIN_SMOKE_OK'
    ;;
  train|resume)
    sha256sum -c "$OUT/smoke.sha256"
    extra=()
    if [ "$STAGE" = resume ]; then
      test -s "$OUT/train/last_checkpoint"
      test -s "$(cat "$OUT/train/last_checkpoint")"
      test ! -s "$FINAL"
      extra+=(--resume)
    else
      test ! -e "$OUT/train" || { echo 'Train directory exists; use resume if it has a saved state' >&2; exit 1; }
    fi
    "$PY" tools/train.py "$CONFIG" --work-dir "$OUT/train" "${extra[@]}" \
      2>&1 | tee -a "$OUT/train.log"
    test -s "$FINAL"
    sha256sum "$FINAL" > "$OUT/final.sha256"
    echo 'REIN_TRAIN_OK'
    ;;
  test)
    sha256sum -c "$OUT/final.sha256"
    test ! -e "$OUT/evidence.tar.gz.sha256" || { echo 'This run is already evaluated' >&2; exit 1; }
    ATTEMPT=$(mktemp -d "$OUT/eval_XXXXXX")
    "$PY" tools/test.py "$CONFIG" "$FINAL" --backbone "$BACKBONE" \
      --work-dir "$ATTEMPT/cityscapes_final" 2>&1 | tee "$ATTEMPT/cityscapes_final.log"
    grep -E 'mIoU:[[:space:]]+[0-9]+([.][0-9]+)?' "$ATTEMPT/cityscapes_final.log" \
      | tail -n 1 | tee "$OUT/cityscapes_metrics.txt"
    cp "$ATTEMPT/cityscapes_final.log" "$OUT/cityscapes_final.log"
    sha256sum -c "$OUT/final.sha256"
    tar -czf "$OUT/evidence.tar.gz" -C "$OUT" -- project_sha.txt rein_sha.txt pip_freeze.txt \
      gta_cityscapes.py data_report.json pairs.tsv inputs.sha256 prepare.log smoke.log smoke.sha256 \
      train.log final.sha256 cityscapes_final.log cityscapes_metrics.txt
    (cd "$OUT" && sha256sum evidence.tar.gz > evidence.tar.gz.sha256)
    echo "REIN_TEST_OK: $OUT/evidence.tar.gz"
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
