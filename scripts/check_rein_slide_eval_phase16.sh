#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
SHA=$(git rev-parse --short HEAD)
BASE=/root/autodl-tmp/outputs/CausalQ_DG
REPORT="$BASE/analysis/rein_phase16_slide_eval_${SHA}_v1.json"
LOG="$BASE/analysis/rein_phase16_slide_eval_${SHA}_v1.stderr.log"
test ! -e "$REPORT"
test ! -e "$LOG"
set +e
PYTHONPATH="$PWD" /root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python \
  -m tools.check_rein_slide_eval \
  --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
  --weights /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth \
  --data-report "$BASE/analysis/rein_phase16_data_protocol_b645793_v1.json" \
  --schedule-report "$BASE/analysis/rein_phase16_schedule_3ca6731_v1.json" \
  --checkpoint "$BASE/REIN_SCHEDULE_SMOKE_80MICRO_SEED0_3ca6731/checkpoint_20updates.pth" \
  > "$REPORT" 2> "$LOG"
CODE=$?
cat "$REPORT"
tail -n 25 "$LOG"
sha256sum "$REPORT" "$LOG"
echo "slide_eval_gate_exit_code=$CODE"
exit "$CODE"
