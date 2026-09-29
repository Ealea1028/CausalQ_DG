#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
BASE=/root/autodl-tmp/outputs/CausalQ_DG
SHA=$(git rev-parse --short HEAD)
RUN="$BASE/REIN_FRESH_PROCESS_GATE_$SHA"
REPORT="$BASE/analysis/rein_phase16_fresh_process_${SHA}_v1.json"
LOG="$BASE/analysis/rein_phase16_fresh_process_${SHA}_v1.stderr.log"
test ! -e "$RUN"
test ! -e "$REPORT"
test ! -e "$LOG"
set +e
"$PY" -m tools.rein_fresh_process \
  --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
  --weights /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth \
  --data-report "$BASE/analysis/rein_phase16_data_protocol_b645793_v1.json" \
  --checkpoint "$BASE/REIN_CONTINUATION_GATE_20UPDATES_SEED0_5fee730/checkpoint_20updates.pth" \
  --output-dir "$RUN" > "$REPORT" 2> "$LOG"
CODE=$?
cat "$REPORT"
tail -n 25 "$LOG"
sha256sum "$REPORT" "$LOG"
echo "fresh_process_gate_exit_code=$CODE"
df -h /root/autodl-tmp
exit "$CODE"
