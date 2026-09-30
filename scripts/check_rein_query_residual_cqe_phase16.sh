#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
BASE=/root/autodl-tmp/outputs/CausalQ_DG
SHA=$(git rev-parse --short HEAD)
RUN="$BASE/REIN_QUERY_RESIDUAL_CQE_PAIRED_SMOKE_20UPDATES_$SHA"
REPORT="$BASE/analysis/rein_phase16_query_residual_cqe_${SHA}_v1.json"
LOG="$BASE/analysis/rein_phase16_query_residual_cqe_${SHA}_v1.stderr.log"
test ! -e "$RUN"
test ! -e "$REPORT"
test ! -e "$LOG"
set +e
"$PY" -m tools.check_rein_query_residual_cqe \
  --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
  --weights /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth \
  --data-report "$BASE/analysis/rein_phase16_data_protocol_b645793_v1.json" \
  --baseline-audit "$BASE/analysis/rein_phase16_source40k_d6fc52c_saved_audit.json" \
  --baseline-checkpoint "$BASE/REIN_SOURCE_ONLY_SEED0_40000_d6fc52c/last.pth" \
  --run-dir "$RUN" > "$REPORT" 2> "$LOG"
CODE=$?
cat "$REPORT"
tail -n 30 "$LOG"
sha256sum "$REPORT" "$LOG"
if [ -d "$RUN" ]; then
  find "$RUN" -maxdepth 1 -type f -printf '%s %f\n' | sort
  sha256sum "$RUN"/*.pth "$RUN"/*.jsonl 2>/dev/null || true
fi
echo "paired_cqe_smoke_exit_code=$CODE"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
exit "$CODE"
