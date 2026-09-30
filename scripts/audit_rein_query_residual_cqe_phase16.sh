#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
SHA=3644785
BASE=/root/autodl-tmp/outputs/CausalQ_DG
RUN="$BASE/REIN_QUERY_RESIDUAL_CQE_PAIRED_SMOKE_20UPDATES_$SHA"
REPORT="$BASE/analysis/rein_phase16_query_residual_cqe_${SHA}_v1.json"
LOG="$BASE/analysis/rein_phase16_query_residual_cqe_${SHA}_v1.stderr.log"
AUDIT="$BASE/analysis/rein_phase16_query_residual_cqe_${SHA}_saved_audit.json"
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
test -f "$REPORT" && test -f "$LOG" && test -d "$RUN" && test -x "$PY"
test ! -e "$AUDIT" || { echo "existing_audit=$AUDIT"; exit 1; }
set +e
"$PY" -m tools.audit_rein_query_residual_cqe \
  --report "$REPORT" --stderr-log "$LOG" --run-dir "$RUN" > "$AUDIT"
CODE=$?
cat "$AUDIT"
sha256sum "$AUDIT" "$REPORT" "$LOG"
sha256sum "$RUN"/*.pth "$RUN"/*.jsonl
echo "paired_cqe_saved_audit_exit_code=$CODE"
git rev-parse HEAD
df -h /root/autodl-tmp
exit "$CODE"
