#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
REIN=/root/autodl-tmp/external/rein-phase16-dc063429
BASE=/root/autodl-tmp/outputs/CausalQ_DG
SHA=$(git rev-parse --short=7 HEAD)
RUN_ID="REIN_QUERY_RESIDUAL_CQE_FORMAL_SEED0_${SHA}"
RUN="$BASE/$RUN_ID"
REPORT="$BASE/analysis/rein_phase16_query_residual_cqe_formal_${SHA}_v1.json"
LOG="$BASE/analysis/rein_phase16_query_residual_cqe_formal_${SHA}_v1.stderr.log"
AUDIT="$BASE/analysis/rein_phase16_query_residual_cqe_formal_${SHA}_audit.json"
AUDIT_LOG="$BASE/analysis/rein_phase16_query_residual_cqe_formal_${SHA}_audit.log"
TRAIN_EXIT="$BASE/analysis/rein_phase16_query_residual_cqe_formal_${SHA}.exit"

test -x "$PY"
test -d "$REIN"
test -d "$BASE/analysis"
test -z "$(git status --porcelain)" || { echo "dirty_project_checkout"; exit 1; }
test ! -e "$RUN" || { echo "existing_run=$RUN"; exit 1; }
test ! -e "$REPORT" || { echo "existing_report=$REPORT"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }
test ! -e "$AUDIT" || { echo "existing_audit=$AUDIT"; exit 1; }
test ! -e "$AUDIT_LOG" || { echo "existing_audit_log=$AUDIT_LOG"; exit 1; }
test ! -e "$TRAIN_EXIT" || { echo "existing_exit_marker=$TRAIN_EXIT"; exit 1; }
FREE_KIB=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
test "$FREE_KIB" -ge 8388608 || { echo "need_at_least_8GiB_free_kib=$FREE_KIB"; exit 1; }

set +e
"$PY" -u tools/train_rein_query_residual_cqe_formal.py \
  --rein-root "$REIN" \
  --weights /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth \
  --data-report "$BASE/analysis/rein_phase16_data_protocol_b645793_v1.json" \
  --baseline-audit "$BASE/analysis/rein_phase16_source40k_d6fc52c_saved_audit.json" \
  --baseline-checkpoint "$BASE/REIN_SOURCE_ONLY_SEED0_40000_d6fc52c/last.pth" \
  --run-dir "$RUN" > "$REPORT" 2> "$LOG"
TRAIN_CODE=$?
set -e
printf '%s\n' "$TRAIN_CODE" > "$TRAIN_EXIT"
echo "formal_training_exit_code=$TRAIN_CODE"

if [ "$TRAIN_CODE" -eq 0 ]; then
  set +e
  "$PY" -u tools/audit_rein_query_residual_cqe_formal.py \
    --report "$REPORT" --run-dir "$RUN" --output "$AUDIT" \
    --bootstrap-replicates 2000 > "$AUDIT_LOG" 2>&1
  AUDIT_CODE=$?
  set -e
  echo "formal_audit_exit_code=$AUDIT_CODE"
  tail -n 80 "$AUDIT_LOG"
else
  echo "formal_audit_skipped=true"
  tail -n 80 "$LOG"
  AUDIT_CODE=1
fi

echo "===== formal report ====="
cat "$REPORT"
echo "===== artifact hashes ====="
sha256sum "$REPORT" "$LOG" "$TRAIN_EXIT"
if [ -f "$AUDIT" ]; then sha256sum "$AUDIT"; fi
if [ -d "$RUN" ]; then
  find "$RUN" -maxdepth 2 -type f -printf '%s %p\n' | sort -k2
  find "$RUN" -maxdepth 2 -type f \( -name 'final_branch.pth' -o -name 'train.jsonl' \) -print0 | xargs -0 -r sha256sum
fi
echo "===== source and disk ====="
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
test "$TRAIN_CODE" -eq 0
test "$AUDIT_CODE" -eq 0
