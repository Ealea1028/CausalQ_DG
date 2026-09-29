#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
BASE=/root/autodl-tmp/outputs/CausalQ_DG
# Prevent duplicate launches of this baseline while an earlier invocation runs.
exec 9> "$BASE/analysis/rein_phase16_training.lock"
flock -n 9 || { echo 'rein_training_already_running'; exit 1; }
SHA=$(git rev-parse --short HEAD)
RUN="$BASE/REIN_SOURCE_ONLY_SEED0_40000_$SHA"
REPORT="$BASE/analysis/rein_phase16_source40k_${SHA}_v1.json"
LOG="$BASE/analysis/rein_phase16_source40k_${SHA}_v1.stderr.log"
EXIT_FILE="$BASE/analysis/rein_phase16_source40k_${SHA}_v1.exit.txt"
test -z "$(git status --porcelain)"
test ! -e "$RUN"
test ! -e "$REPORT"
test ! -e "$LOG"
test ! -e "$EXIT_FILE"
# Read-only audit must succeed before optimization. The Python runner repeats it.
"$PY" -m tools.audit_rein_runner \
  --report "$BASE/analysis/rein_phase16_runner_7a3a1c6_v1.json" \
  --stderr-log "$BASE/analysis/rein_phase16_runner_7a3a1c6_v1.stderr.log" \
  --run-dir "$BASE/REIN_RUNNER_GATE_20UPDATES_SEED0_7a3a1c6"
set +e
"$PY" -m tools.rein_runner_gate --formal-source-only \
  --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
  --weights /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth \
  --data-report "$BASE/analysis/rein_phase16_data_protocol_b645793_v1.json" \
  --fresh-report "$BASE/analysis/rein_phase16_fresh_process_1671f1a_v1.json" \
  --runner-report "$BASE/analysis/rein_phase16_runner_7a3a1c6_v1.json" \
  --runner-log "$BASE/analysis/rein_phase16_runner_7a3a1c6_v1.stderr.log" \
  --runner-run "$BASE/REIN_RUNNER_GATE_20UPDATES_SEED0_7a3a1c6" \
  --run-dir "$RUN" > "$REPORT" 2> "$LOG"
CODE=$?
printf '%s\n' "$CODE" > "$EXIT_FILE"
cat "$REPORT"
tail -n 25 "$LOG"
sha256sum "$REPORT" "$LOG"
find "$RUN" -maxdepth 1 -type f -name '*.pth' -printf '%s %f\n'
echo "train_exit_code=$CODE"
df -h /root/autodl-tmp
exit "$CODE"
