#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
BASE="/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_data_protocol_$(git rev-parse --short HEAD)_v1"
test -z "$(git status --porcelain)" || { echo 'dirty_worktree'; exit 1; }
for SUFFIX in json stderr.log; do
  test ! -e "$BASE.$SUFFIX" || { echo "existing_output=$BASE.$SUFFIX"; exit 1; }
done
test -x "$PY"
set +e
"$PY" -m tools.check_rein_data_protocol \
  --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
  --data-root /root/autodl-tmp/datasets \
  --inventory /root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_protocol_ede170c_v1.json \
  > "$BASE.json" 2> "$BASE.stderr.log"
GATE_EXIT=$?
set -e
cat "$BASE.json"
tail -n 40 "$BASE.stderr.log"
sha256sum "$BASE.json" "$BASE.stderr.log"
echo "data_protocol_exit_code=$GATE_EXIT"
exit "$GATE_EXIT"
