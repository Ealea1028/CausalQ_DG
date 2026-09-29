#!/usr/bin/env bash
# Read pinned source only. No training, model/weight loading or dataset decoding.
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
BASE="/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_protocol_$(git rev-parse --short HEAD)_v1"
test -z "$(git status --porcelain)" || { echo 'dirty_worktree'; exit 1; }
for SUFFIX in json stderr.log; do
  test ! -e "$BASE.$SUFFIX" || { echo "existing_output=$BASE.$SUFFIX"; exit 1; }
done
test -x "$PY"
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
set +e
"$PY" -m tools.inspect_rein_protocol \
  --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
  > "$BASE.json" 2> "$BASE.stderr.log"
PROTOCOL_EXIT=$?
set -e
cat "$BASE.json"
tail -n 30 "$BASE.stderr.log"
sha256sum "$BASE.json" "$BASE.stderr.log"
echo "protocol_inventory_exit_code=$PROTOCOL_EXIT"
exit "$PROTOCOL_EXIT"
