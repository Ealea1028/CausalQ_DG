#!/usr/bin/env bash
# Run as a child shell. No installations, datasets, optimizer or checkpoints.
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
REIN_PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
REIN_ROOT=/root/autodl-tmp/external/rein-phase16-dc063429
WEIGHTS=/root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth
BASE="/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_segmentor_$(git rev-parse --short HEAD)_v1"
test -z "$(git status --porcelain)" || { echo 'dirty_project_worktree'; exit 1; }
test -x "$REIN_PY" && test -f "$WEIGHTS"
for SUFFIX in json stderr.log; do
  test ! -e "$BASE.$SUFFIX" || { echo "existing_output=$BASE.$SUFFIX"; exit 1; }
done
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
set +e
"$REIN_PY" -m tools.check_rein_segmentor --rein-root "$REIN_ROOT" --weights "$WEIGHTS" \
  > "$BASE.json" 2> "$BASE.stderr.log"
GATE_EXIT=$?
set -e
cat "$BASE.json"
tail -n 80 "$BASE.stderr.log"
sha256sum "$BASE.json" "$BASE.stderr.log" "$WEIGHTS"
echo "segmentor_gate_exit_code=$GATE_EXIT"
exit "$GATE_EXIT"
