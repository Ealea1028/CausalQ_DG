#!/usr/bin/env bash
# Child shell only. Bounded real-data smoke; no checkpoints or package changes.
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
REIN_PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
BASE="/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_real_data_$(git rev-parse --short HEAD)_v1"
test -z "$(git status --porcelain)" || { echo 'dirty_project_worktree'; exit 1; }
for SUFFIX in json stderr.log; do
  test ! -e "$BASE.$SUFFIX" || { echo "existing_output=$BASE.$SUFFIX"; exit 1; }
done
test -x "$REIN_PY"
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
set +e
"$REIN_PY" -m tools.check_rein_segmentor \
  --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
  --weights /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth \
  --data-root /root/autodl-tmp/datasets \
  > "$BASE.json" 2> "$BASE.stderr.log"
GATE_EXIT=$?
set -e
cat "$BASE.json"
tail -n 80 "$BASE.stderr.log"
sha256sum "$BASE.json" "$BASE.stderr.log"
echo "real_data_gate_exit_code=$GATE_EXIT"
exit "$GATE_EXIT"
