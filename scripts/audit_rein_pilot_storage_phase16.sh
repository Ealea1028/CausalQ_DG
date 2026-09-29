#!/usr/bin/env bash
# Read-only: no installs, training, checkpoint loading or cleanup.
set -eu
cd /root/autodl-tmp/CausalQ_DG
export OMP_NUM_THREADS=1
BASE=/root/autodl-tmp/outputs/CausalQ_DG
PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
set +e
"$PY" -m tools.audit_rein_pilot \
  --report "$BASE/analysis/rein_phase16_pilot_b4e292d_v1.json" \
  --stderr-log "$BASE/analysis/rein_phase16_pilot_b4e292d_v1.stderr.log" \
  --run-dir "$BASE/REIN_SOURCE_PILOT_500UPDATES_SEED0_b4e292d"
AUDIT_EXIT=$?
set -e
echo "pilot_saved_audit_exit_code=$AUDIT_EXIT"
echo '===== directory usage (read-only) ====='
for DIR in /root/.cache/pip /root/miniconda3/pkgs /root/autodl-tmp/envs \
  /root/autodl-tmp/pretrained /root/autodl-tmp/uploads /root/autodl-tmp/external; do
  if [ -d "$DIR" ]; then du -sh "$DIR"; fi
done
du -h --max-depth=1 "$BASE" | sort -h
echo '===== pip cache (read-only) ====='
"$PY" -m pip cache info || true
echo '===== conda safe-cache dry run ====='
if [ -x /root/miniconda3/bin/conda ]; then
  /root/miniconda3/bin/conda clean --tarballs --index-cache --dry-run --json || true
fi
echo '===== cache and data mount identities ====='
df -h /root/autodl-tmp /root/.cache /root/miniconda3
git rev-parse HEAD
git status --short
exit "$AUDIT_EXIT"
