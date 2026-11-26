#!/usr/bin/env bash
set -Eeuo pipefail
SHA=${1:?pass the exact cleanup release SHA}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:-/root/autodl-tmp/envs/causalq-lora1114/bin/python}
test -x "$PY"
AUDIT=/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114/disk_audits
test -d "$AUDIT"
OUT="$AUDIT/phase46_checkpoint_cleanup_${SHA:0:7}_$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
exec > >(tee "$OUT.log") 2>&1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STATUS=0
nice -n 10 "$PY" -u -m tools.apply_lora_checkpoint_cleanup \
  --expected-sha "$SHA" --output "$OUT" --apply || STATUS=$?
if [[ -f "$OUT" ]]; then
  sha256sum "$OUT" | tee "$OUT.sha256"
  echo "Return both files: $OUT and $OUT.sha256"
fi
if (( STATUS != 0 )); then
  echo "Cleanup failed with exit code $STATUS. Return log: $OUT.log"
fi
exit "$STATUS"
