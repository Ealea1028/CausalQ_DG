#!/usr/bin/env bash
# Verify the reviewed Phase42–44 candidate payloads without changing checkpoints.
set -Eeuo pipefail
SHA=${1:?pass the exact release SHA}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=no)"
PY=${PY:-/root/autodl-tmp/envs/causalq-lora1114/bin/python}
test -x "$PY"
BASE=/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114
AUDIT="$BASE/disk_audits"
test -d "$BASE"
mkdir -p "$AUDIT"
OUT="$AUDIT/phase46_checkpoint_verify_${SHA:0:7}_$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
exec > >(tee "$OUT.log") 2>&1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "Checking the 568 listed payloads plus the retained L0 final"
nice -n 10 "$PY" -u -m tools.verify_lora_checkpoint_preview \
  --expected-sha "$SHA" --output "$OUT"
sha256sum "$OUT" | tee "$OUT.sha256"
echo "Return both files: $OUT and $OUT.sha256"
