# Next AutoDL action

Status: the Phase 9 A4 run is complete and stable at `0.5917505963` Cityscapes mIoU. It missed both `A4 > A3` and the predefined one-point safety floor. Retain the fixed result without tuning. Evaluate the remaining Phase 9 target, cross-style normalized query-effect variance, on the A3 and A4 final checkpoints. Do not begin Phase 10.

## Fixed evaluation protocol

- Dataset: all 500 Cityscapes validation images in sorted order.
- Views: original, photometric, and Fourier, with identical per-sample random seeds for A3 and A4.
- Effect API: `model.get_query_effect(...)`; analysis does not access private tensors.
- Masking: valid pixels only and ground-truth-present classes only.
- Normalization: L2 per class map over valid spatial positions, matching Phase 9 training.
- Metric: population variance across the three normalized view maps, summed spatially, then averaged over all present class maps.
- Seed: `20260918`.
- Target: `variance(A4) < variance(A3)`.

## Reclaim A4 intermediate-checkpoint space

Verify the completed run before deleting only its 79 intermediate checkpoints:

```bash
A4_RUN=/root/autodl-tmp/outputs/CausalQ_DG/A4_CQE_SEED0_40000_f05cdc3
A4_RESOLVED="$(realpath "$A4_RUN")"

case "$A4_RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/A4_CQE_SEED0_40000_f05cdc3) ;;
  *) echo "unsafe path: $A4_RESOLVED"; exit 1 ;;
esac

test -f "$A4_RUN/summary.json"
test -f "$A4_RUN/checkpoints/iter_040000.pth"
test "$(find "$A4_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 80
test "$(find "$A4_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)" -eq 79

du -sh "$A4_RUN/checkpoints"
```

After all checks succeed:

```bash
find "$A4_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$A4_RUN/checkpoints/iter_040000.pth"
test "$(find "$A4_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$A4_RUN/checkpoints"
df -h /root/autodl-tmp
```

## Checkout and preflight

```bash
cd /root/autodl-tmp/CausalQ_DG
git status --short
git fetch origin
git checkout --detach <EXACT_SHA_FROM_HANDOFF>
git rev-parse HEAD
git status --short

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
python -m pytest
```

Both Git status checks must be empty and the full test suite must pass. Verify inputs:

```bash
A3_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/A3_PRED_CONS_SEED0_40000_4a95ebd/checkpoints/iter_040000.pth
A4_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/A4_CQE_SEED0_40000_f05cdc3/checkpoints/iter_040000.pth

test -f "$A3_CKPT"
test -f "$A4_CKPT"

python - "$A3_CKPT" "$A4_CKPT" <<'PY'
import sys
from causalq.utils.checkpoint import load_training_checkpoint

expected = {
    sys.argv[1]: "4a95ebd7dd627bd4695f201553a5e84aa709db09",
    sys.argv[2]: "f05cdc37fc1a95bc83d1b4a90441250c657e8c51",
}
for path, git_sha in expected.items():
    payload = load_training_checkpoint(path)
    assert payload["iteration"] == 40000
    assert payload["metadata"]["git_sha"] == git_sha
    assert payload["trainable_model"]
    print(path, git_sha, "OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run the variance comparison

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

A3_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/A3_PRED_CONS_SEED0_40000_4a95ebd/checkpoints/iter_040000.pth
A4_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/A4_CQE_SEED0_40000_f05cdc3/checkpoints/iter_040000.pth
REPORT=/root/autodl-tmp/outputs/CausalQ_DG/analysis/A3_A4_effect_variance_seed20260918.json
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/A3_A4_effect_variance_seed20260918.log

mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
test ! -e "$REPORT"
test ! -e "$LOG"

set -o pipefail
python tools/compare_effect_variance.py \
  --config configs/causalq/gta_dinov3l_causalq.yaml \
  --reference-name A3_PRED_CONS \
  --reference-checkpoint "$A3_CKPT" \
  --candidate-name A4_CQE \
  --candidate-checkpoint "$A4_CKPT" \
  --max-samples 500 \
  --seed 20260918 \
  --output "$REPORT" \
  2>&1 | tee "$LOG"

EVAL_EXIT=${PIPESTATUS[0]}
echo "eval_exit_code=$EVAL_EXIT"
```

## Evidence

```bash
cat "$REPORT"

python - "$REPORT" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
a3 = report["models"]["A3_PRED_CONS"]
a4 = report["models"]["A4_CQE"]

assert report["ok"] is True
assert a3["sample_count"] == 500
assert a4["sample_count"] == 500
assert a3["present_class_map_count"] == a4["present_class_map_count"]
assert a3["iteration"] == a4["iteration"] == 40000

print("a3_effect_variance:", a3["effect_variance"])
print("a4_effect_variance:", a4["effect_variance"])
print("candidate_minus_reference:", report["candidate_minus_reference"])
print("candidate_to_reference_ratio:", report["candidate_to_reference_ratio"])
print("passes_lower_variance_target:", report["passes_lower_variance_target"])
print("a3_peak_allocated_gib:", a3["peak_allocated_gib"])
print("a4_peak_allocated_gib:", a4["peak_allocated_gib"])
PY

git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- Intermediate cleanup retains only A4 `iter_040000.pth` and recovers roughly 15 GiB.
- Full tests pass; A3/A4 checkpoints have the expected training SHAs and iteration 40,000.
- `eval_exit_code=0`, report `ok=true`, and each model evaluates all 500 images.
- Both models use the same number of present class maps and the fixed seed/protocol.
- Both variances are finite and non-negative.
- Phase 9 second target passes only if `A4_CQE.effect_variance < A3_PRED_CONS.effect_variance`.
- Exact evaluation Git SHA, clean Git status, and remaining disk are reported.

Return cleanup, tests/preflight, complete report, provenance, and disk output. Stop after this comparison; do not begin Phase 10.
