# AutoDL next step: paired learned-null effect localization

The Phase 14 seed-0 40k run passes at training SHA
`d85db4e0dcf411681c1ddad09080dfeed1c419bd`, reaching `0.655364` final
Cityscapes mIoU (`+0.7968` points versus paired Static). Before repeat seeds or
new objectives, compare factual-minus-zero and factual-minus-learned-null
effects on the same 500 validation images.

This step is read-only. It does not train, modify, or replace the final
checkpoint.

## 1. Check out the exact analysis implementation

```bash
cd /root/autodl-tmp/CausalQ_DG

test -z "$(git status --porcelain)"

ANALYSIS_SHA=549a9753914b9868a852a65983fb8cf0f246462b

git fetch origin main
git cat-file -e "${ANALYSIS_SHA}^{commit}"
git checkout --detach "$ANALYSIS_SHA"

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

test "$(git rev-parse HEAD)" = "$ANALYSIS_SHA"
test -z "$(git status --porcelain)"

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -
```

Expected: `96 passed` and the DINOv3-L hash reports `OK`.

## 2. Run the paired 500-image analysis

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

NULL_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/B4B_NULL_IQE_SEED0_40000_d85db4e/checkpoints/iter_040000.pth
REPORT=/root/autodl-tmp/outputs/CausalQ_DG/analysis/B4B_NULL_IQE_SEED0_paired_effect_localization_549a975.json
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/B4B_NULL_IQE_SEED0_paired_effect_localization_549a975.log

test -f "$NULL_CKPT"
test ! -e "$REPORT"
test ! -e "$LOG"

mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis

set -o pipefail

python tools/analyze_learned_null_effect.py \
  --config configs/learned_null/gta_dinov3l_static_null.yaml \
  --checkpoint "$NULL_CKPT" \
  --max-samples 500 \
  --output "$REPORT" \
  2>&1 | tee "$LOG"

ANALYSIS_EXIT=${PIPESTATUS[0]}
echo "analysis_exit_code=$ANALYSIS_EXIT"
test "$ANALYSIS_EXIT" -eq 0
```

## 3. Validate and summarize the report

```bash
python - "$REPORT" <<'PY'
import json
import math
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text())

expected_analysis_sha = "549a9753914b9868a852a65983fb8cf0f246462b"
expected_training_sha = "d85db4e0dcf411681c1ddad09080dfeed1c419bd"

assert report["ok"] is True
assert report["evaluation_git_sha"] == expected_analysis_sha
assert report["dataset"] == "cityscapes_val"
assert report["requested_max_samples"] == 500
assert report["metric"]["name"] == \
    "paired_zero_and_learned_null_effect_localization"

model = report["model"]
assert model["training_git_sha"] == expected_training_sha
assert model["training_seed"] == 0
assert model["iteration"] == 40000
assert model["sample_count"] == 500
assert model["present_class_map_count"] == 6005

zero = model["zero_ablation"]
learned_null = model["learned_null"]
paired = model["paired"]

numeric = [
    *zero.values(),
    *learned_null.values(),
    *paired.values(),
    model["alpha"],
    model["peak_allocated_gib"],
    model["peak_reserved_gib"],
]
assert all(math.isfinite(float(value)) for value in numeric)
assert zero["mean_inside_absolute_mean"] >= 0.0
assert zero["mean_outside_absolute_mean"] >= 0.0
assert learned_null["mean_inside_absolute_mean"] >= 0.0
assert learned_null["mean_outside_absolute_mean"] >= 0.0
assert 0.0 <= paired["fraction_learned_null_higher_absolute_ratio"] <= 1.0
assert 0.0 <= paired["fraction_learned_null_lower_outside_absolute_mean"] <= 1.0

print("paired_learned_null_effect_analysis_ok=true")
print("===== provenance =====")
print("evaluation_git_sha:", report["evaluation_git_sha"])
print("training_git_sha:", model["training_git_sha"])
print("training_seed:", model["training_seed"])
print("checkpoint_sha256:", model["checkpoint_sha256"])
print("===== coverage =====")
print("sample_count:", model["sample_count"])
print("present_class_map_count:", model["present_class_map_count"])
print("===== zero ablation =====")
print(zero)
print("===== learned null =====")
print(learned_null)
print("===== paired comparison =====")
print(paired)
print(
    "learned_null_passes_gt_region_magnitude_target:",
    model["learned_null_passes_gt_region_magnitude_target"],
)
print("alpha:", model["alpha"])
print("peak_allocated_gib:", model["peak_allocated_gib"])
print("peak_reserved_gib:", model["peak_reserved_gib"])
PY
```

## 4. Return evidence and stop

```bash
echo "===== report ====="
cat "$REPORT"

echo "===== report integrity ====="
sha256sum "$REPORT" "$NULL_CKPT"

echo "===== error scan ====="
grep -E \
  'Traceback|AssertionError|FloatingPointError|CUDA out of memory' \
  "$LOG" || true

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the validator output and report. Stop after this analysis. Do not launch
seed 1/2, effect-invariance training, sufficiency, specificity, or semantic
counterfactual experiments until the paired localization result is reviewed.
