# AutoDL next step: Static zero-effect localization audit

Phase 13 is complete: Bidirectional seed 0 is a negative ablation and Static
remains selected. Before implementing learned-null, audit whether the current
factual-minus-zero class-logit effect is stronger inside the matching GT region
than outside it. Use all three selected Static seeds and all 500 Cityscapes
validation images. This step is read-only and must not train or modify a model.

Run the analysis from exact commit
`cf83e7750a25b005d2c4940dcb829565dda16aae`.

## 1. Reclaim only Bidirectional intermediate checkpoints

The accepted result is recorded in Git. Preserve the final checkpoint,
metadata, summary, trace, and external log; delete only the 79 intermediate
checkpoints.

```bash
cd /root/autodl-tmp/CausalQ_DG

BIDIR_FULL_RUN=/root/autodl-tmp/outputs/CausalQ_DG/I2_BIDIRECTIONAL_QUERY_SEED0_40000_1366716

test -f "$BIDIR_FULL_RUN/metadata.json"
test -f "$BIDIR_FULL_RUN/summary.json"
test -f "$BIDIR_FULL_RUN/checkpoints/iter_040000.pth"
test "$(wc -l < "$BIDIR_FULL_RUN/train.jsonl")" -eq 40000

BIDIR_CHECKPOINT_COUNT="$(
  find "$BIDIR_FULL_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

BIDIR_INTERMEDIATE_COUNT="$(
  find "$BIDIR_FULL_RUN/checkpoints" \
    -maxdepth 1 -type f \
    -name 'iter_*.pth' \
    ! -name 'iter_040000.pth' \
    | wc -l
)"

echo "bidirectional_checkpoint_count=$BIDIR_CHECKPOINT_COUNT"
echo "bidirectional_intermediate_count=$BIDIR_INTERMEDIATE_COUNT"

test "$BIDIR_CHECKPOINT_COUNT" -eq 80
test "$BIDIR_INTERMEDIATE_COUNT" -eq 79

find "$BIDIR_FULL_RUN/checkpoints" \
  -maxdepth 1 -type f \
  -name 'iter_*.pth' \
  ! -name 'iter_040000.pth' \
  -print -delete

test -f "$BIDIR_FULL_RUN/checkpoints/iter_040000.pth"
test "$(
  find "$BIDIR_FULL_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)" -eq 1

du -sh "$BIDIR_FULL_RUN"
df -h /root/autodl-tmp
```

## 2. Check out and verify the analysis implementation

```bash
cd /root/autodl-tmp/CausalQ_DG

test -z "$(git status --porcelain)"
git fetch origin
git checkout cf83e7750a25b005d2c4940dcb829565dda16aae

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

test "$(git rev-parse HEAD)" = \
  "cf83e7750a25b005d2c4940dcb829565dda16aae"
test -z "$(git status --porcelain)"

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

Expected: `88 passed`, the DINOv3-L hash passes, and the GPU is the RTX
4090 D. Stop if any check fails.

## 3. Verify the three retained Static final checkpoints

```bash
STATIC0_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED0_40000_367694e/checkpoints/iter_040000.pth
STATIC1_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED1_40000_3860e69/checkpoints/iter_040000.pth
STATIC2_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED2_40000_3860e69/checkpoints/iter_040000.pth

test -f "$STATIC0_CKPT"
test -f "$STATIC1_CKPT"
test -f "$STATIC2_CKPT"

sha256sum "$STATIC0_CKPT" "$STATIC1_CKPT" "$STATIC2_CKPT"
```

If any checkpoint is absent, stop and return the output. Do not substitute an
intermediate checkpoint.

## 4. Run the common 500-image zero-effect audit

```bash
REPORT=/root/autodl-tmp/outputs/CausalQ_DG/analysis/STATIC_ZERO_EFFECT_LOCALIZATION_SEEDS012_20260926.json
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/STATIC_ZERO_EFFECT_LOCALIZATION_SEEDS012_20260926.log

mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis

test ! -e "$REPORT"
test ! -e "$LOG"

set -o pipefail

python tools/analyze_zero_effect.py \
  --config configs/query_interaction/gta_dinov3l_static.yaml \
  --checkpoint "STATIC_SEED0=$STATIC0_CKPT" \
  --checkpoint "STATIC_SEED1=$STATIC1_CKPT" \
  --checkpoint "STATIC_SEED2=$STATIC2_CKPT" \
  --max-samples 500 \
  --output "$REPORT" \
  2>&1 | tee "$LOG"

ANALYSIS_EXIT=${PIPESTATUS[0]}
echo "analysis_exit_code=$ANALYSIS_EXIT"
test "$ANALYSIS_EXIT" -eq 0
```

## 5. Validate and return the report

```bash
python - "$REPORT" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())

assert report["ok"] is True
assert report["evaluation_git_sha"] == \
    "cf83e7750a25b005d2c4940dcb829565dda16aae"
assert report["dataset"] == "cityscapes_val"
assert report["requested_max_samples"] == 500
assert report["metric"]["name"] == "zero_ablation_effect_localization"
assert set(report["models"]) == {
    "STATIC_SEED0",
    "STATIC_SEED1",
    "STATIC_SEED2",
}

expected = {
    "STATIC_SEED0": (0, "367694ee8bc2e670be0a896f40e089760fe2177a"),
    "STATIC_SEED1": (1, "3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497"),
    "STATIC_SEED2": (2, "3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497"),
}
numeric_keys = (
    "mean_inside_absolute_mean",
    "mean_outside_absolute_mean",
    "mean_absolute_ratio",
    "mean_normalized_absolute_contrast",
    "mean_inside_signed_mean",
    "mean_outside_signed_mean",
    "localized_class_map_fraction",
    "alpha",
    "peak_allocated_gib",
    "peak_reserved_gib",
)

for name, result in report["models"].items():
    seed, training_sha = expected[name]
    assert result["sample_count"] == 500
    assert result["present_class_map_count"] > 0
    assert result["iteration"] == 40000
    assert result["training_seed"] == seed
    assert result["training_git_sha"] == training_sha
    assert all(math.isfinite(float(result[key])) for key in numeric_keys)
    assert 0.0 <= result["localized_class_map_fraction"] <= 1.0

assert math.isfinite(float(report["across_seed_mean_absolute_ratio"]))
assert math.isfinite(float(report["across_seed_sample_std_absolute_ratio"]))
assert math.isfinite(
    float(report["across_seed_mean_localized_class_map_fraction"])
)

print("static_zero_effect_localization_audit_ok=true")
print(json.dumps(report, indent=2))
PY

echo "===== report hash ====="
sha256sum "$REPORT"

echo "===== analysis log tail ====="
tail -n 30 "$LOG"

echo "===== retained Static checkpoints ====="
find \
  /root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED0_40000_367694e/checkpoints \
  /root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED1_40000_3860e69/checkpoints \
  /root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED2_40000_3860e69/checkpoints \
  -maxdepth 1 -type f -name 'iter_040000.pth' \
  -printf '%s %p\n'

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the complete JSON validator output plus all evidence blocks. Stop after
the audit. Do not delete the three Static final checkpoints, train learned-null,
or add sufficiency/specificity. The audit determines whether the existing query
effect has enough class-localized semantics to justify learned-null as the next
intervention baseline.
