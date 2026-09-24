# AutoDL next step: Phase 13 Static Query smoke test

Phase 12 is complete and selects two queries per class. Phase 13 begins the
query-interaction ablation with one isolated control: static grouped queries.
The run keeps the selected photometric-only protocol and removes only
Query-to-Image cross-attention. Run the 500-iteration smoke below and stop.

## 1. Checkout the exact implementation commit

```bash
cd /root/autodl-tmp/CausalQ_DG

git status --short
test -z "$(git status --porcelain)"

git fetch origin
git checkout 367694ee8bc2e670be0a896f40e089760fe2177a
test "$(git rev-parse HEAD)" = "367694ee8bc2e670be0a896f40e089760fe2177a"

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

python -m pytest
```

Expected: 81 tests pass.

## 2. Verify data, weights, GPU, and free space

```bash
python tools/check_environment.py

test "$(find /root/autodl-tmp/datasets/gta5/images/images -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 24966
test "$(find /root/autodl-tmp/datasets/gta5/labels_trainIds/labels -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 24966
test "$(find /root/autodl-tmp/datasets/cityscapes/leftImg8bit/val -type f -name '*_leftImg8bit.png' | wc -l)" -eq 500
test "$(find /root/autodl-tmp/datasets/cityscapes/gtFine/val -type f -name '*_gtFine_labelTrainIds.png' | wc -l)" -eq 500

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

The environment report may still show missing optional paths unless the
activation script exports them; CUDA, the RTX 4090 D, file counts, and weight
hash are the required gates here.

## 3. Run only the 500-iteration smoke

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="I1_STATIC_QUERY_SMOKE_500_${RUN_SHA}"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test "$RUN_SHA" = "367694e"
test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail

MAX_ITERATIONS=500 \
VALIDATION_MAX_SAMPLES=50 \
SEED=0 \
RUN_ID="$RUN_ID" \
bash scripts/train_static_query.sh \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

Do not replace `MAX_ITERATIONS=500` with 40,000 in this step.

## 4. Validate and return the smoke evidence

```bash
python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metadata = json.loads((run_dir / "metadata.json").read_text())
summary = json.loads((run_dir / "summary.json").read_text())
records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]

assert metadata["phase"] == 13
assert metadata["git_sha"] == "367694ee8bc2e670be0a896f40e089760fe2177a"
assert metadata["seed"] == 0
assert metadata["max_iterations"] == 500
assert metadata["query"]["queries_per_class"] == 2
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["cross_attention_layers"] == 0
assert metadata["style"]["views"] == ["original", "photometric"]
assert "prediction_consistency" not in metadata
assert "causal_query_effect" not in metadata
assert "query_diversity" not in metadata

assert summary["ok"] is True
assert len(records) == 500
assert [row["iteration"] for row in records] == list(range(1, 501))

tracked = ("loss", "loss_original", "loss_photometric", "gradient_norm", "alpha")
for row in records:
    assert all(math.isfinite(float(row[key])) for key in tracked)
    expected = float(row["loss_original"]) + float(row["loss_photometric"])
    assert abs(float(row["loss"]) - expected) <= 1e-5

validations = summary["validation"]
assert len(validations) == 1
assert validations[0]["iteration"] == 500
assert validations[0]["sample_count"] == 50
assert math.isfinite(float(validations[0]["miou"]))

checkpoints = sorted((run_dir / "checkpoints").glob("iter_*.pth"))
assert [path.name for path in checkpoints] == ["iter_000500.pth"]

print("static_query_smoke_ok=true")
print("metadata:", metadata)
print("summary:", {key: value for key, value in summary.items() if key != "validation"})
print("first_record:", records[0])
print("last_record:", records[-1])
print("validation:", validations[0])
print("checkpoint:", checkpoints[0])
PY

echo "===== final log ====="
tail -n 30 "$LOG_FILE"

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
du -sh "$RUN_DIR"
df -h /root/autodl-tmp
```

Return the complete validator output, final log, provenance, and disk output.
Stop afterward: do not start the 40k run and do not implement or run a
bidirectional interaction on AutoDL.
