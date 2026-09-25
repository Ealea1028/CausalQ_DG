# AutoDL next step: one-way R=2 seed-2 paired full run

Static Query seed 1 completed at exact implementation commit
`3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497` with `0.669958` final
Cityscapes mIoU, 4.1601 percentage points above its paired one-way result.
The seed-0 and seed-1 differences both favor Static but vary substantially, so
the prespecified seed-2 pair remains required. Run only the one-way R=2 seed-2
member in this step. Do not start Static seed 2 or bidirectional interaction.

## 1. Verify source, tests, weights, and GPU

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

test "$(git rev-parse HEAD)" = \
  "3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497"
test -z "$(git status --porcelain)"

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

Expected: `82 passed`, the DINOv3-L hash passes, and the GPU is the RTX
4090 D. Stop if any check fails.

## 2. Keep the Static seed-1 final checkpoint and reclaim intermediates

The Static seed-1 result is recorded in Git. Delete only its 79 intermediate
checkpoints, retaining `iter_040000.pth`, metadata, summary, trace, and log.

```bash
STATIC_RUN=/root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED1_40000_3860e69

test -f "$STATIC_RUN/metadata.json"
test -f "$STATIC_RUN/summary.json"
test -f "$STATIC_RUN/checkpoints/iter_040000.pth"
test "$(wc -l < "$STATIC_RUN/train.jsonl")" -eq 40000

STATIC_CHECKPOINT_COUNT="$(
  find "$STATIC_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

STATIC_INTERMEDIATE_COUNT="$(
  find "$STATIC_RUN/checkpoints" \
    -maxdepth 1 -type f \
    -name 'iter_*.pth' \
    ! -name 'iter_040000.pth' \
    | wc -l
)"

echo "static_checkpoint_count=$STATIC_CHECKPOINT_COUNT"
echo "static_intermediate_count=$STATIC_INTERMEDIATE_COUNT"

test "$STATIC_CHECKPOINT_COUNT" -eq 80
test "$STATIC_INTERMEDIATE_COUNT" -eq 79

find "$STATIC_RUN/checkpoints" \
  -maxdepth 1 -type f \
  -name 'iter_*.pth' \
  ! -name 'iter_040000.pth' \
  -print -delete

test -f "$STATIC_RUN/checkpoints/iter_040000.pth"

STATIC_REMAINING="$(
  find "$STATIC_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

echo "static_remaining_checkpoint_count=$STATIC_REMAINING"
test "$STATIC_REMAINING" -eq 1
du -sh "$STATIC_RUN"
df -h /root/autodl-tmp
```

## 3. Run one-way R=2 seed 2 from random initialization

```bash
RUN_ID=Q2_QUERY_COUNT_SEED2_40000_3860e69
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail

QUERY_COUNT=2 \
MAX_ITERATIONS=40000 \
VALIDATION_MAX_SAMPLES=500 \
SEED=2 \
RUN_ID="$RUN_ID" \
bash scripts/train_query_count.sh \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

## 4. Validate and return run evidence

```bash
python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metadata = json.loads((run_dir / "metadata.json").read_text())
summary = json.loads((run_dir / "summary.json").read_text())
records = [
    json.loads(line)
    for line in (run_dir / "train.jsonl").read_text().splitlines()
    if line.strip()
]
validations = summary["validation_results"]
checkpoints = sorted((run_dir / "checkpoints").glob("iter_*.pth"))

assert metadata["experiment_id"] == "Q2_QUERY_COUNT_SEED2_40000_3860e69"
assert metadata["phase"] == 12
assert metadata["git_sha"] == \
    "3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497"
assert metadata["seed"] == 2
assert metadata["max_iterations"] == 40000
assert metadata["query"]["queries_per_class"] == 2
assert metadata["query"]["interaction"] == "one_way"
assert metadata["query"]["cross_attention_layers"] == 1
assert metadata["style"]["views"] == ["original", "photometric"]
assert "prediction_consistency" not in metadata
assert "causal_query_effect" not in metadata
assert "query_diversity" not in metadata

assert summary["ok"] is True
assert len(records) == 40000
assert [row["iteration"] for row in records] == list(range(1, 40001))

tracked = (
    "loss",
    "loss_original",
    "loss_photometric",
    "gradient_norm",
    "alpha",
)
assert all(
    math.isfinite(float(row[key]))
    for row in records
    for key in tracked
)

reconstruction_errors = [
    abs(
        float(row["loss"])
        - float(row["loss_original"])
        - float(row["loss_photometric"])
    )
    for row in records
]
assert max(reconstruction_errors) <= 1e-5

assert len(validations) == 80
assert all(item["sample_count"] == 500 for item in validations)
assert all(math.isfinite(float(item["miou"])) for item in validations)
assert validations[-1]["iteration"] == 40000
assert len(checkpoints) == 80
assert checkpoints[-1].name == "iter_040000.pth"

best = max(validations, key=lambda item: item["miou"])
final = validations[-1]

print("one_way_r2_seed2_full_ok=true")
print("metadata:", metadata)
print(
    "summary:",
    {key: value for key, value in summary.items() if key != "validation_results"},
)
print("record_count:", len(records))
print("maximum_objective_reconstruction_error:", max(reconstruction_errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("validation_count:", len(validations))
print("final_validation:", final)
print("best_validation:", best)
print("checkpoint_count:", len(checkpoints))
print("final_checkpoint:", checkpoints[-1])
PY

echo "===== checkpoint evidence ====="
find "$RUN_DIR/checkpoints" \
  -maxdepth 1 -type f \
  -printf '%s %f\n' \
  | sort -k2 \
  | tail -n 5

du -sh "$RUN_DIR/checkpoints"

echo "===== paired final-checkpoint preservation ====="
test -f "$STATIC_RUN/checkpoints/iter_040000.pth"
find "$STATIC_RUN/checkpoints" \
  -maxdepth 1 -type f -name 'iter_*.pth' \
  | wc -l

echo "===== final log ====="
tail -n 30 "$LOG_FILE"

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the complete validator output and final evidence blocks. Stop after the
run: do not delete the one-way seed-2 checkpoints and do not start Static seed
2 or bidirectional interaction.
