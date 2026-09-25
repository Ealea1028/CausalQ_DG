# AutoDL next step: Static Query seed-1 paired full run

The repaired one-way R=2 seed-1 run completed at exact implementation commit
`3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497` with `0.628357` final
Cityscapes mIoU. Run the paired Static Query seed-1 40k experiment under the
same preprocessing, style protocol, and seed. Do not add bidirectional
interaction or start seed 2 in this step.

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

## 2. Keep the one-way final checkpoint and reclaim its intermediates

The one-way seed-1 result is recorded in Git. Delete only its 79 intermediate
checkpoints, retaining `iter_040000.pth`, metadata, summary, trace, and log.

```bash
ONE_WAY_RUN=/root/autodl-tmp/outputs/CausalQ_DG/Q2_QUERY_COUNT_SEED1_40000_3860e69

test -f "$ONE_WAY_RUN/metadata.json"
test -f "$ONE_WAY_RUN/summary.json"
test -f "$ONE_WAY_RUN/checkpoints/iter_040000.pth"
test "$(wc -l < "$ONE_WAY_RUN/train.jsonl")" -eq 40000

ONE_WAY_CHECKPOINT_COUNT="$(
  find "$ONE_WAY_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

ONE_WAY_INTERMEDIATE_COUNT="$(
  find "$ONE_WAY_RUN/checkpoints" \
    -maxdepth 1 -type f \
    -name 'iter_*.pth' \
    ! -name 'iter_040000.pth' \
    | wc -l
)"

echo "one_way_checkpoint_count=$ONE_WAY_CHECKPOINT_COUNT"
echo "one_way_intermediate_count=$ONE_WAY_INTERMEDIATE_COUNT"

test "$ONE_WAY_CHECKPOINT_COUNT" -eq 80
test "$ONE_WAY_INTERMEDIATE_COUNT" -eq 79

find "$ONE_WAY_RUN/checkpoints" \
  -maxdepth 1 -type f \
  -name 'iter_*.pth' \
  ! -name 'iter_040000.pth' \
  -print -delete

test -f "$ONE_WAY_RUN/checkpoints/iter_040000.pth"

ONE_WAY_REMAINING="$(
  find "$ONE_WAY_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

echo "one_way_remaining_checkpoint_count=$ONE_WAY_REMAINING"
test "$ONE_WAY_REMAINING" -eq 1
du -sh "$ONE_WAY_RUN"
df -h /root/autodl-tmp
```

## 3. Run Static Query seed 1 from random initialization

```bash
RUN_ID=I1_STATIC_QUERY_SEED1_40000_3860e69
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail

MAX_ITERATIONS=40000 \
VALIDATION_MAX_SAMPLES=500 \
SEED=1 \
RUN_ID="$RUN_ID" \
bash scripts/train_static_query.sh \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

## 4. Validate and return paired-run evidence

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
]
validations = summary["validation_results"]
checkpoints = sorted((run_dir / "checkpoints").glob("iter_*.pth"))

assert metadata["experiment_id"] == "I1_STATIC_QUERY_SEED1_40000_3860e69"
assert metadata["phase"] == 13
assert metadata["git_sha"] == \
    "3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497"
assert metadata["seed"] == 1
assert metadata["max_iterations"] == 40000
assert metadata["query"]["queries_per_class"] == 2
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["cross_attention_layers"] == 0
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
one_way_seed1 = 0.6283567654313249

print("static_query_seed1_full_ok=true")
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
print("one_way_seed1_miou:", one_way_seed1)
print("static_minus_one_way_seed1:", final["miou"] - one_way_seed1)
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

echo "===== paired checkpoint preservation ====="
test -f "$ONE_WAY_RUN/checkpoints/iter_040000.pth"
find "$ONE_WAY_RUN/checkpoints" \
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
run: do not delete Static Query checkpoints and do not start seed 2.
