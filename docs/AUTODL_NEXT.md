# AutoDL next step: Phase 13 paired interaction repeat, one-way seed 1

The Static Query seed-0 full run passed at implementation commit
`367694ee8bc2e670be0a896f40e089760fe2177a` and reached `0.647396`
Cityscapes mIoU. Its advantage over the existing one-way R=2 seed-0 result is
only 0.3671 percentage points, below the project's `<0.5 mIoU` repeat trigger.
Run the one-way R=2 seed-1 member of the paired repeat on the same exact
implementation commit. Do not run static seed 1 or add bidirectional
interaction in the same step.

## 1. Verify the exact source and environment

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

test "$(git rev-parse HEAD)" = "367694ee8bc2e670be0a896f40e089760fe2177a"
test -z "$(git status --porcelain)"

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

Expected: 81 tests pass, the DINOv3-L hash passes, and the GPU is the RTX
4090 D.

## 2. Reclaim only recorded Static Query seed-0 intermediate checkpoints

The Static Query seed-0 result is now recorded in Git. Keep its final
checkpoint and delete only the 79 exact-path intermediate checkpoints. This
should reclaim about 3.4 GB.

```bash
STATIC0_RUN=/root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED0_40000_367694e

test -f "$STATIC0_RUN/metadata.json"
test -f "$STATIC0_RUN/summary.json"
test -f "$STATIC0_RUN/checkpoints/iter_040000.pth"

STATIC0_COUNT="$(
  find "$STATIC0_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

STATIC0_INTERMEDIATE_COUNT="$(
  find "$STATIC0_RUN/checkpoints" \
    -maxdepth 1 -type f \
    -name 'iter_*.pth' \
    ! -name 'iter_040000.pth' \
    | wc -l
)"

echo "static0_checkpoint_count=$STATIC0_COUNT"
echo "static0_intermediate_count=$STATIC0_INTERMEDIATE_COUNT"

test "$STATIC0_COUNT" -eq 80
test "$STATIC0_INTERMEDIATE_COUNT" -eq 79

find "$STATIC0_RUN/checkpoints" \
  -maxdepth 1 -type f \
  -name 'iter_*.pth' \
  ! -name 'iter_040000.pth' \
  -print -delete

test -f "$STATIC0_RUN/checkpoints/iter_040000.pth"

STATIC0_REMAINING="$(
  find "$STATIC0_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

echo "static0_remaining_checkpoint_count=$STATIC0_REMAINING"
test "$STATIC0_REMAINING" -eq 1
du -sh "$STATIC0_RUN"
df -h /root/autodl-tmp
```

Do not delete the final checkpoint, metadata, summary, trace, or log.

## 3. Run one-way R=2 seed 1

This uses the already validated Phase 12 R=2 one-way configuration; only the
seed and unique run ID change.

```bash
RUN_ID=Q2_QUERY_COUNT_SEED1_40000_367694e
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail

QUERY_COUNT=2 \
MAX_ITERATIONS=40000 \
VALIDATION_MAX_SAMPLES=500 \
SEED=1 \
RUN_ID="$RUN_ID" \
bash scripts/train_query_count.sh \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

## 4. Validate and return full-run evidence

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

assert metadata["experiment_id"] == "Q2_QUERY_COUNT_SEED1_40000_367694e"
assert metadata["phase"] == 12
assert metadata["git_sha"] == "367694ee8bc2e670be0a896f40e089760fe2177a"
assert metadata["seed"] == 1
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

print("one_way_r2_seed1_full_ok=true")
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

echo "===== final log ====="
tail -n 30 "$LOG_FILE"

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the complete validator output and final evidence blocks. Stop after
this run: do not delete its checkpoints and do not start Static Query seed 1.
