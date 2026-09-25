# AutoDL next step: Bidirectional Query seed-0 full run

The isolated Bidirectional Query GPU smoke passes at exact implementation
commit `13667167b641209a35ddf437616ba11fb6d5c246`. Run one 40,000-iteration
seed-0 experiment on the same commit. Do not add learned-null, start seed
repeats, delete retained final checkpoints, or change any hyperparameter.

## 1. Verify the exact implementation and available space

The smoke was already run on the required commit. Do not pull or check out the
newer documentation-only commit before training.

```bash
cd /root/autodl-tmp/CausalQ_DG

test "$(git rev-parse HEAD)" = \
  "13667167b641209a35ddf437616ba11fb6d5c246"
test -z "$(git status --porcelain)"

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

Expected: `85 passed`, the DINOv3-L hash passes, the GPU is the RTX 4090 D,
and at least 20 GiB is free. The reported 29 GiB is sufficient for the expected
roughly 15 GiB of full-run checkpoints. Stop if any check fails.

## 2. Run Bidirectional Query seed 0 for 40,000 iterations

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

BIDIR_FULL_RUN_ID=I2_BIDIRECTIONAL_QUERY_SEED0_40000_1366716
BIDIR_FULL_RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${BIDIR_FULL_RUN_ID}"
BIDIR_FULL_LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${BIDIR_FULL_RUN_ID}.log"

echo "BIDIR_FULL_RUN_DIR=$BIDIR_FULL_RUN_DIR"
echo "BIDIR_FULL_LOG_FILE=$BIDIR_FULL_LOG_FILE"

test ! -e "$BIDIR_FULL_RUN_DIR"
test ! -e "$BIDIR_FULL_LOG_FILE"

set -o pipefail

MAX_ITERATIONS=40000 \
VALIDATION_MAX_SAMPLES=500 \
SEED=0 \
RUN_ID="$BIDIR_FULL_RUN_ID" \
bash scripts/train_bidirectional_query.sh \
  2>&1 | tee "$BIDIR_FULL_LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

This is a new run from random initialization. Do not initialize it from the
500-iteration smoke checkpoint or any Static/One-way checkpoint.

## 3. Validate the completed run and compare seed 0

Run this block only after training exits with code 0.

```bash
python - "$BIDIR_FULL_RUN_DIR" <<'PY'
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

assert metadata["experiment_id"] == \
    "I2_BIDIRECTIONAL_QUERY_SEED0_40000_1366716"
assert metadata["phase"] == 13
assert metadata["git_sha"] == \
    "13667167b641209a35ddf437616ba11fb6d5c246"
assert metadata["seed"] == 0
assert metadata["max_iterations"] == 40000
assert metadata["query"]["queries_per_class"] == 2
assert metadata["query"]["interaction"] == "bidirectional"
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
assert [row["iteration"] for row in validations] == \
    list(range(500, 40001, 500))
assert all(row["sample_count"] == 500 for row in validations)
assert all(math.isfinite(float(row["miou"])) for row in validations)

assert len(checkpoints) == 80
assert checkpoints[-1].name == "iter_040000.pth"

final = validations[-1]
best = max(validations, key=lambda row: float(row["miou"]))
one_way_seed0 = 0.6437254689928078
static_seed0 = 0.6473964462159979

print("bidirectional_query_full_seed0_ok=true")
print("===== metadata =====")
print("experiment_id:", metadata["experiment_id"])
print("phase:", metadata["phase"])
print("git_sha:", metadata["git_sha"])
print("seed:", metadata["seed"])
print("query:", metadata["query"])
print("style_views:", metadata["style"]["views"])
print("===== summary =====")
print(
    {key: value for key, value in summary.items()
     if key != "validation_results"}
)
print("===== trace =====")
print("record_count:", len(records))
print("iterations_contiguous:", True)
print("maximum_objective_reconstruction_error:", max(reconstruction_errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("===== validation =====")
print("validation_count:", len(validations))
print("final_validation:", final)
print("best_validation:", best)
print("===== seed-0 comparison =====")
print("one_way_seed0_miou:", one_way_seed0)
print("static_seed0_miou:", static_seed0)
print("bidirectional_seed0_miou:", final["miou"])
print(
    "bidirectional_minus_one_way_percentage_points:",
    100 * (float(final["miou"]) - one_way_seed0),
)
print(
    "bidirectional_minus_static_percentage_points:",
    100 * (float(final["miou"]) - static_seed0),
)
print("===== checkpoints =====")
print("checkpoint_count:", len(checkpoints))
print("final_checkpoint:", checkpoints[-1])
PY
```

## 4. Return evidence and stop

```bash
echo "===== checkpoint evidence ====="

find "$BIDIR_FULL_RUN_DIR/checkpoints" \
  -maxdepth 1 -type f \
  -printf '%s %f\n' \
  | sort -k2 \
  | tail -n 5

BIDIR_CHECKPOINT_COUNT="$(
  find "$BIDIR_FULL_RUN_DIR/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

echo "bidirectional_checkpoint_count=$BIDIR_CHECKPOINT_COUNT"
test "$BIDIR_CHECKPOINT_COUNT" -eq 80 \
  && echo "bidirectional_checkpoint_count_ok=true"

test -f "$BIDIR_FULL_RUN_DIR/checkpoints/iter_040000.pth" \
  && echo "bidirectional_final_checkpoint_ok=true"

du -sh "$BIDIR_FULL_RUN_DIR/checkpoints"

echo "===== final log ====="
tail -n 30 "$BIDIR_FULL_LOG_FILE"

echo "===== training exit evidence ====="
grep -E \
  'train_exit_code=|Traceback|FloatingPointError|CUDA out of memory' \
  "$BIDIR_FULL_LOG_FILE" || true

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the complete validator output and all evidence blocks. Stop after this
run. Do not delete its checkpoints, start seed 1/2, or add learned-null. The
seed-0 result determines whether paired repeats are justified.
