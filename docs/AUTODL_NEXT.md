# AutoDL next step: Bidirectional Query GPU smoke

The three-seed interaction comparison selects Static over one-way. The remaining
planned Phase 13 structural control is bidirectional query-image self-attention,
implemented at exact commit
`13667167b641209a35ddf437616ba11fb6d5c246`. Run only a 500-iteration,
50-image GPU smoke. Do not start a full run or add learned-null.

## 1. Check out and verify the exact implementation

```bash
cd /root/autodl-tmp/CausalQ_DG

test -z "$(git status --porcelain)"
git fetch origin
git checkout 13667167b641209a35ddf437616ba11fb6d5c246

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

test "$(git rev-parse HEAD)" = \
  "13667167b641209a35ddf437616ba11fb6d5c246"
test -z "$(git status --porcelain)"

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

Expected: `85 passed`, the DINOv3-L hash passes, and the GPU is the RTX
4090 D. Stop if any check fails.

## 2. Keep the Static seed-2 final checkpoint and reclaim intermediates

The Static seed-2 result is recorded in Git. Delete only its 79 intermediate
checkpoints, retaining `iter_040000.pth`, metadata, summary, trace, and log.

```bash
STATIC2_RUN=/root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED2_40000_3860e69

test -f "$STATIC2_RUN/metadata.json"
test -f "$STATIC2_RUN/summary.json"
test -f "$STATIC2_RUN/checkpoints/iter_040000.pth"
test "$(wc -l < "$STATIC2_RUN/train.jsonl")" -eq 40000

STATIC2_CHECKPOINT_COUNT="$(
  find "$STATIC2_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

STATIC2_INTERMEDIATE_COUNT="$(
  find "$STATIC2_RUN/checkpoints" \
    -maxdepth 1 -type f \
    -name 'iter_*.pth' \
    ! -name 'iter_040000.pth' \
    | wc -l
)"

echo "static2_checkpoint_count=$STATIC2_CHECKPOINT_COUNT"
echo "static2_intermediate_count=$STATIC2_INTERMEDIATE_COUNT"

test "$STATIC2_CHECKPOINT_COUNT" -eq 80
test "$STATIC2_INTERMEDIATE_COUNT" -eq 79

find "$STATIC2_RUN/checkpoints" \
  -maxdepth 1 -type f \
  -name 'iter_*.pth' \
  ! -name 'iter_040000.pth' \
  -print -delete

test -f "$STATIC2_RUN/checkpoints/iter_040000.pth"

STATIC2_REMAINING="$(
  find "$STATIC2_RUN/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

echo "static2_remaining_checkpoint_count=$STATIC2_REMAINING"
test "$STATIC2_REMAINING" -eq 1
du -sh "$STATIC2_RUN"
df -h /root/autodl-tmp
```

## 3. Run the isolated Bidirectional smoke

```bash
BIDIR_RUN_ID=I2_BIDIRECTIONAL_QUERY_SMOKE_500_1366716
BIDIR_RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${BIDIR_RUN_ID}"
BIDIR_LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${BIDIR_RUN_ID}.log"

echo "BIDIR_RUN_DIR=$BIDIR_RUN_DIR"
echo "BIDIR_LOG_FILE=$BIDIR_LOG_FILE"

test ! -e "$BIDIR_RUN_DIR"
test ! -e "$BIDIR_LOG_FILE"

set -o pipefail

MAX_ITERATIONS=500 \
VALIDATION_MAX_SAMPLES=50 \
SEED=0 \
RUN_ID="$BIDIR_RUN_ID" \
bash scripts/train_bidirectional_query.sh \
  2>&1 | tee "$BIDIR_LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

## 4. Validate and return smoke evidence

```bash
python - "$BIDIR_RUN_DIR" <<'PY'
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
    "I2_BIDIRECTIONAL_QUERY_SMOKE_500_1366716"
assert metadata["phase"] == 13
assert metadata["git_sha"] == \
    "13667167b641209a35ddf437616ba11fb6d5c246"
assert metadata["seed"] == 0
assert metadata["max_iterations"] == 500
assert metadata["query"]["queries_per_class"] == 2
assert metadata["query"]["interaction"] == "bidirectional"
assert metadata["query"]["cross_attention_layers"] == 1
assert metadata["style"]["views"] == ["original", "photometric"]
assert "prediction_consistency" not in metadata
assert "causal_query_effect" not in metadata
assert "query_diversity" not in metadata

assert summary["ok"] is True
assert len(records) == 500
assert [row["iteration"] for row in records] == list(range(1, 501))

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

assert len(validations) == 1
assert validations[0]["iteration"] == 500
assert validations[0]["sample_count"] == 50
assert math.isfinite(float(validations[0]["miou"]))
assert len(checkpoints) == 1
assert checkpoints[0].name == "iter_000500.pth"

print("bidirectional_query_smoke_ok=true")
print("metadata:", metadata)
print(
    "summary:",
    {key: value for key, value in summary.items() if key != "validation_results"},
)
print("record_count:", len(records))
print("maximum_objective_reconstruction_error:", max(reconstruction_errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("validation:", validations[0])
print("checkpoint_count:", len(checkpoints))
print("checkpoint:", checkpoints[0])
PY

echo "===== checkpoint evidence ====="
find "$BIDIR_RUN_DIR/checkpoints" \
  -maxdepth 1 -type f \
  -printf '%s %f\n' \
  | sort -k2

du -sh "$BIDIR_RUN_DIR/checkpoints"

echo "===== Static seed-2 final checkpoint ====="
test -f "$STATIC2_RUN/checkpoints/iter_040000.pth"
find "$STATIC2_RUN/checkpoints" \
  -maxdepth 1 -type f -name 'iter_*.pth' \
  -printf '%s %f\n'

echo "===== final log ====="
tail -n 30 "$BIDIR_LOG_FILE"

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the complete validator output and evidence blocks. Stop after the smoke:
do not delete its checkpoint, do not start a 40k bidirectional run, and do not
add learned-null.
