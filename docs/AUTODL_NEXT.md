# AutoDL next step: Phase 14 learned-null seed-0 40k run

The exact-isolation smoke passes at implementation commit
`d85db4e0dcf411681c1ddad09080dfeed1c419bd`: 500 records are contiguous and
finite, objective reconstruction error is `1.790e-7`, peak reserved memory is
`2.713 GiB`, and null calibration loss decreases by approximately `84.8%`.

Run one seed-0 40k experiment with the identical mechanism. Do not add effect
invariance, sufficiency, specificity, semantic counterfactuals, or repeat
seeds. The fixed paired factual-path reference is Static seed 0 at final mIoU
`0.6473964462159979`.

## 1. Check out the exact implementation

```bash
cd /root/autodl-tmp/CausalQ_DG

test -z "$(git status --porcelain)"

IMPL_SHA=d85db4e0dcf411681c1ddad09080dfeed1c419bd

git fetch origin main
git cat-file -e "${IMPL_SHA}^{commit}"
git checkout --detach "$IMPL_SHA"

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

test "$(git rev-parse HEAD)" = "$IMPL_SHA"
test -z "$(git status --porcelain)"

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

Expected: `94 passed`, the DINOv3-L hash passes, the repository is clean, and
at least 10 GiB remains free. The observed 29 GiB is sufficient for the
approximately 3.4 GiB compact checkpoint set.

## 2. Run seed 0 for 40k iterations

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

test "$(git rev-parse HEAD)" = \
  "d85db4e0dcf411681c1ddad09080dfeed1c419bd"

NULL_RUN_ID=B4B_NULL_IQE_SEED0_40000_d85db4e
NULL_RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${NULL_RUN_ID}"
NULL_LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${NULL_RUN_ID}.log"

echo "NULL_RUN_DIR=$NULL_RUN_DIR"
echo "NULL_LOG_FILE=$NULL_LOG_FILE"

test ! -e "$NULL_RUN_DIR"
test ! -e "$NULL_LOG_FILE"

set -o pipefail

MAX_ITERATIONS=40000 \
VALIDATION_MAX_SAMPLES=500 \
SEED=0 \
RUN_ID="$NULL_RUN_ID" \
bash scripts/train_learned_null.sh \
  2>&1 | tee "$NULL_LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

## 3. Validate the complete run

```bash
python - "$NULL_RUN_DIR" <<'PY'
import json
import math
import statistics
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

expected_id = "B4B_NULL_IQE_SEED0_40000_d85db4e"
expected_sha = "d85db4e0dcf411681c1ddad09080dfeed1c419bd"

assert metadata["experiment_id"] == expected_id
assert metadata["git_sha"] == expected_sha
assert metadata["phase"] == 14
assert metadata["seed"] == 0
assert metadata["max_iterations"] == 40000
assert metadata["total_parameters"] == 306931476
assert metadata["trainable_parameters"] == 3801876

assert metadata["query"]["queries_per_class"] == 2
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["cross_attention_layers"] == 0
assert metadata["style"]["views"] == ["original", "photometric"]
assert metadata["learned_null"] == {
    "enabled": True,
    "shared_across_classes": True,
    "slots": "match_queries_per_class",
    "calibration_target": "mean_factual_query_state",
    "stop_gradient_target": True,
    "loss": "smooth_l1",
    "lambda_null": 1.0,
}
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
    "loss_null",
    "loss_null_weighted",
    "gradient_norm",
    "alpha",
)
assert all(
    math.isfinite(float(row[key]))
    for row in records
    for key in tracked
)

errors = [
    abs(
        float(row["loss"])
        - float(row["loss_original"])
        - float(row["loss_photometric"])
        - float(row["loss_null_weighted"])
    )
    for row in records
]
assert max(errors) <= 1e-5

assert len(validations) == 80
assert [row["iteration"] for row in validations] == \
    list(range(500, 40001, 500))
assert all(row["sample_count"] == 500 for row in validations)
assert all(math.isfinite(float(row["miou"])) for row in validations)

assert len(checkpoints) == 80
assert checkpoints[-1].name == "iter_040000.pth"

first_20_null = statistics.fmean(
    float(row["loss_null"]) for row in records[:20]
)
last_20_null = statistics.fmean(
    float(row["loss_null"]) for row in records[-20:]
)
best_validation = max(validations, key=lambda row: float(row["miou"]))
final_validation = validations[-1]
static_seed0 = 0.6473964462159979

print("learned_null_40k_ok=true")
print("===== metadata =====")
print(metadata)
print("===== summary =====")
print({k: v for k, v in summary.items() if k != "validation_results"})
print("===== trace =====")
print("record_count:", len(records))
print("iterations_contiguous:", True)
print("all_tracked_values_finite:", True)
print("maximum_objective_reconstruction_error:", max(errors))
print("first_20_null_loss_mean:", first_20_null)
print("last_20_null_loss_mean:", last_20_null)
print("null_loss_decreased:", last_20_null < first_20_null)
print("first_record:", records[0])
print("last_record:", records[-1])
print("===== validation =====")
print("validation_count:", len(validations))
print("final_validation:", final_validation)
print("best_validation:", best_validation)
print("===== factual-path comparison =====")
print("static_seed0_miou:", static_seed0)
print("learned_null_seed0_miou:", final_validation["miou"])
print(
    "learned_null_minus_static:",
    float(final_validation["miou"]) - static_seed0,
)
print(
    "learned_null_minus_static_percentage_points:",
    100.0 * (float(final_validation["miou"]) - static_seed0),
)
print("===== checkpoints =====")
print("checkpoint_count:", len(checkpoints))
print("final_checkpoint:", checkpoints[-1])
PY
```

## 4. Return evidence and stop

```bash
echo "===== checkpoint evidence ====="

CHECKPOINT_COUNT="$(
  find "$NULL_RUN_DIR/checkpoints" \
    -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

echo "checkpoint_count=$CHECKPOINT_COUNT"
test "$CHECKPOINT_COUNT" -eq 80

find "$NULL_RUN_DIR/checkpoints" \
  -maxdepth 1 -type f \
  -printf '%s %f\n' \
  | sort -k2 \
  | tail -n 5

test -f "$NULL_RUN_DIR/checkpoints/iter_040000.pth" \
  && echo "final_checkpoint_ok=true"

du -sh "$NULL_RUN_DIR/checkpoints"

echo "===== final log ====="
tail -n 30 "$NULL_LOG_FILE"

echo "===== provenance ====="
git rev-parse HEAD
git status --short

python - "$NULL_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metadata = json.loads((run_dir / "metadata.json").read_text())
print("experiment_id:", metadata["experiment_id"])
print("metadata_git_sha:", metadata["git_sha"])
PY

echo "===== error scan ====="
grep -E \
  'Traceback|AssertionError|FloatingPointError|CUDA out of memory' \
  "$NULL_LOG_FILE" || true

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the complete validator output and evidence blocks. Stop after this
seed-0 run. Do not start seed 1/2 or any learned-null effect analysis until the
factual-path result and null calibration trace have been reviewed.
