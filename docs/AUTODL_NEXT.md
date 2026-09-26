# AutoDL next step: Phase 14 learned-null GPU smoke

The three-seed zero-ablation audit passes the semantic localization gate. Phase
14 adds only a class-agnostic R=2 learned-null bank to the selected Static Query
model. The factual segmentation path is unchanged. A detached factual-query
centroid calibrates the null slots; prediction consistency, CQE, diversity,
effect invariance, sufficiency, and specificity remain disabled.

Run only a 500-iteration, 50-image GPU smoke from exact commit
`2882c87fcd825754937e5346923bd64cf9b1d676`.

## 1. Check out and verify the exact implementation

```bash
cd /root/autodl-tmp/CausalQ_DG

test -z "$(git status --porcelain)"
git fetch origin
git checkout 2882c87fcd825754937e5346923bd64cf9b1d676

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

test "$(git rev-parse HEAD)" = \
  "2882c87fcd825754937e5346923bd64cf9b1d676"
test -z "$(git status --porcelain)"

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

Expected: `93 passed`, the DINOv3-L hash passes, the GPU is the RTX 4090 D,
and the three retained Static final checkpoints remain untouched.

## 2. Run the isolated 500-iteration learned-null smoke

```bash
NULL_RUN_ID=B4B_NULL_IQE_SMOKE_500_2882c87
NULL_RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${NULL_RUN_ID}"
NULL_LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${NULL_RUN_ID}.log"

echo "NULL_RUN_DIR=$NULL_RUN_DIR"
echo "NULL_LOG_FILE=$NULL_LOG_FILE"

test ! -e "$NULL_RUN_DIR"
test ! -e "$NULL_LOG_FILE"

set -o pipefail

MAX_ITERATIONS=500 \
VALIDATION_MAX_SAMPLES=50 \
SEED=0 \
RUN_ID="$NULL_RUN_ID" \
bash scripts/train_learned_null.sh \
  2>&1 | tee "$NULL_LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

## 3. Validate the smoke

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

assert metadata["experiment_id"] == "B4B_NULL_IQE_SMOKE_500_2882c87"
assert metadata["phase"] == 14
assert metadata["git_sha"] == \
    "2882c87fcd825754937e5346923bd64cf9b1d676"
assert metadata["seed"] == 0
assert metadata["max_iterations"] == 500
assert metadata["total_parameters"] == 306931476
assert metadata["trainable_parameters"] == 3801876
assert metadata["query"]["queries_per_class"] == 2
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["cross_attention_layers"] == 0
assert metadata["style"]["views"] == ["original", "photometric"]

null_config = metadata["learned_null"]
assert null_config == {
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
assert len(records) == 500
assert [row["iteration"] for row in records] == list(range(1, 501))

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
assert all(float(row["loss_null"]) > 0.0 for row in records)

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

first_20_null = statistics.fmean(
    float(row["loss_null"]) for row in records[:20]
)
last_20_null = statistics.fmean(
    float(row["loss_null"]) for row in records[-20:]
)

assert len(validations) == 1
assert validations[0]["iteration"] == 500
assert validations[0]["sample_count"] == 50
assert math.isfinite(float(validations[0]["miou"]))
assert len(checkpoints) == 1
assert checkpoints[0].name == "iter_000500.pth"

print("learned_null_smoke_ok=true")
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
print(validations[0])
print("===== checkpoint =====")
print("checkpoint_count:", len(checkpoints))
print("checkpoint:", checkpoints[0])
PY
```

## 4. Return evidence and stop

```bash
echo "===== checkpoint evidence ====="
find "$NULL_RUN_DIR/checkpoints" \
  -maxdepth 1 -type f \
  -printf '%s %f\n' \
  | sort -k2

du -sh "$NULL_RUN_DIR/checkpoints"

echo "===== final log ====="
tail -n 30 "$NULL_LOG_FILE"

echo "===== retained Static final checkpoints ====="
find \
  /root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED0_40000_367694e/checkpoints \
  /root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED1_40000_3860e69/checkpoints \
  /root/autodl-tmp/outputs/CausalQ_DG/I1_STATIC_QUERY_SEED2_40000_3860e69/checkpoints \
  -maxdepth 1 -type f -name 'iter_040000.pth' \
  -printf '%s %p\n'

echo "===== training exit evidence ====="
grep -E \
  'train_exit_code=|Traceback|FloatingPointError|CUDA out of memory' \
  "$NULL_LOG_FILE" || true

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the complete validator output and evidence blocks. Stop after the smoke.
Do not start a 40k run and do not add effect invariance, sufficiency,
specificity, or semantic counterfactuals.
