# AutoDL next step: valid-crop fallback GPU smoke

The first one-way R=2 seed-1 repeat stopped after 3,316 iterations at commit
`367694e` because all random crop attempts for one supervised sample missed
its valid pixels. The complete 24,966-label audit passed, so commit
`3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497` adds only a fallback crop that is
guaranteed to include a real valid pixel when every ordinary random attempt is
empty. Run a fresh 500-iteration GPU smoke before restarting any 40k run.

## 1. Fetch and select the exact implementation commit

```bash
cd /root/autodl-tmp/CausalQ_DG

test -z "$(git status --porcelain)"

git fetch origin main
git checkout 3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497

test "$(git rev-parse HEAD)" = \
  "3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497"
test -z "$(git status --porcelain)"

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

python -m pytest

echo 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors' \
  | sha256sum -c -

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h /root/autodl-tmp
```

Expected: `82 passed`, the DINOv3-L hash passes, and the GPU is the RTX
4090 D. Do not continue if any check fails.

## 2. Run a fresh 500-iteration one-way R=2 seed-1 smoke

Do not reuse or resume the failed run directory.

```bash
RUN_ID=Q2_ONE_WAY_SEED1_CROP_FALLBACK_SMOKE_500_3860e69
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail

QUERY_COUNT=2 \
MAX_ITERATIONS=500 \
VALIDATION_MAX_SAMPLES=50 \
SEED=1 \
RUN_ID="$RUN_ID" \
bash scripts/train_query_count.sh \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
```

## 3. Validate and return smoke evidence

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

assert metadata["experiment_id"] == \
    "Q2_ONE_WAY_SEED1_CROP_FALLBACK_SMOKE_500_3860e69"
assert metadata["phase"] == 12
assert metadata["git_sha"] == \
    "3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497"
assert metadata["seed"] == 1
assert metadata["max_iterations"] == 500
assert metadata["query"]["queries_per_class"] == 2
assert metadata["query"]["interaction"] == "one_way"
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

print("crop_fallback_gpu_smoke_ok=true")
print("metadata:", metadata)
print(
    "summary:",
    {key: value for key, value in summary.items() if key != "validation_results"},
)
print("record_count:", len(records))
print("maximum_objective_reconstruction_error:", max(reconstruction_errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("final_validation:", validations[0])
print("checkpoint_count:", len(checkpoints))
print("checkpoint:", checkpoints[0])
PY

echo "===== failed-run preservation ====="
FAILED_RUN=/root/autodl-tmp/outputs/CausalQ_DG/Q2_QUERY_COUNT_SEED1_40000_367694e
test -d "$FAILED_RUN"
wc -l "$FAILED_RUN/train.jsonl"
find "$FAILED_RUN/checkpoints" \
  -maxdepth 1 -type f -name 'iter_*.pth' \
  | wc -l

echo "===== smoke checkpoint ====="
find "$RUN_DIR/checkpoints" \
  -maxdepth 1 -type f \
  -printf '%s %f\n'

echo "===== final log ====="
tail -n 30 "$LOG_FILE"

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

Return the complete validator output and evidence blocks. Stop after this
smoke: do not delete the failed run yet and do not start the 40k retry.
