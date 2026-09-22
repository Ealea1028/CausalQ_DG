# Next AutoDL action

Status: the Phase 12 seed-0 R=2 full run completed stably at `0.643725`
Cityscapes mIoU. It exceeds R=1 by 4.0014 percentage points and the fixed
photometric-only R=3 reference by 0.5439 points. R=2 is the current best
candidate, but Phase 12 is not complete. Run only the seed-0 R=4 40k experiment
next; query-behavior analysis follows after all four final checkpoints exist.

## Reclaim R=2 intermediate-checkpoint space

Retain the final R=2 checkpoint and delete only its 79 intermediates.

```bash
cd /root/autodl-tmp/CausalQ_DG

R2_RUN=/root/autodl-tmp/outputs/CausalQ_DG/Q2_QUERY_COUNT_SEED0_40000_198898f
R2_RESOLVED="$(realpath "$R2_RUN")"

case "$R2_RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/Q2_QUERY_COUNT_SEED0_40000_198898f) ;;
  *) echo "unsafe path: $R2_RESOLVED"; exit 1 ;;
esac

test -f "$R2_RUN/metadata.json"
test -f "$R2_RUN/summary.json"
test -f "$R2_RUN/checkpoints/iter_040000.pth"
test "$(find "$R2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 80
test "$(find "$R2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)" -eq 79
du -sh "$R2_RUN/checkpoints"
```

Only after every check succeeds:

```bash
find "$R2_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$R2_RUN/checkpoints/iter_040000.pth"
test "$(find "$R2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$R2_RUN/checkpoints"
df -h /root/autodl-tmp
```

## Checkout and preflight

Replace `<EXACT_SHA_FROM_HANDOFF>` with the exact SHA supplied in the handoff.

```bash
cd /root/autodl-tmp/CausalQ_DG
git fetch origin
git checkout --detach <EXACT_SHA_FROM_HANDOFF>

test "$(git rev-parse HEAD)" = "<EXACT_SHA_FROM_HANDOFF>"
test -z "$(git status --porcelain)"

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

python tools/check_environment.py
python -m pytest
```

Verify that only query count changes from the accepted control:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, resolve_query_count, resolve_seed, style_view_weights

config = load_config(Path("configs/query_count/gta_dinov3l.yaml"))
assert config["experiment"]["phase"] == 12
assert resolve_seed(config, None) == 0
assert resolve_query_count(config, 4) == 4
assert style_view_weights(config["style"]) == {
    "original": 1.0,
    "photometric": 1.0,
}
assert "prediction_consistency" not in config
assert "causal_query_effect" not in config
assert "query_diversity" not in config
print("PHASE12_R4_FULL_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run the R=4 full experiment

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export QUERY_COUNT=4
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500
export SEED=0

test "$QUERY_COUNT" -eq 4
test "$MAX_ITERATIONS" -eq 40000
test "$VALIDATION_MAX_SAMPLES" -eq 500
test "$SEED" -eq 0

RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="Q4_QUERY_COUNT_SEED0_40000_${RUN_SHA}"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
QUERY_COUNT="$QUERY_COUNT" RUN_ID="$RUN_ID" \
  bash scripts/train_query_count.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0
test -f "$RUN_DIR/summary.json"
```

Do not interrupt training. Continue only after `train_exit_code=0`.

## Extract R=4 evidence

```bash
python - "$RUN_DIR" <<'PY'
import json
import math
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
records = [
    json.loads(line)
    for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
validations = summary["validation_results"]
errors = [
    abs(row["loss"] - (row["loss_original"] + row["loss_photometric"]))
    for row in records
]

assert metadata["phase"] == 12
assert metadata["seed"] == 0
assert metadata["max_iterations"] == 40000
assert metadata["query"]["queries_per_class"] == 4
assert metadata["style"]["views"] == ["original", "photometric"]
assert "prediction_consistency" not in metadata
assert "causal_query_effect" not in metadata
assert "query_diversity" not in metadata
assert summary["ok"] is True
assert len(records) == 40000
assert [row["iteration"] for row in records] == list(range(1, 40001))
for key in ("loss", "loss_original", "loss_photometric", "gradient_norm", "alpha"):
    assert all(math.isfinite(row[key]) for row in records), key
assert max(errors) < 1e-5
assert len(validations) == 80
assert all(row["sample_count"] == 500 for row in validations)

print("===== metadata =====")
print(metadata)
print("===== summary =====")
for key in (
    "ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
    "finite_losses", "last_gradient_norm", "final_alpha",
    "peak_allocated_gib", "peak_reserved_gib",
):
    print(f"{key}:", summary[key])
print("===== trace =====")
print("record_count:", len(records))
print("iterations_contiguous:", True)
print("all_tracked_values_finite:", True)
print("maximum_objective_reconstruction_error:", max(errors))
print("===== validation =====")
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
print("best_validation:", max(validations, key=lambda row: row["miou"]))
print("===== complete query-count mIoU table =====")
values = {
    1: 0.6037110117161197,
    2: 0.6437254689928078,
    3: 0.6382865707103561,
    4: validations[-1]["miou"],
}
for count, value in values.items():
    print(f"R={count}:", value)
best_count = max(values, key=values.get)
print("best_query_count_by_final_miou:", best_count)
print("r4_minus_r2:", values[4] - values[2])
print("r4_minus_r3:", values[4] - values[3])
PY

CHECKPOINT_COUNT="$(
  find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' \
    | wc -l
)"

INTERMEDIATE_COUNT="$(
  find "$RUN_DIR/checkpoints" -maxdepth 1 -type f \
    -name 'iter_*.pth' ! -name 'iter_040000.pth' \
    | wc -l
)"

echo "checkpoint_count=$CHECKPOINT_COUNT"
echo "intermediate_checkpoint_count=$INTERMEDIATE_COUNT"
du -sh "$RUN_DIR/checkpoints"

test "$CHECKPOINT_COUNT" -eq 80 \
  && echo "checkpoint_count_ok=true"
test "$INTERMEDIATE_COUNT" -eq 79 \
  && echo "intermediate_count_ok=true"
test -f "$RUN_DIR/checkpoints/iter_040000.pth" \
  && echo "final_checkpoint_ok=true"

echo "===== provenance ====="
cd /root/autodl-tmp/CausalQ_DG
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

## Acceptance criteria

- R=2 cleanup retains only `iter_040000.pth` and recovers roughly 15 GiB.
- Exact handoff SHA, clean Git status, passing tests, and the expected DINOv3
  hash are recorded.
- R=4 metadata records Phase 12, seed 0, 40,000 iterations, query count 4, and
  only original/photometric views.
- No prediction consistency, CQE, or query-diversity configuration is present.
- Exit code is zero; all 40,000 records and tracked values are finite and
  contiguous; objective reconstruction error is below `1e-5`.
- There are 80 complete 500-image validations and 80 checkpoints, including
  `iter_040000.pth`.
- The complete R=1/2/3/4 final-mIoU table, best query count, exact Git SHA, and
  disk status are printed.

Return all R=4 evidence. Stop afterward: do not delete its checkpoints. The
next local step will implement and verify the planned cross-count query
similarity, active-query, and effect-variance analysis before Phase 12 closes.
