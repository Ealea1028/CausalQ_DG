# Next AutoDL action

Status: the Phase 12 `R=1,2,4` query-count smoke tests passed at `be450a1`.
Run only the seed-0 `R=1` 40k experiment next. The selected photometric-only
style protocol remains fixed, and the existing seed-0 `R=3` result
(`0.6382865707103561` final Cityscapes mIoU) is the reference. Do not launch
`R=2` or `R=4` full training in this step.

## Reclaim the three smoke checkpoints

Each smoke is already recorded in Git. Validate the exact paths and artifacts,
then remove only the three `iter_000500.pth` files.

```bash
cd /root/autodl-tmp/CausalQ_DG

for QUERY_COUNT in 1 2 4; do
  RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/Q${QUERY_COUNT}_QUERY_COUNT_SMOKE_500_be450a1"
  RESOLVED="$(realpath "$RUN_DIR")"
  EXPECTED="/root/autodl-tmp/outputs/CausalQ_DG/Q${QUERY_COUNT}_QUERY_COUNT_SMOKE_500_be450a1"

  test "$RESOLVED" = "$EXPECTED"
  test -f "$RUN_DIR/metadata.json"
  test -f "$RUN_DIR/summary.json"
  test -f "$RUN_DIR/checkpoints/iter_000500.pth"
  test "$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
  du -sh "$RUN_DIR/checkpoints"
done
```

Only after all checks above succeed:

```bash
for QUERY_COUNT in 1 2 4; do
  RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/Q${QUERY_COUNT}_QUERY_COUNT_SMOKE_500_be450a1"
  rm -- "$RUN_DIR/checkpoints/iter_000500.pth"
  test "$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 0
done

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

Verify the single-variable protocol and pretrained checkpoint:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, resolve_query_count, resolve_seed, style_view_weights

config = load_config(Path("configs/query_count/gta_dinov3l.yaml"))
assert config["experiment"]["phase"] == 12
assert resolve_seed(config, None) == 0
assert resolve_query_count(config, 1) == 1
assert style_view_weights(config["style"]) == {
    "original": 1.0,
    "photometric": 1.0,
}
assert "prediction_consistency" not in config
assert "causal_query_effect" not in config
assert "query_diversity" not in config
print("PHASE12_R1_FULL_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run the R=1 full experiment

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export QUERY_COUNT=1
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500
export SEED=0

test "$QUERY_COUNT" -eq 1
test "$MAX_ITERATIONS" -eq 40000
test "$VALIDATION_MAX_SAMPLES" -eq 500
test "$SEED" -eq 0

RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="Q1_QUERY_COUNT_SEED0_40000_${RUN_SHA}"
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

## Extract R=1 evidence

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
assert metadata["query"]["queries_per_class"] == 1
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
print("===== R=3 reference comparison =====")
reference = 0.6382865707103561
candidate = validations[-1]["miou"]
print("r1_miou:", candidate)
print("r3_photometric_miou:", reference)
print("r1_minus_r3:", candidate - reference)
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

- The three smoke checkpoint files are removed only after their summaries and
  exact paths are validated.
- The exact handoff SHA is checked out with clean Git status, all tests pass,
  and the DINOv3 hash matches.
- R=1 metadata records Phase 12, seed 0, 40,000 iterations, query count 1, and
  only original/photometric views.
- No prediction consistency, CQE, or query-diversity configuration is present.
- Exit code is zero; all 40,000 records and component values are finite and
  contiguous; objective reconstruction error is below `1e-5`.
- There are 80 complete 500-image validations and 80 checkpoints, including
  `iter_040000.pth`.
- Final and best Cityscapes mIoU, the fixed R=3 comparison, exact Git SHA, and
  disk status are printed.

Return all R=1 evidence. Stop after this experiment: do not delete its
checkpoints and do not start R=2 or R=4 until R=1 has been recorded in Git.
