# Next AutoDL action

Status: the Phase 8 prediction-consistency smoke test at implementation commit `ec93fbbc754c21803971d98514e3b4e2d7c43b26` is operator-confirmed as accepted. Run a fresh 40,000-iteration seed-0 Phase 8 control next. Do not resume the smoke checkpoint and do not add CQE.

## Fixed comparison

- Reference: `A2_QUERY_STYLE_SEED0_40000_293b330`, Cityscapes mIoU `0.639610125136919`.
- Experiment: `A3_PRED_CONS`.
- Same backbone, query branch, style views, supervised losses, optimizer, schedule, seed, and validation protocol as A2.
- Only addition: valid-pixel `KL(P_original.detach() || P_style)`, averaged over photometric and Fourier views.
- `lambda_pred=1.0`, temperature `1.0`.
- No CQE or diversity objective.
- Acceptance floor: `0.629610125136919` (no more than one percentage point below A2).

## Checkout and preflight

```bash
cd /root/autodl-tmp/CausalQ_DG
git status --short
git fetch origin
git checkout --detach <EXACT_SHA_FROM_HANDOFF>
git rev-parse HEAD
git status --short

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
python tools/check_environment.py
python -m pytest tests/test_prediction_consistency.py
```

Both status checks must be empty. Validate the accepted smoke artifact without relying on unreported copied metrics:

```bash
python - <<'PY'
import json
import math
from pathlib import Path

output = Path("/root/autodl-tmp/outputs/CausalQ_DG")
smoke_dir = output / "A3_PRED_CONS_SMOKE_500_ec93fbb"
metadata = json.loads((smoke_dir / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((smoke_dir / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (smoke_dir / "train.jsonl").read_text(
    encoding="utf-8"
).splitlines() if line.strip()]

assert metadata["phase"] == 8
assert metadata["git_sha"] == "ec93fbbc754c21803971d98514e3b4e2d7c43b26"
assert metadata["prediction_consistency"]["enabled"] is True
assert metadata["prediction_consistency"]["lambda_pred"] == 1.0
assert "causal_query_effect" not in metadata
assert summary["ok"] is True
assert summary["max_iterations"] == 500
assert summary["finite_losses"] is True
assert len(records) == 500
assert [record["iteration"] for record in records] == list(range(1, 501))
assert len(summary["validation_results"]) == 1
assert summary["validation_results"][0]["sample_count"] == 50
assert (smoke_dir / "checkpoints/iter_000500.pth").is_file()

keys = (
    "loss", "loss_original", "loss_photometric", "loss_fourier",
    "loss_prediction", "loss_prediction_photometric",
    "loss_prediction_fourier", "gradient_norm", "alpha",
)
for key in keys:
    assert all(math.isfinite(record[key]) for record in records), key

errors = [abs(record["loss"] - (
    record["loss_original"]
    + 0.5 * record["loss_photometric"]
    + 0.5 * record["loss_fourier"]
    + record["loss_prediction"]
)) for record in records]
assert max(errors) < 1e-6
assert min(record["loss_prediction"] for record in records) > -1e-6

style_dir = output / "A2_QUERY_STYLE_SEED0_40000_293b330"
style = json.loads((style_dir / "summary.json").read_text(encoding="utf-8"))
assert style["ok"] is True
assert style["validation_results"][-1]["miou"] == 0.639610125136919
assert (style_dir / "checkpoints/iter_040000.pth").is_file()

print("PHASE8_FULL_RUN_GATES_OK")
print("smoke_miou:", summary["validation_results"][-1]["miou"])
print("smoke_final_alpha:", summary["final_alpha"])
print("maximum_objective_error:", max(errors))
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
```

The weight hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

Check storage using the measured smoke checkpoint and retain a 5 GiB reserve:

```bash
SMOKE_CHECKPOINT=/root/autodl-tmp/outputs/CausalQ_DG/A3_PRED_CONS_SMOKE_500_ec93fbb/checkpoints/iter_000500.pth
CHECKPOINT_BYTES="$(stat -c '%s' "$SMOKE_CHECKPOINT")"
ESTIMATED_CHECKPOINT_BYTES="$((CHECKPOINT_BYTES * 80))"
FREE_BYTES="$(df --output=avail -B1 /root/autodl-tmp | tail -n 1 | tr -d ' ')"
SAFETY_BYTES="$((5 * 1024 * 1024 * 1024))"
REQUIRED_BYTES="$((ESTIMATED_CHECKPOINT_BYTES + SAFETY_BYTES))"

echo "checkpoint_bytes=$CHECKPOINT_BYTES"
echo "estimated_80_checkpoints_bytes=$ESTIMATED_CHECKPOINT_BYTES"
echo "free_bytes=$FREE_BYTES"
echo "required_with_5gib_reserve=$REQUIRED_BYTES"
test "$FREE_BYTES" -gt "$REQUIRED_BYTES"
df -h /root/autodl-tmp
```

Stop if the storage assertion fails.

## Full run

Use a persistent terminal such as `tmux new -s causalq_a3_full`, then run:

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A3_PRED_CONS_SEED0_40000_${RUN_SHA}"
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500

RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
bash scripts/train_pred_cons.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "run_id=$RUN_ID"
echo "train_exit_code=$TRAIN_EXIT"
echo "log_file=$LOG_FILE"
```

## Completion evidence

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A3_PRED_CONS_SEED0_40000_${RUN_SHA}"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"

python - "$RUN_DIR" <<'PY'
import json
import math
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text(
    encoding="utf-8"
).splitlines() if line.strip()]

keys = (
    "loss", "loss_original", "loss_photometric", "loss_fourier",
    "loss_prediction", "loss_prediction_photometric",
    "loss_prediction_fourier", "gradient_norm", "alpha",
)
iterations = [record["iteration"] for record in records]
errors = [abs(record["loss"] - (
    record["loss_original"]
    + 0.5 * record["loss_photometric"]
    + 0.5 * record["loss_fourier"]
    + record["loss_prediction"]
)) for record in records]
validations = summary["validation_results"]
style_miou = 0.639610125136919
pred_miou = validations[-1]["miou"]

print("metadata:", metadata)
for key in ("ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
            "finite_losses", "last_gradient_norm", "final_alpha",
            "peak_allocated_gib", "peak_reserved_gib"):
    print(f"{key}: {summary[key]}")
print("record_count:", len(records))
print("iterations_contiguous:", iterations == list(range(1, 40001)))
for key in keys:
    print(f"all_{key}_finite:", all(math.isfinite(record[key]) for record in records))
print("minimum_prediction_loss:", min(record["loss_prediction"] for record in records))
print("maximum_objective_reconstruction_error:", max(errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
print("style_miou:", style_miou)
print("prediction_consistency_miou:", pred_miou)
print("prediction_consistency_minus_style:", pred_miou - style_miou)
print("passes_minus_1pp_floor:", pred_miou >= style_miou - 0.01)
print("has_cqe_metadata:", "causal_query_effect" in metadata)
PY

echo "===== checkpoint evidence ====="
CHECKPOINT_COUNT="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)"
echo "checkpoint_count=$CHECKPOINT_COUNT"
du -sh "$RUN_DIR/checkpoints"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2 | tail -n 5
test "$CHECKPOINT_COUNT" -eq 80 && echo "checkpoint_count_ok=true"
test -f "$RUN_DIR/checkpoints/iter_040000.pth" && echo "final_checkpoint_ok=true"

echo "===== provenance and disk ====="
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- All smoke, reference, test, weight, and storage gates pass.
- `train_exit_code=0`, `summary.ok=true`, and 40,000 iterations are contiguous.
- All supervised/prediction losses, gradients, and alpha values are finite.
- Prediction KL is non-negative up to negligible numerical error.
- Objective reconstruction error is negligible.
- Metadata contains prediction consistency and no CQE.
- Eighty full 500-image validations and 80 checkpoints complete.
- Final mIoU is at least `0.629610125136919`.
- Exact Git SHA and clean status are recorded.

Return all gate/storage output, training exit code, compact evidence, checkpoint evidence, Git SHA/status, and disk usage. Stop after Phase 8 full training; do not begin Phase 9.
