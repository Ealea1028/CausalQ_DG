# Next AutoDL action

Status: Phase 7 `A2_QUERY_STYLE_SEED0_40000_293b330` is accepted at `0.6396101251` Cityscapes mIoU. Phase 8 adds only ordinary prediction consistency under the same Query + Style setup. Run the 500-iteration Phase 8 smoke test next; do not start full training or add CQE.

## Fixed Phase 8 control

- Original-view prediction is the detached teacher.
- Photometric and Fourier predictions are students.
- `Lpred = mean(KL(P_original || P_style))` over valid label pixels.
- `lambda_pred=1.0`, temperature `1.0`.
- Total loss is `Lseg + Lseg_cf + Lpred`.
- Geometry, model, optimizer, seed, style strengths, and sequential forwards are unchanged from Phase 7.

## Reclaim Phase 7 intermediate-checkpoint space

Preview the exact accepted run before deletion:

```bash
STYLE_RUN=/root/autodl-tmp/outputs/CausalQ_DG/A2_QUERY_STYLE_SEED0_40000_293b330
RESOLVED="$(realpath "$STYLE_RUN")"

case "$RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/A2_QUERY_STYLE_SEED0_40000_293b330) ;;
  *) echo "unsafe path: $RESOLVED"; exit 1 ;;
esac

test -f "$STYLE_RUN/summary.json"
test -f "$STYLE_RUN/checkpoints/iter_040000.pth"

echo "checkpoint_count_before=$(find "$STYLE_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)"
echo "intermediate_count=$(find "$STYLE_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)"
du -sh "$STYLE_RUN/checkpoints"
```

Require 80 total and 79 intermediate checkpoints. Then retain only the final checkpoint:

```bash
find "$STYLE_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$STYLE_RUN/checkpoints/iter_040000.pth"
test "$(find "$STYLE_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1

du -sh "$STYLE_RUN/checkpoints"
df -h /root/autodl-tmp
```

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

Both Git status checks must be empty and all prediction-consistency tests must pass. Verify the Phase 7 reference and fixed Phase 8 configuration:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml

root = Path("/root/autodl-tmp/outputs/CausalQ_DG")
style_dir = root / "A2_QUERY_STYLE_SEED0_40000_293b330"
style = json.loads((style_dir / "summary.json").read_text(encoding="utf-8"))

assert style["ok"] is True
assert style["git_sha"] == "293b3309e5dd4c95086e95c9a35576754444e7e5"
assert style["max_iterations"] == 40000
assert len(style["validation_results"]) == 80
assert style["validation_results"][-1]["miou"] == 0.639610125136919
assert (style_dir / "checkpoints/iter_040000.pth").is_file()

config = yaml.safe_load(Path(
    "configs/consistency/gta_dinov3l_pred_cons.yaml"
).read_text(encoding="utf-8"))
prediction = config["prediction_consistency"]

assert config["experiment"]["phase"] == 8
assert config["style"]["lambda_cf"] == 1.0
assert prediction["enabled"] is True
assert prediction["lambda_pred"] == 1.0
assert prediction["temperature"] == 1.0
assert prediction["divergence"] == "kl_reference_to_view"
assert prediction["reference_stop_gradient"] is True
assert prediction["valid_pixels_only"] is True
assert "causal_query_effect" not in config

print("PHASE8_SMOKE_GATES_OK")
print("phase7_miou:", style["validation_results"][-1]["miou"])
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The weight hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run the 500-iteration smoke test

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A3_PRED_CONS_SMOKE_500_${RUN_SHA}"
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50

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

## Smoke evidence

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A3_PRED_CONS_SMOKE_500_${RUN_SHA}"
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

print("metadata:", metadata)
for key in ("ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
            "finite_losses", "last_gradient_norm", "final_alpha",
            "peak_allocated_gib", "peak_reserved_gib"):
    print(f"{key}: {summary[key]}")
print("record_count:", len(records))
print("iterations_contiguous:", iterations == list(range(1, 501)))
for key in keys:
    print(f"all_{key}_finite:", all(math.isfinite(record[key]) for record in records))
print("minimum_prediction_loss:", min(record["loss_prediction"] for record in records))
print("maximum_objective_reconstruction_error:", max(errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("validation_count:", len(summary["validation_results"]))
print("final_validation:", summary["validation_results"][-1])
print("has_cqe_metadata:", "causal_query_effect" in metadata)
PY

echo "===== checkpoint ====="
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2
test -f "$RUN_DIR/checkpoints/iter_000500.pth" && echo "final_checkpoint_ok=true"

echo "===== provenance and disk ====="
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- Cleanup retains the Phase 7 final checkpoint and frees its 79 intermediates.
- All gates and unit tests pass.
- `train_exit_code=0`, `summary.ok=true`, and 500 iterations are contiguous.
- Total, segmentation, prediction, gradient, and alpha values are finite.
- Prediction KL is non-negative up to negligible floating-point error.
- Objective reconstruction error is negligible.
- Metadata includes prediction consistency and no CQE.
- One 50-image validation and `iter_000500.pth` exist.
- Peak memory remains within RTX 4090D capacity.
- Exact Git SHA and clean final status are reported.

Return cleanup, gates/tests, training exit code, compact evidence, checkpoint, provenance, and disk output. Stop after the smoke test; do not begin the full Phase 8 run.
