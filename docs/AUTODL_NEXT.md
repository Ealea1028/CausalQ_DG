# Next AutoDL action

Status: Phase 8 is a complete, stable control at `0.6212242964` Cityscapes mIoU; it missed the predefined floor and is retained without post-hoc tuning. Phase 9 adds only cross-style causal-query-effect distillation. Run the 500-iteration CQE smoke test next; do not begin full training or add prediction consistency/diversity.

## Fixed Phase 9 mechanism

- Public effect API: `model.get_query_effect(...)`.
- Logit effect: `E(c,s) = alpha * delta_logits(c,s)`.
- Original effect is the detached reference.
- Each class map is L2-normalized over valid pixels.
- SmoothL1 is averaged over photometric/Fourier views and GT-present classes.
- `lambda_cqe=1.0`; supervised style loss remains unchanged.
- Prediction consistency is disabled, making A4 a clean comparison with A3.

## Reclaim Phase 8 intermediate-checkpoint space

```bash
PRED_RUN=/root/autodl-tmp/outputs/CausalQ_DG/A3_PRED_CONS_SEED0_40000_4a95ebd
RESOLVED="$(realpath "$PRED_RUN")"

case "$RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/A3_PRED_CONS_SEED0_40000_4a95ebd) ;;
  *) echo "unsafe path: $RESOLVED"; exit 1 ;;
esac

test -f "$PRED_RUN/summary.json"
test -f "$PRED_RUN/checkpoints/iter_040000.pth"
echo "checkpoint_count_before=$(find "$PRED_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)"
echo "intermediate_count=$(find "$PRED_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)"
du -sh "$PRED_RUN/checkpoints"
```

Require 80 total and 79 intermediates, then retain only the final checkpoint:

```bash
find "$PRED_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$PRED_RUN/checkpoints/iter_040000.pth"
test "$(find "$PRED_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$PRED_RUN/checkpoints"
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
python -m pytest tests/test_causal_effect.py tests/test_cqe_loss.py
```

Both Git status checks must be empty and all CQE tests must pass. Verify references and mechanism isolation:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml

output = Path("/root/autodl-tmp/outputs/CausalQ_DG")
references = {
    "A2_QUERY_STYLE_SEED0_40000_293b330": 0.639610125136919,
    "A3_PRED_CONS_SEED0_40000_4a95ebd": 0.621224296375965,
}
for run, expected_miou in references.items():
    run_dir = output / run
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["max_iterations"] == 40000
    assert len(summary["validation_results"]) == 80
    assert summary["validation_results"][-1]["miou"] == expected_miou
    assert (run_dir / "checkpoints/iter_040000.pth").is_file()

config = yaml.safe_load(Path(
    "configs/causalq/gta_dinov3l_causalq.yaml"
).read_text(encoding="utf-8"))
cqe = config["causal_query_effect"]
assert config["experiment"]["phase"] == 9
assert config["style"]["lambda_cf"] == 1.0
assert cqe["enabled"] is True
assert cqe["lambda_cqe"] == 1.0
assert cqe["effect_space"] == "logits"
assert cqe["reference_stop_gradient"] is True
assert cqe["normalization"] == "l2_per_class_map"
assert cqe["present_classes_only"] is True
assert cqe["valid_pixels_only"] is True
assert cqe["loss"] == "smooth_l1"
assert "prediction_consistency" not in config

print("PHASE9_SMOKE_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The weight hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run the 500-iteration CQE smoke test

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A4_CQE_SMOKE_500_${RUN_SHA}"
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50

RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
bash scripts/train_causalq.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "run_id=$RUN_ID"
echo "train_exit_code=$TRAIN_EXIT"
echo "log_file=$LOG_FILE"
```

## Smoke evidence

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A4_CQE_SMOKE_500_${RUN_SHA}"
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
    "loss_cqe", "loss_cqe_photometric", "loss_cqe_fourier",
    "gradient_norm", "alpha",
)
iterations = [record["iteration"] for record in records]
errors = [abs(record["loss"] - (
    record["loss_original"]
    + 0.5 * record["loss_photometric"]
    + 0.5 * record["loss_fourier"]
    + record["loss_cqe"]
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
print("minimum_cqe_loss:", min(record["loss_cqe"] for record in records))
print("maximum_cqe_loss:", max(record["loss_cqe"] for record in records))
print("last_20_cqe_mean:", sum(record["loss_cqe"] for record in records[-20:]) / 20)
print("maximum_objective_reconstruction_error:", max(errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("validation_count:", len(summary["validation_results"]))
print("final_validation:", summary["validation_results"][-1])
print("has_prediction_consistency:", "prediction_consistency" in metadata)
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

- Cleanup retains the A3 final checkpoint and removes only 79 intermediates.
- All gates and CQE unit tests pass.
- `train_exit_code=0`, `summary.ok=true`, and 500 iterations are contiguous.
- Supervised/CQE losses, gradients, and alpha are finite.
- CQE is non-negative, becomes non-zero after alpha leaves zero, and does not explode.
- Objective reconstruction error is negligible.
- Metadata contains CQE and no prediction consistency.
- One 50-image validation and `iter_000500.pth` exist.
- Peak memory remains within RTX 4090D capacity.
- Exact Git SHA and clean final status are reported.

Return cleanup, gates/tests, training exit code, compact evidence, checkpoint, provenance, and disk output. Stop after the smoke test; do not begin the full Phase 9 run.
