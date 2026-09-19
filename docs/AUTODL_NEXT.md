# Next AutoDL action

Status: Phase 9 is complete with a failed accuracy target and a passed effect-variance target. Phase 10 composes Query + Style + prediction consistency + CQE and adds only the planned query-residual diversity loss. Run the 500-iteration A5 smoke test next; do not begin the 40k run.

## Fixed Phase 10 mechanism

- Diversity target: the learned `query_residuals`, not class anchors or contextual query states.
- Loss: mean squared off-diagonal cosine similarity within each class.
- Fixed weight: `lambda_div=0.01`.
- Prediction consistency and CQE retain their accepted Phase 8/9 definitions and weights.
- The diversity loss is added once per training batch, not once per style view.

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
python -m pytest
```

Both Git status checks must be empty and all 59 tests must pass. Verify the fixed full objective and Phase 9 evidence:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml

config = yaml.safe_load(Path(
    "configs/full/gta_dinov3l_full.yaml"
).read_text(encoding="utf-8"))

assert config["experiment"]["phase"] == 10
assert config["train"]["max_iterations"] == 40000
assert config["train"]["seed"] == 0
assert config["style"]["lambda_cf"] == 1.0
assert config["prediction_consistency"]["enabled"] is True
assert config["prediction_consistency"]["lambda_pred"] == 1.0
assert config["causal_query_effect"]["enabled"] is True
assert config["causal_query_effect"]["lambda_cqe"] == 1.0
assert config["query_diversity"] == {
    "enabled": True,
    "lambda_div": 0.01,
    "target": "query_residuals",
    "loss": "off_diagonal_cosine_squared",
}

output = Path("/root/autodl-tmp/outputs/CausalQ_DG")
a4 = output / "A4_CQE_SEED0_40000_f05cdc3"
a4_summary = json.loads((a4 / "summary.json").read_text(encoding="utf-8"))
assert a4_summary["ok"] is True
assert a4_summary["validation_results"][-1]["miou"] == 0.5917505963345144
assert (a4 / "checkpoints/iter_040000.pth").is_file()

variance = json.loads((
    output / "analysis/A3_A4_effect_variance_seed20260918.json"
).read_text(encoding="utf-8"))
assert variance["ok"] is True
assert variance["passes_lower_variance_target"] is True
assert variance["models"]["A3_PRED_CONS"]["sample_count"] == 500
assert variance["models"]["A4_CQE"]["sample_count"] == 500

print("PHASE10_SMOKE_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run the 500-iteration A5 smoke test

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A5_FULL_SMOKE_500_${RUN_SHA}"
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50

RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
bash scripts/train_full.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "run_id=$RUN_ID"
echo "train_exit_code=$TRAIN_EXIT"
echo "log_file=$LOG_FILE"
```

## Smoke evidence

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A5_FULL_SMOKE_500_${RUN_SHA}"
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
    "loss_prediction_fourier", "loss_cqe", "loss_cqe_photometric",
    "loss_cqe_fourier", "loss_diversity", "loss_diversity_weighted",
    "gradient_norm", "alpha",
)
iterations = [row["iteration"] for row in records]
errors = [abs(row["loss"] - (
    row["loss_original"]
    + 0.5 * row["loss_photometric"]
    + 0.5 * row["loss_fourier"]
    + row["loss_prediction"]
    + row["loss_cqe"]
    + row["loss_diversity_weighted"]
)) for row in records]

print("===== metadata =====")
print(metadata)
print("===== summary =====")
for key in ("ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
            "finite_losses", "last_gradient_norm", "final_alpha",
            "peak_allocated_gib", "peak_reserved_gib"):
    print(f"{key}: {summary[key]}")
print("===== trace =====")
print("record_count:", len(records))
print("iterations_contiguous:", iterations == list(range(1, 501)))
for key in keys:
    print(f"all_{key}_finite:", all(math.isfinite(row[key]) for row in records))
print("first_diversity_loss:", records[0]["loss_diversity"])
print("last_diversity_loss:", records[-1]["loss_diversity"])
print("minimum_diversity_loss:", min(row["loss_diversity"] for row in records))
print("maximum_diversity_loss:", max(row["loss_diversity"] for row in records))
print("nonzero_diversity_count:", sum(row["loss_diversity"] > 0 for row in records))
print("maximum_objective_reconstruction_error:", max(errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("===== validation =====")
print("validation_count:", len(summary["validation_results"]))
print("final_validation:", summary["validation_results"][-1])
print("===== isolation =====")
print("prediction_consistency:", metadata.get("prediction_consistency"))
print("causal_query_effect:", metadata.get("causal_query_effect"))
print("query_diversity:", metadata.get("query_diversity"))
PY

echo "===== checkpoint ====="
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2
test -f "$RUN_DIR/checkpoints/iter_000500.pth" && echo "final_checkpoint_ok=true"

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

## Acceptance criteria

- Preflight prints `PHASE10_SMOKE_GATES_OK`, all 59 tests pass, and the backbone hash matches.
- `train_exit_code=0`, `summary.ok=true`, and all 500 iterations are contiguous.
- Supervised, prediction, CQE, diversity, gradient, and alpha values are finite.
- Diversity loss is non-negative, nonzero, and does not explode.
- Total loss reconstructs from the logged supervised, prediction, CQE, and weighted-diversity components with negligible error.
- Metadata contains all three auxiliary mechanisms with `lambda_div=0.01`.
- One 50-image validation and `iter_000500.pth` exist.
- Peak memory remains within RTX 4090D capacity.
- Exact Git SHA, clean status, and disk space are reported.

Return gates/tests, training exit code, compact evidence, checkpoint, provenance, and disk output. Stop after the smoke test; do not begin the 40k Phase 10 run.
