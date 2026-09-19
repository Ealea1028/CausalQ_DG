# Next AutoDL action

Status: both Phase 11 smoke tests passed at commit `b0d6e14`. Run only the fresh 40,000-iteration photometric-only S1 experiment next. Do not start S2 Fourier-only in the same step.

## Fixed protocol

- Config: `configs/style_ablation/gta_dinov3l_photo.yaml`.
- Views: original + photometric.
- The sole counterfactual view has weight `1.0`, preserving total `lambda_cf=1.0` parity with A2.
- Prediction consistency, CQE, and diversity are disabled.
- Seed 0; 40,000 GTA5 iterations; all 500 Cityscapes validation images every 500 iterations.
- Start from random head/query initialization. Do not resume any checkpoint.

## Checkout and preflight

Replace `<EXACT_SHA_FROM_HANDOFF>` with the exact handoff SHA.

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

Both Git status outputs must be empty and all 63 tests must pass. Verify isolation and the checkpoint:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, style_view_weights

config = load_config(Path("configs/style_ablation/gta_dinov3l_photo.yaml"))
assert config["experiment"]["phase"] == 11
assert config["style"]["views"] == ["original", "photometric"]
assert style_view_weights(config["style"]) == {
    "original": 1.0,
    "photometric": 1.0,
}
assert "prediction_consistency" not in config
assert "causal_query_effect" not in config
assert "query_diversity" not in config
print("S1_FULL_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`. At least 17 GiB free is required; the reported 32 GiB is sufficient for one full run.

## Run S1 photometric-only 40k

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
unset MAX_ITERATIONS
unset VALIDATION_MAX_SAMPLES

RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="S1_STYLE_PHOTO_SEED0_40000_${RUN_SHA}"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
RUN_ID="$RUN_ID" bash scripts/train_style_photo.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
```

Do not interrupt the process. Continue only when `train_exit_code=0` and `summary.json` exists.

## Evidence extraction

```bash
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
validations = summary["validation_results"]
errors = [abs(row["loss"] - (
    row["loss_original"] + row["loss_photometric"]
)) for row in records]

print("===== metadata =====")
for key in ("experiment_id", "phase", "git_sha", "gpu", "pytorch", "cuda",
            "backbone", "pretrained_checkpoint_sha256", "seed", "max_iterations"):
    print(f"{key}:", metadata[key])
print("style:", metadata["style"])
print("has_prediction_consistency:", "prediction_consistency" in metadata)
print("has_cqe:", "causal_query_effect" in metadata)
print("has_diversity:", "query_diversity" in metadata)

print("===== summary =====")
for key in ("ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
            "finite_losses", "last_gradient_norm", "final_alpha",
            "peak_allocated_gib", "peak_reserved_gib"):
    print(f"{key}:", summary[key])

print("===== trace =====")
print("record_count:", len(records))
print("iterations_contiguous:", [row["iteration"] for row in records] == list(range(1, 40001)))
for key in ("loss", "loss_original", "loss_photometric", "gradient_norm", "alpha"):
    print(f"all_{key}_finite:", all(math.isfinite(row[key]) for row in records))
print("maximum_objective_reconstruction_error:", max(errors))
print("first_record:", records[0])
print("last_record:", records[-1])

print("===== validation =====")
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
PY

CHECKPOINT_COUNT="$(
  find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l
)"
echo "checkpoint_count=$CHECKPOINT_COUNT"
du -sh "$RUN_DIR/checkpoints"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' \
  | sort -k2 | tail -n 5
test "$CHECKPOINT_COUNT" -eq 80 && echo "checkpoint_count_ok=true"
test -f "$RUN_DIR/checkpoints/iter_040000.pth" && echo "final_checkpoint_ok=true"

tail -n 20 "$LOG_FILE"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- Exit code zero and `ok=true`.
- Exactly 40,000 contiguous records; all total/original/photometric losses, gradients, and alpha values finite.
- Objective reconstruction error negligible.
- Metadata contains only original + photometric style supervision and no prediction consistency, CQE, or diversity.
- Exactly 80 full 500-image validations and 80 checkpoints; `iter_040000.pth` exists.
- Exact Git SHA, clean status, and disk report are present.

Return the complete evidence above. Stop after S1; do not delete its intermediate checkpoints and do not begin S2 until the result has been recorded and the next cleanup step is supplied.
