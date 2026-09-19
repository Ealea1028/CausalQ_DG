# Next AutoDL action

Status: the fixed S1 photometric-only 40k run completed at `0.638287` Cityscapes mIoU. The disk has only 17 GiB free. First retain S1's final checkpoint and delete its 79 verified intermediate checkpoints; then run the fresh S2 Fourier-only 40k experiment. Do not delete `iter_040000.pth`.

## Reclaim S1 intermediate-checkpoint space

```bash
S1_RUN=/root/autodl-tmp/outputs/CausalQ_DG/S1_STYLE_PHOTO_SEED0_40000_d335694
S1_RESOLVED="$(realpath "$S1_RUN")"

case "$S1_RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/S1_STYLE_PHOTO_SEED0_40000_d335694) ;;
  *) echo "unsafe path: $S1_RESOLVED"; exit 1 ;;
esac

test -f "$S1_RUN/summary.json"
test -f "$S1_RUN/checkpoints/iter_040000.pth"
test "$(find "$S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 80
test "$(find "$S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)" -eq 79
du -sh "$S1_RUN/checkpoints"
```

Only after every check succeeds:

```bash
find "$S1_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$S1_RUN/checkpoints/iter_040000.pth"
test "$(find "$S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$S1_RUN/checkpoints"
df -h /root/autodl-tmp
```

Approximately 15 GiB should be recovered.

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

Both Git status outputs must be empty and all 63 tests must pass. Verify isolation and weights:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, style_view_weights

config = load_config(Path("configs/style_ablation/gta_dinov3l_fourier.yaml"))
assert config["experiment"]["phase"] == 11
assert config["style"]["views"] == ["original", "fourier"]
assert style_view_weights(config["style"]) == {
    "original": 1.0,
    "fourier": 1.0,
}
assert "prediction_consistency" not in config
assert "causal_query_effect" not in config
assert "query_diversity" not in config
print("S2_FULL_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run S2 Fourier-only 40k

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500

test "$MAX_ITERATIONS" -eq 40000
test "$VALIDATION_MAX_SAMPLES" -eq 500
echo "max_iterations=$MAX_ITERATIONS"
echo "validation_max_samples=$VALIDATION_MAX_SAMPLES"

RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="S2_STYLE_FOURIER_SEED0_40000_${RUN_SHA}"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
RUN_ID="$RUN_ID" bash scripts/train_style_fourier.sh 2>&1 | tee "$LOG_FILE"
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
    row["loss_original"] + row["loss_fourier"]
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
for key in ("loss", "loss_original", "loss_fourier", "gradient_norm", "alpha"):
    print(f"all_{key}_finite:", all(math.isfinite(row[key]) for row in records))
print("maximum_objective_reconstruction_error:", max(errors))
print("first_record:", records[0])
print("last_record:", records[-1])

print("===== validation =====")
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
print("best_validation:", max(validations, key=lambda row: row["miou"]))

a2_miou = 0.639610125136919
s1_miou = 0.6382865707103561
s2_miou = validations[-1]["miou"]
print("===== comparison =====")
print("a2_combined_miou:", a2_miou)
print("s1_photometric_miou:", s1_miou)
print("s2_fourier_miou:", s2_miou)
print("s2_minus_a2:", s2_miou - a2_miou)
print("s2_minus_s1:", s2_miou - s1_miou)
PY

CHECKPOINT_COUNT="$(
  find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l
)"
INTERMEDIATE_COUNT="$(
  find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l
)"
echo "checkpoint_count=$CHECKPOINT_COUNT"
echo "intermediate_checkpoint_count=$INTERMEDIATE_COUNT"
du -sh "$RUN_DIR/checkpoints"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' \
  | sort -k2 | tail -n 5
test "$CHECKPOINT_COUNT" -eq 80 && echo "checkpoint_count_ok=true"
test "$INTERMEDIATE_COUNT" -eq 79 && echo "intermediate_count_ok=true"
test -f "$RUN_DIR/checkpoints/iter_040000.pth" && echo "final_checkpoint_ok=true"

tail -n 20 "$LOG_FILE"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- S1 cleanup retains `iter_040000.pth`, removes exactly 79 intermediate checkpoints, and recovers roughly 15 GiB.
- S2 exit code zero and `ok=true`.
- Exactly 40,000 contiguous records; all total/original/Fourier losses, gradients, and alpha values finite.
- Objective reconstruction error negligible.
- Metadata contains only original + Fourier style supervision and no prediction consistency, CQE, or diversity.
- Exactly 80 full 500-image validations and 80 checkpoints; `iter_040000.pth` exists.
- Exact Git SHA, clean status, and disk report are present.

Return the cleanup output and complete S2 evidence. Stop after S2; do not delete its checkpoints until the result has been recorded.
