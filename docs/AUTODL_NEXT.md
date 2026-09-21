# Next AutoDL action

Status: combined-view A2 seed 2 completed at `0.629514` Cityscapes mIoU. Run the paired photometric-only S1 seed 2 next. This is the final GPU run needed to close Phase 11. First retain the A2 seed-2 final checkpoint and reclaim its 79 intermediates.

## Reclaim A2 seed-2 intermediate-checkpoint space

```bash
cd /root/autodl-tmp/CausalQ_DG
git rev-parse HEAD
git status --short

A2S2_RUN=/root/autodl-tmp/outputs/CausalQ_DG/A2_QUERY_STYLE_SEED2_40000_fe41a13
A2S2_RESOLVED="$(realpath "$A2S2_RUN")"

case "$A2S2_RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/A2_QUERY_STYLE_SEED2_40000_fe41a13) ;;
  *) echo "unsafe path: $A2S2_RESOLVED"; exit 1 ;;
esac

test -f "$A2S2_RUN/summary.json"
test -f "$A2S2_RUN/checkpoints/iter_040000.pth"
test "$(find "$A2S2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 80
test "$(find "$A2S2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)" -eq 79
du -sh "$A2S2_RUN/checkpoints"
```

The Git SHA must be `fe41a138d1a56ce0b0a41641ca79eaf6ca5d5b8f` and status must be empty. Only after all checks succeed:

```bash
find "$A2S2_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$A2S2_RUN/checkpoints/iter_040000.pth"
test "$(find "$A2S2_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$A2S2_RUN/checkpoints"
df -h /root/autodl-tmp
```

## Checkout and preflight

Replace `<EXACT_SHA_FROM_HANDOFF>` with the exact handoff SHA.

```bash
cd /root/autodl-tmp/CausalQ_DG
git fetch origin
git checkout --detach <EXACT_SHA_FROM_HANDOFF>
git rev-parse HEAD
git status --short

source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
python tools/check_environment.py
python -m pytest
```

Git status must be empty and all 64 tests must pass. Verify seed 2 and photometric-only weights:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, resolve_seed, style_view_weights

config = load_config(Path("configs/style_ablation/gta_dinov3l_photo.yaml"))
assert resolve_seed(config, 2) == 2
assert style_view_weights(config["style"]) == {
    "original": 1.0,
    "photometric": 1.0,
}
assert "prediction_consistency" not in config
assert "causal_query_effect" not in config
assert "query_diversity" not in config
print("S1_SEED2_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run photometric-only S1 seed 2

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500
export SEED=2

test "$MAX_ITERATIONS" -eq 40000
test "$VALIDATION_MAX_SAMPLES" -eq 500
test "$SEED" -eq 2
echo "max_iterations=$MAX_ITERATIONS"
echo "validation_max_samples=$VALIDATION_MAX_SAMPLES"
echo "seed=$SEED"

RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="S1_STYLE_PHOTO_SEED2_40000_${RUN_SHA}"
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

## Evidence extraction and final paired statistics

```bash
python - "$RUN_DIR" <<'PY'
import json
import math
from pathlib import Path
from statistics import mean, stdev
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
print(metadata)
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
print("===== validation =====")
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
print("best_validation:", max(validations, key=lambda row: row["miou"]))

combined = [0.639610125136919, 0.6510211122004892, 0.629513840900438]
photo = [0.6382865707103561, 0.6514857398288119, validations[-1]["miou"]]
paired = [c - p for c, p in zip(combined, photo)]
print("===== final three-seed comparison =====")
print("combined_values:", combined)
print("photometric_values:", photo)
print("paired_combined_minus_photo:", paired)
print("combined_mean:", mean(combined))
print("combined_sample_std:", stdev(combined))
print("photometric_mean:", mean(photo))
print("photometric_sample_std:", stdev(photo))
print("paired_difference_mean:", mean(paired))
print("paired_difference_sample_std:", stdev(paired))
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
test "$CHECKPOINT_COUNT" -eq 80 && echo "checkpoint_count_ok=true"
test "$INTERMEDIATE_COUNT" -eq 79 && echo "intermediate_count_ok=true"
test -f "$RUN_DIR/checkpoints/iter_040000.pth" && echo "final_checkpoint_ok=true"

cd /root/autodl-tmp/CausalQ_DG
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- A2 seed-2 cleanup retains only `iter_040000.pth` and recovers roughly 15 GiB.
- S1 metadata records seed 2, 40,000 iterations, and original/photometric views with weights `1/1`.
- Exit code zero, `ok=true`, 40,000 contiguous finite records, reconstruction error `0`, and 80 full validations/checkpoints.
- No prediction consistency, CQE, or diversity is present.
- Three-seed paired mean and sample standard deviation are printed.
- Exact Git SHA, clean status, and disk report are present.

Return cleanup output and complete S1 seed-2 evidence. Stop after this run; do not delete its checkpoints until Phase 11 is recorded.
