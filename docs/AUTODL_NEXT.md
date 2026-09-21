# Next AutoDL action

Status: combined-view A2 seed 1 completed at `0.651021` Cityscapes mIoU. Run the paired photometric-only S1 seed 1 next. First verify repository provenance from the project directory and reclaim A2 seed-1 intermediate checkpoints; retain `iter_040000.pth`.

## Verify the completed run and reclaim space

```bash
cd /root/autodl-tmp/CausalQ_DG
git rev-parse HEAD
git status --short

A2S1_RUN=/root/autodl-tmp/outputs/CausalQ_DG/A2_QUERY_STYLE_SEED1_40000_a4aefd3
A2S1_RESOLVED="$(realpath "$A2S1_RUN")"

case "$A2S1_RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/A2_QUERY_STYLE_SEED1_40000_a4aefd3) ;;
  *) echo "unsafe path: $A2S1_RESOLVED"; exit 1 ;;
esac

test -f "$A2S1_RUN/summary.json"
test -f "$A2S1_RUN/checkpoints/iter_040000.pth"
test "$(find "$A2S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 80
test "$(find "$A2S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)" -eq 79
du -sh "$A2S1_RUN/checkpoints"
```

The Git SHA must be `a4aefd39527bed4044a4569cb90b161594a971f4` and status must be empty. Only after all checks succeed:

```bash
find "$A2S1_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$A2S1_RUN/checkpoints/iter_040000.pth"
test "$(find "$A2S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$A2S1_RUN/checkpoints"
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

Git status must be empty and all 64 tests must pass. Verify the paired seed and photometric-only weights:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, resolve_seed, style_view_weights

config = load_config(Path("configs/style_ablation/gta_dinov3l_photo.yaml"))
assert resolve_seed(config, None) == 0
assert resolve_seed(config, 1) == 1
assert style_view_weights(config["style"]) == {
    "original": 1.0,
    "photometric": 1.0,
}
assert "prediction_consistency" not in config
assert "causal_query_effect" not in config
assert "query_diversity" not in config
print("S1_SEED1_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run photometric-only S1 seed 1

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500
export SEED=1

test "$MAX_ITERATIONS" -eq 40000
test "$VALIDATION_MAX_SAMPLES" -eq 500
test "$SEED" -eq 1
echo "max_iterations=$MAX_ITERATIONS"
echo "validation_max_samples=$VALIDATION_MAX_SAMPLES"
echo "seed=$SEED"

RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="S1_STYLE_PHOTO_SEED1_40000_${RUN_SHA}"
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
print("best_validation:", max(validations, key=lambda row: row["miou"]))

combined_seed1 = 0.6510211122004892
photo_seed1 = validations[-1]["miou"]
print("===== paired seed-1 comparison =====")
print("combined_seed1_miou:", combined_seed1)
print("photometric_seed1_miou:", photo_seed1)
print("combined_minus_photometric:", combined_seed1 - photo_seed1)
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

cd /root/autodl-tmp/CausalQ_DG
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- A2 seed-1 cleanup retains only `iter_040000.pth` and recovers roughly 15 GiB.
- S1 metadata records seed 1, 40,000 iterations, and original/photometric views with weights `1/1`.
- Exit code zero, `ok=true`, exactly 40,000 contiguous finite records, reconstruction error `0`, and 80 full validations/checkpoints.
- No prediction consistency, CQE, or diversity is present.
- Exact Git SHA, clean status, and disk report are present.

Return cleanup output and complete S1 seed-1 evidence. Stop after this run; do not delete its checkpoints until the result is recorded.
