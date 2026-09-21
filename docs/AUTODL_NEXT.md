# Next AutoDL action

Status: the Phase 11 seed-1 pair is complete: combined `0.651021`, photometric-only `0.651486`. Their ordering reverses seed 0, so the seed-2 pair is required. Run combined-view A2 seed 2 next after retaining the S1 seed-1 final checkpoint and reclaiming its 79 intermediates.

## Reclaim S1 seed-1 intermediate-checkpoint space

```bash
cd /root/autodl-tmp/CausalQ_DG
git rev-parse HEAD
git status --short

S1S1_RUN=/root/autodl-tmp/outputs/CausalQ_DG/S1_STYLE_PHOTO_SEED1_40000_54d00c1
S1S1_RESOLVED="$(realpath "$S1S1_RUN")"

case "$S1S1_RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/S1_STYLE_PHOTO_SEED1_40000_54d00c1) ;;
  *) echo "unsafe path: $S1S1_RESOLVED"; exit 1 ;;
esac

test -f "$S1S1_RUN/summary.json"
test -f "$S1S1_RUN/checkpoints/iter_040000.pth"
test "$(find "$S1S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 80
test "$(find "$S1S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)" -eq 79
du -sh "$S1S1_RUN/checkpoints"
```

The Git SHA must be `54d00c1a6964807fee4945c48b56e042aba3c906` and status must be empty. Only after all checks succeed:

```bash
find "$S1S1_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$S1S1_RUN/checkpoints/iter_040000.pth"
test "$(find "$S1S1_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$S1S1_RUN/checkpoints"
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

Git status must be empty and all 64 tests must pass. Verify seed 2 and combined-view weights:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, resolve_seed, style_view_weights

config = load_config(Path("configs/style/gta_dinov3l_style.yaml"))
assert resolve_seed(config, 2) == 2
assert style_view_weights(config["style"]) == {
    "original": 1.0,
    "photometric": 0.5,
    "fourier": 0.5,
}
assert "prediction_consistency" not in config
assert "causal_query_effect" not in config
assert "query_diversity" not in config
print("A2_SEED2_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run combined-view A2 seed 2

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
RUN_ID="A2_QUERY_STYLE_SEED2_40000_${RUN_SHA}"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
RUN_ID="$RUN_ID" bash scripts/train_style.sh 2>&1 | tee "$LOG_FILE"
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
    row["loss_original"]
    + 0.5 * row["loss_photometric"]
    + 0.5 * row["loss_fourier"]
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
for key in ("loss", "loss_original", "loss_photometric", "loss_fourier",
            "gradient_norm", "alpha"):
    print(f"all_{key}_finite:", all(math.isfinite(row[key]) for row in records))
print("maximum_objective_reconstruction_error:", max(errors))
print("===== validation =====")
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
print("best_validation:", max(validations, key=lambda row: row["miou"]))
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

- S1 seed-1 cleanup retains only `iter_040000.pth` and recovers roughly 15 GiB.
- A2 metadata records seed 2, 40,000 iterations, and combined views with weights `1/0.5/0.5`.
- Exit code zero, `ok=true`, 40,000 contiguous finite records, reconstruction error `0`, and 80 full validations/checkpoints.
- No prediction consistency, CQE, or diversity is present.
- Exact Git SHA, clean status, and disk report are present.

Return cleanup output and complete A2 seed-2 evidence. Stop after this run; do not delete its checkpoints until the result is recorded.
