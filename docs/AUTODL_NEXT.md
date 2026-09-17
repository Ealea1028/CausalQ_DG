# Next AutoDL action

Status: Phase 6 Query-only is accepted at `0.6195300842` Cityscapes mIoU, an absolute gain of `0.0178023483` over the accepted baseline. Phase 7 implements style intervention only. Do not add prediction consistency or CQE.

## 1. Reclaim accepted-run checkpoint space

Only the final checkpoints of the accepted Phase 5 and Phase 6 runs are needed for reproducibility. First verify the two exact directories and preview the files that will be removed:

```bash
BASE_DIR=/root/autodl-tmp/outputs/CausalQ_DG/A0_DINOV3L_BASE_SEED0_40000_REPAIRED_7eee12b
QUERY_DIR=/root/autodl-tmp/outputs/CausalQ_DG/A1_QUERY_SEED0_40000_543ab2e

for RUN_DIR in "$BASE_DIR" "$QUERY_DIR"; do
  RESOLVED="$(realpath "$RUN_DIR")"
  case "$RESOLVED" in
    /root/autodl-tmp/outputs/CausalQ_DG/*) ;;
    *) echo "unsafe path: $RESOLVED"; exit 1 ;;
  esac
  test -f "$RUN_DIR/summary.json"
  test -f "$RUN_DIR/checkpoints/iter_040000.pth"
  echo "run=$RESOLVED"
  echo "checkpoint_count_before=$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)"
  echo "intermediate_count=$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)"
  du -sh "$RUN_DIR/checkpoints"
done
```

Both runs must report 80 checkpoints and 79 intermediates. Then delete only those 158 accepted intermediate checkpoints:

```bash
find "$BASE_DIR/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete
find "$QUERY_DIR/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

for RUN_DIR in "$BASE_DIR" "$QUERY_DIR"; do
  test -f "$RUN_DIR/checkpoints/iter_040000.pth"
  test "$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
  du -sh "$RUN_DIR/checkpoints"
done
df -h /root/autodl-tmp
```

This retains both final checkpoints, summaries, metadata, logs, and traces. The removed scheduled intermediate checkpoints are not recoverable unless the runs are repeated.

## 2. Check out the exact Phase 7 implementation

Use the exact Git SHA supplied in the hand-off:

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
```

Both Git status checks must be empty. The reported GPU must be the RTX 4090 D and CUDA must be available.

## 3. Generate and inspect aligned style examples

```bash
STYLE_EXAMPLE_DIR=/root/autodl-tmp/outputs/CausalQ_DG/analysis/style_examples
mkdir -p "$STYLE_EXAMPLE_DIR"

set -o pipefail
python tools/check_style.py \
  --sample-index 0 \
  --seed 0 \
  --output-dir "$STYLE_EXAMPLE_DIR" \
  | tee "$STYLE_EXAMPLE_DIR/report.json"
STYLE_CHECK_EXIT=${PIPESTATUS[0]}

echo "style_check_exit_code=$STYLE_CHECK_EXIT"
find "$STYLE_EXAMPLE_DIR" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2
```

Open or download this image with the AutoDL file browser and inspect it before training:

```text
/root/autodl-tmp/outputs/CausalQ_DG/analysis/style_examples/style_montage.png
```

The four panels are original, photometric, Fourier, and GT. Require `ok=true`, `shared_geometry=true`, identical image/label spatial sizes, positive differences for both interventions, no displaced boundaries, no crop/flip mismatch between panels, and recognizable unchanged semantic layout. Stop if the visualization violates these conditions.

## 4. Run the 500-iteration Phase 7 smoke test

Only after the visualization is accepted:

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A2_QUERY_STYLE_SMOKE_500_${RUN_SHA}"
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50

RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
bash scripts/train_style.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "run_id=$RUN_ID"
echo "train_exit_code=$TRAIN_EXIT"
echo "log_file=$LOG_FILE"
```

Collect compact evidence:

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A2_QUERY_STYLE_SMOKE_500_${RUN_SHA}"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"

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

iterations = [record["iteration"] for record in records]
keys = ("loss", "loss_original", "loss_photometric", "loss_fourier", "gradient_norm", "alpha")
objective_errors = [
    abs(record["loss"] - (
        record["loss_original"]
        + 0.5 * record["loss_photometric"]
        + 0.5 * record["loss_fourier"]
    ))
    for record in records
]

print("metadata:", metadata)
for key in (
    "ok",
    "elapsed_seconds",
    "first_20_loss_mean",
    "last_20_loss_mean",
    "finite_losses",
    "last_gradient_norm",
    "final_alpha",
    "peak_allocated_gib",
    "peak_reserved_gib",
):
    print(f"{key}: {summary[key]}")
print("record_count:", len(records))
print("iterations_contiguous:", iterations == list(range(1, 501)))
for key in keys:
    print(f"all_{key}_finite:", all(math.isfinite(record[key]) for record in records))
print("maximum_objective_reconstruction_error:", max(objective_errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("validation_count:", len(summary["validation_results"]))
print("final_validation:", summary["validation_results"][-1])
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

- Style visualization passes the geometry and semantic-layout inspection.
- `train_exit_code=0` and `summary.ok=true`.
- Exactly 500 contiguous iterations complete.
- Total, original, photometric, Fourier, gradient, and alpha values are all finite.
- Reconstructed objective error is negligible (floating-point rounding only).
- Exactly one 50-image Cityscapes validation and `iter_000500.pth` exist.
- Metadata contains only Query + Style with `lambda_cf=1.0`; no prediction-consistency or CQE field exists.
- Peak memory remains within RTX 4090D capacity.
- Exact Git SHA and clean final Git status are reported.

Return the cleanup output, style report, visual inspection result, smoke exit code, compact evidence, checkpoint evidence, exact Git SHA/status, and disk usage. Stop after this smoke test; do not start the full Phase 7 run yet.
