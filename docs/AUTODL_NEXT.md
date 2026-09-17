# Next AutoDL action

Status: Phase 7 five-sample style visualization and the 500-iteration `A2_QUERY_STYLE_SMOKE_500_18f64e8` run are accepted. Run the fresh 40,000-iteration seed-0 Query + Style comparison next. Do not resume the smoke checkpoint and do not add prediction consistency or CQE.

## Fixed comparison

- Reference: `A1_QUERY_SEED0_40000_543ab2e`, Cityscapes mIoU `0.6195300841614141`.
- Experiment: `A2_QUERY_STYLE`.
- Same DINOv3-L backbone, query branch, data, geometry transform, optimizer, schedule, seed, and validation protocol as A1.
- Only addition: aligned photometric/Fourier counterfactual views and `L_original + 0.5 L_photometric + 0.5 L_fourier`.
- All views are processed sequentially.
- Acceptance floor: `0.6095300841614141` (no more than one percentage point below A1).

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
```

Both status checks must be empty. Verify the accepted data, weights, smoke run, final reference checkpoints, and multi-sample style reports:

```bash
python - <<'PY'
import json
from pathlib import Path

output = Path("/root/autodl-tmp/outputs/CausalQ_DG")

data = json.loads(
    (output / "data_check/gta5_after_20217_repair.json").read_text(encoding="utf-8")
)
gta5 = data["datasets"]["gta5"]
assert data["ok"] is True
assert gta5["paired_count"] == 24966
assert gta5["unreadable_count"] == 0

smoke = json.loads(
    (output / "A2_QUERY_STYLE_SMOKE_500_18f64e8/summary.json").read_text(
        encoding="utf-8"
    )
)
assert smoke["ok"] is True
assert smoke["git_sha"] == "18f64e8612f46133cd66feda8522ee98b0b45411"
assert smoke["max_iterations"] == 500
assert smoke["finite_losses"] is True
assert smoke["style"]["views"] == ["original", "photometric", "fourier"]
assert smoke["style"]["lambda_cf"] == 1.0

multi = output / "analysis/style_examples_multi"
reports = sorted(multi.glob("sample_*/report.json"))
assert len(reports) == 5
for path in reports:
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["shared_geometry"] is True
    assert report["photometric_mean_absolute_difference"] > 0
    assert report["fourier_mean_absolute_difference"] > 0
assert (multi / "style_montage_multi.png").is_file()

for run, checkpoint in (
    ("A0_DINOV3L_BASE_SEED0_40000_REPAIRED_7eee12b", "iter_040000.pth"),
    ("A1_QUERY_SEED0_40000_543ab2e", "iter_040000.pth"),
):
    assert (output / run / "checkpoints" / checkpoint).is_file()

print("PHASE7_FULL_RUN_GATES_OK")
print("style_report_count:", len(reports))
print("smoke_miou:", smoke["validation_results"][-1]["miou"])
print("smoke_final_alpha:", smoke["final_alpha"])
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
```

The weight hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

Check storage using the measured Phase 7 checkpoint size and retain a 5 GiB reserve:

```bash
SMOKE_CHECKPOINT=/root/autodl-tmp/outputs/CausalQ_DG/A2_QUERY_STYLE_SMOKE_500_18f64e8/checkpoints/iter_000500.pth
test -f "$SMOKE_CHECKPOINT"

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

## Full run

Optionally use `tmux new -s causalq_a2_full`. Start a unique run from iteration zero:

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A2_QUERY_STYLE_SEED0_40000_${RUN_SHA}"
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500

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

## Completion evidence

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A2_QUERY_STYLE_SEED0_40000_${RUN_SHA}"
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

keys = ("loss", "loss_original", "loss_photometric", "loss_fourier", "gradient_norm", "alpha")
iterations = [record["iteration"] for record in records]
errors = [abs(record["loss"] - (
    record["loss_original"]
    + 0.5 * record["loss_photometric"]
    + 0.5 * record["loss_fourier"]
)) for record in records]
validations = summary["validation_results"]
query_miou = 0.6195300841614141
style_miou = validations[-1]["miou"]

print("metadata:", metadata)
for key in ("ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
            "finite_losses", "last_gradient_norm", "final_alpha",
            "peak_allocated_gib", "peak_reserved_gib"):
    print(f"{key}: {summary[key]}")
print("record_count:", len(records))
print("iterations_contiguous:", iterations == list(range(1, 40001)))
for key in keys:
    print(f"all_{key}_finite:", all(math.isfinite(record[key]) for record in records))
print("maximum_objective_reconstruction_error:", max(errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
print("query_miou:", query_miou)
print("style_miou:", style_miou)
print("style_minus_query:", style_miou - query_miou)
print("passes_minus_1pp_floor:", style_miou >= query_miou - 0.01)
PY

echo "===== checkpoint evidence ====="
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l
du -sh "$RUN_DIR/checkpoints"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2 | tail -n 5
test -f "$RUN_DIR/checkpoints/iter_040000.pth" && echo "final_checkpoint_ok=true"

echo "===== provenance and disk ====="
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- All preflight and storage gates pass.
- `train_exit_code=0`, `summary.ok=true`, and 40,000 iterations are contiguous.
- Total/component losses, gradients, and alpha values are finite.
- The recorded loss exactly follows the fixed Phase 7 objective up to rounding.
- Eighty full 500-image validations and 80 checkpoints complete, including `iter_040000.pth`.
- Final mIoU is at least `0.6095300841614141`.
- Peak memory stays within RTX 4090D capacity.
- Exact Git SHA and clean Git status are recorded.

Return all gate/storage output, training exit code, compact evidence, checkpoint evidence, Git SHA/status, and disk usage. Stop after Phase 7 full training; do not begin Phase 8.
