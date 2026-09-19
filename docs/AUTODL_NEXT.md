# Next AutoDL action

Status: the A0--A5 core sequence is complete. A5 is stable but misses the diversity-retention threshold, so diversity is excluded from the final method. Phase 11 begins the planned style ablation. Run 500-iteration photometric-only and Fourier-only smoke tests next; do not begin either 40k run.

## Fixed Phase 11 protocol

- Base: accepted Query-only architecture and A2 training protocol.
- S1 views: original + photometric.
- S2 views: original + Fourier.
- No prediction consistency, CQE, or diversity.
- Total counterfactual weight remains `lambda_cf=1.0`; the sole intervention receives weight 1.0.
- The existing combined A2 run remains the original + 0.5 photometric + 0.5 Fourier reference.

## Reclaim A5 intermediate-checkpoint space

```bash
A5_RUN=/root/autodl-tmp/outputs/CausalQ_DG/A5_FULL_SEED0_40000_973922b
A5_RESOLVED="$(realpath "$A5_RUN")"

case "$A5_RESOLVED" in
  /root/autodl-tmp/outputs/CausalQ_DG/A5_FULL_SEED0_40000_973922b) ;;
  *) echo "unsafe path: $A5_RESOLVED"; exit 1 ;;
esac

test -f "$A5_RUN/summary.json"
test -f "$A5_RUN/checkpoints/iter_040000.pth"
test "$(find "$A5_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 80
test "$(find "$A5_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' ! -name 'iter_040000.pth' | wc -l)" -eq 79
du -sh "$A5_RUN/checkpoints"
```

After all checks succeed:

```bash
find "$A5_RUN/checkpoints" -maxdepth 1 -type f \
  -name 'iter_*.pth' ! -name 'iter_040000.pth' -delete

test -f "$A5_RUN/checkpoints/iter_040000.pth"
test "$(find "$A5_RUN/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)" -eq 1
du -sh "$A5_RUN/checkpoints"
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
python -m pytest
```

Both Git status checks must be empty and all 63 tests must pass. Verify isolation and weight parity:

```bash
python - <<'PY'
from pathlib import Path
from tools.train import load_config, style_view_weights

root = Path("configs/style_ablation")
expected = {
    "gta_dinov3l_photo.yaml": ["original", "photometric"],
    "gta_dinov3l_fourier.yaml": ["original", "fourier"],
}
for filename, views in expected.items():
    config = load_config(root / filename)
    assert config["experiment"]["phase"] == 11
    assert config["style"]["views"] == views
    assert style_view_weights(config["style"]) == {
        "original": 1.0,
        views[1]: 1.0,
    }
    assert "prediction_consistency" not in config
    assert "causal_query_effect" not in config
    assert "query_diversity" not in config

combined = load_config(Path("configs/style/gta_dinov3l_style.yaml"))
assert style_view_weights(combined["style"]) == {
    "original": 1.0,
    "photometric": 0.5,
    "fourier": 0.5,
}
print("PHASE11_SMOKE_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The DINOv3 hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

## Run both smoke tests

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50

PHOTO_ID="S1_STYLE_PHOTO_SMOKE_500_${RUN_SHA}"
PHOTO_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${PHOTO_ID}"
PHOTO_LOG="/root/autodl-tmp/outputs/CausalQ_DG/${PHOTO_ID}.log"
test ! -e "$PHOTO_DIR"
test ! -e "$PHOTO_LOG"

set -o pipefail
RUN_ID="$PHOTO_ID" bash scripts/train_style_photo.sh 2>&1 | tee "$PHOTO_LOG"
PHOTO_EXIT=${PIPESTATUS[0]}
echo "photo_exit_code=$PHOTO_EXIT"
```

Only after `photo_exit_code=0`, run Fourier:

```bash
FOURIER_ID="S2_STYLE_FOURIER_SMOKE_500_${RUN_SHA}"
FOURIER_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${FOURIER_ID}"
FOURIER_LOG="/root/autodl-tmp/outputs/CausalQ_DG/${FOURIER_ID}.log"
test ! -e "$FOURIER_DIR"
test ! -e "$FOURIER_LOG"

set -o pipefail
RUN_ID="$FOURIER_ID" bash scripts/train_style_fourier.sh 2>&1 | tee "$FOURIER_LOG"
FOURIER_EXIT=${PIPESTATUS[0]}
echo "fourier_exit_code=$FOURIER_EXIT"
```

## Smoke evidence

```bash
python - "$PHOTO_DIR" photometric "$FOURIER_DIR" fourier <<'PY'
import json
import math
from pathlib import Path
import sys

for index in (1, 3):
    run_dir = Path(sys.argv[index])
    intervention = sys.argv[index + 1]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    expected_keys = ("loss", "loss_original", f"loss_{intervention}",
                     "gradient_norm", "alpha")
    errors = [abs(row["loss"] - (
        row["loss_original"] + row[f"loss_{intervention}"]
    )) for row in records]

    print(f"===== {intervention} metadata =====")
    print(metadata)
    print(f"===== {intervention} summary =====")
    for key in ("ok", "elapsed_seconds", "first_20_loss_mean",
                "last_20_loss_mean", "finite_losses", "last_gradient_norm",
                "final_alpha", "peak_allocated_gib", "peak_reserved_gib"):
        print(f"{key}: {summary[key]}")
    print("record_count:", len(records))
    print("iterations_contiguous:", [row["iteration"] for row in records] == list(range(1, 501)))
    for key in expected_keys:
        print(f"all_{key}_finite:", all(math.isfinite(row[key]) for row in records))
    print("maximum_objective_reconstruction_error:", max(errors))
    print("first_record:", records[0])
    print("last_record:", records[-1])
    print("validation_count:", len(summary["validation_results"]))
    print("final_validation:", summary["validation_results"][-1])
    print("checkpoint_exists:", (run_dir / "checkpoints/iter_000500.pth").is_file())
    print("has_prediction_consistency:", "prediction_consistency" in metadata)
    print("has_cqe:", "causal_query_effect" in metadata)
    print("has_diversity:", "query_diversity" in metadata)
PY

git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- A5 cleanup retains only `iter_040000.pth` and recovers roughly 15 GiB.
- Preflight prints `PHASE11_SMOKE_GATES_OK`, all 63 tests pass, and the weight hash matches.
- Both exits are zero; both summaries are `ok=true`; each has 500 contiguous records.
- Each run logs only original plus its configured intervention, with finite losses, gradients, and alpha.
- Each total loss reconstructs as `loss_original + loss_intervention` with negligible error.
- Metadata contains no prediction consistency, CQE, or diversity.
- Each run has one 50-image validation and `iter_000500.pth`.
- Exact Git SHA, clean status, and disk are reported.

Return cleanup, gates/tests, both smoke summaries/traces, provenance, and disk output. Stop after both smoke tests; do not begin the 40k style ablations.
