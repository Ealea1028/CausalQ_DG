# Next AutoDL action

Status: the Phase 10 A5 full-objective smoke test passed at exact implementation SHA `9cbaf8b85202b2ecaf274df6e2cd93427096d7fb`. Run a fresh 40,000-iteration seed-0 A5 experiment next. Do not resume the smoke checkpoint and do not change any loss weight.

## Fixed comparison

- A5: Query + Style + prediction consistency + CQE + query diversity.
- Fixed weights: `lambda_cf=1`, `lambda_pred=1`, `lambda_cqe=1`, `lambda_div=0.01`.
- Primary references: A3 prediction consistency `0.621224296375965`; A4 CQE `0.5917505963345144`; A2 Query + Style `0.639610125136919`.
- The run must be retained as the fixed A5 result regardless of outcome.
- Per the project plan, diversity is retained in the final method only if the full run is stable and gains at least 0.2 mIoU percentage points over A4; because A5 also composes the existing prediction term, interpret attribution conservatively.

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

Both Git status checks must be empty and all 59 tests must pass. Verify the smoke run and fixed mechanism:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml

output = Path("/root/autodl-tmp/outputs/CausalQ_DG")
smoke = output / "A5_FULL_SMOKE_500_9cbaf8b"
metadata = json.loads((smoke / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((smoke / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (smoke / "train.jsonl").read_text(
    encoding="utf-8"
).splitlines() if line.strip()]

assert metadata["git_sha"] == "9cbaf8b85202b2ecaf274df6e2cd93427096d7fb"
assert summary["ok"] is True
assert len(records) == 500
assert [row["iteration"] for row in records] == list(range(1, 501))
assert sum(row["loss_diversity"] > 0 for row in records) == 500
assert (smoke / "checkpoints/iter_000500.pth").is_file()

config = yaml.safe_load(Path(
    "configs/full/gta_dinov3l_full.yaml"
).read_text(encoding="utf-8"))
assert config["experiment"]["phase"] == 10
assert config["train"]["max_iterations"] == 40000
assert config["train"]["seed"] == 0
assert config["style"]["lambda_cf"] == 1.0
assert config["prediction_consistency"]["lambda_pred"] == 1.0
assert config["causal_query_effect"]["lambda_cqe"] == 1.0
assert config["query_diversity"] == {
    "enabled": True,
    "lambda_div": 0.01,
    "target": "query_residuals",
    "loss": "off_diagonal_cosine_squared",
}
print("PHASE10_FULL_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors

SMOKE_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/A5_FULL_SMOKE_500_9cbaf8b/checkpoints/iter_000500.pth
CKPT_BYTES="$(stat -c %s "$SMOKE_CKPT")"
REQUIRED_BYTES="$((CKPT_BYTES * 80 + 5 * 1024 * 1024 * 1024))"
AVAILABLE_BYTES="$(df --output=avail -B1 /root/autodl-tmp | tail -n 1)"
echo "estimated_required_bytes=$REQUIRED_BYTES"
echo "available_bytes=$AVAILABLE_BYTES"
test "$AVAILABLE_BYTES" -ge "$REQUIRED_BYTES" && echo "storage_ok=true"
df -h /root/autodl-tmp
```

The backbone hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`, and `storage_ok=true` is required.

## Start the full run

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A5_FULL_SEED0_40000_${RUN_SHA}"
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500

RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

tmux new-session -d -s causalq_a5 \
  "cd /root/autodl-tmp/CausalQ_DG && source scripts/activate_autodl.sh && export OMP_NUM_THREADS=1 RUN_ID='${RUN_ID}' MAX_ITERATIONS=40000 VALIDATION_MAX_SAMPLES=500 && set -o pipefail && bash scripts/train_full.sh 2>&1 | tee '${LOG_FILE}'; code=\${PIPESTATUS[0]}; echo train_exit_code=\$code | tee -a '${LOG_FILE}'; exit \$code"

tmux ls
tmux attach -t causalq_a5
```

Detach with `Ctrl+B`, then `D`. Monitor with:

```bash
tmux capture-pane -pt causalq_a5 -S -30
tail -n 30 "/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
nvidia-smi
```

## Completion evidence

Run only after `summary.json` exists:

```bash
cd /root/autodl-tmp/CausalQ_DG
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A5_FULL_SEED0_40000_${RUN_SHA}"
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
errors = [abs(row["loss"] - (
    row["loss_original"]
    + 0.5 * row["loss_photometric"]
    + 0.5 * row["loss_fourier"]
    + row["loss_prediction"]
    + row["loss_cqe"]
    + row["loss_diversity_weighted"]
)) for row in records]
validations = summary["validation_results"]
miou = validations[-1]["miou"]
a3 = 0.621224296375965
a4 = 0.5917505963345144
a2 = 0.639610125136919

print("===== metadata =====")
print(metadata)
print("===== summary =====")
for key in ("ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
            "finite_losses", "last_gradient_norm", "final_alpha",
            "peak_allocated_gib", "peak_reserved_gib"):
    print(f"{key}: {summary[key]}")
print("===== trace =====")
print("record_count:", len(records))
print("iterations_contiguous:", [row["iteration"] for row in records] == list(range(1, 40001)))
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
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
print("===== comparison =====")
print("a5_miou:", miou)
print("a5_minus_a3_prediction:", miou - a3)
print("a5_minus_a4_cqe:", miou - a4)
print("a5_minus_a2_style:", miou - a2)
print("passes_diversity_retention_threshold_vs_a4:", miou >= a4 + 0.002)
print("===== isolation =====")
print("prediction_consistency:", metadata.get("prediction_consistency"))
print("causal_query_effect:", metadata.get("causal_query_effect"))
print("query_diversity:", metadata.get("query_diversity"))
PY

echo "===== checkpoint evidence ====="
CHECKPOINT_COUNT="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'iter_*.pth' | wc -l)"
echo "checkpoint_count=$CHECKPOINT_COUNT"
du -sh "$RUN_DIR/checkpoints"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2 | tail -n 5
test "$CHECKPOINT_COUNT" -eq 80 && echo "checkpoint_count_ok=true"
test -f "$RUN_DIR/checkpoints/iter_040000.pth" && echo "final_checkpoint_ok=true"

echo "===== final log ====="
tail -n 20 "/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

echo "===== provenance ====="
git rev-parse HEAD
git status --short

echo "===== disk ====="
df -h /root/autodl-tmp
```

## Acceptance criteria

- Preflight prints `PHASE10_FULL_GATES_OK`, all 59 tests pass, the weight hash matches, and storage is sufficient.
- Training exits zero; `summary.ok=true`; 40,000 records and 80 validations/checkpoints exist.
- All supervised, prediction, CQE, diversity, gradient, and alpha values are finite.
- Diversity remains non-negative/nonzero without destabilizing training.
- Objective reconstruction error is negligible and metadata contains all three auxiliary mechanisms.
- A5 is compared with fixed A2, A3, and A4 results; no post-hoc rerun or weight change is allowed.
- The project-plan diversity retention threshold is at least `+0.002` mIoU versus A4, with attribution caveated because A5 also composes prediction consistency.
- Exact Git SHA, clean status, final checkpoint, and disk are reported.

Return preflight, training exit code, compact evidence, comparisons, checkpoint, provenance, and disk output. Stop after A5; do not start secondary ablations.
