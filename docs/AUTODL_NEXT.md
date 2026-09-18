# Next AutoDL action

Status: the Phase 9 CQE smoke test passed at exact implementation SHA `fc063040ed9d0df19cae3cd059f68a6903be3c2c`. Run the fresh 40,000-iteration seed-0 CQE experiment next. Do not resume the smoke checkpoint and do not add prediction consistency or diversity.

## Controlled comparison

- A4: Query + Style + CQE, `lambda_cqe=1.0`.
- Primary control A3 prediction consistency: `0.621224296375965` Cityscapes mIoU.
- Secondary reference A2 Query + Style: `0.639610125136919` Cityscapes mIoU.
- First target: A4 exceeds A3.
- Safety floor: A4 is no more than 1 percentage point below A3, namely `0.611224296375965`.
- The cross-style query-effect variance comparison is performed after this full run; it is not a reason to alter the training objective.

## Checkout and preflight

Replace `<EXACT_SHA_FROM_HANDOFF>` with the commit supplied in the handoff. The checkout must remain clean.

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

Verify the accepted smoke run, reference runs, mechanism isolation, weights, and storage:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml

output = Path("/root/autodl-tmp/outputs/CausalQ_DG")

smoke = output / "A4_CQE_SMOKE_500_fc06304"
smoke_meta = json.loads((smoke / "metadata.json").read_text(encoding="utf-8"))
smoke_summary = json.loads((smoke / "summary.json").read_text(encoding="utf-8"))
smoke_records = [json.loads(line) for line in (smoke / "train.jsonl").read_text(
    encoding="utf-8"
).splitlines() if line.strip()]
assert smoke_meta["git_sha"] == "fc063040ed9d0df19cae3cd059f68a6903be3c2c"
assert smoke_summary["ok"] is True
assert len(smoke_records) == 500
assert [row["iteration"] for row in smoke_records] == list(range(1, 501))
assert sum(row["loss_cqe"] > 0 for row in smoke_records) == 499
assert (smoke / "checkpoints/iter_000500.pth").is_file()

references = {
    "A2_QUERY_STYLE_SEED0_40000_293b330": 0.639610125136919,
    "A3_PRED_CONS_SEED0_40000_4a95ebd": 0.621224296375965,
}
for run, expected_miou in references.items():
    run_dir = output / run
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["validation_results"][-1]["miou"] == expected_miou
    assert (run_dir / "checkpoints/iter_040000.pth").is_file()

config = yaml.safe_load(Path(
    "configs/causalq/gta_dinov3l_causalq.yaml"
).read_text(encoding="utf-8"))
cqe = config["causal_query_effect"]
assert config["experiment"]["phase"] == 9
assert config["train"]["max_iterations"] == 40000
assert config["train"]["seed"] == 0
assert config["style"]["lambda_cf"] == 1.0
assert cqe == {
    "enabled": True,
    "lambda_cqe": 1.0,
    "effect_space": "logits",
    "reference_stop_gradient": True,
    "normalization": "l2_per_class_map",
    "present_classes_only": True,
    "valid_pixels_only": True,
    "loss": "smooth_l1",
    "smooth_l1_beta": 1.0,
}
assert "prediction_consistency" not in config
print("PHASE9_FULL_GATES_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors

SMOKE_CKPT=/root/autodl-tmp/outputs/CausalQ_DG/A4_CQE_SMOKE_500_fc06304/checkpoints/iter_000500.pth
CKPT_BYTES="$(stat -c %s "$SMOKE_CKPT")"
REQUIRED_BYTES="$((CKPT_BYTES * 80 + 5 * 1024 * 1024 * 1024))"
AVAILABLE_BYTES="$(df --output=avail -B1 /root/autodl-tmp | tail -n 1)"
echo "estimated_required_bytes=$REQUIRED_BYTES"
echo "available_bytes=$AVAILABLE_BYTES"
test "$AVAILABLE_BYTES" -ge "$REQUIRED_BYTES" && echo "storage_ok=true"
df -h /root/autodl-tmp
```

The weight hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`, and `storage_ok=true` is required.

## Start the full run

Use `tmux` so training survives a browser or SSH disconnect. The run starts from random head/query initialization; it does not load the 500-iteration smoke checkpoint.

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A4_CQE_SEED0_40000_${RUN_SHA}"
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500

RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

tmux new-session -d -s causalq_a4 \
  "cd /root/autodl-tmp/CausalQ_DG && source scripts/activate_autodl.sh && export OMP_NUM_THREADS=1 RUN_ID='${RUN_ID}' MAX_ITERATIONS=40000 VALIDATION_MAX_SAMPLES=500 && set -o pipefail && bash scripts/train_causalq.sh 2>&1 | tee '${LOG_FILE}'; code=\${PIPESTATUS[0]}; echo train_exit_code=\$code | tee -a '${LOG_FILE}'; exit \$code"

tmux ls
tmux attach -t causalq_a4
```

Detach without stopping training by pressing `Ctrl+B`, then `D`. To monitor later:

```bash
tmux capture-pane -pt causalq_a4 -S -30
tail -n 30 "/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
nvidia-smi
```

## Completion evidence

Run this only after training finishes and `summary.json` exists:

```bash
cd /root/autodl-tmp/CausalQ_DG
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A4_CQE_SEED0_40000_${RUN_SHA}"
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
    "loss_cqe", "loss_cqe_photometric", "loss_cqe_fourier",
    "gradient_norm", "alpha",
)
iterations = [row["iteration"] for row in records]
errors = [abs(row["loss"] - (
    row["loss_original"]
    + 0.5 * row["loss_photometric"]
    + 0.5 * row["loss_fourier"]
    + row["loss_cqe"]
)) for row in records]
validations = summary["validation_results"]
final_miou = validations[-1]["miou"]
pred_miou = 0.621224296375965
style_miou = 0.639610125136919

print("===== metadata =====")
print(metadata)
print("===== summary =====")
for key in ("ok", "elapsed_seconds", "first_20_loss_mean", "last_20_loss_mean",
            "finite_losses", "last_gradient_norm", "final_alpha",
            "peak_allocated_gib", "peak_reserved_gib"):
    print(f"{key}: {summary[key]}")
print("===== trace =====")
print("record_count:", len(records))
print("iterations_contiguous:", iterations == list(range(1, 40001)))
for key in keys:
    print(f"all_{key}_finite:", all(math.isfinite(row[key]) for row in records))
print("minimum_cqe_loss:", min(row["loss_cqe"] for row in records))
print("maximum_cqe_loss:", max(row["loss_cqe"] for row in records))
print("last_20_cqe_mean:", sum(row["loss_cqe"] for row in records[-20:]) / 20)
print("nonzero_cqe_count:", sum(row["loss_cqe"] > 0 for row in records))
print("maximum_objective_reconstruction_error:", max(errors))
print("first_record:", records[0])
print("last_record:", records[-1])
print("===== validation =====")
print("validation_count:", len(validations))
print("final_validation:", validations[-1])
print("===== comparison =====")
print("prediction_consistency_miou:", pred_miou)
print("query_style_miou:", style_miou)
print("cqe_miou:", final_miou)
print("cqe_minus_prediction_consistency:", final_miou - pred_miou)
print("cqe_minus_query_style:", final_miou - style_miou)
print("passes_primary_target:", final_miou > pred_miou)
print("passes_minus_1pp_floor:", final_miou >= pred_miou - 0.01)
print("===== isolation =====")
print("causal_query_effect:", metadata.get("causal_query_effect"))
print("has_prediction_consistency:", "prediction_consistency" in metadata)
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

- The preflight prints `PHASE9_FULL_GATES_OK`, the full test suite passes, the weight hash matches, and storage is sufficient.
- Training exits with code zero; `summary.ok=true`; exactly 40,000 records and 80 validations/checkpoints exist.
- Every supervised/CQE loss, gradient norm, and alpha value is finite.
- CQE becomes nonzero without exploding; objective reconstruction error is negligible.
- Metadata contains CQE and no prediction consistency.
- Primary target: final mIoU exceeds `0.621224296375965`.
- Safety floor: final mIoU is at least `0.611224296375965`.
- Exact Git SHA, clean Git status, final checkpoint, and remaining disk are reported.

Return the preflight, training exit code, compact completion evidence, comparison, checkpoint, provenance, and disk output. Stop after the full Phase 9 run; do not begin effect-variance analysis or Phase 10.
