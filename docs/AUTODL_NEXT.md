# Next AutoDL action

Status: Phase 5 is accepted at `0.6017277359` Cityscapes mIoU. The Phase 6 Query-only real-weight contract check and 500-iteration smoke run are operator-confirmed as accepted. Run the full 40,000-iteration seed-0 Query-only experiment next; do not begin Phase 7.

## Fixed comparison

- Baseline experiment: `A0_DINOV3L_BASE_SEED0_40000_REPAIRED_7eee12b`.
- Baseline Cityscapes mIoU: `0.6017277358761972`.
- Query experiment: `A1_QUERY`.
- Same DINOv3-L backbone, data, geometry augmentation, decoder, optimizer, schedule, seed, and validation protocol.
- Query-only addition: 19 classes x 3 queries, one one-way eight-head cross-attention layer, temperature `0.07`, `logsumexp`, and zero-initialized learnable `alpha`.
- No style intervention, prediction consistency, CQE, or diversity loss.
- Acceptance floor: final Query-only mIoU must be at least `0.5967277358761972` (no more than 0.5 percentage points below baseline).
- Ideal range: approximately `0.6067` to `0.6217` (+0.5 to +2.0 percentage points).

## Commands

Run the exact Git commit supplied in the hand-off:

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

Both Git status checks must be empty. Reconfirm the accepted data, weight, and smoke artifacts:

```bash
python - <<'PY'
import json
from pathlib import Path

data_report = Path(
    "/root/autodl-tmp/outputs/CausalQ_DG/"
    "data_check/gta5_after_20217_repair.json"
)
report = json.loads(data_report.read_text(encoding="utf-8"))
gta5 = report["datasets"]["gta5"]
assert report["ok"] is True
assert gta5["ok"] is True
assert gta5["paired_count"] == 24966
assert gta5["unreadable_count"] == 0

query_report = Path(
    "/root/autodl-tmp/outputs/CausalQ_DG/query_check/phase6_query.json"
)
query = json.loads(query_report.read_text(encoding="utf-8"))
assert query["ok"] is True
assert query["alpha"] == 0.0
assert query["max_abs_full_vs_baseline"] < 1e-6

smoke_dir = Path(
    "/root/autodl-tmp/outputs/CausalQ_DG/A1_QUERY_SMOKE_500_956f692"
)
smoke = json.loads((smoke_dir / "summary.json").read_text(encoding="utf-8"))
assert smoke["ok"] is True
assert smoke["max_iterations"] == 500
assert smoke["finite_losses"] is True
assert len(smoke["validation_results"]) == 1
assert smoke["validation_results"][0]["sample_count"] == 50

print("PHASE6_FULL_RUN_GATES_OK")
print("smoke_final_alpha:", smoke["final_alpha"])
print("smoke_miou:", smoke["validation_results"][0]["miou"])
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
```

The checkpoint hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

Estimate full-run checkpoint storage from the smoke checkpoint and require a 5 GiB safety reserve:

```bash
SMOKE_CHECKPOINT="/root/autodl-tmp/outputs/CausalQ_DG/A1_QUERY_SMOKE_500_956f692/checkpoints/iter_000500.pth"
test -f "$SMOKE_CHECKPOINT"

CHECKPOINT_BYTES="$(stat -c '%s' "$SMOKE_CHECKPOINT")"
ESTIMATED_CHECKPOINT_BYTES="$((CHECKPOINT_BYTES * 80))"
FREE_BYTES="$(df --output=avail -B1 /root/autodl-tmp | tail -n 1 | tr -d ' ')"
SAFETY_BYTES="$((5 * 1024 * 1024 * 1024))"
REQUIRED_BYTES="$((ESTIMATED_CHECKPOINT_BYTES + SAFETY_BYTES))"

echo "smoke_checkpoint_bytes=${CHECKPOINT_BYTES}"
echo "estimated_80_checkpoints_bytes=${ESTIMATED_CHECKPOINT_BYTES}"
echo "free_bytes=${FREE_BYTES}"
echo "required_with_5gib_reserve=${REQUIRED_BYTES}"

test "$FREE_BYTES" -gt "$REQUIRED_BYTES"
df -h /root/autodl-tmp
```

Stop if the storage test fails. Otherwise, optionally create a stable terminal with `tmux new -s causalq_a1_full`, then start a unique run from iteration zero:

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A1_QUERY_SEED0_40000_${RUN_SHA}"
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500

RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
bash scripts/train_query.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "run_id=${RUN_ID}"
echo "train_exit_code=${TRAIN_EXIT}"
echo "log_file=${LOG_FILE}"
```

After completion, collect compact evidence:

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A1_QUERY_SEED0_40000_${RUN_SHA}"
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
losses = [record["loss"] for record in records]
gradients = [record["gradient_norm"] for record in records]
alphas = [record["alpha"] for record in records]
validations = summary["validation_results"]
final_validation = validations[-1]
baseline_miou = 0.6017277358761972
query_miou = final_validation["miou"]

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
print("iterations_contiguous:", iterations == list(range(1, 40001)))
print("all_losses_finite:", all(math.isfinite(value) for value in losses))
print("all_gradients_finite:", all(math.isfinite(value) for value in gradients))
print("all_alphas_finite:", all(math.isfinite(value) for value in alphas))
print("first_alpha:", alphas[0])
print("last_alpha:", alphas[-1])
print("minimum_alpha:", min(alphas))
print("maximum_alpha:", max(alphas))
print("validation_count:", len(validations))
print("final_validation:", final_validation)
print("baseline_miou:", baseline_miou)
print("query_miou:", query_miou)
print("query_minus_baseline:", query_miou - baseline_miou)
print("passes_minus_0_5pp_floor:", query_miou >= baseline_miou - 0.005)
print("first_record:", records[0])
print("last_record:", records[-1])
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

- All preflight gates and the storage test pass.
- `train_exit_code=0` and `summary.ok=true`.
- Exactly 40,000 contiguous iterations complete with finite losses, gradients, and alpha values.
- `alpha` remains finite and the query path trains without persistent instability.
- Eighty full 500-image Cityscapes validations complete with finite class IoUs and mIoU.
- Exactly 80 compact checkpoints exist, including `iter_040000.pth`.
- Final Query-only mIoU is at least `0.5967277358761972`.
- Peak memory remains within the RTX 4090D capacity.
- Exact Git SHA and DINOv3 checkpoint hash are recorded; final Git status is empty.

Return the gate/storage output, training exit code, compact result output, checkpoint evidence, exact Git SHA/status, and disk usage. Stop after the full Phase 6 run. If the acceptance floor fails, inspect alpha, query scale, temperature, logsumexp, and learning rate before adding any new loss or moving to Phase 7.
