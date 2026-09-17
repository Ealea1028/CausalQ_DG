# Next AutoDL action

Status: Phase 5 is accepted at 60.17% Cityscapes mIoU. Phase 6 Query-only is implemented locally and must now pass a real DINOv3-L GPU contract check followed by a 500-iteration smoke run. Do not start the full Phase 6 schedule yet.

## Phase 6 scope

- Experiment: `A1_QUERY`.
- Frozen DINOv3-L/16 backbone and the same baseline decoder.
- Nineteen class anchors with three residual queries per class: 57 queries total.
- One one-way, eight-head query-to-image cross-attention layer.
- Normalized feature/query similarity with temperature `0.07` and within-class `logsumexp` aggregation.
- Additive query residual controlled by a learnable scalar `alpha`, initialized to exactly zero.
- GTA5 source training and Cityscapes validation.
- No style intervention, prediction consistency, CQE, or diversity loss.

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

Both Git status checks must be empty. Reuse the accepted data and weight gates:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path(
    "/root/autodl-tmp/outputs/CausalQ_DG/"
    "data_check/gta5_after_20217_repair.json"
)
report = json.loads(path.read_text(encoding="utf-8"))
gta5 = report["datasets"]["gta5"]
assert report["ok"] is True
assert gta5["ok"] is True
assert gta5["paired_count"] == 24966
assert gta5["scanned_label_count"] == 24966
assert gta5["unreadable_count"] == 0
print("GTA5_POST_REPAIR_GATE_OK")
PY

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The weight hash must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

Run the real-weight query contract check:

```bash
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/query_check
set -o pipefail

python tools/check_query.py \
  --config configs/query/gta_dinov3l_query.yaml \
  --image-size 512 512 \
  --batch-size 1 \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/query_check/phase6_query.json

QUERY_CHECK_EXIT=${PIPESTATUS[0]}
echo "query_check_exit_code=${QUERY_CHECK_EXIT}"
```

The report must have `ok=true`, logits and delta logits shaped `[1,19,512,512]`, query states shaped `[1,19,3,1024]`, `alpha=0`, finite outputs, and `max_abs_full_vs_baseline < 1e-6`.

Only after that check passes, start a unique 500-iteration smoke run with 50-image validation:

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A1_QUERY_SMOKE_500_${RUN_SHA}"
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50

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

Collect compact evidence after successful completion:

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A1_QUERY_SMOKE_500_${RUN_SHA}"
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
validation = summary["validation_results"][-1]

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
print("iterations_contiguous:", [r["iteration"] for r in records] == list(range(1, 501)))
print("all_gradients_finite:", all(math.isfinite(r["gradient_norm"]) for r in records))
print("first_alpha:", records[0]["alpha"])
print("last_alpha:", records[-1]["alpha"])
print("validation:", validation)
PY

find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' | sort
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- Query checker exits zero with the specified shapes, finite values, zero alpha, and baseline equivalence below `1e-6`.
- Smoke training exits zero and records `summary.ok=true`.
- Exactly 500 contiguous iterations complete with finite losses and gradients.
- `alpha` starts from zero and becomes finite; query training does not produce persistent instability.
- The late loss distribution is credibly below the early distribution.
- Validation completes on 50 Cityscapes images with finite class IoUs and mIoU.
- The iteration-500 compact checkpoint exists.
- Peak memory remains below the RTX 4090D capacity.
- Exact Git SHA and accepted DINOv3 checkpoint hash are recorded; final Git status is empty.

Return the complete query-check JSON and exit code, smoke training exit code, compact evidence output, checkpoint listing, exact Git SHA/status, and disk usage. Stop after the Phase 6 smoke run; the full query-only schedule begins only after local review.
