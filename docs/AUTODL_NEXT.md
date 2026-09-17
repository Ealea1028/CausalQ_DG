# Next AutoDL action

Status: the Phase 5 500-iteration smoke run is accepted. GTA5 image `20217.png` was restored from the official part-9 archive, and the operator-confirmed full decoding recheck passed. Start a fresh 40,000-iteration seed-0 baseline run; never reuse the earlier interrupted run directory.

## Accepted prerequisites

- Corrected GTA5 palette labels passed semantic validation.
- Full GTA5 scan covers 24,966 paired samples.
- The only truncated image, `20217.png`, was replaced from the integrity-checked official archive.
- The post-repair scan reports no unreadable images or labels, missing pairs, invalid train IDs, all-ignore labels, or geometry mismatches.
- The 62 scale-equivalent size mismatches remain accepted warnings.
- The DINOv3-L/16 checkpoint SHA-256 is `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
- The 500-iteration smoke run completed with finite loss, decreasing loss means, 50-image Cityscapes mIoU `0.322154`, and 1.349 GiB peak allocated VRAM.

## Full-run scope

- Experiment: `A0_DINOV3L_BASE`.
- Seed: 0.
- GTA5 source training; all 500 Cityscapes val images for validation.
- Frozen DINOv3-L/16 and the baseline segmentation decoder only.
- 40,000 iterations, 512x512 crops, batch size 1, bfloat16 autocast, AdamW, and polynomial learning-rate decay.
- Validation and compact checkpoint every 500 iterations.
- No query branch, style intervention, prediction consistency, or CQE.

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

Both Git status checks must be empty. Enforce the saved post-repair report as a training gate:

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

assert report["ok"] is True, report
assert gta5["ok"] is True, gta5
assert gta5["paired_count"] == 24966, gta5["paired_count"]
assert gta5["scanned_label_count"] == 24966, gta5["scanned_label_count"]
assert gta5["scan_scope"] == "all", gta5["scan_scope"]
assert gta5["unreadable_count"] == 0, gta5["unreadable"]
assert gta5["missing_label_count"] == 0
assert gta5["missing_image_count"] == 0
assert gta5["invalid_train_ids"] == []
assert gta5["all_ignore_label_count"] == 0
assert gta5["geometry_mismatch_count"] == 0
print("GTA5_POST_REPAIR_GATE_OK")
PY
```

Fully decode the installed repaired image once more and verify the remaining prerequisites:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image

path = Path("/root/autodl-tmp/datasets/gta5/images/images/20217.png")
with Image.open(path) as image:
    image.load()
    print("repaired_image_ok:", path, image.mode, image.size, path.stat().st_size)
PY

test "$(find /root/autodl-tmp/datasets/gta5/images/images -type f -name '*.png' | wc -l)" -eq 24966
test "$(find /root/autodl-tmp/datasets/gta5/labels_trainIds -type f -name '*.png' | wc -l)" -eq 24966
test "$(find /root/autodl-tmp/datasets/cityscapes/leftImg8bit/val -type f -name '*_leftImg8bit.png' | wc -l)" -eq 500

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The weight hash must match the accepted value above. Keep the repair archive, staging directory, quarantine file, and failed run until the new full run is accepted.

For a stable terminal, optionally start `tmux new -s causalq_a0_full_r2` before running the following block. Start a uniquely named run from iteration zero:

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A0_DINOV3L_BASE_SEED0_40000_REPAIRED_${RUN_SHA}"
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500

RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

test ! -e "$RUN_DIR"
test ! -e "$LOG_FILE"

set -o pipefail
bash scripts/train_baseline.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "run_id=${RUN_ID}"
echo "train_exit_code=${TRAIN_EXIT}"
echo "log_file=${LOG_FILE}"
```

Do not resume or overwrite `A0_DINOV3L_BASE_SEED0_40000_c085de9`; it remains a failed pre-repair run.

After completion, collect compact evidence:

```bash
RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A0_DINOV3L_BASE_SEED0_40000_REPAIRED_${RUN_SHA}"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"

echo "===== metadata ====="
cat "$RUN_DIR/metadata.json"

echo "===== summary ====="
cat "$RUN_DIR/summary.json"

echo "===== checkpoint evidence ====="
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name '*.pth' | wc -l
du -sh "$RUN_DIR/checkpoints"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2 | tail -n 5
test -f "$RUN_DIR/checkpoints/iter_040000.pth"

echo "===== first and last training records ====="
head -n 5 "$RUN_DIR/train.jsonl"
tail -n 30 "$RUN_DIR/train.jsonl"

echo "===== provenance and disk ====="
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- The post-repair JSON gate and direct `20217.png` decode both pass before training.
- `train_exit_code=0` and `summary.json` has `"ok": true`.
- Exact hand-off Git SHA and accepted DINOv3-L checkpoint SHA-256 are recorded.
- All 40,000 iterations complete with finite loss and gradient checks.
- The late-stage loss distribution remains credibly below its beginning without persistent instability.
- Validation covers all 500 Cityscapes val images and returns finite class IoUs and mIoU.
- Exactly 80 compact checkpoints exist from iteration 500 through 40,000, including `iter_040000.pth`.
- Peak memory remains below the RTX 4090D capacity.
- Final `git status --short` is empty.

Return the preflight gate output, training exit code, complete `metadata.json` and `summary.json`, checkpoint evidence, first and last training records, exact Git SHA/status, and disk usage. Stop after the full run so the baseline can be published before Phase 6 begins.
