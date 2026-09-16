# Next AutoDL action

Status: the Phase 5 source-only baseline passed its 500-iteration RTX 4090D smoke test. Run the full 40,000-iteration seed-0 schedule next; do not begin Phase 6 until the complete baseline is accepted.

## Accepted smoke evidence

- Run: `A0_DINOV3L_BASE_SMOKE_500_0a758f8`.
- Exact Git SHA: `0a758f86b55398711033707093a5b45fe35d2c19`.
- All 500 losses were finite.
- First/last 20-loss means: `1.640873` / `0.687394`.
- Cityscapes mIoU on the bounded 50-image smoke validation: `0.322154`.
- Peak allocated/reserved VRAM: `1.349` / `1.795 GiB`.
- The iteration-500 checkpoint is 19,742,619 bytes and does not duplicate the frozen backbone.
- Corrected GTA5 and Cityscapes data, DINOv3-L weights, and Git provenance passed their prerequisites.

## Full-run scope

- Experiment: `A0_DINOV3L_BASE`.
- Seed: 0.
- Source: GTA5.
- Validation: all 500 Cityscapes val images.
- Frozen DINOv3-L/16 plus the baseline segmentation decoder.
- 40,000 iterations, 512x512 crops, batch size 1, bfloat16 autocast, AdamW, and polynomial learning-rate decay.
- Validation and compact checkpoint every 500 iterations.
- No query branch, style intervention, prediction consistency, or CQE.

## Commands

Use the exact Git commit supplied in the hand-off:

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

Both `git status --short` checks must be empty. Confirm the fixed data, weight identity, and available space:

```bash
GTA_IMAGE_COUNT="$(find /root/autodl-tmp/datasets/gta5/images/images -type f -name '*.png' | wc -l)"
GTA_LABEL_COUNT="$(find /root/autodl-tmp/datasets/gta5/labels_trainIds -type f -name '*.png' | wc -l)"
CITY_VAL_COUNT="$(find /root/autodl-tmp/datasets/cityscapes/leftImg8bit/val -type f -name '*_leftImg8bit.png' | wc -l)"

echo "gta_image_count=${GTA_IMAGE_COUNT}"
echo "gta_label_count=${GTA_LABEL_COUNT}"
echo "cityscapes_val_count=${CITY_VAL_COUNT}"

test "$GTA_IMAGE_COUNT" -eq 24966
test "$GTA_LABEL_COUNT" -eq 24966
test "$CITY_VAL_COUNT" -eq 500

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The checkpoint SHA-256 must be `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`. The run creates 80 compact checkpoints and requires roughly 2 GiB beyond transient overhead; the currently reported 41 GiB free is sufficient.

Start the full run:

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh

export OMP_NUM_THREADS=1
export RUN_SHA="$(git rev-parse --short HEAD)"
export RUN_ID="A0_DINOV3L_BASE_SEED0_40000_${RUN_SHA}"
export MAX_ITERATIONS=40000
export VALIDATION_MAX_SAMPLES=500

LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

set -o pipefail
bash scripts/train_baseline.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "run_id=${RUN_ID}"
echo "train_exit_code=${TRAIN_EXIT}"
echo "log_file=${LOG_FILE}"
```

Do not reuse an existing run ID. The current trainer intentionally does not claim interruption-resume support; run it in a stable terminal session. If desired, create a `tmux` session before starting with `tmux new -s causalq_a0_full`, and detach with `Ctrl-b` then `d`.

After completion, collect the evidence:

```bash
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"

echo "===== metadata ====="
cat "$RUN_DIR/metadata.json"

echo "===== summary ====="
cat "$RUN_DIR/summary.json"

echo "===== checkpoint count and size ====="
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name '*.pth' | wc -l
du -sh "$RUN_DIR/checkpoints"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2 | tail -n 5

echo "===== first and last training records ====="
head -n 5 "$RUN_DIR/train.jsonl"
tail -n 30 "$RUN_DIR/train.jsonl"

echo "===== provenance and disk ====="
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- `train_exit_code=0` and `summary.json` has `"ok": true`.
- Exact hand-off Git SHA and accepted DINOv3-L checkpoint SHA-256 are recorded.
- All 40,000 iterations complete with finite loss and gradient checks.
- The training trace remains stable and shows a credible lower late-stage loss distribution than its beginning.
- Validation covers all 500 Cityscapes val images and returns finite class IoUs and mIoU.
- There are 80 compact checkpoints from iteration 500 through 40,000; the final `iter_040000.pth` exists.
- Peak memory remains below the RTX 4090D capacity.
- Final `git status --short` is empty.

Return the training exit code, complete `metadata.json` and `summary.json`, checkpoint count/size, first and last training records, exact Git SHA/status, and disk usage. Stop after the full Phase 5 run so its final baseline result can be reviewed and published before Phase 6 begins.
