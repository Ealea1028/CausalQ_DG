# Next AutoDL action

Status: Phase 3 is accepted again after the corrected GTA5 palette-index conversion passed full validation. Phase 5 resumes with the specified 500-iteration source-only baseline smoke test on the RTX 4090D.

## Accepted prerequisites

- Corrected GTA5 derived labels: 24,966 images, labels, and pairs.
- Full scan: 24,966 labels; IDs are exactly `0..18` plus `255`.
- Zero missing pairs, invalid train IDs, unreadable files, geometry mismatches, and all-ignore labels.
- The 62 reported shape mismatches are scale-equivalent; the dataset loader aligns masks to images with nearest-neighbor resampling.
- Valid-pixel fraction: minimum `0.134332...`, mean `0.888130...`.
- Cityscapes train/val: 3,475 valid pairs with a full label scan.
- DINOv3 ViT-L/16 checkpoint SHA-256: `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
- Available AutoDL data-disk space after archive cleanup: 86 GiB.

## Scope

This run contains only the frozen DINOv3-L source-only baseline:

- source training data: GTA5;
- validation data: Cityscapes val;
- no query branch;
- no style intervention;
- no prediction consistency or CQE loss;
- 500 training iterations and a bounded 50-image validation smoke check.

## Commands

Run the exact Git commit supplied in the hand-off. A clean worktree is mandatory:

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

Confirm the corrected GTA5 labels and ViT-L checkpoint are still present:

```bash
test "$(find /root/autodl-tmp/datasets/gta5/images/images -type f -name '*.png' | wc -l)" -eq 24966
test "$(find /root/autodl-tmp/datasets/gta5/labels_trainIds -type f -name '*.png' | wc -l)" -eq 24966

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
df -h /root/autodl-tmp
```

The SHA-256 must match the accepted value above. Then start a uniquely named smoke run and retain the pipeline exit status:

```bash
cd /root/autodl-tmp/CausalQ_DG
set -o pipefail

RUN_SHA="$(git rev-parse --short HEAD)"
RUN_ID="A0_DINOV3L_BASE_SMOKE_500_${RUN_SHA}"
LOG_FILE="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"

export RUN_ID
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50

bash scripts/train_baseline.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "run_id=${RUN_ID}"
echo "train_exit_code=${TRAIN_EXIT}"
echo "log_file=${LOG_FILE}"
```

Do not reuse an existing run ID. If the command reports that the run directory already exists, append `_R2` to `RUN_ID` and rerun.

After a successful run, collect the compact evidence:

```bash
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}"

cat "$RUN_DIR/metadata.json"
cat "$RUN_DIR/summary.json"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%s %p\n' | sort
tail -n 30 "$RUN_DIR/train.jsonl"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

## Acceptance criteria

- `train_exit_code=0` and `summary.json` has `"ok": true`.
- The recorded Git SHA equals the exact hand-off commit.
- The recorded checkpoint SHA-256 equals the accepted DINOv3-L value.
- Exactly 500 iterations complete with finite losses and gradients.
- The final 20-loss mean is lower than the first 20-loss mean, or the trace otherwise shows a credible downward trend without instability.
- Validation completes on 50 Cityscapes images and reports a finite mIoU.
- The iteration-500 checkpoint exists and contains only the intended compact training state rather than a duplicate frozen backbone.
- Peak GPU memory is recorded and remains within the RTX 4090D capacity.
- Final `git status --short` is empty.

Return the environment report, complete `metadata.json` and `summary.json`, checkpoint listing, last 30 training records, training exit code, final Git SHA/status, and disk usage. Stop after this smoke run; the full 40k schedule begins only after its evidence is reviewed locally.
