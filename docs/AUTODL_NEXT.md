# Next AutoDL action

Status: Phases 1 through 4 are accepted. Phase 5 source-only baseline implementation is complete locally and awaits a 500-iteration GPU smoke run.

## Phase 4 conclusion

- Verified commit: `881a5cd2539a132669616975ef51cbd013a44981`.
- RTX 4090D, PyTorch `2.7.0+cu126`, CUDA runtime `12.6`, BF16.
- ViT-B/16: `[1, 1024, 768]` patch tokens, `[1, 768, 32, 32]` map, four valid intermediate maps, 0.209 GiB peak allocated.
- ViT-L/16: `[1, 1024, 1024]` patch tokens, `[1, 1024, 32, 32]` map, four valid intermediate maps, 0.642 GiB peak allocated.
- Both models exposed five prefix tokens, had zero trainable backbone parameters, and contained no NaN or Inf values.
- ViT-B SHA-256: `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
- ViT-L SHA-256: `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
- The files came from the corresponding `facebook` namespace on ModelScope, recorded as a secondary distribution source subject to the DINOv3 License.

## Phase 5 smoke goal

Train only the segmentation decoder for 500 optimizer iterations on GTA5, with the DINOv3-L backbone frozen. Use standard shared geometric augmentation only: no causal queries, style intervention, prediction consistency, or CQE. Validate on 50 Cityscapes validation images to verify the complete inference and mIoU path. Do not start the 40k schedule yet.

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
python tools/check_environment.py
nvidia-smi
```

Confirm persistent inputs:

```bash
test -f /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
test -d /root/autodl-tmp/datasets/gta5/images
test -d /root/autodl-tmp/datasets/gta5/labels_trainIds
test -d /root/autodl-tmp/datasets/cityscapes/leftImg8bit/val
test -d /root/autodl-tmp/datasets/cityscapes/gtFine/val

sha256sum /root/autodl-tmp/pretrained/dinov3_vitl16/model.safetensors
```

Run the smoke test with a unique output name:

```bash
cd /root/autodl-tmp/CausalQ_DG
export MAX_ITERATIONS=500
export VALIDATION_MAX_SAMPLES=50
export RUN_ID="A0_DINOV3L_BASE_SMOKE_500_$(git rev-parse --short HEAD)"

set -o pipefail
bash scripts/train_baseline.sh \
  | tee "/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}.log"
echo "train_exit_code=${PIPESTATUS[0]}"
```

Inspect the artifacts:

```bash
cat "/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}/metadata.json"
cat "/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}/summary.json"
tail -n 30 "/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}/train.jsonl"
find "/root/autodl-tmp/outputs/CausalQ_DG/${RUN_ID}/checkpoints" \
  -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
git status --short
```

## Acceptance criteria

- `train_exit_code=0` and `summary.json` contains `"ok": true`.
- Git SHA and ViT-L SHA-256 exactly match the hand-off and manifest.
- `max_iterations=500`, source is GTA5, and validation is Cityscapes val with 50 samples.
- Total loss and gradient norm remain finite; gradient norm is nonzero for ordinary iterations.
- The last-20 loss mean is lower than the first-20 mean. Small short-term fluctuations are acceptable.
- Validation returns a finite mIoU and non-empty per-class IoUs. This smoke run is not a reportable benchmark.
- Backbone remains frozen; trainable parameters belong only to the decoder.
- At least one compact `.pth` checkpoint exists and does not contain a duplicate DINOv3 backbone.
- Peak allocated/reserved VRAM is recorded without out-of-memory errors.
- `git status --short` is empty.

Return the complete `metadata.json`, `summary.json`, the last 30 training records, checkpoint sizes, training exit code, and final Git status. Stop after the smoke run; review the evidence locally before authorizing the full 40k baseline.
