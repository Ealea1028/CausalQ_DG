# Next AutoDL action

Status: Phases 1 through 3 are accepted. In Phase 4, a direct ViT-B diagnostic loaded the ModelScope-distributed checkpoint and produced the expected dense features on the RTX 4090D. The formal checker failed before loading because PyTorch 2.7.0 rejected a `torch.device` argument in its CUDA memory-statistics API; the local fix now uses an integer CUDA index. The formal B/L checker awaits rerun.

## Phase 3 conclusion

- GTA5 full validation is accepted under the user's stated completion assumption.
- Cityscapes train/val: 3,475 image-mask pairs checked.
- Cityscapes train IDs: all `0..18` plus ignore index `255` observed.
- Missing pairs, invalid IDs, unreadable files, and geometry mismatches: zero.
- All 5,000 Cityscapes train-ID masks already existed, so the repeat conversion safely skipped them.

## Goal

Record the hashes of the already-downloaded LVD-1689M DINOv3 ViT-B/16 and ViT-L/16 checkpoints and verify frozen dense features for a 512×512 input on the RTX 4090D. The files were acquired from the corresponding `facebook` namespace on ModelScope after the official Hugging Face gating request was rejected. Treat ModelScope as a secondary distribution source and preserve that provenance in every report. Do not implement or train the segmentation baseline yet.

## Weight provenance

The local snapshots correspond to these upstream model identities:

- `https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`
- `https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m`

They were downloaded from:

- `https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m`
- `https://modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m`

Use is subject to the DINOv3 License. Do not describe the secondary distribution as proof of official Meta or Hugging Face access. Record the exact checkpoint SHA-256 values returned by the checker.

## Commands

Run the exact Git commit from the hand-off:

```bash
cd /root/autodl-tmp/CausalQ_DG
git status --short
git fetch origin
git checkout --detach <EXACT_SHA_FROM_HANDOFF>
git rev-parse HEAD

source scripts/activate_autodl.sh
python tools/check_environment.py
nvidia-smi
```

Confirm both existing snapshots and record their hashes:

```bash
find /root/autodl-tmp/pretrained/dinov3_vitb16 \
  -maxdepth 1 -type f -name '*.safetensors' -exec sha256sum {} \;
find /root/autodl-tmp/pretrained/dinov3_vitl16 \
  -maxdepth 1 -type f -name '*.safetensors' -exec sha256sum {} \;
```

Run the B/16 smoke test first, followed by the primary L/16 test:

```bash
mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/backbone_check
set -o pipefail

python tools/check_backbone.py \
  --model dinov3_vitb16 \
  --weights /root/autodl-tmp/pretrained/dinov3_vitb16 \
  --image-size 512 512 \
  --batch-size 1 \
  --dtype bfloat16 \
  --intermediate-indices 3 6 9 12 \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/backbone_check/vitb16.json
echo "vitb_exit_code=${PIPESTATUS[0]}"

python tools/check_backbone.py \
  --model dinov3_vitl16 \
  --weights /root/autodl-tmp/pretrained/dinov3_vitl16 \
  --image-size 512 512 \
  --batch-size 1 \
  --dtype bfloat16 \
  --intermediate-indices 6 12 18 24 \
  | tee /root/autodl-tmp/outputs/CausalQ_DG/backbone_check/vitl16.json
echo "vitl_exit_code=${PIPESTATUS[0]}"

git status --short
```

## Acceptance criteria

Both reports must show:

- `ok: true`, `weights_loaded: true`, and the exact hand-off Git SHA.
- Input shape `[1, 3, 512, 512]`, patch size `[16, 16]`, and five model-derived prefix tokens.
- ViT-B patch tokens `[1, 1024, 768]` and patch map `[1, 768, 32, 32]`.
- ViT-L patch tokens `[1, 1024, 1024]` and patch map `[1, 1024, 32, 32]`.
- Four intermediate maps with the same spatial grid and model hidden size.
- `nan_count: 0`, `inf_count: 0`, and `trainable_parameters: 0`.
- Non-empty checkpoint SHA-256 mappings and recorded peak allocated/reserved VRAM.
- Empty `git status --short`.

Return both complete JSON reports, the two shell exit codes, the `sha256sum` output, and the final Git status. Stop after this check; Phase 5 baseline implementation begins only after the real ViT-L result is reviewed locally.
