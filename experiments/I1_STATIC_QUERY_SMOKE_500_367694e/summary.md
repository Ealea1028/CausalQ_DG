# Phase 13 Static Query smoke

- Status: passed.
- Exact Git SHA: `367694ee8bc2e670be0a896f40e089760fe2177a`.
- Isolation: R=2 static grouped queries, zero cross-attention layers,
  original/photometric supervision, and no prediction consistency, CQE, or
  diversity loss.
- Stability: 500 contiguous finite records and exact objective reconstruction.
- Validation: 50 Cityscapes images at iteration 500; diagnostic mIoU
  `0.333036`.
- Peak allocated/reserved VRAM: `1.568/2.258 GiB`.
- Final alpha: `-0.024463`.
- Checkpoint: one `iter_000500.pth` file (`45,615,343` bytes).

The finite final gradient norm is recorded before the trainer's fixed
max-norm-1.0 clipping step. The smoke mIoU is not used for model selection.
Proceed to the isolated seed-0 40k Static Query run.
