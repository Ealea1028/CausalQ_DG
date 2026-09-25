# Phase 13 Bidirectional Query smoke

- Status: passed.
- Exact Git SHA: `13667167b641209a35ddf437616ba11fb6d5c246`.
- Isolation: R=2 bidirectional joint query-patch self-attention, one interaction
  layer, original/photometric supervision, and no prediction consistency, CQE,
  or diversity loss.
- Stability: 500 contiguous finite records and exact objective reconstruction.
- Validation: 50 Cityscapes images at iteration 500; diagnostic mIoU
  `0.324368`.
- Peak allocated/reserved VRAM: `1.763/2.455 GiB`.
- Final alpha: `-0.024750`.
- Checkpoint: one `iter_000500.pth` file (`196,783,903` bytes).

The smoke mIoU is not used for model selection. The run verifies the intended
mechanism, training path, GPU memory, and serialization behavior. Proceed to
the isolated seed-0 40k Bidirectional Query run; do not add learned-null or
start seed repeats before evaluating that run.
