# Phase 13 paired repeat: one-way R=2 seed 2

- Status: complete and accepted for the paired interaction comparison.
- Exact Git SHA: `3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497`.
- Protocol: one-way R=2, photometric-only, seed 2, 40,000 iterations, and
  all 500 Cityscapes validation images every 500 iterations.
- Stability: 40,000 contiguous finite records, exact objective reconstruction,
  and 80 complete validations.
- Final Cityscapes mIoU: `0.628523`; best observed mIoU: `0.632286` at
  iteration 21,000.
- Final alpha: `-0.261834`; peak allocated/reserved VRAM:
  `1.726/2.436 GiB`.
- Checkpoints: 80, including `iter_040000.pth`.

The seed-2 one-way member is complete. Run the paired Static Query seed-2
experiment under the same repaired preprocessing commit before computing the
three-seed interaction comparison or introducing bidirectional interaction.
