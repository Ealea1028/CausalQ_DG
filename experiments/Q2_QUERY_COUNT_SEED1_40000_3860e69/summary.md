# Phase 13 paired repeat: one-way R=2 seed 1

- Status: complete and accepted for the paired interaction comparison.
- Exact Git SHA: `3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497`.
- Protocol: one-way R=2, photometric-only, seed 1, 40,000 iterations, and
  all 500 Cityscapes validation images every 500 iterations.
- Stability: 40,000 contiguous finite records, exact objective reconstruction,
  and 80 complete validations.
- Final Cityscapes mIoU: `0.628357`; best observed mIoU: `0.637596` at
  iteration 23,500.
- Final alpha: `-0.264000`; peak allocated/reserved VRAM:
  `1.726/2.436 GiB`.
- Checkpoints: 80, including `iter_040000.pth`.

The final sample produces a finite loss of `7.186951` and finite pre-clipping
gradient norm of `72.1414`. The trainer applies the fixed max-norm `1.0` clip
before the optimizer step, all 40,000 tracked values remain finite, and the
objective reconstructs exactly; this is not a numerical failure. Run the
paired Static Query seed-1 experiment before interpreting the seed effect.
