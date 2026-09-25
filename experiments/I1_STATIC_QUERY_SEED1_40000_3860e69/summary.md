# Phase 13 Static Query seed 1

- Status: complete and accepted for the paired interaction comparison.
- Exact Git SHA: `3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497`.
- Protocol: seed 1, 40,000 GTA5 iterations, R=2 static grouped queries,
  original/photometric views at weights `1/1`, and all 500 Cityscapes
  validation images every 500 iterations.
- Isolation: Query-to-Image attention, prediction consistency, CQE, query
  diversity, bidirectional interaction, and learned-null intervention are absent.
- Stability: 40,000 contiguous finite records; objective reconstruction error
  is `0.0`.
- Final alpha: `0.291529`; peak allocated/reserved VRAM:
  `1.568/2.258 GiB`.
- Final Cityscapes mIoU: `0.669958`; best observed mIoU: `0.676172` at
  iteration 37,000.
- Static exceeds the paired seed-1 one-way R=2 result by 4.1601 percentage
  points.
- Checkpoints: 80, including `iter_040000.pth`.

Across seeds 0 and 1, the paired Static-minus-One-way differences are
`+0.3671/+4.1601` percentage points, with mean `+2.2636` and sample standard
deviation `2.6821` points. Both signs favor Static, but the effect size is
variable and the prespecified seed-2 pair remains required. Run one-way R=2
seed 2 next; do not introduce bidirectional interaction yet.
