# Phase 13 Static Query seed 2

- Status: complete and accepted; closes the three-seed Static-versus-One-way
  comparison.
- Exact Git SHA: `3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497`.
- Protocol: seed 2, 40,000 GTA5 iterations, R=2 static grouped queries,
  original/photometric views, and all 500 Cityscapes validation images every
  500 iterations.
- Stability: 40,000 contiguous finite records; objective reconstruction error
  is `0.0`.
- Final alpha: `-0.298329`; peak allocated/reserved VRAM:
  `1.568/2.258 GiB`.
- Final Cityscapes mIoU: `0.641193`; best observed mIoU: `0.656751` at
  iteration 35,000.
- Static exceeds paired one-way seed 2 by 1.2670 percentage points.
- Checkpoints: 80, including `iter_040000.pth`.

Across seeds 0/1/2, one-way reaches `0.633535 ± 0.008826` and Static reaches
`0.652849 ± 0.015138`. Static-minus-One-way is
`+0.019314 ± 0.019819`, or `+1.9314 ± 1.9819` percentage points, and is
positive for all three seeds. Static is selected over one-way. The remaining
planned Phase 13 structure control is bidirectional query-image self-attention.
