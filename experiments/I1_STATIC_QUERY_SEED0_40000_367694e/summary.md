# Phase 13 Static Query seed 0

- Status: complete and accepted for the paired interaction comparison.
- Exact Git SHA: `367694ee8bc2e670be0a896f40e089760fe2177a`.
- Protocol: seed 0, 40,000 GTA5 iterations, R=2 static grouped queries,
  original/photometric views at weights `1/1`, and all 500 Cityscapes
  validation images every 500 iterations.
- Isolation: Query-to-Image attention, prediction consistency, CQE, query
  diversity, bidirectional interaction, and learned-null intervention are absent.
- Stability: 40,000 contiguous finite records; objective reconstruction error
  is `0.0`.
- Final alpha: `-0.296545`; peak allocated/reserved VRAM:
  `1.568/2.258 GiB`.
- Final Cityscapes mIoU: `0.647396`; best observed mIoU: `0.657943` at
  iteration 32,000.
- Static exceeds the existing seed-0 one-way R=2 reference by 0.3671
  percentage points.
- Checkpoints: 80, including `iter_040000.pth`.

The final difference is below the project's 0.5-point repeat threshold, so a
single seed does not establish that static interaction is superior. Repeat the
one-way and static variants as paired seeds 1 and 2 before selecting an
interaction or adding the bidirectional control.
