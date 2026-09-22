# Phase 12 query-count R=2

- Status: complete, stable, and currently the strongest query-count candidate.
- Exact Git SHA: `198898f580516e3cf508b588d299b0bd3a95c55f`.
- Protocol: seed 0, 40,000 GTA5 iterations, two residual queries per class,
  original/photometric views at weights `1/1`, and all 500 Cityscapes
  validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and query diversity are absent.
- Stability: 40,000 contiguous records; all tracked values are finite;
  objective reconstruction error is `0.0`.
- Final alpha: `-0.263620`; peak allocated/reserved VRAM:
  `1.725/2.436 GiB`.
- Final Cityscapes mIoU: `0.643725`; best observed mIoU: `0.646860` at
  iteration 37,000.
- R=2 exceeds R=1 by 4.0014 percentage points and the fixed R=3
  photometric-only reference by 0.5439 points.
- Checkpoints: 80, including `iter_040000.pth`.

The R=2 versus R=3 final gap is just above the project's 0.5-point repeat
threshold. R=4 and the planned query-behavior diagnostics remain necessary
before closing the ablation.
