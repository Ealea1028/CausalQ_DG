# Phase 12 query-count R=4

- Status: complete, stable, and retained as a negative capacity ablation.
- Exact Git SHA: `84268a115400ef360270c2ed9d4a229041bfbbe5`.
- Protocol: seed 0, 40,000 GTA5 iterations, four residual queries per class,
  original/photometric views at weights `1/1`, and all 500 Cityscapes
  validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and query diversity are absent.
- Stability: 40,000 contiguous records; all tracked values are finite;
  objective reconstruction error is `0.0`.
- Final alpha: `0.264185`; peak allocated/reserved VRAM:
  `1.726/2.438 GiB`.
- Final Cityscapes mIoU: `0.624083`; best observed mIoU: `0.639086` at
  iteration 32,500.
- R=4 trails R=2 by 1.9642 percentage points and the fixed R=3 reference by
  1.4203 points.
- Checkpoints: 80, including `iter_040000.pth`.

The complete final-mIoU order is R=2, R=3, R=4, R=1. Query similarity,
active-query behavior, and cross-style query-effect variance remain required
before Phase 12 is closed.
