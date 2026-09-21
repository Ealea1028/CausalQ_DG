# Phase 12 query-count R=1

- Status: complete, stable, and retained as a negative query-count ablation.
- Exact Git SHA: `9f477261d4cf1bb7e9618b0d9d4fb2c1f6227d3c`.
- Protocol: seed 0, 40,000 GTA5 iterations, one residual query per class,
  original/photometric views at weights `1/1`, and all 500 Cityscapes
  validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and query diversity are absent.
- Stability: 40,000 contiguous records; all tracked values are finite;
  objective reconstruction error is `0.0`.
- Final alpha: `0.272815`; peak allocated/reserved VRAM:
  `1.725/2.436 GiB`.
- Final Cityscapes mIoU: `0.603711`; best observed mIoU: `0.612727` at
  iteration 33,000.
- Fixed R=3 photometric-only reference: `0.638287`; R=1 is lower by `0.034576`
  mIoU, or 3.4576 percentage points.
- Checkpoints: 80, including `iter_040000.pth`.

The performance deficit is not accompanied by numerical instability or a
protocol mismatch. R=1 is therefore a valid negative capacity ablation rather
than a failed run.
