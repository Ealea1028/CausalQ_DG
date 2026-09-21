# S1 photometric-only style seed 2

- Status: complete and accepted for the Phase 11 paired-seed comparison.
- Exact Git SHA: `4662cbadd4553728e521c351682da53545996a03`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: seed 2, 40,000 GTA5 iterations with original and photometric views at weights `1/1`, and all 500 Cityscapes validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and diversity are disabled.
- Stability: 40,000 contiguous records; all component losses, gradients, and alpha values finite; objective reconstruction error `0.0`.
- Final alpha: `-0.262352`; peak allocated/reserved VRAM: `1.725/2.436 GiB`.
- Final Cityscapes mIoU: `0.618831`; best observed validation mIoU: `0.634745` at iteration 31,500.
- Paired comparison: combined-view seed 2 exceeds photometric-only by 1.0683 percentage points.
- Checkpoints: 80 trainable-state checkpoints, including `iter_040000.pth`.

The last logged gradient norm is a finite pre-clipping value and is clipped at max-norm `1.0`. The complete three-seed comparison is recorded separately.
