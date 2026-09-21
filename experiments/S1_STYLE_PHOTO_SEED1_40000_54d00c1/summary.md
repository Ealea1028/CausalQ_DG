# S1 photometric-only style seed 1

- Status: complete and accepted for the Phase 11 paired-seed comparison.
- Exact Git SHA: `54d00c1a6964807fee4945c48b56e042aba3c906`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: seed 1, 40,000 GTA5 iterations with original and photometric views at weights `1/1`, and all 500 Cityscapes validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and diversity are disabled.
- Stability: 40,000 contiguous records; all component losses, gradients, and alpha values finite; objective reconstruction error `0.0`.
- Final alpha: `-0.264797`; peak allocated/reserved VRAM: `1.725/2.436 GiB`.
- Final Cityscapes mIoU: `0.651486`; best observed validation mIoU: `0.652874` at iteration 31,000.
- Paired comparison: photometric-only exceeds combined-view seed 1 by 0.0465 percentage points, reversing the seed-0 ordering.
- Checkpoints: 80 trainable-state checkpoints, including `iter_040000.pth`.

The opposing seed-0 and seed-1 rankings confirm that the close ablation cannot be decided from a single seed. The seed-2 pair remains required.
