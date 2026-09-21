# A2 combined style seed 2

- Status: complete and accepted for the Phase 11 paired-seed comparison.
- Exact Git SHA: `fe41a138d1a56ce0b0a41641ca79eaf6ca5d5b8f`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: seed 2, 40,000 GTA5 iterations, sequential original/photometric/Fourier forwards with weights `1/0.5/0.5`, and all 500 Cityscapes validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and diversity are disabled.
- Stability: 40,000 contiguous records; all component losses, gradients, and alpha values finite; objective reconstruction error `0.0`.
- Final alpha: `-0.263139`; peak allocated/reserved VRAM: `1.730/2.391 GiB`.
- Final Cityscapes mIoU: `0.629514`; best observed validation mIoU: `0.634381` at iteration 31,500.
- Checkpoints: 80 trainable-state checkpoints, including `iter_040000.pth`.

The final logged gradient norm `31.1846` is the pre-clipping norm; the trainer applies max-norm `1.0` clipping before the optimizer step. It is finite and accompanied by a complete stable run, so it is retained rather than treated as divergence. The paired photometric-only seed-2 result remains required.
