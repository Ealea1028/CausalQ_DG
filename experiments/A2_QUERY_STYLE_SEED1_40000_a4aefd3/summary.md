# A2 combined style seed 1

- Status: complete and accepted for the Phase 11 paired-seed comparison.
- Exact Git SHA: `a4aefd39527bed4044a4569cb90b161594a971f4`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: seed 1, 40,000 GTA5 iterations, sequential original/photometric/Fourier forwards with weights `1/0.5/0.5`, and all 500 Cityscapes validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and diversity are disabled.
- Stability: 40,000 contiguous records; all component losses, gradients, and alpha values finite; objective reconstruction error `0.0`.
- Final alpha: `-0.262852`; peak allocated/reserved VRAM: `1.730/2.391 GiB`.
- Final Cityscapes mIoU: `0.651021`; best observed validation mIoU: `0.654834` at iteration 32,000.
- Checkpoints: 80 trainable-state checkpoints, including `iter_040000.pth`.

The negative alpha is valid because the residual scale is unconstrained and its sign can be absorbed by the learned query residual. This result must be paired with photometric-only seed 1 before updating the close-ablation comparison.
