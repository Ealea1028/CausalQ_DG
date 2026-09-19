# S2 Fourier-only style ablation

- Status: complete and accepted as the fixed Fourier-only result.
- Exact Git SHA: `2a6199cf363fde194b5c62f78500428959dcaaa1`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 40,000 GTA5 iterations with original and Fourier views; the sole counterfactual view has weight `1.0`; all 500 Cityscapes validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and diversity are disabled.
- Stability: 40,000 contiguous records; all component losses, gradients, and alpha values finite; objective reconstruction error `0.0`.
- Final alpha: `0.272216`; peak allocated/reserved VRAM: `1.727/2.443 GiB`.
- Final Cityscapes mIoU: `0.627076`; best observed validation mIoU: `0.638286` at iteration 29,500.
- Controlled comparison: the final result is 1.2534 percentage points below combined-view A2 and 1.1211 points below photometric-only S1.
- Checkpoints: 80 trainable-state checkpoints, including `iter_040000.pth`.

The final-iteration score is the primary result. Fourier-only is separated from the other variants by more than the plan's 0.5-point repeat threshold, while the combined-versus-photometric gap requires seeds 1 and 2.
