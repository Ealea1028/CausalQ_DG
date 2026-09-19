# S1 photometric-only style ablation

- Status: complete and accepted as the fixed photometric-only result.
- Exact Git SHA: `d335694f33b0266a338c91fe9bfe6885fb48467c`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 40,000 GTA5 iterations with original and photometric views; the sole counterfactual view has weight `1.0`; all 500 Cityscapes validation images every 500 iterations.
- Isolation: prediction consistency, CQE, and diversity are disabled.
- Stability: 40,000 contiguous records; all component losses, gradients, and alpha values finite; objective reconstruction error `0.0`.
- Final alpha: `0.270177`; peak allocated/reserved VRAM: `1.725/2.436 GiB`.
- Final Cityscapes mIoU: `0.638287`; best observed validation mIoU: `0.641253` at iteration 30,000.
- Controlled comparison: the final result is 0.1324 percentage points below the combined-view A2 result (`0.639610`) and 1.8756 points above Query-only (`0.619530`).
- Checkpoints: 80 trainable-state checkpoints, including `iter_040000.pth`.

The fixed final-iteration score is used for the primary ablation comparison. The best intermediate score is retained as descriptive evidence and does not replace the prespecified final result.
