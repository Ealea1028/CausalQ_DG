# A5 Full objective

- Status: technically complete and stable; diversity retention threshold not met.
- Exact Git SHA: `973922b857177223cb5c6c231a8f1560c4c4a9b3`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 40,000 GTA5 iterations with Query + Style + prediction consistency + CQE + query diversity; all 500 Cityscapes validation images every 500 iterations.
- Stability: every loss component, gradient, and alpha value finite; objective reconstruction error `8.02e-7`.
- Diversity: active on all 40,000 iterations, falling from `0.00110865` to `1.07e-7`.
- Final Cityscapes mIoU: `0.592124`.
- Comparison: `+0.000374` versus A4, `-0.029100` versus A3, and `-0.047486` versus A2.
- Peak allocated/reserved VRAM: `1.827/2.535 GiB`.
- Checkpoints: 80, including `iter_040000.pth`.

The gain over A4 is only 0.037 percentage points, below the predefined 0.2-point retention threshold. The run is retained as a negative ablation result, and query diversity is removed from the final method.
