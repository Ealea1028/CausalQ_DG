# A2 DINOv3-L Query + Style

- Status: accepted.
- Exact Git SHA: `293b3309e5dd4c95086e95c9a35576754444e7e5`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 40,000 GTA5 iterations with sequential original, photometric, and Fourier forwards; all 500 Cityscapes validation images every 500 iterations.
- Objective: `L_original + 0.5 L_photometric + 0.5 L_fourier`; maximum reconstruction error `0.0`.
- Stability: all total/component losses, gradients, and alpha values finite; first-20 loss `4.126116`, last-20 loss `1.170753`.
- Final alpha: `0.270644`.
- Final Cityscapes mIoU: `0.639610`, an absolute gain of `0.020080` over Query-only and `0.037882` over the source-only baseline.
- Peak allocated/reserved VRAM: `1.730/2.391 GiB`.
- Checkpoints: 80 compact trainable-state checkpoints, including `iter_040000.pth`.

This is the accepted Query + Style reference for the Phase 8 prediction-consistency control and subsequent Phase 9 CQE comparison.
