# A5 Full smoke test

- Status: accepted for full training.
- Exact Git SHA: `9cbaf8b85202b2ecaf274df6e2cd93427096d7fb`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 500 GTA5 iterations with Query + Style + prediction consistency + CQE + query diversity; one 50-image Cityscapes validation.
- Stability: all supervised, prediction, CQE, diversity, gradient, and alpha values finite; first-20 loss `4.150551`, last-20 loss `1.255349`.
- Diversity: nonzero on all 500 iterations; first `0.00110865`, last `0.000757529`, maximum `0.00110865`.
- Objective reconstruction error: `1.39e-7`.
- Final alpha: `0.022077`.
- Cityscapes smoke mIoU: `0.329130`.
- Peak allocated/reserved VRAM: `1.827/2.535 GiB`.

This run validates the composed objective, diversity activity, and numerical stability only. The controlled Phase 10 result requires a fresh 40,000-iteration seed-0 run from random initialization.
