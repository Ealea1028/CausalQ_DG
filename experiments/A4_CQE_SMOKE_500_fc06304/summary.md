# A4 CQE smoke test

- Status: accepted for full training.
- Exact Git SHA: `fc063040ed9d0df19cae3cd059f68a6903be3c2c`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 500 GTA5 iterations with Query + Style + logit-level CQE; one 50-image Cityscapes validation.
- Stability: all 500 supervised/CQE losses, gradients, and alpha values finite; first-20 loss `4.134130`, last-20 loss `1.238015`.
- CQE activity: zero at the first near-zero-alpha step, nonzero on the remaining 499 steps, maximum `0.029544`, and last-20 mean `0.0000522`.
- Objective reconstruction error: `2.07e-7`.
- Final alpha: `0.015589`.
- Cityscapes smoke mIoU: `0.328202`.
- Peak allocated/reserved VRAM: `1.807/2.535 GiB`.
- Isolation: prediction consistency is absent; the run contains only Query + Style + CQE.

This run validates mechanism execution, isolation, and numerical stability only. The controlled Phase 9 result requires a fresh 40,000-iteration seed-0 run from random initialization.
