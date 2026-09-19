# S2 Fourier-only style smoke test

- Status: accepted for full training.
- Exact Git SHA: `b0d6e14f0237582bd72fa643994a3fb2f3e3b71c`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 500 GTA5 iterations with original and Fourier views; the sole intervention has weight `1.0`; one 50-image Cityscapes validation.
- Isolation: prediction consistency, CQE, and diversity are absent.
- Stability: 500 contiguous iterations; all losses, gradients, and alpha values are finite; first-20 loss `4.073564`, last-20 loss `1.104635`.
- Objective reconstruction error: exactly `0.0`.
- Final alpha: `0.025409`.
- Cityscapes smoke mIoU: `0.339582`.
- Peak allocated/reserved VRAM: `1.727/2.443 GiB`.

This run validates Phase 11 Fourier-only isolation and numerical stability. Its 50-image mIoU is not used to rank the two interventions; that requires the controlled 40,000-iteration runs.
