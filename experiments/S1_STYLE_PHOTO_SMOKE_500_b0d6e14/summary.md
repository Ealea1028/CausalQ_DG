# S1 photometric-only style smoke test

- Status: accepted for full training.
- Exact Git SHA: `b0d6e14f0237582bd72fa643994a3fb2f3e3b71c`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 500 GTA5 iterations with original and photometric views; the sole intervention has weight `1.0`; one 50-image Cityscapes validation.
- Isolation: prediction consistency, CQE, and diversity are absent.
- Stability: 500 contiguous iterations; all losses, gradients, and alpha values are finite; first-20 loss `4.061964`, last-20 loss `1.088749`.
- Objective reconstruction error: exactly `0.0`.
- Final alpha: `0.025661`.
- Cityscapes smoke mIoU: `0.338213`.
- Peak allocated/reserved VRAM: `1.725/2.436 GiB`.

This run validates Phase 11 photometric-only isolation and numerical stability. It does not establish the final ablation result; that requires a fresh 40,000-iteration seed-0 run.
