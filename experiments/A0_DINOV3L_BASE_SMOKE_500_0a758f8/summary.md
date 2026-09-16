# A0 DINOv3-L baseline smoke result

- Status: accepted.
- Exact Git SHA: `0a758f86b55398711033707093a5b45fe35d2c19`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: GTA5 source-only training for 500 iterations; 50-image Cityscapes validation.
- Loss: first-20 mean `1.640873`; last-20 mean `0.687394`; all losses finite.
- Cityscapes smoke mIoU: `0.322154`.
- Peak allocated/reserved VRAM: `1.349/1.795 GiB`.
- Checkpoint: `iter_000500.pth`, 19,742,619 bytes; the frozen backbone is not duplicated.

The loss trace contains occasional crop-dependent spikes, but it completed without non-finite values and the optimizer applies gradient clipping. The run satisfies the Phase 5 smoke criteria and authorizes the full 40k schedule. The reported mIoU is a smoke metric over 50 validation images, not a final benchmark result.
