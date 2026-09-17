# A0 DINOv3-L source-only baseline

- Status: accepted.
- Exact Git SHA: `7eee12b3ad8c923de0e8557d3126f41c7b6cc617`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: GTA5 source-only training for 40,000 iterations; all 500 Cityscapes validation images every 500 iterations.
- Loss: first-20 mean `1.638739`; last-20 mean `0.129330`; all 40,000 losses and gradients finite.
- Final Cityscapes mIoU: `0.601728`.
- Peak allocated/reserved VRAM: `1.349/1.795 GiB`.
- Checkpoints: 80 compact trainable-state checkpoints, including `iter_040000.pth`.

The result is the Phase 5 reference for the Phase 6 query-only comparison. The original full-run exit shell variable was unavailable after reconnecting, but the valid final summary, contiguous 40,000-record trace, 80 scheduled validations/checkpoints, and final checkpoint establish successful completion.
