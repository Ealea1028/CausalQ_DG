# A1 DINOv3-L Query-only

- Status: accepted.
- Exact Git SHA: `543ab2e5ad302127716888f13c520786c3b2479c`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: GTA5 source-only training for 40,000 contiguous iterations; all 500 Cityscapes validation images every 500 iterations.
- Mechanism: 19 classes x 3 queries, one-way eight-head cross-attention, normalized logsumexp residual logits, and learnable residual scale.
- Loss: first-20 mean `2.067257`; last-20 mean `0.476843`; all losses, gradients, and alpha values finite.
- Final alpha: `0.261552`.
- Final Cityscapes mIoU: `0.619530`, an absolute gain of `0.017802` (1.78 percentage points) over the accepted source-only baseline.
- Peak allocated/reserved VRAM: `1.723/2.369 GiB`.
- Checkpoints: 80 compact trainable-state checkpoints, including `iter_040000.pth`.

The result passes the Phase 6 acceptance floor and lies within the project's ideal gain range. It is the controlled Query-only reference for Phase 7.
