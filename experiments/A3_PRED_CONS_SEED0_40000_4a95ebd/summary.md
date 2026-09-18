# A3 Prediction Consistency control

- Status: technically valid control; predefined performance floor not met.
- Exact Git SHA: `4a95ebd7dd627bd4695f201553a5e84aa709db09`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 40,000 GTA5 iterations with Query + Style + valid-pixel prediction KL; all 500 Cityscapes validation images every 500 iterations.
- Stability: all supervised/prediction losses, gradients, and alpha values finite; objective reconstruction error `4.42e-7`.
- Final alpha: `0.273463`.
- Final Cityscapes mIoU: `0.621224`.
- Comparison: `-0.018386` versus Query + Style, but `+0.001694` versus Query-only.
- Peak allocated/reserved VRAM: `1.749/2.434 GiB`.
- Checkpoints: 80, including `iter_040000.pth`.

The run is retained without post-hoc tuning as the ordinary prediction-consistency control for the Phase 9 CQE comparison.
