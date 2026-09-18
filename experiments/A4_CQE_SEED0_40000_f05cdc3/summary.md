# A4 Causal Query Effect

- Status: technically complete and stable; primary target and safety floor not met.
- Exact Git SHA: `f05cdc37fc1a95bc83d1b4a90441250c657e8c51`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 40,000 GTA5 iterations with Query + Style + logit-level CQE; all 500 Cityscapes validation images every 500 iterations.
- Stability: all 40,000 supervised/CQE losses, gradients, and alpha values finite; objective reconstruction error `4.43e-7`.
- CQE activity: nonzero on 39,998 iterations; maximum `0.175106`; last-20 mean `0.0000108`.
- Final alpha: `0.044348`.
- Final Cityscapes mIoU: `0.591751`.
- Comparison: `-0.029474` versus prediction consistency and `-0.047860` versus Query + Style.
- Peak allocated/reserved VRAM: `1.807/2.537 GiB`.
- Checkpoints: 80, including `iter_040000.pth`.

The result is retained without post-hoc tuning. It fails both the primary `A4 > A3` target and the predefined `A4 >= A3 - 0.01` floor. The planned cross-style effect-variance comparison remains necessary to evaluate Phase 9's second target.
