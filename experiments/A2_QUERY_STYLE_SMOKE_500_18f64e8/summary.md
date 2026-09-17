# A2 Query + Style smoke test

- Status: accepted for full training.
- Exact Git SHA: `18f64e8612f46133cd66feda8522ee98b0b45411`.
- Hardware: NVIDIA GeForce RTX 4090 D; PyTorch 2.7.0+cu126; CUDA 12.6.
- Protocol: 500 GTA5 iterations with sequential original, photometric, and Fourier forwards; one 50-image Cityscapes validation.
- Objective: `L_original + 0.5 L_photometric + 0.5 L_fourier`; reconstruction error `0.0`.
- Stability: all 500 total/component losses, gradients, and alpha values finite; first-20 loss `4.132003`, last-20 loss `1.127431`.
- Final alpha: `0.025434`.
- Cityscapes smoke mIoU: `0.339556`.
- Peak allocated/reserved VRAM: `1.730/2.391 GiB`.
- Five-sample visual review: exact geometry preserved, meaningful appearance variation on normal street scenes, and no structural Fourier artifacts.

This run validates execution and numerical stability only. The controlled Phase 7 result requires a fresh 40,000-iteration seed-0 run from random initialization.
