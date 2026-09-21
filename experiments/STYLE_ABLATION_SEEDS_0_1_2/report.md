# Phase 11 style ablation conclusion

Final-iteration Cityscapes mIoU over paired seeds 0, 1, and 2:

| Variant | Values | Mean ± sample std |
|---|---|---|
| Combined photometric + Fourier | 0.639610, 0.651021, 0.629514 | 0.640048 ± 0.010760 |
| Photometric-only | 0.638287, 0.651486, 0.618831 | 0.636201 ± 0.016427 |
| Paired Combined − Photometric | 0.001324, -0.000465, 0.010683 | 0.003847 ± 0.005987 |

Combined has a 0.3847-percentage-point mean advantage, but the paired sample standard deviation is 0.5987 points and the ordering reverses at seed 1. The evidence does not support a stable superiority claim. Following the prespecified decision rule, photometric-only is preferred as the simpler and cheaper default; combined remains a controlled ablation and Fourier-only remains a negative single-style result (`0.627076` at seed 0).
