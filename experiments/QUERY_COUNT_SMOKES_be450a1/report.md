# Phase 12 query-count smoke tests

The isolated 500-iteration GPU smoke tests for `R=1,2,4` passed at
`be450a19eb457bac91b9f61a8f56d52bb3f689dd` on an RTX 4090 D.

| Queries per class | Trainable parameters | Final alpha | 50-image mIoU | Peak reserved GiB |
|---:|---:|---:|---:|---:|
| 1 | 16,376,596 | 0.024317 | 0.349869 | 2.434 |
| 2 | 16,396,052 | -0.024133 | 0.324659 | 2.436 |
| 4 | 16,434,964 | 0.023747 | 0.333064 | 2.438 |

Every run contains 500 contiguous records, finite losses/gradients/alpha,
zero objective-reconstruction error, one 50-image validation, and one final
checkpoint. Metadata confirms seed 0, the selected original/photometric style
protocol, the intended query count, and the absence of prediction consistency,
CQE, and query diversity. The `R=1` last gradient norm is recorded before the
trainer's fixed max-norm `1.0` clipping and is not a failure.

Smoke mIoU is not used to rank query counts. Full 40k runs with all 500
Cityscapes validation images are required.
