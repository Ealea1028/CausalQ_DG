# Phase 12 query-count mechanism analysis

- Evaluation commit: `07ef4ce71d04cac432259f10d05693cdcecfdca8`.
- Environment: RTX 4090 D, PyTorch `2.7.0+cu126`, CUDA `12.6`.
- Protocol: the same 500 Cityscapes validation images, seed `20260924`, and
  original/photometric views for all four final checkpoints.
- Final mIoU ranks `R=2 > R=3 > R=4 > R=1`.
- R=2 uses about `1.91/2` effective queries, with residual/contextual mean
  cosine `0.320/0.931` and effect variance `9.03e-4`.
- R=3 and R=4 use nearly uniform responsibilities, but their contextual
  within-class cosine is approximately `0.999`, indicating near-duplicate
  contextual states rather than useful specialization.
- R=3 has the lowest effect variance (`7.88e-4`), but trails R=2 by 0.5439
  mIoU percentage points and uses one additional query per class.
- R=4 both loses 1.9642 mIoU points to R=2 and has the largest effect variance
  (`1.40e-3`). R=1 lacks capacity and loses 4.0014 points to R=2.

Phase 12 selects `R=2`: it has the highest task accuracy, substantially less
query redundancy than R=3/R=4, active participation from both queries, and
better cross-style effect stability than R=1/R=4. The slightly lower variance
of R=3 does not outweigh its lower accuracy and near-duplicate query states.
