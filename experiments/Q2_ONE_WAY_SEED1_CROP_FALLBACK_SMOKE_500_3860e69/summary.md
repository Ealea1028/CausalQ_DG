# Valid-crop fallback GPU smoke

- Status: passed.
- Exact Git SHA: `3860e69b560ad8ce6f4d7ba1d7b83eb7448b2497`.
- Protocol: one-way R=2, photometric-only, seed 1, 500 iterations, and one
  50-image Cityscapes validation.
- Completion: 500 records and one `iter_000500.pth` checkpoint
  (`196,783,903` bytes).
- Diagnostic Cityscapes mIoU: `0.341777`.
- Final alpha: `-0.025015`.
- The excluded failed attempt remains preserved with 3,316 records and six
  intermediate checkpoints.

The smoke confirms that the isolated valid-crop fallback integrates with the
full GPU training path. Its short-run mIoU is not used for model selection.
Restart the one-way R=2 seed-1 full run from random initialization under this
exact implementation commit; do not resume the excluded checkpoint.
