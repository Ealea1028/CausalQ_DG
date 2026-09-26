# Phase 13 Bidirectional Query seed 0

- Status: complete negative ablation.
- Exact Git SHA: `13667167b641209a35ddf437616ba11fb6d5c246`.
- Isolation: R=2 joint query-patch self-attention, one interaction layer,
  original/photometric supervision, and no auxiliary consistency loss.
- Stability: 40,000 contiguous finite records, exact objective reconstruction,
  and 80 complete 500-image Cityscapes validations.
- Final/best mIoU: `0.630394` / `0.636628` (best at iteration 33,000).
- Peak allocated/reserved VRAM: `1.763/2.457 GiB`.
- Final alpha: `-0.210633`.
- Checkpoints: 80; final file `iter_040000.pth` (`196,783,903` bytes).

At the prespecified final iteration, Bidirectional is 1.3332 percentage points
below the seed-0 One-way result and 1.7003 points below Static. Even its best
intermediate result remains below both final references. The deficit exceeds
the 0.5-point repeat band, so no seed-1/2 Bidirectional repeats are authorized.
Phase 13 closes with Static as the selected interaction.
