# Phase 14 learned-null seed-0 40k

- Status: complete; paired effect analysis pending.
- Exact training SHA: `d85db4e0dcf411681c1ddad09080dfeed1c419bd`.
- Stability: 40,000 contiguous finite records, 80 complete validations, and
  maximum objective reconstruction error `2.357e-7`.
- Null calibration: first/last 20-step means `2.139e-4/2.614e-13`.
- Final Cityscapes mIoU: `0.655364`; best mIoU: `0.662705` at iteration
  30,500.
- Factual-path comparison: `+0.7968` percentage points over the fixed Static
  seed-0 final result (`0.647396`).
- Peak allocated/reserved VRAM: `1.974/2.713 GiB`.
- Final alpha: `0.293385`.
- Checkpoints: 80 compact files; final `iter_040000.pth` is `45,641,331`
  bytes.

The run passes the Phase 14 training gate. Before authorizing repeat seeds or a
new loss, evaluate zero-ablation and learned-null class-logit effects on the
same 500 Cityscapes validation images. This read-only comparison must establish
whether learned-null retains matching-region localization and how it changes
background effect magnitude.
