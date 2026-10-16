# Phase 17: source structure review (2026-10-16)

Evidence provenance: operator-pasted complete console summary and successful
`sha256sum -c` output. Full inventory bytes are still on AutoDL, not locally
audited. The next export command must verify their exact SHA256 before use.

- Producer Git: `f41dbeea92d67bffc571e2c3ef103de7b3693e00`; clean checkout.
- Python: 3.12.3, `/root/miniconda3/bin/python3`, Linux x86_64/glibc 2.35.
- Inventory: `/root/autodl-tmp/outputs/CausalQ_DG/phase17/GTA5_SOURCE_STRUCTURE_f41dbee_20261016T135221_54521/inventory.json`.
- Inventory SHA256: `7224adc8aaafbcaac412aa5ee3904c11ba10d3c81aac7dffea231259e21e0d25`.
- Pair-record SHA256: `92a67a0e32a297d2f41618c6919a0573cfde6ec23890b5dd9a6ae7353842662f`.
- Console log SHA256: `f080f77c22a6875c461fff7f586739c99a21afab9a067d7e4be6324476993b09`.
- Exit 0, `ok: true`, 24,966 image/converted-label pairs; no missing or duplicate
  IDs. Images resolve to `images/images`; labels to `labels_trainIds`.
- All logical IDs are flat numeric filenames; metadata candidate count is zero.
  The GTA5 subtree contains 74,900 PNG files (including original-label assets).
- No image decoding, label-value validation, target access or optimization.

| Requested comparison | Observation |
|---|---|
| Source mIoU / target mIoU | Not measured |
| Effect variance / magnitude | Not measured |
| Localization / responsibility | Not measured |
| Sequence/scene grouping | No usable evidence in the inventoried subtree |

Decision: accept the operator-reported filesystem gate; no model hypothesis is
tested. Under 1016 §6, prepare the configurable deterministic 10% filename
fallback. No training seed participates in membership. Do not infer that
numeric adjacency proves a video sequence, or that flat IDs are independent
scenes. This fallback cannot guarantee separation of correlated scenes/frames;
disclose that limitation. Metadata outside the inventoried GTA5 subtree has
not been investigated and is not claimed absent globally.

Next gate: verify exact saved inventory bytes, internal pair digest and fields;
export full candidate ID manifests. They are not frozen/usable for training
until returned, verified locally and committed. Do not invent IDs from the
count (e.g. assume a gapless 1..24966 range). Pixel-level dataset checks and
the strict training route remain later work in the same Priority 0 phase.
