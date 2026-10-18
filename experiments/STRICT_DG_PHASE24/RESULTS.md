# M2 formal seed0 source result

Producer:`72b168ae2a3995a711394418354682ed97cdaafc`.
Accepted locally2026-10-18.40000 updates,2497 frozen source holdout images,
fixed-final iteration40000. Source mIoU=73.71820469715%;
delta vsM1=-0.13595781695 percentage points. No source gain or target claim.

Report SHA256:355a4b5c2e109a35989675aba4d7bf1e0bb416a1b621339d3e9c06bc0b49f5f7.
Checksum SHA256:fcfc2263307242b8da2defa9aca9f380a4fd035944462c7b15f667d884fdd578.
Returned bytes unchanged; current frozen config and reserialized embedded
metadata reproduce all original checksum entries. Independent19-class
integer confusion reduction reproduces per-class and mean source IoUs.
Same initial head/splits/weights/runtime/budget as accepted M0/M1.
No target access. Full traces/checkpoints remain on AutoDL,not locally replayed.

Next phase checks raw anchor stability at the existing FORMAL M1 final
checkpoint before enabling M3,following the plan section46 stopping rule.
The old20-step failure does not establish the trained-state ordering.
Do not infer effect collapse or DG failure from source mIoU alone.
