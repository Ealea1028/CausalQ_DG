# Phase 16: source schedule/checkpoint boundary

Accepted operator evidence: adapted data protocol producer
`b645793ff8fcdd12b0cdd8078b64dc929f107a70`, exit 0, report SHA
`b893b991c4bc85eb6105e8d8ae1f39df670b256ce12a10aaca8c1a7c89d71c69`.
The complete JSON is checked remotely; local code does not assume access to
the data volume. Optional ConvNeXt and deprecation warnings are not failures.

Implementation adds one bounded integration gate, not a learning mechanism:
80 physical source batches, accumulation four, 20 optimizer updates, the
accepted paramwise AdamW, 40k-update PolyLR, clip norm one and float32.
Schedule units are explicit; future formal training needs 160k physical batches
for 40k updates. Batch accumulation is an adaptation, not exact four-GPU REIN
reproduction. Target labels never enter optimization.

One compact checkpoint retains trainable tensors and all buffers plus optimizer
and scheduler states; frozen backbone stays external and hash-pinned. The GPU
gate tests parameter restoration and output roundtrip. Exact resumability and
accuracy are deliberately not claimed. Runtime memory and checkpoint size are
unknown until AutoDL executes. No formal run is authorized by this handoff.

## Failed first attempt and repair

At `d5558e1`, the runner failed to construct Mask2Former because JSON lost
recursive ConfigDict containers (`dict` has no `layer_cfg` attribute). Zero
optimizer updates occurred. Report SHA
`0984148e8dd906eb37d6517d61ee2f039a25f07940438888a38ad52f6493c164`;
stderr SHA `effe107293daf24abbf0caec31531b86488b4cb7689470ece011b3295af9c6fe`.
Restoring the accepted configuration through mmengine.Config fixes the caller
contract without changing model values or experiment protocol. CPU tests cover
the restoration boundary; no GPU success is inferred from them.

The `9ac1f80` retry successfully constructs the model, but dataset construction
fails because JSON also erased tuple types required by RandomCrop. Exit 1;
report SHA `8639e7def5339968bb8d9a46d8d8b78dffc3e5e27fcee561f5b134f6b6e2bbd8`,
stderr SHA `bd5ed92d5e2c8b6cce6f2d69d882e1ad26df3e41121857e7bda088ad6e00fd16`.
The current repair reconstructs the full typed pinned-source configuration via
the same adapter and rejects any JSON-normalized value difference from the
accepted report. Both recursive attribute semantics and all original tuples
are preserved, while ordinary lists remain lists. GPU integration is pending.
