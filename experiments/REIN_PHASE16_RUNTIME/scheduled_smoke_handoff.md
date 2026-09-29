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
