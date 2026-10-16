# Phase 17: local source membership verification

Received `source_split_export/` as extracted metadata files. Both actual manifest
hashes match the operator-pinned values; report checksum is
`d9b029a46fe2ba48ba7f70eee6e38edb416ca08327c17150c763e44f1d1f24da`.
All checksum entries, report/producer identity, counts, canonical LF bytes and
unique/disjoint membership pass. Independent recomputation of the versioned
SHA256 ranking yields exactly the returned train and validation lists.

Verified counts: source24966, train22469, val2497; intersection empty. The
exact returned ID spellings are preserved. No consecutive range was assumed.
`manifests/gta5_source_train.txt`, `gta5_source_val.txt` and provenance
`gta5_source_split.json` are ready for permanent Git freeze. `.gitattributes`
forces LF for frozen files so checkout cannot change the pinned digests.

The original ZIP itself was not received, so its bytes/hash were NOT locally
verified. Provenance marks `input_format: extracted_directory` and
`zip_bytes_verified: false`. The ZIP hash remains operator-reported. The
actual manifest bytes and report fields were verified directly; no data or
weights were accessed. Existing model/loss/trainer behavior remains unchanged.

Source split is fixed across seeds by the salt/rule, not a scene-level
independence claim. This is a protocol gate only: source/target mIoU, effect
statistics, localization and responsibility remain unmeasured. Next verify
frozen membership against actual AutoDL payloads and label/geometry validity,
then integrate the source-only development route. No training authorized yet.
