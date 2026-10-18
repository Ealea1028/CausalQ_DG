# Formal M1 hypothesis stop;M3 not run

Producer:`148746b6e5d6a36b413a8b6d104652860cab5a36`.
Returned job SHA0184a3d57d88c284a5a18eda49ed121fb0de2a367b90de6ad228541652343460.
Returned checksum SHAa1ecb2cfb9155a4f12f490139f820c7d699f0470e08b70c30f392fbbb4aa356c.
Embedded stability report reproduces its original checksum entry exactly.

Accepted formal M1 iteration40000 checkpoint:
c2559a431830d6d9160f5c9e75bebe9b8676d73d983a4674a2f9659ce9c00af2.
2497 source images,4994 forwards,zero optimizer updates,read-only state per
producer. Raw anchor variance0.0009380707222222419;residual variance
0.0000339865702574878;ratio27.601217631413384.0/2497 images favor anchor.
The predeclared raw criterion failed. M3 was NOT started;no M3 score exists.

Returned certificate/hash/ratio/provenance are checked locally. The per-image
sample_index.jsonl remains remote at hashe5cf43fe7ec03bc9caf79f504a35eabf602c81812984f59d54e570af6583e140;
its raw reductions have not been independently replayed locally. Evidence
is descriptive for this single trained seed and fixed source/style setup.

Follow plan46/55:stop adding losses/optional mechanisms. Freeze completed
formal M0/M1/P1/M2 and final target evaluation policy. No unsupported M3/M4
or three-seed-M3 training. No target-feedback-driven redesign after evaluation.
