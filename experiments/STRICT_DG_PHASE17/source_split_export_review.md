# Phase 17: source split export review (2026-10-16)

Evidence: operator-pasted complete console output. No real ZIP attachment was
available locally when this review was written. Files found in ignored test
directories are synthetic fixtures and must never become dataset manifests.

- Producer: `57a724428b60d0f6636db056071bc620d45f45e0`; clean Git status.
- Python 3.12.3, `/root/miniconda3/bin/python3`, Linux x86_64/glibc 2.35.
- Inventory hash: `7224adc8aaafbcaac412aa5ee3904c11ba10d3c81aac7dffea231259e21e0d25`.
- Pair-record hash: `92a67a0e32a297d2f41618c6919a0573cfde6ec23890b5dd9a6ae7353842662f`.
- Source count 24966; train22469; val2497; overlap0; exact partition true.
- Fixed salt `causalq-gta5-source-split-v1`, f=0.1, ceil(N*f), same rule across
  training seeds. Report flags retain no dataset reads, no target access,
  no training and no Git freeze.
- Train SHA256: `c33c1316eb1e547199aa0be79f749d6af9a7c8a39d6ab40313e410b1fe021ac9`.
- Val SHA256: `57dec87d8081f50162e29738a7052cd014404d23bf6cd2268c77b23bc0f011cb`.
- ZIP SHA256: `a555c740546a992ec72199f314a750699acf3782ceda1c98e9cf8908206b96db`.
- Log SHA256: `5daa7c9068bb6b5019bb5af675d2c77e09c049b15c9ec0278679ae47bb8b45c6`.
- Output: `/root/autodl-tmp/outputs/CausalQ_DG/phase17/GTA5_SOURCE_SPLIT_57a7244_20261016T140258_54816/`.
- Exit 0, ok true, train/val/report checksum checks all OK.

| Question | Evidence / limit |
|---|---|
| Export engineering gate | Operator reports success with expected counts and digests |
| Independent local byte/membership audit | Pending real ZIP transfer |
| Source/target mIoU | Not measured |
| Effect variance/magnitude, localization/responsibility | Not measured |
| Scene/sequence independence | Not established by filename fallback |

Decision: do not repeat the successful AutoDL export. Transfer the existing
metadata ZIP. `tools/import_gta5_source_split.py` now checks the operator-pinned
ZIP and manifest hashes, archive/checksum/report contract, canonical ID bytes,
disjoint/exact membership, and independently recomputes the versioned split
before creating any manifest directory. Then stage/commit exact manifests and
provenance locally. Hash strings cannot reconstruct missing ID bytes. Do not
guess a consecutive numeric ID range, substitute fixtures or claim frozen
membership before those actual files enter Git. No model hypothesis is tested
by this export; strict training and source pixel inspection remain pending.
