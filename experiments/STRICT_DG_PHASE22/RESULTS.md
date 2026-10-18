# M0/M1 formal seed0 source result

Producer: `2537653ae9be0d84afbb86e22b5ec2200d280885`.
Accepted locally on2026-10-18 from the returned compressed reports.

| Variant | Updates | Source holdout images | Source mIoU (%) |
| --- | ---: | ---: | ---: |
| M0 Static R2 |40000|2497|73.8861626669|
| M1 centered Static R2 |40000|2497|73.8541625141|

M1-M0=-0.0320001528 percentage points. This is a single-seed source-domain
result, not a target-domain result or evidence of a generalization gain.
Both start at head hash e15b11e3b4d645fc17d936576a5d1fee49b418ce30d5ef265d899788f72d6c7a,
use identical frozen weights/splits/runtime and select fixed iteration40000.
Backbone/training isolation is recorded by the producer. No target data
accessed; no checkpoint/method chosen on target feedback.

Returned pair SHA256:49b13eed277266542f79d59ff0d7c7fa3edc4eb6d78cee53f1f8292ab79b9b64.
Original checksum file SHA256:07fbe0f7b6a34bff33151df50ea50d0d5f8be70c6b7f87dc1222169fa68acf7a.
Individual report bytes reconstructed from the embedded objects reproduce
both original individual checksum entries exactly. These small metadata
files are stored with canonical LF endings. All19 per-class IoUs and mIoUs
independently recompute from the returned integer confusion matrices.

Checkpoints, full traces and complete run directories remain on AutoDL;
they have not been locally downloaded or replayed. Earlier Phase21 readiness
remains user-assumed, not retrospectively claimed as a measured replay.
Next phase is P1 prediction-consistency formal control, following section27
and the first-round seed0 full-budget experiment list in section31.
