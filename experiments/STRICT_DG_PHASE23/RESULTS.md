# P1 formal seed0 source result

Producer:`268fda1b340f8bd5ceded83f9d9b9c6bf0009c07`.
Accepted locally2026-10-18 from returned report/checksum bytes.
40000 updates,2497 frozen GTA5 holdout images,fixed-final checkpoint.
P1 source mIoU=73.84414154795%;M1=73.85416251410%;
delta(P1-M1)=-0.01002096615 percentage points.
No single-seed source improvement or target-domain conclusion follows.

Report SHA256:9e029d792319a823675a233ad850804f754c0700ef1389f46cfae34796e78d1f.
Checksum SHA256:fa22fa5a6dcfd15f1c29495872d89e538d89e57f7aed3c2717e303f1e6797ca7.
Both files stored unchanged with canonical LF endings. Exact frozen config
and reserialized embedded metadata reproduce the two other checksum entries.
Integer19x19 confusion counts independently reproduce all class IoUs/mIoU.
Initial head matches M0/M1; recipe,runtime and budget match the released P1.
No target access. Full traces/checkpoints remain on AutoDL and have not been
locally replayed. Next is M2 Full CQE control,following the plan's first-round
seed0 full-budget ablations. No P1 weight tuning or short-run repeats.
