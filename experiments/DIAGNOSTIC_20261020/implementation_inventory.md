# Phase28 implementation inventory

2026-10-20; starting checkout `aa39d532a270b9859f51ae1846f74c9df34fba68`.
User requests executing `20261020_Diagnostic_Plan.md`, avoiding redundant
engineering gates and proceeding promptly to formal selective-stability runs.

Existing: formal40000-step M0/M1/P1/M2, exact checkpoints in the immutable
Phase26 freeze; source22469/2497 manifests; deterministic CE,1000-update
recovery,photometric bank,Full-CQE loss,component/responsibility diagnostics.
U0=M0,C0=M1,CP=P1,CF=M2. Do not retrain or repeat their target evaluation.

Missing: controlled uncentered anchor-only access; multi-view amplitude,
normalized variance,covariance/correlation; Query/direct-anchor gradient
attribution; global responsibility aggregation; traced intervention examples;
one resumable source-only baseline diagnostic launcher.

Phase28 implements those diagnostics and freezes the new matrix. It does not
change legacy M3 authorization or train new models. The new study explicitly
tests selective regularization without the old raw-stability ranking gate;
old negative evidence remains intact. Existing target feedback is known.

Local: implementation,synthetic CPU tests,small manifests,Git documentation.
AutoDL: original weights/checkpoints/source data; four old final models through
the new diagnostics; three fixed random intervention groups.
One background launch;no short pilot,repeated resume gate,legacy artifact
roundtrip,or target access. Next phase: UP/UF/UA/UR formal40000-step training.

Disk cleanup: read-only occupancy report plus documented candidates;preserve
the exact final checkpoints,backbone weights and source/target payloads.
Preserve unrelated user files/untracked reports and closed-study metadata.

Implementation complete:controlled protocol,ten planned YAMLs,multi-view
moments/correlation,read-only gradient attribution,weighted responsibility,
traced examples,resumable launcher and read-only disk report. Full CPU suite
619 passed/1 skipped;actual source/CUDA work is pending AutoDL. Per-commit
AutoDL worktrees avoid replacing source underneath a running worker.
