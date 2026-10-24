# Phase28 implementation inventory

2026-10-24 Phase30 follow-up;starting checkout0a0350d025b063cca2668ab0606bd80179f13ea3.
Received phase29_export/ with18 checked files,four40000-update source runs,
matched initial head/environment and full2497-source confusions. Source mIoU
UP/UF/UA/UR=73.8831/73.7590/73.8529/73.8912%;no target results. Independently
reduced integer confusion;no checkpoint/trace replay. Store small acceptance
only;preserve/ignore user export and other unrelated files.

Phase30 adds centered CA/CR frozen recipes and routes the existing selective
update/1000-step runner into the same background launcher and report exporter.
No new loss kernel,model parameter or training RNG draw. Centering flag plus
experiment identity are the only config differences fromUA/UR. C0 initialization
and source/runtime policy matched. Old M3 guard and Phase29 freeze untouched.
Next GPU action is two full40000-step formal controls;no standalone pilot,
diagnostic rerun or target access. Stop after source reports;Phase31 follows.

## Original Phase29 inventory

2026-10-23 Phase29 follow-up;starting checkout5ef6b30e726eb1ace7d89cdb506547133c267155.
Received phase28_report/ contains51 checked export files. All four summaries
report2497 samples,zero updates,unchanged model/RNG,no target access at1086bc3.
Compact acceptance and interpretation are committed;raw exported images/CSVs
remain local and full raw per-image records remain remote. No diagnostic replay.

Reused:two-view KL/Full updates,controlled uncentered component API,source-only
formal runner,deterministic CE,exact1000-step recovery,same seed0 initialization
and source manifests. Added:exclusive Anchor/Residual CQE update selector,
four frozen formal YAMLs/initialization-sampling contract,Phase29 background
mode and reports-only export. No new model parameters or constructor RNG draws.
Initial/final query norms are observed outside optimizer updates. Old M3 guard,
Phase28 planned diagnostic YAMLs and original final freeze remain unchanged.

Next real operation is AutoDL formal UP/UF/UA/UR40000 each,then stop for returned
source reports. CA/CR are plannedPhase30,not implemented/launched in this release.
No additional short GPU gate or target evaluation. Preserve unrelated user files.

## Original Phase28 inventory

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
