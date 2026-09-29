# Phase 16 bounded pilot handoff

Five-image slide evaluation producer: `59d522bcac07a73c4cd0b1cb1185983a389b7281`.
Operator reports exit 0. JSON SHA:
`a97165a09f67d49f38c11e82e9a8341f56720471148b0eabea02ae76ce4fcfc7`.
stderr SHA: `8e3c8dd0e4ad3d07513c6303a5e05ac69a36fd5978244e5d1b54597fbb5b55b2`.
Only the metric tail was supplied locally, so the next preflight must check the
complete hash-pinned JSON. The 20-update checkpoint predicts mainly sky/pole in
these five images; that is not a viable accuracy baseline yet. Metric NaNs for
classes with zero union are distinct from non-finite model scores.

Next: fresh seed-0, 500 optimizer updates / 2,000 source microbatches, accumulation
4, unchanged 40k-update PolyLR horizon and accepted typed upstream protocol.
One compact checkpoint (model, optimizer, scheduler) is saved and roundtripped.
Fifty original-resolution Cityscapes samples are evaluated without gradients
or target optimization. Official IoUMetric is checked against independent
confusion counts. No formal accuracy threshold or checkpoint selection applies.

No new mechanism, dependencies, frozen weights or datasets. No formal 40k run,
CQE transfer claim or exact-resume claim. Source-only REIN baseline accuracy
remains unverified. The supported DINOv3 method is unaffected.

Remote runtime remains Python 3.10 / torch 2.0.1+cu118 / xformers 0.0.20,
with the previously pinned OpenMMLab packages and upstream REIN SHA.
Current 9.6 GiB free covers the one-checkpoint budget; minimum free-space gate
is 2 GiB and maximum checkpoint size is 1 GiB. No artifacts are deleted.

Local verification: 167 CPU tests pass on Python 3.10.20 / torch 2.10.0+cu128
(no GPU execution), using the existing PyTorch interpreter and base Anaconda
pytest module. `git diff --check` passes. Bash execution and all OpenMMLab,
real-data optimization, target inference and memory measurements await AutoDL.
