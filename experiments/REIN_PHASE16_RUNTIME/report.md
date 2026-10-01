# Phase 16 feasibility and runtime acceptance

## Accepted formal source-only baseline and current boundary

The saved-evidence audit at project commit `1da98b5` accepts producer
`d6fc52c`: 40,000 optimizer updates / 160,000 microbatches, final fixed-checkpoint
Cityscapes mIoU `0.656051`, first/last-20 source loss means
`122.297790/20.474319`, peak reserved memory `4.217 GiB`, and 40 checkpoint
history records. The final checkpoint SHA256 is
`84231e98dda68ac4b1fd4f59cc97881887a614de01b01e2697323c1eac80daf2`.
This closes the prerequisite REIN reference; CQE is still absent.

The next controlled addition is only the explicit class-specific residual
branch described in `docs/PHASE16_REIN_TRANSFER.md`. Its first GPU execution is
a 20-update engineering smoke over a frozen baseline. It is not an accuracy
experiment and does not authorize CQE or a full branch run.

## Archived boundary: saved-runner audit before first formal baseline

Runner `7a3a1c699f5b00ed4b6cc76055457beec374e05d` returns exit 0, with
last/previous checkpoints each 283,336,151 bytes. Only a metric tail was supplied;
full integration acceptance remains CONDITIONAL on the new read-only audit.
JSON SHA `b584a6504f75351864922b5f43119950bee9621b2cf30cf75bbb09be6d833a97`;
stderr SHA `22af8fbb574b10d70714716ad83dff186ec8c59708dcfeb0950abe79aa3bb5d5`.
Low five-image IoUs after 20 updates are not accuracy evidence. Missing-class
IoUs may be undefined, distinct from nonfinite logits/losses; saved metrics
must agree with confusion counts, with absent classes represented as null.

Next: audit existing report/metadata/80 records/two retained checkpoint hashes,
then, only if it passes, launch one fresh seed-0 adapted REIN source baseline.
40k optimizer updates = 160k physical-batch1 microbatches, accumulation4,
FP32, fixed 40k PolyLR, rolling saves every1k, final500 Cityscapes evaluation.
No CQE, new targets, target selection, pilot resume or exact-paper claim.
Data disk free7.7 GiB; do not clean historical evidence. Only newly generated
rolling states are replaced, keeping latest/previous. Completion and accuracy
remain pending; DINOv3 comparisons have backbone/head/exposure confounds.

Local CPU verification: 203 tests pass (Python3.10.20/PyTorch2.10.0+cu128,
pytest from base Anaconda site-packages). Existing short-evaluation API still
rejects full500 coverage unless explicitly opted in by the formal runner.
Tests reject incomplete/nonfinite/incorrect-producer integration evidence,
bad metric counts, and preserve an existing run on preflight rejection.
The 160k sampler-budget test spans six full 24,966-image source permutations
and ends at epoch6/cursor10,204 without exhaustion.
No local GPU, dataset or pretrained checkpoint was used; real long training
and full-target evaluation remain AutoDL work.

## Latest: fresh-process replay accepted; runner integration pending

Operator evidence at `1671f1a07441868e464776cfbf9305308187c289` passes:
two independent construction seeds 17/91 restored the saved seed-0 state.
Each worker's repeated update 21 passes; cross-process loss difference zero,
parameter difference `1.7881393432617188e-7`, optimizer difference
`1.0710209608078003e-8`, sampler and post-update RNG equal. Peak reserved
3.877 GiB each, exit 0. This is restore-replay verification only.
Report SHA `586e96ec63c481a9f9bf2b3c62754d96c98001123bc57fa34254f2361a9d01e7`;
stderr SHA `c6d89954053c0a21e6570b4c0dcd33ae594aaee4bceebfd648730ba068d4bca2`.
Snapshots SHA `4854edf61a2e644c49987a510640cb00178d7abe24b55ef5dd2c6067b14878eb`
and `ac9d68a4746fb4b747c58ee374edaafd3a82a55ac8ddf9ae097313d1676131d5`.

Next fresh-source runner gate: 80 microbatches / 20 updates, save at updates
5/10/15/20, keep latest and previous (20/15), five-image diagnostic only.
No previous checkpoint resumes this run. Data-volume 8.3 GiB free is sufficient;
no cleanup. Formal 40k/CQE transfer remain pending, and no REIN-versus-DINOv3
accuracy comparison can be made from the short pilot or integration gate.

Local verification: 192 CPU tests pass (Python 3.10.20 / PyTorch 2.10.0+cu128,
pytest loaded from the base Anaconda site-packages). No local GPU/data/weights
used. Focused tests cover atomic two-file rotation, pending-write preservation,
RNG invariance during saves and accepted-evidence rejection. Existing sampler
tests cover epoch wraparound; the 20-update GPU gate itself does not span an epoch.

## Current boundary: independent-process restore replay (pending AutoDL)

Same-process continuation accepted from operator evidence at
`5fee730e98d0a62535d93dd310a7a6dd334a36a3`: 80 microbatches / 20 updates,
two update-21 diagnostic replays; source indices `[22892,5750,13022,5583]`,
loss difference zero, parameter maximum difference `1.2759119272232056e-7`,
optimizer maximum difference `1.3969838619232178e-8`; peak reserved 3.920 GiB.
Report SHA `bec27b5dfa5a23e09915993ac13c8a403186314d55d03bb1c096d9ae5d11c82e`;
stderr SHA `3b020f67b5f578e7c8ae11e887061a2cd067c893946df362d3c8a5115adc7aa2`;
checkpoint SHA `e160cb17c3a708d2592016961bcde1eaa061abdddb5d9e0980dfe2d823e84678`
(283,344,599 bytes). The earlier failed attempt remains excluded and preserved.

Next gate builds two fresh processes with distinct construction seeds and
restores the same accepted model/optimizer/scheduler/sampler/RNG checkpoint.
Each replays update 21 twice; parent compares update-21 snapshots, including
post-update RNG. Only 16 extra source microbatches; no target evaluation,
formal training or CQE. This tests fresh-process restore replay, not full
uninterrupted-versus-resumed equivalence. No accuracy claim is added.
Operator disk free 8.8 GiB covers two ~270 MiB audit snapshots; no cleanup.

Local CPU suite: 189 tests pass. GPU execution, pinned OpenMMLab model
construction and remote dataset/checkpoint use remain AutoDL-only.

## Archived earlier evidence and handoffs

## Adapted protocol data gate (pending)

Repaired inventory exit 0 at `ede170c4fa2c5c330581771e9cf5c4f659360e9d`.
JSON hash `2cdf617c9370735ac275c4875af52f3a017898dd690ff3cc793298322298e64a`;
stderr hash `df377c839408bd0115d8ec0bc1708542b1ac7b3ff7f59be67e800ebbcab302c6`.
Only the report tail was supplied; next gate verifies full saved JSON before use.
The versioned protocol adapter resolves project paths/train IDs, narrows datasets
and evaluator to Cityscapes and declares batch/clip/crop/storage deviations.
Next AutoDL action checks path pairing and ten pipeline samples without model,
GPU or optimizer. Formal scheduled training and checkpoint loading stay pending.
Local verification: 156 CPU tests pass with the existing PyTorch interpreter
and base-Anaconda pytest workaround. Tests cover nonmutating adaptation,
single-target isolation, batch accumulation, valid-pixel crop fallback and
train-ID rejection. OpenMMLab dataset execution remains AutoDL-only.

Scope: GTA5 to Cityscapes only. Evidence supplied by the AutoDL operator;
no local GPU results are claimed.

## Accepted real-data smoke audit

The subsequent protocol inventory at `4e6a373f803a547a7044e67b45e87c5b4b2ad4b2`
failed only after effective config parsing: our tool assumed a nonexistent
`rein.datasets` package. No installation or training failure is inferred.
JSON/stderr hashes `e0cf9c211494fe91750c18491fbdf8c6fbf38a244362935b4c3fc0ff4d62fad2` /
`df377c839408bd0115d8ec0bc1708542b1ac7b3ff7f59be67e800ebbcab302c6`.
The repair reads the actual MMSeg 1.2.2 dataset/base/annotation/crop classes
without constructing them. Preserve failed reports; refreshed inventory remains
the next AutoDL boundary. Effective GTA5 suffix differs from local `.png`, and
both upstream multi-target dataloader and DG evaluator need explicit narrowing.
Recovery verification: 153 CPU tests pass, including a regression that supplies
MMSeg exports without any `rein.datasets` package. Installed class inspection
is pending AutoDL; no local OpenMMLab environment was built or modified.

Saved-report audit source `b85f105384d07c88f23f89bec1723aa6a6c92c7e` returns
true and exit 0 for producer `17883de0c87abe3ed9bd051b260ba20bc259999e`.
Twenty optimizer steps and five target predictions; maximum objective error
`1.7642974853515625e-05`; normalization round-trip error
`4.76837158203125e-07`; peak allocated/reserved `3.537/3.881 GiB`.
This also accepts adapter-change and frozen-patch invariance producer checks.
The hashes in the historical excerpt below are matched by the audit.
Disk remains 9.9 GiB. No accuracy claim or formal training authorization.

Next boundary: inspect effective configuration and dataset source in the pinned
AutoDL REIN checkout. The [official pinned entry configuration](https://github.com/w1oves/Rein/blob/dc063429c4dadc0da9c6252b3db22fc55a9882ab/configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py)
declares batch 4, appearance augmentation, parameter-wise decay and 40k PolyLR.
The project smoke is not that protocol. Source paths/train-ID mapping, full
target evaluation, single-GPU effective batch and checkpoint storage require
explicit adaptation before claiming a comparable baseline. The next remote
inspection does not build a model or decode any dataset.
Local verification: 152 CPU tests pass with the existing PyTorch interpreter
and base Anaconda pytest workaround. Upstream configuration parsing remains
an AutoDL-only check; no local REIN runtime or weights were loaded.

## Real-data execution excerpt (complete report audit pending)

Operator source `17883de0c87abe3ed9bd051b260ba20bc259999e`; exit 0;
20 contiguous finite loss/norm records with positive valid source pixels.
First/last loss `139.721939/83.330597`; these are different crops and not
an accuracy or convergence result. Maximum printed pre-clip norm `1161.029907`;
the producer clips to 1.0 before every optimizer update.
JSON SHA256 `9ed6e11b7e66e481aac9ea7a7d8b2a8f8d090b592acfa91c97d71d7b8d2a8ab5`;
stderr SHA256 `186be4d21cfb4b3ae1f84d5c08b198ab37404df762bb7b4079c261218b31f2c1`.
Disk free 9.9 GiB. Meshgrid deprecation warning is nonfatal.
Complete target/normalization/frozen-weight/peak-memory fields were not supplied,
so full acceptance and formal training remain pending. Next remote action reads
and audits those existing files without GPU execution or artifact mutation.
Local verification: 150 CPU tests pass using the existing PyTorch interpreter
with pytest loaded from base Anaconda site-packages (same workaround as before).
The new fixtures are fabricated schema tests, not remote experiment evidence.

## Feasibility assessment

Engineering remains feasible: the pinned DINOv2 conversion passed exact
343-tensor audit; the isolated REIN runtime now passes GPU NMS, xformers
attention and model registration. This establishes runtime compatibility,
not full-model or optimization compatibility.

The empirical project remains useful: selected Static R=2 yields final
Cityscapes mIoU `0.652849 ± 0.015138` across three seeds. The two-size study
consistently favors ViT-L in accuracy and normalized effect stability.
These are bounded protocol results, not a scaling law or causal identification.

The original central claim is not validated: A4 CQE lowers normalized style
effect variance by 99.47% but loses 2.95 accuracy percentage points versus A3.
Query diversity fails retention; learned-null worsens localization despite
seed-0 segmentation retention. REIN transfer therefore remains an exploratory
test, not an assumed positive result. Do not add CQE until a credible REIN
baseline is established. Repeated use of Cityscapes validation for development
and the single target constrain generalization claims.

## Accepted runtime evidence

- Project source: `0e0ef7c7bb1588cb0ba190c652fbce926c257121`.
- Upstream REIN: `dc063429c4dadc0da9c6252b3db22fc55a9882ab` (clean).
- Python 3.10.21; torch 2.0.1+cu118; torchvision 0.15.2+cu118;
  numpy 1.26.4; mmcv 2.1.0; mmengine 0.10.7; mmseg 1.2.2;
  mmdet 3.3.0; xformers 0.0.20.
- RTX 4090 D; CUDA 11.8; CUDA NMS indices `[0]`.
- Attention max absolute error `0.0005131959915161133` <= `0.005`.
- Backbone/head registered: true/true; runtime `ok: true`.
- Report SHA256: `2d3498fc8343927da29dfa2e73bb5fb89ad46132248d15e6a323cdef4adcdfcb`.
- stderr SHA256: `d35e74535a57ec10102ddfa16054f5f3f7d61636b7cf166085c58f6ab6fec8f2`.
- Installed freeze SHA256: `47f0e84c38075e5c2f70ffba58746844e0cc8f206f430f2b295fb928008e3245`.
- Available disk approximately 9.9 GiB; no formal training authorized yet.

## Accepted backbone evidence

- Project source: `ed3dcfe202435f4d7893681054473724e19e1ad6`.
- Upstream and all eight runtime versions unchanged from accepted runtime above.
- Converted weights SHA256: `91730ebf59fb634f5572cf5071fef8665473dcffcbef7ba4f4fa497533a8c837`.
- 343 loaded tensors; 2,990,081 trainable adapter parameters.
- Feature shapes: `[1,1024,128,128]`, `[1,1024,64,64]`,
  `[1,1024,32,32]`, `[1,1024,16,16]`; linked queries `[100,256]`.
- Synthetic objective `8.531936645507812`; 11 nonzero adapter gradients;
  frozen-backbone gradients absent. This is not a segmentation loss.
- Peak allocated/reserved memory `2.767/2.982 GiB`; exit 0 and `ok: true`.
- Report SHA256: `317dff2553f4674f9363d8660ce369245863676866fa0cf5ed0745db1bb53706`.
- stderr SHA256: `c4775f166f11b2cbb18b26ecb5487133076edf537e05943acf6e29501284ed1f`.
- Disk available 9.9 GiB. Optional ConvNeXt and TypedStorage warnings are not
  failures of this gate.

Next boundary: synthetic full Mask2Former prediction/loss/backward on AutoDL.
The pinned upstream model config is used unchanged except disabling automatic
pretrained initialization, followed by our safe explicit checkpoint load.
The test has no dataset/optimizer/checkpoints/CQE; real protocol and training
remain pending. MMSegmentation interfaces were checked against
[v1.2.2 EncoderDecoder](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/segmentors/encoder_decoder.py)
and [Mask2FormerHead](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/decode_heads/mask2former_head.py).

Local verification: 142 CPU tests pass using the existing local PyTorch
interpreter with pytest from the base Anaconda site-packages; plain
`python -m pytest` in that interpreter could not find pytest. No OpenMMLab
model/real checkpoint was loaded locally. Bash execution and full GPU model
execution are pending AutoDL verification.

## Accepted full-segmentor synthetic evidence

- Source `86fb3ffdc7a13285965f0df825472625f9e52ed4`; pinned upstream/runtime/weights unchanged.
- Upstream model-config hash `2f51a8ac089848f5e276028801983c95bf5f89e688753c1fff5a9e64bb766868`.
- 343 loaded tensors; 23,569,877 trainable parameters; 19-class scores `[1,19,512,512]`.
- Float32; all 30 losses finite; total `124.92772674560547`.
- Nonzero adapter/head/pixel-decoder gradient counts `11/291/116`; no frozen gradients.
- Peak allocated/reserved `3.400/3.662 GiB`; `ok: true`, stage complete, exit 0.
- Report hash `87ba441bb9f879fd4048c18518477e8294c06dfa27335f7811954bd0c7552301`.
- stderr hash `05a9a5cc9f11d36804cb7a20fa4832afbe547937dde1d678284a69e28cd2bca3`.
- 9.9 GiB disk free. Synthetic loss is NOT an accuracy result.

Next: 20 GTA5 optimizer updates and five Cityscapes center-crop inference
diagnostics in float32. Uses existing project preprocessing rather than
claiming exact upstream training reproduction. The inverse RGB-to-BGR bridge
must round-trip through upstream preprocessing within 1e-5; target labels
never enter optimization. No checkpoints/mIoU/CQE. Formal source-only schedule
and full-target evaluation remain pending. Local CPU tests: 145 pass; actual
isolated-runtime dataset imports and optimizer GPU memory await AutoDL.

## Frozen-REIN class-query residual smoke (Phase 16)

The accepted source-only REIN seed-0 reference at `d6fc52c` has fixed-final
500-image Cityscapes mIoU `0.6560509975`. A separate, explicitly class-indexed
logit-residual branch was then tested at producer `14d3e5a` with the REIN
segmentor frozen. The upstream 100 Mask2Former queries were not relabelled.
The no-CQE source smoke completed 80 GTA5 microbatches / 20 optimizer updates
with accumulation four. The producer reports finite losses, branch gradients
for `alpha`, `query_bank` and `pixel_projection.weight`, final scale
`0.0019659693`, unchanged base parameters, zero effect on nonselected classes,
and selected-class effect error `5.96e-8`. Peak reserved memory was 2.055 GiB.
The branch-only checkpoint is 17,887 bytes.

Report/stderr/checkpoint SHA256 are respectively
`5b566bd4d764109e71a26b4da7abd10f267ad627377de0340ce3c616e1266510`,
`ebec21c1e0c8bd257fa9353337304a3c96e6d7361e7520ba6d43b3711bf040b2`,
and `fade8f3893ca1b8e2559ec154450ad3cb6ff9efee84b17354e6f70149c68d852`.
An independent saved-evidence audit is the next gate. This smoke contains no
target mIoU, CQE mechanism or accuracy claim; it does not authorize a formal
branch run or a causal conclusion.

The smoke's independent saved-evidence audit passes at Git `161f077`. It
checked 80 unique source indices, the 20 optimizer-update boundaries, all
finite trace values, report/metadata agreement, loss summaries, class-isolated
intervention, unchanged base parameters and branch checkpoint SHA256. The
audit artifact SHA256 is
`517ea9e74a1d2ad999182e957b1d20c8588d8d08fec1b70ac4eb14043fd1011b`.

Next is a matched, bounded no-CQE/CQE engineering smoke. Both arms use fresh
identical branch initialization, the same 80 GTA5 records and per-step hashed
original/photometric inputs. Each optimizes mean two-view segmentation loss;
the candidate alone adds the existing normalized CQE objective at weight 1.0.
This is not Cityscapes evaluation, a metric comparison, or formal training.
The first attempt stopped before updates on compact-checkpoint coverage. The
producer now invokes and validates REIN's `train(True)` trainability hook
before restoring the frozen source baseline; failed artifacts are retained.
The next retry stopped on the first batch because PyTorch 2.0 rejects tuple
dimensions in `Tensor.any`. The CQE class-presence reduction is now sequential
and covered for multiple samples and ignored pixels; no optimizer updates
occurred in the failed attempt.
The corrected paired smoke completes at `3644785` with 20 updates per arm,
80 identical matched inputs, finite objectives/gradients, unchanged base
parameters and successful checkpoint roundtrips (2.430 GiB peak reserved).
The first saved audit failed only on float64-versus-float32 scalar
reconstruction; all producer-pinned evidence hashes matched. The verifier fix
at `f18c2c2` now passes the read-only saved-evidence audit; v2 audit JSON SHA256
is `083191fcfe702928f89d5d346e8bcd01525fca53cafc40017bd21bd2bca957da`.
There is no target metric or accuracy conclusion. A full 40k-update-per-arm
comparison requires a separate decision; the smoke artifact explicitly does
not authorize formal training.

## User-approved seed-0 CQE performance comparison (implementation ready)

The user subsequently approved the formal test to answer whether CQE improves
performance in this REIN transfer setting. The experiment is a matched pair,
not a comparison against a different model: both arms start from the accepted
frozen REIN source-only baseline and train the same small class-query residual
for 40,000 optimizer updates / 160,000 GTA5 microbatches, using identical
source order and per-step original/photometric tensors. The only objective
difference is the candidate's existing normalized CQE term (`lambda=1.0`).
Both fixed-final checkpoints will be evaluated on the full 500-image
Cityscapes validation set, without target-based checkpoint selection.

The producer and read-only audit are implemented locally, with paired
image-bootstrap analysis and exact artifact/source-schedule checks. Local
focused tests pass. Full `pytest` cannot collect on this Windows environment
because PyTorch is not installed; this is an environment limitation, not a
test assertion failure. GPU training and full target evaluation have not yet
run. Stop at the AutoDL boundary and follow the exact-commit commands in
`docs/AUTODL_NEXT.md`.

Interpretation limit: a positive seed-0 delta with an image-bootstrap interval
above zero is encouraging but not seed-general evidence. The interval is
conditional on these two trained models; a second paired training seed is
needed before claiming robust improvement. mIoU performance and effect
stability are distinct endpoints.

The first user-approved formal launch used project revision `8c1396c` and
exited before experiment initialization with `ModuleNotFoundError: tools`.
The report is empty by construction; there was no run directory, data/model
load, CUDA work, training record or target evaluation. Preserve the report,
stderr, exit marker and launcher log as excluded failure evidence. The launcher
now uses repository module entry points plus an import preflight. This is an
execution-only repair and does not change either paired arm.
