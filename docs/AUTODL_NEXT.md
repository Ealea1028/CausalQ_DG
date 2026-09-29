# AutoDL next step: Phase 16 bounded continuation state gate

The saved pilot audit PASSES at `3601ecd0ccc5425de3080d44ccca4186d72aebe7`:
2,000 microbatches, 500 updates, 50 targets, diagnostic mIoU 0.2351890098,
checkpoint roundtrip zero, checkpoint 283,250,650 bytes, peak reserved 4.217 GiB.
Checkpoint SHA `b9747a71b1871e201a7730d7b92ca1daf836c7fdc0bcf61b06bd2c9b40fbfda0`.
Formal accuracy/CQE transfer remain unverified.

Storage decision: pip cache 8 MB and conda disposable tarballs 161,065,321 bytes
are on the system overlay, NOT `/dev/md0`. Cleaning them will not increase
the data-volume 9.3 GiB free. Most older full runs already retain only finals.
The three ViT-B runs (2.4 GiB each) and learned-null run (3.5 GiB) could contain
intermediate checkpoints, but must not be blanket-deleted. No cleanup needed
for the next gate; preserve all datasets, environments, weights and evidence.

Next save a serializable source permutation/cursor/generator and primitive-only
Python/NumPy RNG plus CPU/CUDA RNG tensors with the existing compact checkpoint.
Bounded gate: fresh seed 0, 80 microbatches / 20 updates, unchanged 40k PolyLR,
then TWO replays of the next four microbatches / update 21 from the SAME saved
disk state. Extra 8 microbatches are diagnostic branches; neither is committed
to the retained 20-update checkpoint. Check source indices, losses, parameters,
optimizer moments, scheduler and sampler agreement, then restore the saved state.
It does NOT verify fresh-process resume or exact bitwise GPU determinism.
No target evaluation, CQE, formal training, environment installs or deletion.
One checkpoint ~270 MiB, hard cap 1 GiB, minimum free 2 GiB.

```bash
(
cd /root/autodl-tmp/CausalQ_DG || exit 1
export OMP_NUM_THREADS=1
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
test -z "$(git status --porcelain)" || { echo 'dirty_worktree'; exit 1; }
git fetch origin main || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
bash scripts/check_rein_continuation_phase16.sh
RESULT=$?
echo "continuation_gate_exit_code=$RESULT"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
)
```

Return complete JSON, log tail, hashes, exit and disk. Pass: ok true, complete,
optimizer_updates 20, continuation_replay_ok true, replay_count 2, extra microbatches
8, matching indices, maximum loss difference <=1e-4, maximum parameter difference
<=1e-5. Optimizer floating tensors allow atol 1e-6 / rtol 1e-5; integer states and
sampler are exact. Stop for review before formal runner/fresh-process continuation.

## Archived pilot audit and inventory (do not rerun separately)

Pilot producer `b4e292d8da87d186f03ac4a4769f7464a2499ef8` exits 0.
Report SHA `c90169ad57ea39a937bbd2760aff3122758399c28dc6138a6822689212fc789a`;
stderr SHA `83ac1b99558f5fde68b18b5a7f39f5520569d083bf1a91319d472dcfa490ca43`.
Only the per-class metric tail was supplied. Do not rerun the pilot or declare
formal accuracy accepted. Next audit the complete existing report, metadata,
summary, 2,000 trace records, 500 accumulation boundaries, update-based LR,
checkpoint path/size/SHA, roundtrip and 50-image confusion/metric agreement.
No GPU, checkpoint deserialization, image decoding or optimization is required.

Storage cleanup priority:

1. Inventory pip download/wheel cache and conda package tarballs/index cache.
   Use conda dry-run, not `--force-pkgs-dirs` or removal of the package directory.
2. Compare cache mount identities with `/root/autodl-tmp`: freeing system-disk
   cache may NOT increase the data-volume free space.
3. If still necessary, inspect completed experiments' intermediate checkpoints.
   Preserve final, useful best checkpoints, full metadata/trace/reports/logs,
   active jobs and all checkpoints referenced by analysis reports. No blanket
   deletion of `outputs`, `pretrained`, `envs`, raw datasets or staging evidence.
4. Only after the exact space inventory is returned prepare explicit cleanup
   targets and formal runner storage/resumption policy. No cleanup happens in
   this script. Current 9.3 GiB is sufficient for this read-only audit.

```bash
(
cd /root/autodl-tmp/CausalQ_DG || exit 1
export OMP_NUM_THREADS=1
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
test -z "$(git status --porcelain)" || { echo 'dirty_worktree'; exit 1; }
git fetch origin main || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
bash scripts/audit_rein_pilot_storage_phase16.sh
RESULT=$?
echo "audit_and_inventory_exit_code=$RESULT"
)
```

Return the entire compact audit JSON and storage inventory (a text attachment
is fine). Audit pass is engineering evidence only, not a CQE claim, exact paper
reproduction, formal training authorization or accuracy acceptance. Stop here
before formal 40k updates / 160k microbatches; that runner remains unimplemented.

## Archived 500-update pilot instructions (do not rerun)

The operator reports slide evaluation exit 0 at
`59d522bcac07a73c4cd0b1cb1185983a389b7281`.
Report SHA `a97165a09f67d49f38c11e82e9a8341f56720471148b0eabea02ae76ce4fcfc7`;
stderr SHA `8e3c8dd0e4ad3d07513c6303a5e05ac69a36fd5978244e5d1b54597fbb5b55b2`.
Only the metric tail was supplied; the pilot must verify the full saved report
hash, ok/complete state and all five original-resolution geometries before
optimization. Low scores after 20 updates are not accuracy acceptance.

Run only a fresh seed-0 source pilot: 500 optimizer updates, 2,000 microbatches,
accumulation 4, unchanged 40k-update PolyLR horizon, float32, pinned isolated
runtime, unchanged accepted model/data protocol. Then restore one compact
checkpoint and diagnose the first 50 Cityscapes-val images with standard slide
postprocessing and independent versus official metric counts. No CQE, target
optimization, model selection, formal 40k training or exact-resume claim.

One checkpoint is expected near 270 MiB (hard cap 1 GiB); no datasets or frozen
weights are copied. Require at least 2 GiB free; current 9.6 GiB is sufficient
for this bounded action. Preserve all earlier reports/checkpoints. No cleanup.

```bash
(
cd /root/autodl-tmp/CausalQ_DG || exit 1
export OMP_NUM_THREADS=1
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
test -z "$(git status --porcelain)" || { echo 'dirty_worktree'; exit 1; }
git fetch origin main || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
bash scripts/train_rein_pilot_phase16.sh
RESULT=$?
echo "pilot_exit_code=$RESULT"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
)
```

The outer subshell prevents a failing preflight from closing the interactive
terminal. Return full pilot JSON, stderr tail, hashes, exit and disk. Acceptance
requires ok/complete, 2,000 contiguous finite training records, 500 updates,
checkpoint roundtrip <=1e-5, 50 original-resolution predictions and metric
agreement. The diagnostic mIoU is reported without a score acceptance floor.
Stop here until AutoDL returns this evidence.

## Archived five-image evaluation instructions (do not rerun)

The scheduled source gate PASSES at
`3ca67315bc4f1dc91f1891de3f305e3162a0a53a`: 80 microbatches, 20 updates,
float32, 23,569,877 trainable parameters; reload prediction difference zero;
checkpoint 283,250,458 bytes, peak reserved GPU memory 3.920 GiB, exit 0.
Accepted report SHA `ea528c623bd071a7aeb8214e2b1434073fe4ef00055478fc073fd482c3abbfad`;
stderr SHA `8a94f49175dfefd9b611a9cbd8914470b74f2df0549355132c7f652ce89809db`;
compact checkpoint SHA `d0cf1dd59fd7eb2ee9e2aa280b47e03819b8d26d814f12ab9f6162f7bcd410af`.
Preserve this checkpoint and frozen pretrained weights. Do NOT rerun training.

Next load the accepted frozen backbone plus exact compact checkpoint in a fresh
process. Verify complete trainable tensor/buffer coverage and the same typed
adapted protocol. Run ONLY the first five Cityscapes-val samples (dataset still
contains all 500). Input resize stays 1024x512, model policy stays slide crop
512x512/stride341. Use the standard model.test_step and postprocessing to
restore predictions to ORIGINAL 1024x2048 GT geometry. Check finite logits,
unchanged target GT, ignore-255 handling, and official IoUMetric counts against
an independent 19x19 confusion matrix. Five-image mIoU is diagnostic only:
no score threshold, no model selection, no accuracy or transfer claim.

No optimization, installation, copied datasets, prediction images or new
checkpoints. Only small JSON/stderr reports; current 9.6 GiB free is sufficient.
GPU/memory verification remains pending. Formal 40k training remains deferred.

```bash
(
cd /root/autodl-tmp/CausalQ_DG || exit 1
export OMP_NUM_THREADS=1
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
  exit 1
fi
git fetch origin main || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
bash scripts/check_rein_slide_eval_phase16.sh
RESULT=$?
echo "slide_eval_gate_exit_code=$RESULT"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
)
```

Return complete JSON/stderr tail, hashes, exit and disk. Acceptance: ok true,
stage complete, sample_count 5, matching original-resolution prediction/GT,
official and independent metric agreement, finite memory/metrics, exit 0.
Only then prepare the next bounded training/validation integration stage.

## Archived scheduled-smoke instructions (do not rerun)

## Recovery: restore tuple types from pinned source (current action)

The `9ac1f8083f6bb446b4da1ec5c1e1b45d1701aa10` retry builds the model
but FAILS constructing RandomCrop: JSON changed `(512, 512)` into a list.
Exit 1 is not acceptance. This is still a runner serialization defect.
Failed report SHA `8639e7def5339968bb8d9a46d8d8b78dffc3e5e27fcee561f5b134f6b6e2bbd8`;
stderr SHA `bd5ed92d5e2c8b6cce6f2d69d882e1ad26df3e41121857e7bda088ad6e00fd16`.

Now load the full typed configuration from the pinned upstream entry point,
reapply the SAME versioned adapter using accepted physical dataset roots, and
compare every JSON-normalized configuration value with the hash-pinned saved
report. Fail on any drift. Only then wrap in mmengine.Config. This restores
all original tuple/list distinctions (crop/resize/preprocessor/stride/betas),
not just the field encountered so far; no protocol values are changed. Tests
cover actual JSON roundtrip, tuple/list preservation and rejection of drift.
Retry the same 80-microbatch gate; preserve both failed attempts. No installs.

## Previous repair: JSON configuration containers

The attempt at `d5558e19809ebb0b0f51957c8d21276a72cbce18` FAILED before
optimization (`optimizer_updates=0`, exit 1): Mask2Former accesses
`transformer_decoder.layer_cfg` but the saved JSON was passed as ordinary
nested dictionaries. This is our runner integration defect, not an environment
or dataset failure. The runner now reconstructs the accepted full configuration
with `mmengine.Config`, restoring recursive ConfigDict attribute access before
model construction. It checks the exact nested decoder attribute contract.
No architecture, optimizer, data, packages or schedule settings change.

Preserve failed JSON SHA
`0984148e8dd906eb37d6517d61ee2f039a25f07940438888a38ad52f6493c164`
and stderr SHA
`effe107293daf24abbf0caec31531b86488b4cb7689470ece011b3295af9c6fe`.
The child script uses the new producer SHA in every output path, so no failed
artifact is overwritten. Retry the same bounded gate on the handoff commit;
do not advance to formal training. Local regression tests mock the Config
boundary because OpenMMLab is intentionally absent locally; real integration
acceptance still requires AutoDL.

The operator accepts the adapted data gate at
`b645793ff8fcdd12b0cdd8078b64dc929f107a70`, exit 0. Accepted JSON SHA:
`b893b991c4bc85eb6105e8d8ae1f39df670b256ce12a10aaca8c1a7c89d71c69`;
stderr SHA `b7ddc04bfe4985764aca1bcc163b5a1b8dca7298da3e21acaeaee9d5e8f04776`.
The next runner checks that complete saved report before consuming its exact
adapted configuration. No installs or dataset/weight copies are required.

Run only 80 source microbatches, with accumulation 4 (=20 optimizer updates).
Use upstream PEFT paramwise AdamW and clipping 1, float32, seed 0. PolyLR's
40k horizon is explicitly measured in OPTIMIZER UPDATES, stepping once per four
microbatches, not per source sample. Future 40k-update training would require
160k microbatches and a separate runner/authorization; the saved adapted upstream
40k iteration config is not yet a formal accumulation-aware training config.

The gate writes one compact trainable-parameters-plus-buffers checkpoint with
optimizer and scheduler state (maximum 1 GiB), verifies safe reload, perturbs a
parameter before restoration, and compares semantic scores before/after. It
does not claim exact training resumption (sampler/RNG continuation is not saved),
target mIoU, paper reproduction or CQE transfer. At least 2 GiB free is required;
the returned 9.9 GiB is adequate for this bounded gate, not an unrestricted run.
No existing checkpoints are deleted. GPU verification remains pending.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]; then
    bash scripts/check_rein_schedule_phase16.sh
    echo "schedule_gate_exit_code=$?"
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return complete JSON and stderr tail printed by this child script, hashes,
exit code and disk. Pass: ok true, stage complete, optimizer_updates 20,
checkpoint under 1 GiB, prediction_roundtrip_max_error <=1e-5. Trace contains
80 contiguous records and exactly 20 update boundaries. Stop before formal
training; a Cityscapes full-resolution sliding evaluation gate is still pending.

## Completed adapted data-pipeline gate (do not rerun)

The repaired inventory exits 0 at `ede170c4fa2c5c330581771e9cf5c4f659360e9d`;
JSON SHA `2cdf617c9370735ac275c4875af52f3a017898dd690ff3cc793298322298e64a`.
The operator returned only the inventory tail; the next script reads and checks
the complete saved JSON (ok/stage/hash) before adaptation. Do not rerun inventory.

The versioned adapter retains upstream architecture, source shortest-edge resize,
cat-ratio crop, flip, photometric distortion, paramwise AdamW, and 40k PolyLR.
Explicit deviations: physical batch 1/accumulation 4, clipping 1, workers 0,
seed 0, one retained checkpoint; no best-by-target checkpoint selection.
This is an adapted REIN baseline, NOT exact paper reproduction. Future runner
must safely load the accepted frozen weights because automatic init is disabled.
The next action is NOT that runner and does not train or save checkpoints.

GTA5 roots are resolved with existing project pairing logic, `.png` labels are
already train IDs (no remap or reduce-zero). Source-only label alignment uses
the existing nearest resampling/aspect gate before official resize; an all-ignore
random crop alone triggers a valid-pixel fallback. These safety deviations are
explicit, not new learning mechanisms. Cityscapes-only dataset and IoUMetric
replace both upstream multi-target components. Target input retains upstream
1024x512 resize; labels retain ORIGINAL resolution. Model test policy remains
512 slide crops, stride 341. No target labels enter source optimization.

Run only a dataset gate: compare all 24966/500 path pairs with existing project
pairing, then decode five source and five target samples through adapted pipelines.
No model, GPU, weight load, optimizer, installs or data copies; small reports only.
Existing 9.9 GiB free is sufficient. This does not authorize formal training.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]; then
    bash scripts/check_rein_data_protocol_phase16.sh
    echo "data_protocol_exit_code=$?"
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return complete JSON, stderr tail, hashes and Git/disk evidence. Pass: ok true,
stage complete, exact pairing true, source/target counts 24966/500, ten pipeline
samples with valid IDs, matched source image/label shapes and original target GT.
Preserve failures. Stop before scheduled GPU training/accumulation/checkpoint gates.

## Completed upstream protocol inventory (do not repeat)

## Recovery of the inventory tool (current action)

The first inventory at `4e6a373` parsed the full effective configuration but
failed with `ModuleNotFoundError: No module named 'rein.datasets'`. This was
our inspector's incorrect package assumption, NOT a missing training dependency.
The effective dataset is installed MMSegmentation `CityscapesDataset`, even for
GTA5. The repaired inspector reads CityscapesDataset, BaseSegDataset,
LoadAnnotations and RandomCrop from the accepted MMSeg 1.2.2 installation,
including implementation source and file hashes. No new packages are needed.
Preserve the failed JSON/stderr; rerunning on the new commit creates new names.
Use the commands below with the new exact handoff SHA, not `4e6a373`.

Partial protocol evidence: GTA5 requires `_labelTrainIds.png` upstream, whereas
local converted GTA5 labels use `.png`; explicit suffix adaptation is required.
Inherited validation includes three targets and DGIoUMetric: both dataset and
evaluator must be narrowed to Cityscapes, not just one of them. Upstream target
input resize is 1024x512 with full original labels, slide crops 512x512/stride
341, batch 4 and 40k PolyLR. None of this authorizes running the unadapted
config. The following step is still read-only source inspection, not training.

The complete real-data report passed audit at
`b85f105384d07c88f23f89bec1723aa6a6c92c7e`: 20 optimizer steps, five target
predictions, maximum objective reconstruction error `1.7642974853515625e-05`,
normalization error `4.76837158203125e-07`, allocated/reserved memory
`3.537/3.881 GiB`. Adapter update and frozen patch invariance pass. This accepts
the engineering smoke, NOT accuracy, the formal baseline or CQE transfer.

Before defining formal training, inspect the effective pinned upstream config:
`configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py`.
Its leaf source declares batch 4, PhotoMetricDistortion, parameter-specific
decay, PolyLR over 40k, validation every 10k and three retained checkpoints.
These differ from the accepted batch-1, constant-lr, project-transform smoke.
Inherited GTA5 label handling, nested local image/label paths and full target
test pipeline still need inspection. Do not blindly use smoke settings or
remap existing train IDs a second time. Only Cityscapes validation is in scope,
even if the upstream inherited config references other target datasets.

Next operation reads the pinned upstream configuration/source and writes small
JSON/stderr files. It builds NO model/dataset, decodes no images, loads no
weights, and does not install, train, delete or save checkpoints. At 9.9 GiB
free, avoid launching unrestricted checkpoint generation. Formal storage policy
and effective-batch adaptation must be versioned after reviewing this inventory.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]; then
    bash scripts/inspect_rein_protocol_phase16.sh
    echo "protocol_inventory_exit_code=$?"
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the complete JSON (a text attachment is fine), stderr tail, hashes, exit
code and Git/disk evidence. Pass: `ok: true`, stage complete, pinned clean
upstream, effective loaders/optimizer/schedule/test policy and source hashes.
This inventory is not a GPU result or training authorization. Stop here before
adapting/running a fresh scheduled baseline smoke. No CQE or additional targets.

## Completed saved real-data report audit (do not repeat)

The operator returned all 20 finite source losses/norms, positive valid pixels,
exit 0, source `17883de0c87abe3ed9bd051b260ba20bc259999e`, and 9.9 GiB free.
The meshgrid warning is nonfatal. Norms are measured BEFORE clipping to 1.0.
The complete JSON was not returned: target inference, frozen-weight invariance,
normalization and peak memory fields remain unreviewed. Do not call the full
gate accepted or launch formal training based only on this log excerpt.

Next run ONLY the read-only saved-report audit. No GPU, dataset decoding,
installation, download, new reports or checkpoints. It pins the supplied JSON
and stderr hashes, checks source/runtime/weights, 20 complete 30-loss records,
normalization, frozen weight and adapter update flags, five target score shapes
and peak memory. Finite predictions/gradients rely on the hash-matched producer's
checks; this is not an independent rerun. No accuracy claim or training authority.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]; then
    /root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python \
      -m tools.audit_rein_real_data \
      --report /root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_real_data_17883de_v1.json \
      --stderr-log /root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_real_data_17883de_v1.stderr.log
    echo "saved_report_audit_exit_code=$?"
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the full compact audit JSON, exit code and Git/disk evidence. On failure,
return the existing complete real-data JSON; do not overwrite or rerun it.
Stop for review before defining the source-only formal schedule and full-target
evaluation. Only GTA5 to Cityscapes remains in scope.

## Completed execution: real-data 20-step optimization smoke (audit pending)

The full float32 segmentor gate passes at `86fb3ffdc7a13285965f0df825472625f9e52ed4`:
343 loaded tensors; 23,569,877 trainable parameters; scores `[1,19,512,512]`;
all 30 losses finite, total `124.92772674560547`; nonzero gradients in adapter,
head and pixel decoder `11/291/116`; frozen backbone gradients absent.
Peak allocated/reserved memory `3.400/3.662 GiB`, exit 0. Report/stderr hashes:
`87ba441bb9f879fd4048c18518477e8294c06dfa27335f7811954bd0c7552301`,
`05a9a5cc9f11d36804cb7a20fa4832afbe547937dde1d678284a69e28cd2bca3`.

Next run ONLY a 20-update real-data diagnostic, not a formal 40k baseline.
The previously accepted synthetic gate is rerun as a prerequisite in the same
process; its backward gradients are cleared and no synthetic optimizer step
occurs. Existing project GTA5 pairing/geometry/crop-fallback code is reused.
Source crops are 512x512, batch 1, workers 0. AdamW lr `1e-4`, weight decay
`0.05`, betas `(0.9,0.999)`, clip norm `1.0`, constant lr, float32.
These are smoke settings, not a claim of exact official REIN reproduction.
The model initialization seed remains 20260929; data selection/augmentation
seed is 0. No CQE, appearance consistency or diversity is enabled.

The bridge reverses the project's RGB normalization to floating BGR 0..255
before the unchanged upstream preprocessor, with round-trip error <=1e-5.
This prevents double normalization/channel swapping. Pair counts must be
24966 GTA5 / 500 Cityscapes-val; inventory hashes cover paths, NOT full file
contents. Only selected files are decoded. Five Cityscapes samples undergo
aspect-preserving resize and deterministic 512x512 center-crop inference;
target labels never enter a loss or optimizer update. No mIoU is computed:
this cropped inference is not the formal target evaluation protocol.

No checkpoint, data copy, installation or download. Existing 9.9 GiB disk
space suffices for small reports; optimizer GPU memory is still to be measured.
Do not start formal training or clean up evidence as part of this action.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]; then
    bash scripts/check_rein_real_data_phase16.sh
    echo "real_data_gate_exit_code=$?"
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the complete JSON, stderr tail, hashes and Git/disk evidence. Pass:
exit 0; `ok` and `real_data_smoke_ok` true; `optimizer_steps: 20`; 20 finite
contiguous source records, each with 30 finite losses and positive valid pixels;
adapter scale changed; reference frozen patch weight unchanged; five finite
target score maps; normalization round-trip error <=1e-5. Loss decrease over
20 different crops is not a pass criterion. Preserve partial reports on failure;
do not edit source on AutoDL, upgrade packages or reuse existing report paths.
Stop for review before selecting a formal baseline schedule/data protocol.

## Completed full-segmentor synthetic handoff (do not rerun separately)

The backbone gate passed at `ed3dcfe202435f4d7893681054473724e19e1ad6`:
343 pretrained tensors loaded, four expected feature maps and `[100,256]`
linked queries, 11 nonzero adapter gradients, no frozen backbone gradients.
Peak allocated/reserved memory was `2.767/2.982 GiB`, with 9.9 GiB disk free.
Report SHA256: `317dff2553f4674f9363d8660ce369245863676866fa0cf5ed0745db1bb53706`.
stderr SHA256: `c4775f166f11b2cbb18b26ecb5487133076edf537e05943acf6e29501284ed1f`.

Next execute the full upstream EncoderDecoder + REIN + Mask2Former with one
512x512 synthetic image and synthetic labels containing all 19 classes and
255 ignore pixels. This checks preprocessing, full-resolution semantic scores,
all 30 final/auxiliary classification-mask-Dice losses, and a single backward
pass through adapters and the head including the deformable-attention pixel
decoder. Use float32, without autocast, so mixed-precision kernel compatibility
is not confounded with model compatibility. Actual GPU memory remains untested.

No optimizer, dataset loading, checkpoint saving, downloads or installation.
No CQE or other experimental objective. Synthetic losses/scores are NOT accuracy
results. Real GTA5/Cityscapes data protocol, optimizer steps, mixed precision
and formal baseline training remain later gates. Stop after this one test.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]; then
    bash scripts/check_rein_segmentor_phase16.sh
    echo "segmentor_gate_exit_code=$?"
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the JSON, exit code, stderr tail, report hashes, Git and disk evidence.
Pass requires `ok: true`, `stage: complete`, exit 0, 343 loaded tensors,
semantic scores `[1,19,512,512]`, finite 30 losses, positive finite total loss,
nonzero `adapter/head/pixel_decoder` gradient counts and no frozen backbone
gradients. Preserve reports on failure. Do not edit upstream or upgrade packages,
delete data or start training. The child `bash` protects the interactive terminal
from script exits; existing output names are never overwritten.

## Completed backbone handoff (do not rerun)

The xformers recovery runtime gate passed at `0e0ef7c7bb1588cb0ba190c652fbce926c257121`.
CUDA NMS returned `[0]`; fp16 xformers attention max error was
`0.0005131959915161133` (limit `0.005`). All eight package pins and both
REIN registrations passed. The optional missing `mmpretrain` message concerns
ConvNeXt, not this DINOv2 path; do not install it. Runtime report SHA256:
`2d3498fc8343927da29dfa2e73bb5fb89ad46132248d15e6a323cdef4adcdfcb`.
The upstream informational print preceded that old report's JSON, so it is
preserved as textual evidence; the new probe redirects upstream prints to stderr.

Proceed only with a synthetic 512x512 batch-1 backbone test. It builds the
pinned upstream backbone, disables automatic checkpoint initialization, safely
loads all 343 converted tensors, freezes the pretrained path, checks four
feature maps and linked queries, and performs one diagnostic backward pass.
There is no optimizer, dataset, segmentation head, checkpoint write or CQE.
This does NOT validate full Mask2Former/deformable-attention training or prove
CQE transfer. Those gates remain later. Free space is approximately 9.9 GiB;
this test produces only small logs and needs no new installation/download.

Run this block in AutoDL (the `bash` child keeps failure from closing your terminal):

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]; then
    bash scripts/check_rein_backbone_phase16.sh
    echo "backbone_gate_exit_code=$?"
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the JSON, stderr, exit code, hashes and source/disk evidence. Pass requires
`ok: true`, 343 loaded tensors, feature shapes `[1,1024,128,128]`,
`[1,1024,64,64]`, `[1,1024,32,32]`, `[1,1024,16,16]`, queries `[100,256]`,
finite outputs/gradients, nonzero adapter gradients and no frozen-backbone
gradients. If it fails, preserve all artifacts; do not upgrade packages, modify
upstream source, start training or rerun against existing output filenames.

## Historical runtime handoffs (do not rerun)

Phase 15's three-seed ViT-B/ViT-L comparison is complete. The seed-2 paired
report has `ok: true`, 500 matched Cityscapes validation images, 6,005
present-class effect maps per backbone, and report SHA256
`2761a9264593aba81712f9a5ad0ef2006c0a1b533e99a0987cd57654f3ae357b`.
Across seeds 0/1/2, ViT-L has higher final mIoU and lower normalized
query-effect style variance in every pair. See the versioned three-seed
report in `experiments/BACKBONE_SCALING_STATIC_R2_3SEED/report.md`.

The original plan's §43 DINOv2/REIN transfer check is the current single
phase. The converted DINOv2-L checkpoint passed an independent on-disk audit
at Git SHA `8ee5abf9d927743a5a88fa8b4691fd0cd4bb2301`: both pinned hashes
matched, and all 343 tensor keys, shapes, dtypes, and values matched a fresh
conversion. The audit report SHA256 is
`6cdbdf1b503f7f6c7eb58287088d99de1edde8b378e7304add140e5418a4c120`.
There were 476 GiB available system RAM and 16 GiB free data volume.

## Historical action: recover the missing xformers dependency

The runtime recheck at `44d30fa397205b10a75d08fb60b62ffcc055872c`
passed all seven package pins, CUDA availability and MMCV CUDA NMS (`[0]`),
then failed during REIN import with `ModuleNotFoundError: xformers`.
Report/stderr SHA256:
`b7ef215f8ec60e6c4e7d0c06a1074169b39d6fa8a4f77bafa21e505ea45626be`
and `f2d119b131234a5049ca7ad999c4e62fb3fd50cea8233300f09e1d20434f34b3`.
The pinned upstream package imports `eva_02.py`, which unconditionally imports
`xformers.ops`; DINO layers' optional fallback does not avoid this import.

Install only xformers 0.0.20 and its Python helper dependency in the existing
isolated REIN environment. The official PyPI metadata requires torch 2.0.1
and pyre-extensions 0.0.29. The exact CPython 3.10 Linux wheel (109067679
bytes) is hash-verified; core package constraints and `--no-deps` on this wheel
prevent replacement of the validated CUDA/PyTorch/OpenMMLab stack. No source
build, upstream edit, new environment, dataset or weight download is involved.
The gate also executes a tiny fp16 xformers CUDA attention probe and compares
it with a float32 reference before importing REIN. GPU compatibility remains
pending until this AutoDL run. Existing reports and freezes are preserved.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_xformers_recovery_v1.log
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ] \
    || [ -n "$(git status --porcelain)" ] || [ -e "$LOG" ]; then
    echo 'preflight_failed: wrong source or existing log'
  else
    bash scripts/recover_rein_xformers_phase16.sh 2>&1 | tee "$LOG"
    RECOVERY_EXIT=${PIPESTATUS[0]}
    echo "recovery_exit_code=$RECOVERY_EXIT"
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
BASE=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_xformers_recovery_v1
for SUFFIX in json stderr.log; do
  if [ -f "$BASE.$SUFFIX" ]; then cat "$BASE.$SUFFIX"; fi
done
for SUFFIX in before.txt after.txt json stderr.log log; do
  if [ -f "$BASE.$SUFFIX" ]; then sha256sum "$BASE.$SUFFIX"; fi
done
tail -n 50 "$LOG"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the exit code, JSON, stderr, hashes and source/disk evidence. A pass
requires `ok: true`, xformers version 0.0.20, attention max error <=0.005,
CUDA NMS `[0]` and both REIN registrations true. If it fails, keep all outputs
and the installed state; do not blindly rerun installation or upgrade torch.
Stop here; model loading, full deformable attention and training remain later
gates. This recovery requires only 1 GiB free (11 GiB is currently available).

## Previous runtime recheck (failed on missing xformers; do not repeat)

Both CUDA diagnostics passed at `1badc850f35817352e943139baf94b12eee965f4`:
the main PyTorch 2.7/cu126 and isolated PyTorch 2.0.1/cu118 environments
each saw one RTX 4090 D (capability 8.9, driver 595.71.05) and completed
the scalar compute probe. Main/rein log hashes respectively:
`cbf3646e2b74c4205b30b9c57e7d58cb33e1d4ca888a42d7bf1784bbb878cfaf`
and `4f7736a3ac085b86ccadffcf10b47d87b079072ccb57622a4fffe3792c831a39`.
This establishes current basic GPU access, not the cause of the earlier
failure or full REIN compatibility. Preserve `NVIDIA_VISIBLE_DEVICES=void`:
the actual probes succeeded without overriding it. Both runs emitted an
invalid `OMP_NUM_THREADS` warning; set a valid value for the next process.
There are still 11 GiB free; no installation or download is needed.

Run only the existing runtime gate in the installed isolated prefix. It
checks exact package/source pins, CUDA NMS, REIN model registration and
configuration parsing. It does not build a model, load weights or datasets,
or execute deformable-attention forward. New output paths preserve all
previous reports. No training is authorized until this result is reviewed.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
REIN_PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
BASE=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_runtime_gpu_recheck_v1
REPORT="$BASE.json"
ERROR_LOG="$BASE.stderr.log"

if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ] \
    || [ -n "$(git status --porcelain)" ] \
    || [ ! -x "$REIN_PY" ] \
    || [ -e "$REPORT" ] || [ -e "$ERROR_LOG" ]; then
    echo 'preflight_failed: wrong source, missing interpreter or existing outputs'
  else
    OMP_NUM_THREADS=1 "$REIN_PY" -m tools.check_rein_runtime \
      --rein-root /root/autodl-tmp/external/rein-phase16-dc063429 \
      --expected-rein-sha dc063429c4dadc0da9c6252b3db22fc55a9882ab \
      2> "$ERROR_LOG" | tee "$REPORT"
    RUNTIME_EXIT=${PIPESTATUS[0]}
    echo "runtime_exit_code=$RUNTIME_EXIT"
    cat "$ERROR_LOG"
    sha256sum "$REPORT" "$ERROR_LOG"
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the JSON, runtime exit code, stderr, hashes and source/disk evidence.
Success requires `ok: true`, CUDA NMS kept indices `[0]`, and both REIN
registrations true. If it fails, preserve the installed environment and
outputs and return the actual error; do not rerun bootstrap or edit upstream
source. Stop at this remote verification boundary.

## Completed paired CUDA diagnostic handoff (do not repeat)

At `035a3faaa90365286db06f0f728a023fcb1e47a3`, recovery installed all
seven pinned packages and `pip check` passed. The REIN source SHA matched,
but the runtime gate stopped at GPU availability, before MMCV CUDA NMS or
REIN imports. CUDA runtime was already the expected 11.8. The failed report
SHA256 is `a24a1fefce3d2a0a233010e2af567c10094c55d75e1845f0ccc86aa29e4ff44b`;
pip-freeze SHA256 is
`dd00674c5a9b5361730fdcaeaedf25e883befa0cc9d5f1e56df69f7f26d3fbfd`.
The data volume has 11 GiB free. Do not reinstall or create another environment.

The next action is a diagnostic, not a model smoke. It runs the same source
with the main and isolated interpreters, reports selected visibility/library
variables, NVIDIA device nodes and `nvidia-smi`, forces CUDA initialization
to retain its real exception, and attempts a one-scalar GPU computation.
It does not load weights/data or override visibility/library paths. An
installed CUDA wheel alone does not establish GPU access.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ] \
    && [ -z "$(git status --porcelain)" ]; then
    for NAME in main rein; do
      if [ "$NAME" = main ]; then
        DIAG_PY=/root/autodl-tmp/envs/causalq-dg/bin/python
      else
        DIAG_PY=/root/autodl-tmp/envs/rein-phase16-py310-cu118-tuna-retry1/bin/python
      fi
      DIAG_LOG="/root/autodl-tmp/outputs/CausalQ_DG/analysis/cuda_diagnostic_${NAME}_$(git rev-parse --short HEAD).log"
      if [ ! -x "$DIAG_PY" ] || [ -e "$DIAG_LOG" ]; then
        echo "diagnostic_preflight_failed=$NAME"
      else
        "$DIAG_PY" -m tools.diagnose_cuda 2>&1 | tee "$DIAG_LOG"
        DIAG_EXIT=${PIPESTATUS[0]}
        echo "diagnostic_${NAME}_exit_code=$DIAG_EXIT"
        sha256sum "$DIAG_LOG"
      fi
    done
  else
    echo 'preflight_failed: wrong source'
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return both complete logs and exit codes. If `nvidia-smi` and both probes
fail, check the AutoDL console's GPU allocation/boot mode and contact AutoDL
support with the diagnostic; this is not evidence that package replacement
is needed. If the main probe passes but REIN fails, retain the exact init
error and library paths for environment-specific diagnosis. Do not unset
GPU allocation variables, remove drivers, disable the gate, or start training.
If both pass, the isolated REIN runtime gate still must be rerun into a fresh
report before model construction. Stop for remote diagnosis now.

## Previous Conda recovery handoff (installation completed; do not repeat)

At `893c9de308a868e2b9cbe6f9cab054c5916090b9`, the upstream REIN
checkout succeeded, but Conda metadata retrieval from `repo.anaconda.com`
failed with HTTP 000. No runtime gate or model loading was reached. Preserve
the old bootstrap log, upstream checkout, and any partial environment.

The recovery mode verifies and reuses the clean pinned REIN checkout. It
creates a **new** prefix `rein-phase16-py310-cu118-tuna-retry1`, refuses
existing retry outputs, and uses `--override-channels` with only Tsinghua's
main channel for Python/pip creation. It does not change `.condarc`, disable
TLS, delete files, or install into `causalq-dg`. PyTorch and MMCV still use
their previously pinned official wheel sources. A mirror network failure
must stop this gate; it does not authorize a model run.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_bootstrap_tuna_retry1.log

if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ] \
    || [ -n "$(git status --porcelain)" ] || [ -e "$LOG" ]; then
    echo 'preflight_failed: wrong source or existing retry log'
  else
    bash scripts/bootstrap_rein_phase16.sh --recover-conda 2>&1 | tee "$LOG"
    BOOTSTRAP_EXIT=${PIPESTATUS[0]}
    echo "bootstrap_exit_code=$BOOTSTRAP_EXIT"
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi

tail -n 50 "$LOG"
REPORT=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_runtime_tuna_retry1.json
if [ -f "$REPORT" ]; then cat "$REPORT"; fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the exit code, runtime JSON if created, report/freeze hashes printed by
the script, and source/disk evidence. This block does not use `exit` in the
interactive shell. Do not rerun over the retry prefix after a failure;
preserve it and return the log. Stop here until this remote gate is accepted.

## Previous bootstrap handoff (failed; do not repeat)

Use the full Git SHA from the handoff. The versioned bootstrap script checks
both checkpoint hashes and at least 14 GiB free before creating anything.
It clones only the official REIN commit
`dc063429c4dadc0da9c6252b3db22fc55a9882ab` to a separate external
directory, creates a separate Python 3.10 environment, installs pinned
PyTorch/OpenMMLab dependencies without a pip cache, and probes CUDA NMS,
REIN imports, and configuration parsing. It neither edits the upstream
checkout nor loads a segmentor or trains. If any installation fails, preserve
the partial environment and log; do not install into `causalq-dg` or rerun
over the same paths. The official REIN repository is GPL-3.0 and is not
copied into this Git project.

```bash
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
LOG=/root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_bootstrap.log

if [ -n "$(git status --porcelain)" ]; then
  echo 'preflight_failed: Git worktree is not clean'
elif git fetch origin main && git checkout --detach "$EXPECTED_SHA"; then
  if [ "$(git rev-parse HEAD)" != "$EXPECTED_SHA" ] \
    || [ -n "$(git status --porcelain)" ] \
    || [ -e "$LOG" ]; then
    echo 'preflight_failed: wrong source or existing log'
  else
    mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/analysis
    bash scripts/bootstrap_rein_phase16.sh 2>&1 | tee "$LOG"
    BOOTSTRAP_EXIT=${PIPESTATUS[0]}
    echo "bootstrap_exit_code=$BOOTSTRAP_EXIT"
  fi
else
  echo 'git_fetch_or_checkout_failed'
fi
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return `bootstrap_exit_code`, the runtime JSON (if created), report and
environment-freeze SHA256 values, Git SHA/status, and disk space. On failure,
also return the last 50 log lines and preserve all created paths. A passing
runtime gate does not prove the full REIN model loads or fits RTX 4090 D;
the next action is a separate model-build and weight-load smoke after the
GTA5/Cityscapes-only configuration is versioned.

## Completed handoff: DINOv3-B Static R=2 full seed-0 training (do not repeat)

The restored ViT-B Static R=2 500-step smoke on AutoDL commit
`915e872ca080eae74eef387e7990ad6e3cecbdc5` produced a complete
`summary.json` (`ok: true`, finite losses), 500 training records, one
50-image Cityscapes validation at iteration 500 (`mIoU: 0.337162`), and the
final smoke checkpoint. Peak reserved GPU memory was 1.512 GiB; 25 GiB of
disk remained free. That short-run mIoU is diagnostic only. The next action
first audits the saved trace for contiguity and objective reconstruction,
then starts a fresh 40,000-iteration seed-0 run on all 500 validation images.
Do not resume the smoke checkpoint, overwrite an existing run, or start seed
1/2 or another target dataset at this stage.

## Completed smoke and recovery records (do not repeat)

The ViT-B weights have been restored from the distribution recorded in
`pretrained_manifest.yaml`. Their SHA256 is again
`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
On AutoDL commit `d7913cf276bab6c96cfea4846dc6948bd38fb529`, the
separate backbone check passed on RTX 4090 D with bfloat16,
768-dimensional patch features, four requested intermediate maps, and zero
NaN/Inf; 25 GiB remained free. The next action is one 500-iteration training
smoke, not a 40k experiment.

## Completed recovery record (do not repeat)

The scaling smoke did not reach training. AutoDL's read-only inventory found
that `/root/autodl-tmp/pretrained/dinov3_vitb16` is absent, only the ViT-L
safetensors remains on the data volume, no likely Hugging Face cache blob was
found, and the failed smoke produced no run directory. The source commit is
`d7913cf276bab6c96cfea4846dc6948bd38fb529` with a clean worktree; the
disk has 26 GiB free. Preserve both failed-attempt logs.

The checked-in `pretrained_manifest.yaml` records the previously used
ModelScope distribution at
`https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m`.
Only the exact ViT-B safetensors with SHA256
`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`
is admissible. Download into a new staging directory, verify before moving to
the configured path, and stop on any mismatch. Do not substitute ViT-L, use a
different ViT-B conversion, disable the SHA check, or launch training yet.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

DOWNLOAD_ENV=/root/autodl-tmp/envs/modelscope-download-phase15
STAGE=/root/autodl-tmp/pretrained/.dinov3_vitb16_restore_phase15
DEST=/root/autodl-tmp/pretrained/dinov3_vitb16
EXPECTED_SHA=9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b

test ! -e "$DOWNLOAD_ENV" || { echo "existing_download_env=$DOWNLOAD_ENV"; exit 1; }
test ! -e "$STAGE" || { echo "existing_stage=$STAGE"; exit 1; }
test ! -e "$DEST" || { echo "existing_destination=$DEST"; exit 1; }

python -m venv "$DOWNLOAD_ENV" || exit 1
"$DOWNLOAD_ENV/bin/python" -m pip install modelscope-hub || exit 1
"$DOWNLOAD_ENV/bin/ms-hub" download \
  facebook/dinov3-vitb16-pretrain-lvd1689m \
  --include config.json model.safetensors \
  --local-dir "$STAGE" || exit 1

test -s "$STAGE/config.json" || exit 1
test -s "$STAGE/model.safetensors" || exit 1
sha256sum "$STAGE/model.safetensors"
ACTUAL_SHA="$(sha256sum "$STAGE/model.safetensors" | cut -d' ' -f1)"
test "$ACTUAL_SHA" = "$EXPECTED_SHA" || {
  echo "weight_sha_mismatch=$ACTUAL_SHA"; exit 1;
}

python - "$STAGE/config.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert config["hidden_size"] == 768, config["hidden_size"]
assert config["patch_size"] == 16, config["patch_size"]
assert config["num_register_tokens"] == 4, config["num_register_tokens"]
print("vitb16_config_ok=true")
PY
test "$?" -eq 0 || exit 1

test ! -e "$DEST" || exit 1
mv -- "$STAGE" "$DEST" || exit 1
test "$(sha256sum "$DEST/model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_SHA" || exit 1

mkdir -p /root/autodl-tmp/outputs/CausalQ_DG/backbone_check
CHECK_LOG=/root/autodl-tmp/outputs/CausalQ_DG/backbone_check/vitb16_restored_phase15.json
test ! -e "$CHECK_LOG" || { echo "existing_check_log=$CHECK_LOG"; exit 1; }
set -o pipefail
python tools/check_backbone.py \
  --model dinov3_vitb16 \
  --weights "$DEST" \
  --image-size 512 512 \
  --batch-size 1 \
  --dtype bfloat16 \
  --intermediate-indices 3 6 9 12 \
  2>&1 | tee "$CHECK_LOG"
CHECK_EXIT=${PIPESTATUS[0]}
echo "backbone_check_exit_code=$CHECK_EXIT"
test "$CHECK_EXIT" -eq 0 || exit 1
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

The recovery and GPU check above passed. These commands are retained for
provenance only; their staging and download-environment directories now exist,
so do not rerun them. Preserve the two failed smoke logs. The commands below
are historical handoffs, not the current Phase 16 action.

## Completed smoke handoff after the weight was restored

Project-plan §41's GTA5 → Cityscapes qualitative figures are provenance-matched
and closed as a diagnostic, not a causal proof. Begin §42 scaling with **one**
isolated DINOv3-B/16, Static R=2, photometric-only seed-0 smoke. The existing
DINOv3-L/16 Static R=2 three-seed result remains the reference. Do not train
on Cityscapes, run BDD100K/Mapillary/ACDC, enable CQE or learned-null, or launch
the 40k B run before this 500-step smoke is accepted.

## 1. Sync exact source and verify inputs

Replace `EXPECTED_SHA` with the full commit SHA provided with this handoff.
Run on AutoDL; never edit source files there.

```bash
cd /root/autodl-tmp/CausalQ_DG
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
git fetch origin main
git checkout --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || exit 1

WEIGHTS=/root/autodl-tmp/pretrained/dinov3_vitb16/model.safetensors
test -f "$WEIGHTS" || exit 1
test -d /root/autodl-tmp/datasets/gta5 || exit 1
test -d /root/autodl-tmp/datasets/cityscapes/leftImg8bit/val || exit 1
test -d /root/autodl-tmp/datasets/cityscapes/gtFine/val || exit 1

sha256sum "$WEIGHTS"
test "$(sha256sum "$WEIGHTS" | cut -d' ' -f1)" = \
  '9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b' || exit 1
df -h /root/autodl-tmp
```

If any command fails, stop and return the output. The weight hash above is
from the previously validated local DINOv3-B safetensors. Leave at least 2 GiB
available before this smoke; do not delete data or existing checkpoints to
make space without a separate reviewed cleanup plan.

## 2. Run only the 500-iteration smoke

```bash
RUN_ID="SCALE_VITB_STATIC_R2_SEED0_SMOKE_500_RESTORED_$(git rev-parse --short HEAD)"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID"
LOG="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID.log"

echo "run_id=$RUN_ID"
test ! -e "$RUN_DIR" || { echo "existing_run=$RUN_DIR"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }

set -o pipefail
python tools/train.py \
  --config configs/scaling/gta_dinov3b_static.yaml \
  --run-id "$RUN_ID" \
  --max-iterations 500 \
  --validation-max-samples 50 \
  --seed 0 \
  2>&1 | tee "$LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0 || exit 1
```

Do not overwrite or resume a failed smoke directory. Keep its logs and report
the traceback for diagnosis. Smoke mIoU on 50 images is not a scaling result.

## 3. Inspect the smoke evidence

```bash
test -f "$RUN_DIR/summary.json" || exit 1
test -f "$RUN_DIR/metadata.json" || exit 1
test -f "$RUN_DIR/checkpoints/iter_000500.pth" || exit 1
wc -l "$RUN_DIR/train.jsonl"

python - "$RUN_DIR" "$EXPECTED_SHA" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected = sys.argv[2]
metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
assert summary["ok"] is True and summary["finite_losses"] is True
assert metadata["git_sha"] == expected and metadata["phase"] == 15
assert metadata["backbone"] == "dinov3_vitb16"
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["queries_per_class"] == 2
assert metadata["style"]["views"] == ["original", "photometric"]
assert len(records) == 500
assert [item["iteration"] for item in records] == list(range(1, 501))
assert len(summary["validation_results"]) == 1
assert summary["validation_results"][0]["sample_count"] == 50
print("scaling_smoke_ok=true")
print("metadata:", metadata)
print("summary_without_validation:", {k: v for k, v in summary.items() if k != "validation_results"})
print("validation:", summary["validation_results"][0])
PY

sha256sum "$RUN_DIR/checkpoints/iter_000500.pth"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return `train_exit_code`, the audit output, checkpoint hash, final Git SHA and
status, and any errors. Only after checking those results may the 40k B run be
considered. §42's question about effect variance requires a later matched
full-validation analysis; this smoke cannot answer it.

## Historical action: audit smoke, then run one 40k seed-0 experiment

Replace `EXPECTED_SHA` below with the exact commit from the new handoff. Run
from a clean AutoDL checkout. The local smoke audit must pass before the full
run begins. The full run starts from random head initialization and the
verified frozen ViT-B weights; it must not resume the smoke checkpoint.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
source scripts/activate_autodl.sh
export OMP_NUM_THREADS=1

EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
git fetch origin main
git checkout --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || exit 1

SMOKE=/root/autodl-tmp/outputs/CausalQ_DG/SCALE_VITB_STATIC_R2_SEED0_SMOKE_500_RESTORED_915e872
python - "$SMOKE" <<'PY'
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
assert metadata["git_sha"] == "915e872ca080eae74eef387e7990ad6e3cecbdc5"
assert metadata["phase"] == 15 and metadata["backbone"] == "dinov3_vitb16"
assert metadata["pretrained_checkpoint_sha256"] == "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
assert metadata["seed"] == 0 and metadata["max_iterations"] == 500
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["queries_per_class"] == 2
assert metadata["style"]["views"] == ["original", "photometric"]
assert all(key not in metadata for key in ("prediction_consistency", "causal_query_effect", "query_diversity", "learned_null"))
assert summary["ok"] is True and summary["finite_losses"] is True
assert [record["iteration"] for record in records] == list(range(1, 501))
assert all(math.isfinite(record[key]) for record in records for key in ("loss", "loss_original", "loss_photometric", "gradient_norm", "alpha"))
error = max(abs(record["loss"] - record["loss_original"] - record["loss_photometric"]) for record in records)
assert error < 1e-5, error
assert len(summary["validation_results"]) == 1
assert summary["validation_results"][0]["sample_count"] == 50
assert summary["validation_results"][0]["iteration"] == 500
assert (run / "checkpoints" / "iter_000500.pth").is_file()
print("smoke_audit_ok=true")
print("maximum_objective_reconstruction_error:", error)
print("smoke_validation:", summary["validation_results"][0])
PY
test "$?" -eq 0 || exit 1

WEIGHTS=/root/autodl-tmp/pretrained/dinov3_vitb16/model.safetensors
test -f "$WEIGHTS" || exit 1
test "$(sha256sum "$WEIGHTS" | cut -d' ' -f1)" = \
  '9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b' || exit 1
test -d /root/autodl-tmp/datasets/gta5 || exit 1
test -d /root/autodl-tmp/datasets/cityscapes/leftImg8bit/val || exit 1
test -d /root/autodl-tmp/datasets/cityscapes/gtFine/val || exit 1
df -h /root/autodl-tmp

RUN_ID="SCALE_VITB_STATIC_R2_SEED0_40000_$(git rev-parse --short HEAD)"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID"
LOG="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID.log"
echo "run_id=$RUN_ID"
test ! -e "$RUN_DIR" || { echo "existing_run=$RUN_DIR"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }

set -o pipefail
python tools/train.py \
  --config configs/scaling/gta_dinov3b_static.yaml \
  --run-id "$RUN_ID" \
  --max-iterations 40000 \
  --validation-max-samples 500 \
  --seed 0 \
  2>&1 | tee "$LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "train_exit_code=$TRAIN_EXIT"
test "$TRAIN_EXIT" -eq 0 || exit 1
```

Do not delete intermediate checkpoints before auditing the run. After the
command returns, report `train_exit_code`, the following evidence, and any
traceback. A final mIoU alone is not sufficient to accept a full run.

```bash
RUN_ID="SCALE_VITB_STATIC_R2_SEED0_40000_$(git rev-parse --short HEAD)"
RUN_DIR="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID"
LOG="/root/autodl-tmp/outputs/CausalQ_DG/$RUN_ID.log"

python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
validations = summary["validation_results"]
checkpoints = sorted((run / "checkpoints").glob("iter_*.pth"))
assert summary["ok"] is True and summary["finite_losses"] is True
assert metadata["phase"] == 15 and metadata["max_iterations"] == 40000
assert metadata["backbone"] == "dinov3_vitb16" and metadata["seed"] == 0
assert metadata["query"]["interaction"] == "static"
assert metadata["query"]["queries_per_class"] == 2
assert metadata["style"]["views"] == ["original", "photometric"]
assert [record["iteration"] for record in records] == list(range(1, 40001))
assert all(math.isfinite(record[key]) for record in records for key in ("loss", "loss_original", "loss_photometric", "gradient_norm", "alpha"))
error = max(abs(record["loss"] - record["loss_original"] - record["loss_photometric"]) for record in records)
assert error < 1e-5, error
assert len(validations) == 80
assert all(item["sample_count"] == 500 for item in validations)
assert validations[-1]["iteration"] == 40000
assert len(checkpoints) == 80 and checkpoints[-1].name == "iter_040000.pth"
print("full_run_audit_ok=true")
print("experiment_id:", metadata["experiment_id"])
print("training_git_sha:", metadata["git_sha"])
print("pretrained_checkpoint_sha256:", metadata["pretrained_checkpoint_sha256"])
print("maximum_objective_reconstruction_error:", error)
print("peak_reserved_gib:", summary["peak_reserved_gib"])
print("final_alpha:", summary["final_alpha"])
print("final_validation:", validations[-1])
print("best_validation:", max(validations, key=lambda item: item["miou"]))
print("checkpoint_count:", len(checkpoints))
PY

sha256sum "$RUN_DIR/checkpoints/iter_040000.pth"
tail -n 20 "$LOG"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

After the 40k evidence is accepted, compare ViT-B and the fixed ViT-L Static
R=2 seed-0 result on the same 500 Cityscapes validation images. A later paired
style-effect analysis is also required before any scale-dependent variance
claim. Do not start these analyses or additional seeds as part of this handoff.
