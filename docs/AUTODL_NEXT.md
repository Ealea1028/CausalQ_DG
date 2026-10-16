# AutoDL next: Phase 17 — bounded M0 source-only GPU smoke

Only this section is active. All older data/export/Phase16 instructions below
the archive marker are inactive. Do not repeat the accepted data scan.

## Returned evidence accepted

The supplied source-data report matches SHA256
`cf01d813f6125c42a8e6712e449bfdba72ce5f287222357c975780d7cad8fab3`.
Its exact producer is clean `cb5207101915c97b23da82792608b2e2c24c123e`.
Pinned local audit accepts complete train22469/val2497 scans, legal train IDs,
zero unreadable/all-ignore/incompatible geometry, 62 declared scale-equivalent
differences, all2497 holdout transforms and32 train-photo inputs. The report
and read-only review are versioned under `experiments/STRICT_DG_PHASE17/`.
No source/target mIoU or effect statistic was measured. This accepts source
data compatibility, not scene independence or any model-performance hypothesis.

## Current bounded authorization

Use only the dedicated source smoke: fresh seed0 Static R=2, 20 optimizer
updates, batch1/accum1, workers0, original+photometric CE sum, unchanged frozen
DINOv3-L and BF16 AMP. AdamW1e-4/wd0.01, clip1 and the SAME 40k poly0.9
horizon; the schedule is not compressed to20. Source permutation is fixed for
this smoke. This is not worker4 long-run replay or exact resume evidence.

The tool checks alpha0 identity, finite losses/gradients/parameters and nonzero
head/query/scale gradients, unchanged backbone state, and a single final
compact checkpoint. Eight GTA5 holdout inputs check only finite compatibility.
No mIoU, target data, checkpoint selection, CQE, centered Query, diversity,
learned-null or REIN. Restore uses newly initialized head/query parameters
over the same frozen backbone, verifies exact model/optimizer state and
prediction error <=1e-5. GPU memory measurements include model construction
and restoration. The old general M0 training entry point stays blocked.

## AutoDL exact checkout and launch

Use the full literal SHA from Codex's final response. Existing environment:
`/root/autodl-tmp/envs/causalq-dg/bin/python`, torch2.7.0+cu126 and
transformers4.56.1. The tool checks those versions, disk free>=1GiB and the
fixed DINOv3-L safetensors digest before loading weights. No installation,
network download or source edits are permitted; return any preflight error.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
EXPECTED_SHA=FULL_40_CHARACTER_SHA_FROM_CODEX
git fetch origin codex/strict-dg-source-split || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || { echo dirty_checkout; git status --short; exit 1; }
export CAUSALQ_DATA_ROOT=/root/autodl-tmp/datasets
export CAUSALQ_PRETRAINED_ROOT=/root/autodl-tmp/pretrained
export CAUSALQ_OUTPUT_ROOT=/root/autodl-tmp/outputs/CausalQ_DG
nvidia-smi
BASE="$CAUSALQ_OUTPUT_ROOT"
mkdir -p "$BASE/phase17"
LAUNCH_LOG="$BASE/phase17/M0_STRICT_SOURCE_SMOKE_${EXPECTED_SHA:0:7}_launcher.log"
test ! -e "$LAUNCH_LOG" || { echo existing_launcher_log; exit 1; }
nohup bash scripts/check_strict_source_smoke_phase17.sh "$EXPECTED_SHA" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "source_smoke_pid=$!"
echo "launcher_log=$LAUNCH_LOG"
```

The process continues after terminal disconnect. It prints one record per
update; backbone hashing/checkpoint restore can take additional time. Do not
launch twice or overwrite any previous evidence. From a new terminal:

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
SHA=$(git rev-parse --short=7 HEAD)
tail -n 40 "/root/autodl-tmp/outputs/CausalQ_DG/phase17/M0_STRICT_SOURCE_SMOKE_${SHA}_launcher.log"
```

## Outputs, return and acceptance

Under `/root/autodl-tmp/outputs/CausalQ_DG/phase17/`:

```text
M0_STRICT_SOURCE_SMOKE_<sha7>/smoke_report.json
M0_STRICT_SOURCE_SMOKE_<sha7>/train.jsonl
M0_STRICT_SOURCE_SMOKE_<sha7>/SHA256SUMS
M0_STRICT_SOURCE_SMOKE_<sha7>/final.pth
M0_STRICT_SOURCE_SMOKE_<sha7>.log
M0_STRICT_SOURCE_SMOKE_<sha7>_launcher.log
```

Return the full JSON report, train.jsonl, SHA256SUMS and final launcher output;
on failure include its complete log. Keep final.pth on AutoDL; it does not
enter Git. The report/trace are small experiment metadata, not raw datasets.

Acceptance requires `source_smoke_exit_code=0`, ok true, exact clean SHA and
environment, matching source-data audit/manifests/weight identity, exactly20
updates on20 unique train IDs, eight disjoint holdout compatibility records,
finite optimization, nonzero required gradients, alpha-zero error0, unchanged
backbone, exact model/optimizer roundtrips, prediction error<=1e-5, one compact
checkpoint and valid checksums, plus measured peak GPU memory. Retain
formal_training_authorized=false, target_data_accessed=false, source_miou=null,
target_miou=null and exact_resume_verified=false. There is no accuracy threshold
and no interpretation of low short-run source scores as a model result.

Stop even on success. Next local work starts with read-only saved-evidence
audit, then completes source-runner/checkpoint/selection and final-only target
evaluation preparation as justified. No40k training or new mechanism is implied.

Scientific question: can the frozen-split source-only Static R=2 optimizer,
holdout and compact-checkpoint paths run with finite CUDA optimization while
preserving the backbone and avoiding all target data?

## Local verification

The full CPU suite is **333 passed, 1 skipped**, 11.50s, in Python3.10.20 / torch2.10.0+cu128 /
pytest8.4.2 environment (CUDA hidden, OMP threads1). New tests use synthetic
features, exercise the same20-update engine and checkpoint roundtrip, verify
data-audit rejection cases and ensure the real launcher refuses CPU before
weight loading. No local DINO weights or GPU work. One existing test skips
because Windows denies unprivileged symlink creation. Shell syntax, Python
compilation and diff whitespace checks also pass; this is not AutoDL runtime acceptance.

---

# Archive: completed full source pixel/preprocessing gate — INACTIVE

Only this section is active. All transfer/export and Phase16 commands below
the archive marker are inactive. Do not repeat the successful exports.

## Local completion and remaining evidence

Real extracted files arrived in `source_split_export/`. All individual hashes,
checksums, producer/report fields, canonical IDs and deterministic membership
recomputation pass. The actual train22469/val2497 manifests and provenance are
frozen in Git at `4314d61`. `.gitattributes` preserves LF on all checkouts.
ZIP bytes were not received and are not claimed locally verified.

New source-only data routing and M0 data-gate config are implemented. The
config blocks training before CUDA/data access. Legacy configs preserve their
previous behavior. Query representation, model/loss mathematics, old results,
and REIN remain unchanged. No new mIoU/effect measurement exists.

Next verification requires the actual remote GTA5 payload: full image/label
decoding, train-ID and geometry checks, every fixed source-holdout transform,
32 fixed training crop/photo samples. No GPU, DINO weights, model forward,
optimizer updates or target access. This is not M0 training or a final test.

## Launch on exact commit

Use the literal full SHA from Codex's final response. The script requires the
existing DINOv3 environment at `/root/autodl-tmp/envs/causalq-dg/bin/python`;
it uses that environment's CPU packages with CUDA hidden. Do not install or
modify environments if the preflight fails; return the error.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
EXPECTED_SHA=FULL_40_CHARACTER_SHA_FROM_CODEX
git fetch origin codex/strict-dg-source-split || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || { echo dirty_checkout; git status --short; exit 1; }
BASE=/root/autodl-tmp/outputs/CausalQ_DG
mkdir -p "$BASE/phase17"
LAUNCH_LOG="$BASE/phase17/STRICT_SOURCE_DATA_${EXPECTED_SHA:0:7}_launcher.log"
test ! -e "$LAUNCH_LOG" || { echo existing_launcher_log; exit 1; }
nohup bash scripts/check_strict_source_data_phase17.sh "$EXPECTED_SHA" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "source_data_pid=$!"
echo "launcher_log=$LAUNCH_LOG"
```

This decodes all 24,966 pairs sequentially, then all 2,497 holdout inputs;
it may take time and continues if the terminal disconnects. No source or
dataset file is written. Evidence output is exclusively outside both trees.
Source membership/path/size digest must still match the accepted inventory.

Inspect progress from any new terminal (same checkout):

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
SHA=$(git rev-parse --short=7 HEAD)
tail -n 40 "/root/autodl-tmp/outputs/CausalQ_DG/phase17/STRICT_SOURCE_DATA_${SHA}_launcher.log"
```

The gate prints split start/end and holdout progress, then the full report and
`source_data_exit_code`. If any check fails, preserve evidence and return it.
Do not modify source/dataset files, retry under a new run ID, or start training.

## Return outputs and acceptance

```text
/root/autodl-tmp/outputs/CausalQ_DG/phase17/
  STRICT_SOURCE_DATA_<sha7>/source_data_report.json
  STRICT_SOURCE_DATA_<sha7>/SHA256SUMS
  STRICT_SOURCE_DATA_<sha7>.log
  STRICT_SOURCE_DATA_<sha7>_launcher.log
```

Return the complete JSON, SHA256SUMS and final launcher output (and the error
log if any). Small files may be copied into the chat; no source images,
weights or complete run directories should enter Git.

Acceptance requires exit0, ok true, clean exact SHA, recorded environment,
unchanged pair/manifest digests, complete train/val scans 22469/2497, zero
invalid label IDs/unreadable/all-ignore/incompatible geometry counts,
successful 2497 holdout and32 train-photo checks, finite tensors and unchanged
holdout torch RNG. Scale-equivalent label alignment is explicitly allowed and
counted. Holdout uses full-image aspect resize to longest edge1024, nearest GT
and bottom/right ignore padding to16; it is not original-resolution GT.
Report retains target_data_accessed=false, training_authorized=false and
optimizer_updates=0. Stop even on success. Pixel/content checks do not prove
scene independence, anchor effect hypotheses or target generalization.

Next local work after evidence: read-only audit; complete the strict optimizer,
checkpoint/source-selection evidence and bounded GPU smoke preparation. A new
final-only target evaluator and model/hyperparameter freeze still precede any
Cityscapes access. No long training or Query decomposition is authorized now.

## Change verification

Full CPU suite: **322 passed, 1 skipped**, 12.28s. Includes actual frozen hash
checks, synthetic strict versus legacy routing, forbidden target/config drift,
pre-CUDA training block, deterministic holdout geometry/RNG, invalid train IDs,
full synthetic pixel/preprocessing gate and inventory drift rejection. The
skip is the existing Windows unprivileged symlink test. The extracted-file
import test also passes. Shell syntax and diff whitespace checks pass.
Local environment is Python3.10.20 / torch2.10.0+cu128 / pytest8.4.2, CUDA
hidden and OMP threads1; no GPU/runtime acceptance is claimed.

Scientific question: do frozen source memberships match actual complete GTA5
payloads, and do train/holdout preprocessing satisfy label/geometry constraints
without using target data? This is Priority0 protocol acceptance only.

---

# Archive: completed local transfer and freeze instructions — INACTIVE

Only this section is active. The successful inventory/export commands below
the archive marker must not be rerun. No training or target access is needed.

## Accepted operator return; local bytes still missing

The `57a724428b60d0f6636db056071bc620d45f45e0` producer reports exit 0,
22469 train / 2497 val out of 24966, overlap0, exact partition and valid
train/val/report checksums. The read-only result review is in
`../experiments/STRICT_DG_PHASE17/source_split_export_review.md`.
No source/target mIoU or effect diagnostic was measured. This result supports
the membership-export engineering gate only, not model hypotheses or scene
independence. Full membership has not been independently verified locally.

The user supplied the console output but no real metadata ZIP attachment.
Repository search found only synthetic test fixtures, which must not be used.
The missing ZIP is the exact evidence required to complete local Git freeze;
hash strings and sample counts cannot reconstruct the actual filenames.

## AutoDL operation: obtain the existing file, no checkout or rerun

Keep the producer checkout `57a7244`; the new importer runs locally after
transfer, not on AutoDL. To locate and confirm the file, run:

```bash
BUNDLE=/root/autodl-tmp/outputs/CausalQ_DG/phase17/GTA5_SOURCE_SPLIT_57a7244_20261016T140258_54816/source_split_export.zip
ls -lh "$BUNDLE"
sha256sum "$BUNDLE"
```

Expected SHA256:

```text
a555c740546a992ec72199f314a750699acf3782ceda1c98e9cf8908206b96db
```

Download this exact ZIP through AutoDL's file panel, then attach it to this
chat or put it at a named path within the local project and tell Codex that
path. The ZIP contains only two ID manifests, report and checksums. It does
not contain raw data or model weights. Preserve the existing output and log.
No new output file, environment setup or GPU action is required.

## Local completion prepared

`tools/import_gta5_source_split.py` is ready. After receiving the actual ZIP,
Codex will run (locally):

```powershell
python -m tools.import_gta5_source_split --bundle <ACTUAL_LOCAL_ZIP_PATH>
```

It verifies the pinned ZIP/manifest hashes, archive contents, all checksums,
producer/report identity, canonical IDs, counts, disjoint membership, and
independently recomputes H(i)=SHA256(fixed salt + LF + exact ID) with the
first ceil(N*0.1) ranked IDs as holdout. It validates before creating a
destination and refuses to overwrite an existing manifest directory.
It then writes `manifests/gta5_source_train.txt`, `gta5_source_val.txt` and
`gta5_source_split.json` for local Git commit. Training remains unauthorized
until the subsequent source-only protocol/data/runner gates are prepared.

This turn changes only the importer, focused tests, result review and status
documentation. Existing model, losses, trainer, mathematical definitions and
historical experiments are unchanged. It does not manufacture or freeze any
real manifests while the actual returned bytes are missing.

Scientific/engineering question: do the returned manifest bytes match the
operator's evidence and the exact versioned split rule? Its answer awaits the
real ZIP. Scene independence and segmentation accuracy remain untested.

Local verification: 10 new importer tests pass; full CPU suite **306 passed,
1 skipped** in 9.80s, using the existing Python 3.10.20 / torch2.10.0 /
pytest8.4.2 environment, CUDA hidden and OMP threads=1. The skip is the existing
Windows unprivileged symlink test. This is not the pinned AutoDL DINOv3 runtime.
`git diff --check` and Python compilation also pass.

---

# Archive: completed source split export instructions — INACTIVE

Updated 2026-10-16. This is the only active action. All inventory and Phase 16
commands below the archive heading are inactive. Continue to keep Phase 16 closed.

## Returned result and decision

The operator reports exit 0, clean `f41dbee`, 24,966 complete flat-numeric pairs
and no metadata candidates. No model/effect metric was measured. See
`../experiments/STRICT_DG_PHASE17/source_inventory_review.md` for the read-only
review and exact input hashes. The full ID list is still remote, so source
membership has not been frozen in Git. Do not infer IDs from the count.

The next command verifies the original inventory SHA256 and pair digest,
producer SHA/status and expected structural fields, then exports the plan's
configurable deterministic 10% filename fallback. H(i)=SHA256(fixed salt + LF
+ exact ID); first ceil(N*0.1) ranked IDs are holdout. Expected train/val counts
are 22469/2497, independent of all training seeds. This does not establish
sequence/scene independence; correlated frames may cross the filename split.

## Exact command

Use the full SHA provided in Codex's final response (the copyable final block
contains the literal value). No packages, weights or GPU are required. Python
3.10+ standard library only; default python3 is the previously working 3.12.3.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
EXPECTED_SHA=FULL_40_CHARACTER_SHA_FROM_CODEX
git fetch origin codex/strict-dg-source-split || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || { echo dirty_checkout; git status --short; exit 1; }
PY=python3 bash scripts/build_gta5_source_split_phase17.sh "$EXPECTED_SHA"
```

Pinned input (preserve exactly):

```text
/root/autodl-tmp/outputs/CausalQ_DG/phase17/GTA5_SOURCE_STRUCTURE_f41dbee_20261016T135221_54521/inventory.json
SHA256: 7224adc8aaafbcaac412aa5ee3904c11ba10d3c81aac7dffea231259e21e0d25
```

The launcher locates this file under CAUSALQ_OUTPUT_ROOT if explicitly set.
Do not regenerate inventory, modify its JSON, alter the fraction, edit source
code, inspect Cityscapes or start training. On any failure preserve the new
failure report/log and return the error. Output directories are exclusive and
timestamped; previous artifacts are not overwritten.

## Outputs and acceptance

Under `/root/autodl-tmp/outputs/CausalQ_DG/phase17/`:

```text
GTA5_SOURCE_SPLIT_<sha7>_<UTCstamp>_<pid>/
  gta5_source_train.txt
  gta5_source_val.txt
  split_report.json
  SHA256SUMS
  source_split_export.zip
GTA5_SOURCE_SPLIT_<sha7>_<UTCstamp>_<pid>.log
```

The ZIP contains only the two ID manifests, report and checksums, not the raw
inventory, datasets, checkpoints or code. Return the **ZIP and complete console
output**, including its SHA256, plus the log on error. Download the ZIP from
the printed path in AutoDL's file panel and attach it here; the actual ID bytes
are required for local Git freeze. A console summary alone cannot supply them.

Acceptance: `split_exit_code=0`, `ok: true`, source count 24966, train22469,
val2497, `overlap_count=0`, `exact_partition: true`, all three file checksum
checks OK, original inventory and pair digests match, clean current Git SHA.
Report must retain `candidate_pending_local_git_freeze`, `frozen_in_git: false`,
`training_authorized: false`, `dataset_files_opened: false`,
`target_data_accessed: false`. Stop even on success. This is a deterministic
membership engineering gate, not a dataset-decoding or accuracy gate.

Scientific question: can the reviewed full source ID inventory be partitioned
reproducibly with exact coverage and no membership overlap, independently of
training seed? Scene independence is outside what these data can establish.

Next local action after the ZIP returns: read-only hash/membership verification,
recompute the rule, commit the actual manifests and frozen metadata, then
continue strict source-only loader/trainer preparation in Priority 0. Query
decomposition is still later, and no historical experiment is changed.

## Local verification for this change

Full CPU suite: **296 passed, 1 skipped**, 9.33s; 28 new tests cover membership,
seed/order independence, exact fractional rounding, golden membership,
malformed IDs, overlapping/incomplete splits, pinned evidence verification,
metadata/target flag rejection, ZIP/checksum roundtrip, failure preservation,
exclusive outputs and CLI operation without third-party packages. Skip is the
previous Windows unprivileged-symlink test; simulated link rejection still passes.
Shell syntax and `git diff --check` pass.

Environment: existing `E:/Anaconda/envs/pytorch/python.exe`, Python 3.10.20,
torch 2.10.0+cu128, pytest 8.4.2 from ignored `.test-deps-strict-dg/`;
CUDA hidden, OMP threads=1. This differs from the pinned DINOv3 GPU runtime;
only CPU checks are claimed. The default Python 3.12 lacks torch, as recorded
in the archived initial gate below.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.test-deps-strict-dg')
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
& E:/Anaconda/envs/pytorch/python.exe -m pytest -p no:cacheprovider --basetemp=.test-tmp-strict-dg-split-full
```

---

# Archive: completed source structure inventory instructions — INACTIVE

Updated 2026-10-16 under `CausalQ_DG_1016.txt`. Phase 16 remains closed.
Only the following source inventory is active. Every Phase 16 command below
the archive marker is INACTIVE and must not be executed.

## Current action

Scientific question: does the available GTA5 structure preserve sequence or
scene grouping suitable for an independent source holdout?

Local work: repository audit, new standard-library-only inventory command,
focused tests, strict-DG boundary documentation. Query/loss/training behavior
is unchanged. No M0–M4/P1 metrics exist yet. Do not start training, load weights,
or inspect Cityscapes. Even a successful inventory leaves split review pending.

## Exact checkout and command

Use the full 40-character SHA in Codex's handoff message for EXPECTED_SHA.
Do not substitute a floating branch tip. The final response contains the
complete copyable block with the literal SHA filled in.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
EXPECTED_SHA=FULL_40_CHARACTER_SHA_FROM_CODEX
git fetch origin codex/strict-dg-source-audit || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || { echo dirty_checkout; exit 1; }
PY=python3 bash scripts/inspect_gta5_source_phase17.sh "$EXPECTED_SHA"
```

No package installation or GPU allocation is needed. `python3` needs only its
standard library (Python 3.10+). The report records exact Git SHA, Git status,
Python executable/version and platform; CUDA is hidden by the launcher.
Default source root is `/root/autodl-tmp/datasets/gta5`. If GTA5 moved, set only
GTA5_ROOT to its actual source subtree before the launcher. Do not point it to
the common multi-domain dataset root. Source links/junctions are rejected to
avoid traversing into another dataset; report the error rather than moving or
deleting data. No source files are changed on AutoDL.

## Outputs and return checklist

Under `/root/autodl-tmp/outputs/CausalQ_DG/phase17/`:

- `GTA5_SOURCE_STRUCTURE_<sha7>_<UTCstamp>_<pid>/inventory.json`: complete pair
  IDs and paths, file sizes, directory counts, filename patterns, source-side
  metadata candidates and bounded text previews, provenance and pending status.
- The same directory's `SHA256SUMS`: SHA256 of inventory.json.
- Adjacent `<directory>.log`: console summary. Earlier files are not overwritten.

Return all three files (or the full inventory JSON plus hashes and complete
console output). The inventory itself is metadata, not raw images or labels.
Do not commit the full AutoDL output directory. Subsequent local review will
extract and commit only the necessary split evidence and manifests.

Acceptance: `inventory_exit_code=0`, `ok: true`, positive paired count, empty
missing/duplicate lists, successful `sha256sum -c`, exact clean Git SHA,
`target_data_accessed: false`, `images_decoded: false`,
`training_authorized: false`, `split_status: pending_metadata_review`.
`ok: true` certifies filesystem pairing only; pixel decoding, label values,
grouping independence and source split remain unverified.
On failure preserve files and return the error; do not repair source code or
dataset layout on AutoDL. Stop here even on success.

Next local step after returned evidence: read-only audit; decide defensible
grouping; implement and freeze deterministic source train/val manifests; then
repair the new training route to use source-only selection. Centered Query
decomposition follows only after the protocol gate.

## Local verification

- Default `python -m pytest` (Python 3.12.4, pytest 7.4.4) was attempted and
  stopped with 33 collection errors because that interpreter lacks torch.
- Reused existing `E:/Anaconda/envs/pytorch/python.exe`, Python 3.10.20,
  torch 2.10.0+cu128, numpy/Pillow/PyYAML; pytest 8.4.2 installed into ignored
  `.test-deps-strict-dg/`. CUDA hidden; OMP threads=1.
- Final full CPU suite: **268 passed, 1 skipped**, 9.64s. Skip: Windows host
  does not permit creating an unprivileged symlink for the traversal test;
  a separate simulated-link rejection test passes on this host.
- This local environment differs from the Python 3.12 / torch 2.7 pinned
  DINOv3 runtime. No GPU/model/runtime compatibility claim is made.

Reproduce local CPU checks in PowerShell:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.test-deps-strict-dg')
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
& E:/Anaconda/envs/pytorch/python.exe -m pytest -p no:cacheprovider --basetemp=.test-tmp-strict-dg-full
```

---

# Archive: Phase 16 completed handoffs — ALL INACTIVE

## Current action — none

The hash-pinned three-seed aggregate completed successfully at
`2e096bdf40d00d3a03f53f9ca2171ffd8c112aa8`. Its JSON SHA256 is
`43e30e8cbd5165b48210750d0a770f25a87d6ccc9e5755cbf06ae2a4385f3d8c`
and its log SHA256 is
`39a6dbc947e586281826526c937cdb6d42df94ce4d709bbe5ce920ef75d79a8e`.
It reports `ok: true` and `no_additional_long_training_authorized: true`.

Phase 16 is closed. The current project plan defines no subsequent authorized
GPU experiment, so there is no AutoDL command to run. Preserve all accepted
reports, traces and checkpoints. Do not rerun any seed and do not start a new
long run unless the project plan is explicitly extended with a new isolated
phase, hypothesis and acceptance rule.

## Completed three-seed aggregate — inactive

All three matched long runs and their individual audits pass. Their CQE-minus-
control deltas are `0.0027601539`, `0.0001636839`, and `0.0000350110` mIoU.
All directions are positive, but the three-seed mean/sample standard deviation
is `0.0009862829/0.0015375639` and the Student-t 95% interval crosses zero.
Mean CQE mIoU also remains below the unmodified REIN reference. No additional
training is authorized.

The only active action is a read-only aggregate over the three immutable audit
JSON files. It checks their exact SHA256 values, training commits, seeds,
exposure and paired-protocol fields, then records seed-level results, mean/
sample standard deviation, a training-seed Student-t interval, sign-test result
and the bounded conclusion. It does not load a model, dataset or GPU.

Run only the exact commit supplied by Codex:

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
EXPECTED_SHA=FULL_SHA_FROM_CODEX
git fetch origin codex/cqe-three-seed-aggregate || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || { echo "dirty_checkout"; exit 1; }

BASE=/root/autodl-tmp/outputs/CausalQ_DG
PY=/root/autodl-tmp/envs/causalq-dg/bin/python
SHA=$(git rev-parse --short=7 HEAD)
A0="$BASE/analysis/rein_phase16_query_residual_cqe_formal_8640215_audit_v2_96b193d.json"
A1="$BASE/analysis/rein_phase16_query_residual_cqe_formal_seed1_682d2b7_audit.json"
A2="$BASE/analysis/rein_phase16_query_residual_cqe_formal_seed2_e00d5f4_audit.json"
OUTPUT="$BASE/analysis/rein_phase16_query_residual_cqe_three_seed_${SHA}.json"
LOG="$BASE/analysis/rein_phase16_query_residual_cqe_three_seed_${SHA}.log"

test -x "$PY" || exit 1
test -f "$A0" || exit 1
test -f "$A1" || exit 1
test -f "$A2" || exit 1
test ! -e "$OUTPUT" || { echo "existing_output=$OUTPUT"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }

sha256sum "$A0" "$A1" "$A2"
set -o pipefail
"$PY" -m tools.aggregate_rein_query_residual_cqe_seeds \
  --audits "$A0" "$A1" "$A2" \
  --output "$OUTPUT" 2>&1 | tee "$LOG"
AGGREGATE_EXIT=${PIPESTATUS[0]}

echo "aggregate_exit_code=$AGGREGATE_EXIT"
cat "$OUTPUT"
sha256sum "$OUTPUT" "$LOG"
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
```

Return the complete output. Acceptance requires exit 0, `ok: true`, all three
expected seeds and audit hashes, and `no_additional_long_training_authorized:
true`. Stop after this read-only report.

## Completed seed-2 instructions — inactive

The corrected audit of the completed seed-0 producer passes at `96b193d`.
CQE reached `0.6528875888991673` mIoU versus `0.6501274350304209` for
no-CQE, a delta of `+0.0027601538687463956` (`+0.2760` percentage points).
The paired 2000-replicate image-bootstrap 95% interval is
`[0.002118600721601699, 0.003436525542854521]`, entirely above zero.
Inputs match per microbatch, the frozen base is unchanged, and both arms have
40,000 updates / 160,000 records. This is positive evidence conditional on one
training seed, not training-seed uncertainty. CQE remains `0.0031634086` below
the unmodified REIN reference.

Seed-1 at `682d2b7` passes its complete audit: no-CQE/CQE mIoUs are
`0.6502510455774392/0.6504147294870591`, a positive delta of
`0.0001636839096199605` (`0.0164` percentage points). Its paired image
bootstrap 95% interval `[-0.00016424239924645034,
0.0007291624669445709]` crosses zero. Across seeds 0--1, the paired CQE gain is
`0.001461918889183178 +/- 0.0018359815152454602` mIoU (mean +/- sample
standard deviation). Both point estimates are positive, but the magnitude is
not stable and neither CQE branch exceeds the unmodified REIN reference.

One final matched training seed is now authorized. Seed-2 uses integer seed
`20261002`; both no-CQE and CQE arms restart from the same fresh branch
initialization and receive the same new source order and per-step views. The
fixed accepted REIN seed-0 base remains identical. Every other setting remains
unchanged. Do not repeat seed-0/1, resume an old branch, change lambda, or
select a target checkpoint. The launcher and auditor receive the same explicit
seed and write seed-specific paths. Run only the exact commit named in the
Codex handoff.

The command block immediately below is the active seed-2 handoff. Older
seed-0/1 instructions later in this document are retained as history and are
inactive.

The user approved this formal comparison after the paired 20-update engineering
smoke/audit. It asks whether adding only the existing normalized CQE objective
improves Cityscapes mIoU for the same frozen-REIN class-query residual branch.

The dedicated producer and read-only auditor are implemented. Each formal pair
trains 40,000 optimizer updates per arm / 160,000 GTA5 microbatches per arm,
accumulation 4. It holds the accepted frozen REIN baseline, residual branch,
initialization, source order, original/photometric views, optimizer and accepted
40k PolyLR schedule fixed. Only the CQE arm adds normalized CQE at weight 1.0.
Both fixed-final branches are evaluated on all 500 Cityscapes validation
images, without target-based checkpoint selection. The audit reports a paired
image-bootstrap interval, which is conditional on these two models and is not
training-seed uncertainty.

The AutoDL data volume was expanded by 10 GB. Still inspect current free space;
the launcher and trainer both require at least 8 GiB free. Preserve all
existing runs, checkpoints, environments, datasets and evidence. The run is
launched with `nohup` to survive normal web-terminal disconnection. It is not
an exact-resume runner: if it fails, preserve all artifacts and ask before
starting a fresh run ID.

The first formal launch at `8c1396c4eba26757689d4e3b1771a2257deefc47`
is excluded. Python was invoked with a file path, so the repository root was
not on `sys.path` and `tools` could not be imported. The failure happened
before run-directory creation, dataset/model loading, GPU work or optimizer
updates; its empty report, stderr log, exit marker and launcher log remain
failure evidence and must not be overwritten. The corrected launcher performs
an import preflight and starts both producer and auditor with `python -m`.

Run from an AutoDL shell. First replace `FULL_SHA_FROM_CODEX` with the exact
40-character commit SHA provided by Codex; do not type the placeholder itself.

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
export OMP_NUM_THREADS=1
EXPECTED_SHA=FULL_SHA_FROM_CODEX
git fetch origin codex/cqe-seed2-handoff || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1
test -z "$(git status --porcelain)" || { echo "dirty_checkout"; exit 1; }

echo "===== GPU and disk preflight ====="
nvidia-smi
df -h /root/autodl-tmp
FREE_KIB=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
test "$FREE_KIB" -ge 8388608 || { echo "need_at_least_8GiB_free_kib=$FREE_KIB"; exit 1; }

echo "===== pinned input files ====="
test -f /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth || exit 1
test -f /root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_data_protocol_b645793_v1.json || exit 1
test -f /root/autodl-tmp/outputs/CausalQ_DG/analysis/rein_phase16_source40k_d6fc52c_saved_audit.json || exit 1
test -f /root/autodl-tmp/outputs/CausalQ_DG/REIN_SOURCE_ONLY_SEED0_40000_d6fc52c/last.pth || exit 1
sha256sum /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth
sha256sum /root/autodl-tmp/outputs/CausalQ_DG/REIN_SOURCE_ONLY_SEED0_40000_d6fc52c/last.pth

BASE=/root/autodl-tmp/outputs/CausalQ_DG
SHA=$(git rev-parse --short=7 HEAD)
export FORMAL_SEED=20261002
LAUNCH_LOG="$BASE/analysis/rein_phase16_formal_seed2_${SHA}_launcher.log"
PID_FILE="$BASE/analysis/rein_phase16_formal_seed2_${SHA}.pid"
test ! -e "$LAUNCH_LOG" || { echo "existing_launcher_log=$LAUNCH_LOG"; exit 1; }
test ! -e "$PID_FILE" || { echo "existing_pid_file=$PID_FILE"; exit 1; }
nohup bash scripts/run_rein_query_residual_cqe_formal.sh \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "formal_pid=$(cat "$PID_FILE")"
echo "launch_log=$LAUNCH_LOG"
```

To check progress without attaching to the training process, run:

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
BASE=/root/autodl-tmp/outputs/CausalQ_DG
SHA=$(git rev-parse --short=7 HEAD)
PID_FILE="$BASE/analysis/rein_phase16_formal_seed2_${SHA}.pid"
LAUNCH_LOG="$BASE/analysis/rein_phase16_formal_seed2_${SHA}_launcher.log"
LOG="$BASE/analysis/rein_phase16_query_residual_cqe_formal_seed2_${SHA}_v1.stderr.log"
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then echo "formal_process=running"; else echo "formal_process=stopped"; fi
tail -n 20 "$LOG" 2>/dev/null || true
tail -n 30 "$LAUNCH_LOG"
df -h /root/autodl-tmp
```

After completion, return the entire seed-2 audit JSON, both exit codes, report/audit
hashes, both mIoUs and their delta/95% paired image-bootstrap interval, Git
SHA/status and disk output. Stop after seed 2. No further long run is authorized;
the next step is a read-only three-seed aggregate with all seed-level deltas,
mean/sample standard deviation, direction count and training-seed interval.

## Previous producer instructions — completed; do not rerun

The first attempt at this gate stopped during preflight with
`Compact checkpoint coverage mismatch` before any training updates. Preserve
its report and log under the `90f2495` suffix; do not rerun or overwrite them.
The cause was that REIN's trainability hook had not been activated before
checking compact-checkpoint key coverage. The corrected producer calls
`model.train(True)` and validates the exact REIN adapter/decode-head trainable
set before restoring the accepted baseline. The retry uses a new commit suffix.

That retry reached the first source microbatch but stopped while computing
CQE: the pinned PyTorch 2.0 runtime rejects tuple dimensions in `Tensor.any`.
No optimizer updates occurred. Preserve its report/log and partial run under
the `ba61199` suffix. The CQE present-class mask now uses sequential dimension
reductions compatible with the pinned runtime; unit coverage includes multiple
samples and ignored pixels. Retry only at the next new commit suffix.

The no-CQE smoke at `14d3e5a` passed its producer and saved-evidence audits.
The next gate compares fresh branches with and without the existing CQE loss.
Each arm uses the same 80 GTA5 samples, identical initialization and the same
geometry-aligned original/photometric inputs; each runs 20 optimizer updates
with accumulation four. Both use mean two-view segmentation loss. Only the
candidate adds normalized CQE at weight 1.0. The frozen REIN model stays fixed.
This smoke performs no Cityscapes evaluation and gives no accuracy result.

The script creates a new run directory, two trace files, two branch-only
checkpoints, a report and stderr log. It checks the disk/input hashes, captures
a fingerprint for every paired example, and writes final hashes to the
terminal. It needs the isolated CUDA 11.8 environment. Preserve existing runs
and stop if any expected output path already exists. Use a fresh terminal:

```bash
(
cd /root/autodl-tmp/CausalQ_DG || exit 1
export OMP_NUM_THREADS=1
EXPECTED_SHA=<FULL_SHA_FROM_CODEX_HANDOFF>
test -z "$(git status --porcelain)" || { echo dirty_worktree; exit 1; }
git fetch origin main || exit 1
git checkout --detach "$EXPECTED_SHA" || exit 1
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" || exit 1

BASE=/root/autodl-tmp/outputs/CausalQ_DG
RUN="$BASE/REIN_QUERY_RESIDUAL_CQE_PAIRED_SMOKE_20UPDATES_${EXPECTED_SHA:0:7}"
REPORT="$BASE/analysis/rein_phase16_query_residual_cqe_${EXPECTED_SHA:0:7}_v1.json"
LOG="$BASE/analysis/rein_phase16_query_residual_cqe_${EXPECTED_SHA:0:7}_v1.stderr.log"
test -f /root/autodl-tmp/pretrained/dinov2_vitl14_rein_patch16_512.pth || exit 1
test -f "$BASE/analysis/rein_phase16_data_protocol_b645793_v1.json" || exit 1
test -f "$BASE/analysis/rein_phase16_source40k_d6fc52c_saved_audit.json" || exit 1
test -f "$BASE/REIN_SOURCE_ONLY_SEED0_40000_d6fc52c/last.pth" || exit 1
test ! -e "$RUN" || { echo "existing_run=$RUN"; exit 1; }
test ! -e "$REPORT" || { echo "existing_report=$REPORT"; exit 1; }
test ! -e "$LOG" || { echo "existing_log=$LOG"; exit 1; }

bash scripts/check_rein_query_residual_cqe_phase16.sh
git rev-parse HEAD
git status --short
df -h /root/autodl-tmp
)
```

Replace `EXPECTED_SHA` with the full commit SHA supplied in the response. Return
the complete report, per-arm update/loss summaries, `matched_inputs_identical`,
both checkpoint hashes, report/log hashes, exit code, Git SHA/status and disk
output. Acceptance requires exit 0, `ok: true`, 80 matched fingerprints,
20 distinct source indices and 20 updates per arm, finite values, matching
initial branch fingerprints, objective reconstruction error at most `1e-7`,
all required branch gradients, unchanged frozen base parameters and successful
checkpoint roundtrips. Stop after returning this evidence; a short smoke does
not authorize a 40k run.

## Completed branch smoke — do not repeat

Producer `14d3e5a` ran at seed `20260930` with accumulation 4, 80 distinct GTA5
microbatches and 20 optimizer updates. It used the frozen accepted REIN source
baseline and no CQE. Peak reserved GPU memory was 2.055 GiB. Its saved-evidence
audit passed at `161f077`; its audit JSON SHA256 is
`517ea9e74a1d2ad999182e957b1d20c8588d8d08fec1b70ac4eb14043fd1011b`.
It has no target mIoU.

## Completed paired CQE smoke — audit passed

Producer `3644785a22a4c9391bb9076b4cdad606c63be615` completed both arms with
seed `20260931`. Report/stderr hashes:
`c6a7ee96f1e2c4e2defd29e2dc07e8d439c48fde74a4f1e3f7776a4b428f5ddd` /
`15f7e575bf971c025c9cbe64b18e1051f9346458015edb8583f9965940ea591e`.
No-CQE/CQE trace hashes:
`1eb60f2459ea9a5a077303ea6cd8c0f7a66ad1caf3d013c3daecf74ed624e620` /
`f5544d5fa82e81959b9806dbc315921ecfbef4296e48f8c62aa0c8800d488bef`.
Branch checkpoint hashes:
`632929ccce21ee68e6a1b804de900ab30b4019f8391fda7710fd1b81153f78f1` /
`10410867c71843057798a45fb6a2ab86fe57c4e1b0e83491dea7836b79ff69a0`.
No target evaluation or accuracy claim is present.

## Completed source-only audit — do not repeat

The audit JSON reports `source40k_saved_audit_ok: true`; producer
`d6fc52c5af9dcf0d6f57218b8b448c7df7a74ea4`; final mIoU
`0.6560509975136082`; report/stderr hashes
`e1e95369809434603017a9c0539207c8f8db662dad9f0c8c669f8b6980099187` /
`82fe80372ac64042458cc5e5f1f2aceb5a228d8bf52ae799fa843eb5b99e3330`;
final/previous checkpoint hashes
`84231e98dda68ac4b1fd4f59cc97881887a614de01b01e2697323c1eac80daf2` /
`06fe23fde947bf61687f9fde803dbc0ba7ffc8bfebc1c15f480e7841178d7003`.
The audit artifact SHA256 is
`b1a8e15345ca72dac5e9f36550dc5dadb5e4b07423e8bb56d8500fb5fde719d1`.
