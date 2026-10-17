# Phase20 complete — no further transfer or GPU analysis

The extracted directory was supplied locally at analysis_export/ (ignored
by Git). All64 file hashes pass; independent CPU/math.fsum CSV recomputation
reproduces all image/class/slot/global aggregates, masks, denominators and
variance comparisons. All60 PNGs decode; all20 fixed samples were visually
reviewed in three contact sheets, with native-size first/last spot checks.
Pinned local records: raw_review.json and completion_review.json, audited by
tools/audit_query_census_completion.py. No transfer or inference repeat.

Priority3 diagnostic engineering is complete. Raw anchor stability remains
unsupported at this20-step checkpoint; no new Anchor-CQE loss. Next sequential
local scope is M0/M1 formal-comparison readiness, including independent M1
continuation using the schema2 source runner. Formal runs/target access remain
blocked until that dedicated handoff. Earlier transfer requests are archived.

---

# Archive: Phase20 transfer existing analysis archive — INACTIVE

Full census finished at79e440b. Complete report/checksum bytes are locally
verified by tools/audit_query_component_census.py; see
experiments/STRICT_DG_PHASE20/census_review.md. Raw CSVs/sample index/actual
source figures are still absent locally. Do not repeat inference, plots,
report export or training. Current action needs no GPU or checkout/fetch.

The full report states2497 source IDs/4994 forwards, zero updates and60 PNGs.
Raw anchor/residual variance ratio18.67, with0/2497 images showing lower raw
anchor variance (per-image macro over19 class maps). Normalized ordering
differs; it cannot substitute for the raw hypothesis. No Anchor-CQE/new loss
is justified by these engineering-checkpoint numbers. Priority3 remains open
for raw CSV recomputation and visual QA; formal phases have not started.

## Verify the existing file on AutoDL

Paste this entire child-bash block. It only reads/checks the45MiB archive:

```bash
bash <<'BASH'
set -euo pipefail
ARCHIVE=/root/autodl-tmp/outputs/CausalQ_DG/phase20/QUERY_COMPONENT_CENSUS2497_79e440b/analysis_export.zip
EXPECTED=3010b35769c33408eae2af2b57c491d4e65b70880dffb24da35108e935be8d6b
printf '%s  %s\n' "$EXPECTED" "$ARCHIVE" | sha256sum -c -
ls -lh "$ARCHIVE"
BASH
echo "Archive check finished; terminal remains open."
```

In the AutoDL file panel, open that directory, download analysis_export.zip,
then attach the ZIP to this conversation. Returning its hash/text again does
not transfer the CSVs/PNGs. If already downloaded, attach the local ZIP or
provide its exact local path; do not upload pretrained weights/checkpoints.
The archive contains source previews and stays outside Git. Local byte
checker audit_census_archive() is ready; actual CSV recomputation and visual
inspection await the file. If download fails, return the error and preserve
the remote file. All launch/compute steps below are archived.

---

# Archive: Phase20 full-source census launch — INACTIVE

Gate20 accepted separately at8500976. Current action completes Priority3
analysis on the same20-step engineering M1 checkpoint: all2497 source holdout
images, one original and one seeded photo view each,4994 forwards, zero
updates. No new model, loss, checkpoint selection, mIoU or target access.

Local verification:493 passed/1 skipped, Python3.10.20/torch2.10.0+cu128,
CUDA hidden, OMP1/MKL1/sequential MKL. Existing Windows symlink test is skipped.
Synthetic overview/effect/responsibility figures were rendered and inspected;
Bash syntax and Git whitespace checks pass. Actual-source figures/full
metrics remain pending AutoDL.
The old gate20 is not rerun separately; its20 members are checked during the
full census for exact seed/input replay and metric agreement. Parameter CSV
must reproduce its accepted hash. Do not interpret this early checkpoint as
a formal M1/DG result or substitute normalized variance for the raw hypothesis.

Before GPU/data, audit all original files from Phase18,19 and gate20. Require
unchanged model/style/all RNG and original files after analysis. Expected
responsibility/effect CSV row counts189772/142329;19 parameter pairs.
Generate60 PNGs for the same fixed20 images: overview, signed effect maps,
native-grid responsibility. Show road/building/vegetation/car/person in every
sample, including absent/unsuccessful cases. Same signed scale across all6
effect maps for each class; fixed[0,1] responsibility scale.

## Exact checkout and safe launch

Replace FULL_40_CHARACTER_SHA_FROM_CODEX with the literal final handoff SHA.
Paste the entire child-bash block. Preserve all previous run directories.
Matplotlib is already a project dependency; install the pinned version only
if missing. The script records plotting/runtime versions before inference.

```bash
bash <<'BASH'
set -euo pipefail
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=FULL_40_CHARACTER_SHA_FROM_CODEX
git fetch origin codex/strict-dg-source-split
git checkout --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
PY=/root/autodl-tmp/envs/causalq-dg/bin/python
"$PY" -c 'import matplotlib' || "$PY" -m pip install matplotlib==3.10.1
export CAUSALQ_DATA_ROOT=/root/autodl-tmp/datasets
export CAUSALQ_PRETRAINED_ROOT=/root/autodl-tmp/pretrained
export CAUSALQ_OUTPUT_ROOT=/root/autodl-tmp/outputs/CausalQ_DG
bash scripts/analyze_query_components_full_phase20.sh "$EXPECTED_SHA"
BASH
echo "Launch check finished; terminal remains open."
```

Pinned main runtime unchanged: torch2.7.0+cu126/transformers4.56.1, offline
DINOv3-L, BF16, GPU0, workers0, strict deterministic algorithms, TF32true,
cuDNNbenchmarkfalse, CUBLAS :4096:8. MKL sequential/one thread for plotting;
this avoids the local Windows NumPy/MKL native abort seen in initial plotting
QA without changing effect definitions or GPU precision.

## Monitor and return

```bash
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
tail -n 80 "/root/autodl-tmp/outputs/CausalQ_DG/phase20/QUERY_COMPONENT_CENSUS2497_${SHA}_launcher.log"
```

Expect component_census_exit_code=0 and ok=true. Export complete report and
checksum bytes only after completion:

```bash
bash <<'BASH'
set -euo pipefail
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
export COMPONENT_OUT="/root/autodl-tmp/outputs/CausalQ_DG/phase20/QUERY_COMPONENT_CENSUS2497_${SHA}"
cd "$COMPONENT_OUT"
sha256sum -c SHA256SUMS
/root/autodl-tmp/envs/causalq-dg/bin/python - <<'PY'
from pathlib import Path
import os, base64, gzip, hashlib
root = Path(os.environ['COMPONENT_OUT'])
for name, marker in [('component_effect_report.json','CENSUS_REPORT'), ('SHA256SUMS','SHA256SUMS')]:
    raw = (root / name).read_bytes()
    print(marker + '_SHA256=' + hashlib.sha256(raw).hexdigest())
    print(marker + '_GZIP_BASE64=' + base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii'))
PY
ls -lh analysis_export.zip
BASH
echo "Evidence export finished; terminal remains open."
```

Return final console and four hash/base64 lines. Also download/upload
analysis_export.zip using the AutoDL file panel: it contains full3 CSVs,
all2497 sample summaries/input hashes and60 PNGs for local result/visual QA.
It contains source previews and must never enter Git. Report retains compact
per-class/per-slot summaries and the20 replay members, not2497 verbose rows.
No new checkpoint or raw dataset bundle is produced.

If failed, preserve files and return full report/log without rerunning.
Stop here: no training, new loss or target access.

---

# Archive: Phase20 gate20 accepted — previous local scope

Complete returned report/checksum is pinned and accepted by
tools/audit_query_component_diagnostics.py. Do not repeat the20-image gate.
See experiments/STRICT_DG_PHASE20/gate20_review.md: raw anchor variance is
higher than residual variance on this early sample; no stability/utility
conclusion or permission to add a loss. Next local scope is the same
Priority3 full2497 census and systematic20-image visualization. No training
or target access. Launch instructions below are historical.

---

# Archive: Phase20 engineering gate20 — INACTIVE

Updated 2026-10-17. Phase19 accepted at 01f59e9; do not repeat its gate or
exports. Read QUERY_COMPONENT_DIAGNOSTICS.md. Current change is analysis only:
20 hash-selected source holdout images, original/photo views, same immutable
M1 checkpoint, zero optimizer updates. No new loss, target data or mIoU.

Local CPU verification: Python3.10.20 / torch2.10.0+cu128 with CUDA hidden;
474 passed, 1 skipped. Bash syntax and Git whitespace checks pass. The skip
is the existing Windows symlink-permission test; actual source/GPU diagnostics
remain pending AutoDL.

Before CUDA/data, verify exact Phase18 and Phase19 original files and frozen
source manifest. The tool restores model only. Require exactly40 forwards,
sample_count20, finite diagnostics with explicit missing regions, appearance
replay exact, full=anchor+residual <=1e-6, exact class routing and unchanged
model/style/RNG/original files. CSV counts: 19 parameter pairs, 1520
responsibility rows, 1140 effect rows. No threshold on semantic specialization
or effect stability; those are measurements, not correctness criteria.
Full2497 statistics and systematic heatmaps remain subsequent work.

## Exact commit launch

Use the full literal SHA from Codex's final handoff in this complete child-bash
block. A failed check cannot exit the parent terminal; existing results/logs
are preserved. Do not edit source code on AutoDL.

```bash
bash <<'BASH'
set -euo pipefail
cd /root/autodl-tmp/CausalQ_DG
EXPECTED_SHA=FULL_40_CHARACTER_SHA_FROM_CODEX
git fetch origin codex/strict-dg-source-split
git checkout --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
export CAUSALQ_DATA_ROOT=/root/autodl-tmp/datasets
export CAUSALQ_PRETRAINED_ROOT=/root/autodl-tmp/pretrained
export CAUSALQ_OUTPUT_ROOT=/root/autodl-tmp/outputs/CausalQ_DG
bash scripts/analyze_query_components_phase20.sh "$EXPECTED_SHA"
BASH
echo "Launch check finished; terminal remains open."
```

The script launches with nohup and logs early failures. Runtime remains
/root/autodl-tmp/envs/causalq-dg/bin/python, torch2.7.0+cu126,
transformers4.56.1, offline DINOv3-L, GPU0, workers0, BF16, strict deterministic
algorithms, CUBLAS :4096:8, TF32true/cuDNNbenchmarkfalse. Original directories:
phase18/M1_CENTERED_GATE20_9583925 and phase19/ANCHOR_COMPONENT_READONLY_f5ca5b5.
No new checkpoint is produced.

## Progress and return

```bash
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
tail -n 80 "/root/autodl-tmp/outputs/CausalQ_DG/phase20/QUERY_COMPONENT_DIAGNOSTICS20_${SHA}_launcher.log"
```

On completion expect component_diagnostics_exit_code=0 and ok=true. Return
final console output plus complete report/checksum via the block below. Keep
CSVs on AutoDL; return them later if requested. Report contains per-image
effect summaries, full parameter rows, metric counts and all file hashes.

```bash
bash <<'BASH'
set -euo pipefail
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
export COMPONENT_OUT="/root/autodl-tmp/outputs/CausalQ_DG/phase20/QUERY_COMPONENT_DIAGNOSTICS20_${SHA}"
cd "$COMPONENT_OUT"
sha256sum -c SHA256SUMS
/root/autodl-tmp/envs/causalq-dg/bin/python - <<'PY'
from pathlib import Path
import os, base64, gzip, hashlib
root = Path(os.environ['COMPONENT_OUT'])
for name, marker in [('component_effect_report.json','COMPONENT_REPORT'), ('SHA256SUMS','SHA256SUMS')]:
    raw = (root / name).read_bytes()
    print(marker + '_SHA256=' + hashlib.sha256(raw).hexdigest())
    print(marker + '_GZIP_BASE64=' + base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii'))
PY
BASH
echo "Evidence export finished; terminal remains open."
```

If failed, return the complete report and launcher log without removing or
overwriting files. Stop here: no training, new loss or target access.

---

# Archive: Phase19 accepted — no further AutoDL action

Complete report/checksum bytes are now pinned and accepted by
tools/audit_anchor_query_effect.py. All 16 embedded component rows reconstruct
the trace hash; exact source CE replay and lineage pass. Do not repeat the
gate or evidence export. Next local scope is Priority3 diagnostics, separately
committed after this acceptance. No training, new loss or target access.

---

# Archive: Phase19 evidence export — INACTIVE

Updated 2026-10-17 after operator completion at f5ca5b5. Terminal tails
report exit0, checksum OK, CE error0, unchanged model/RNG/files. See
experiments/STRICT_DG_PHASE19/partial_return_review.md. Full report/checksum
bytes are missing locally. No new GPU run or Priority3 work before that audit.

No checkout, fetch, GPU, weights or dataset access is needed. Paste the entire
block; failures remain inside the child bash and preserve the terminal:

```bash
bash <<'BASH'
set -euo pipefail
export ANCHOR_OUT=/root/autodl-tmp/outputs/CausalQ_DG/phase19/ANCHOR_COMPONENT_READONLY_f5ca5b5
cd "$ANCHOR_OUT"
sha256sum -c SHA256SUMS
/root/autodl-tmp/envs/causalq-dg/bin/python - <<'PY'
from pathlib import Path
import os, base64, gzip, hashlib, json
root = Path(os.environ['ANCHOR_OUT'])
report = json.loads((root / 'anchor_report.json').read_bytes())
assert report['git_sha'] == 'f5ca5b5ac604bdae0d233d631001e95ee4b6ac36', 'Unexpected producer'
for name, marker in [('anchor_report.json','ANCHOR_REPORT'), ('SHA256SUMS','SHA256SUMS')]:
    raw = (root / name).read_bytes()
    print(marker + '_SHA256=' + hashlib.sha256(raw).hexdigest())
    print(marker + '_GZIP_BASE64=' + base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii'))
PY
BASH
echo "Evidence export finished; terminal remains open."
```

Return checksum verification output and all four SHA256/base64 lines intact.
If export fails, return the full error and preserve the run directory.
No training, new loss or target access. All launch instructions below are
archived. This step transfers existing evidence, not a new experiment.

---

# Archive: Phase19 Anchor-only computation launch — INACTIVE

Updated 2026-10-17. Only this Phase19 section is active. Phase18 is accepted;
do not repeat its training. Local verification: 446 passed, 1 skipped on
Python3.10.20 / torch2.10.0+cu128, CUDA hidden. Bash syntax checks passed.

## Scope and acceptance

Read the existing Phase18 M1 checkpoint without optimizer construction or
updates. Eight frozen GTA5 holdout images, original and photometric views,
16 forwards total. Full and anchor-only paths share the same image features
and query head; repeat anchors in the same R=2 slots. No new loss, new
checkpoint, formal training, target access or mIoU evaluation.

The gate audits original report, checksum manifest, train trace and checkpoint
before GPU/data access and again after inference. Original run directory:
/root/autodl-tmp/outputs/CausalQ_DG/phase18/M1_CENTERED_GATE20_9583925.
Report SHA256: 3a09ece34eeb17718a90b2d2f53141aacecdce710b6f4aa4be53c3799979926e.
Checkpoint SHA256: 9b31426b13aaecb458866c010fdcf385774c1d085f3de48334485fa0d9996bf9.

Require ok=true, anchor_component_exit_code=0, zero optimizer updates,
sample_count=8, view_count=16 and backbone_forward_calls=16; original CE replay
max error <=1e-5; FP32 full=anchor+residual max error <=1e-6; class routing
exact; model/all RNG/original files unchanged; pinned runtime, weights and
head identity. Canonical effects retain pre-addition alpha*delta semantics.
BF16 base-addition roundoff is reported separately, without an acceptance
threshold. This establishes computation compatibility, not scientific
stability, utility or an independent residual causal effect.

## Exact checkout and launch

Replace FULL_40_CHARACTER_SHA_FROM_CODEX with the literal full SHA in the
final handoff. Preserve existing AutoDL outputs. Paste the ENTIRE block:
the explicit child bash isolates exit/set options from the terminal shell.
Earlier inline exit guards could close the interactive terminal on failure.
The launcher now checks prerequisites and retains existing artifacts/logs;
if a previous attempt exists, return its report/log instead of deleting it.

```bash
bash <<'BASH'
set -euo pipefail
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
bash scripts/launch_anchor_query_effect_phase19.sh "$EXPECTED_SHA"
BASH
echo "Launch check finished; terminal remains open."
```

Runtime remains /root/autodl-tmp/envs/causalq-dg/bin/python,
torch2.7.0+cu126 / transformers4.56.1, offline DINOv3-L, strict deterministic
algorithms, TF32 true, cuDNN benchmark false, CUBLAS workspace :4096:8.

Progress (also works in a newly opened terminal):

```bash
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
tail -n 60 "/root/autodl-tmp/outputs/CausalQ_DG/phase19/ANCHOR_COMPONENT_READONLY_${SHA}_launcher.log"
```

## Return complete evidence

When the process finishes, return the final console output and these complete
compressed report/checksum bytes. components.jsonl stays on AutoDL; the
report contains every component row. Do not upload or edit the checkpoint.

```bash
bash <<'BASH'
set -euo pipefail
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
export ANCHOR_OUT="/root/autodl-tmp/outputs/CausalQ_DG/phase19/ANCHOR_COMPONENT_READONLY_${SHA}"
(cd "$ANCHOR_OUT" && sha256sum -c SHA256SUMS) || exit 1
/root/autodl-tmp/envs/causalq-dg/bin/python - <<'PY'
from pathlib import Path
import os, base64, gzip, hashlib
root = Path(os.environ['ANCHOR_OUT'])
for name, marker in [('anchor_report.json','ANCHOR_REPORT'),('SHA256SUMS','SHA256SUMS')]:
    raw = (root / name).read_bytes()
    print(marker + '_SHA256=' + hashlib.sha256(raw).hexdigest())
    print(marker + '_GZIP_BASE64=' + base64.b64encode(gzip.compress(raw,mtime=0)).decode('ascii'))
PY
BASH
```

If the terminal previously closed, first inspect the old attempt without
rerunning it:

```bash
tail -n 80 /root/autodl-tmp/outputs/CausalQ_DG/phase19/ANCHOR_COMPONENT_READONLY_4347352_launcher.log
ls -ld /root/autodl-tmp/outputs/CausalQ_DG/phase19/ANCHOR_COMPONENT_READONLY_4347352*
```

Missing old logs are diagnostic, not a reason to close the terminal. If an
old report exists, return it before launching the new revision. The fix
changes launch handling only, not the GPU gate or experimental mechanism.

If the gate fails, return anchor_report.json and the launcher log. Stop here:
no training, new loss or target access is authorized by this handoff.

---

# Archive: Phase18 Centered Query engineering accepted — INACTIVE

The complete source gate report/checksum is pinned and accepted. Do not rerun
20-step training or eight-sample compatibility. No further Phase18 GPU work
is needed for this acceptance. Next sequential local scope is anchor-only
effect computation, from the same immutable centered checkpoint, with zero
optimizer updates. No new loss, formal training or target access.

---

# Archive: accepted Phase18 gate20 — INACTIVE

Updated 2026-10-17. Only this top section is active. Phase17 source-development
engineering is accepted with byte-pinned full pilot evidence at a9c0b39. Do
not repeat its smoke, continuation, pilot500 or full-source evaluation. Final
target manifest/evaluator execution remains deferred until architecture and
weights freeze; no target access or formal training is authorized now.

## Current single mechanism

Explicit opt-in centered_residuals=True in the existing bank/segmentor:
e_tilde=e-mean_R(e), q=a+e_tilde. Legacy default remains False; parameter
keys/shapes, initialization RNG consumption and raw get_query_residuals()
are preserved. New getters expose anchors and centered residuals. See
CAUSAL_QUERY_DECOMPOSITION.md for algebra and limits. No new loss, anchor-only
effect, CQE, style/attention change or extra queries. Static R=2, alpha0,
original+photometric CE, deterministic CE backend, AdamW1e-4/wd.01, clip1,
BF16 and absolute 40k PolyLR horizon are unchanged.

M1 config changes only experiment identity and centered flag from strict M0.
The ordinary strict config stays training-blocked. Dedicated CLI has no
budget override: exactly 20 fresh seed0 source updates, eight fixed source
holdout compatibility inputs, one compact checkpoint. No mIoU or accuracy
threshold. Checkpoint metadata pins the semantic centered flag; old M0 states
are not silently converted or reused. A fresh head shares the frozen backbone
for roundtrip checks; this is not independent-process M1 continuation evidence.

Before GPU/data: audit the exact accepted pilot report and original checksum
manifest. Runtime remains /root/autodl-tmp/envs/causalq-dg/bin/python,
torch2.7.0+cu126 / transformers4.56.1 / offline local DINOv3-L. Workers0,
strict deterministic algorithms, TF32 true, cuDNN benchmark false, CUBLAS
workspace :4096:8. Synthetic CUDA CE preflight precedes real source/weights.

## Exact checkout and launch

Use the literal full SHA from Codex's final handoff:

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
mkdir -p "$CAUSALQ_OUTPUT_ROOT/phase18"
LAUNCH_LOG="$CAUSALQ_OUTPUT_ROOT/phase18/M1_CENTERED_GATE20_${EXPECTED_SHA:0:7}_launcher.log"
test ! -e "$LAUNCH_LOG" || { echo existing_launcher_log; exit 1; }
nohup bash scripts/check_centered_query_phase18.sh "$EXPECTED_SHA" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "centered_query_pid=$!"
echo "launcher_log=$LAUNCH_LOG"
```

Progress:

```bash
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
tail -n 60 "/root/autodl-tmp/outputs/CausalQ_DG/phase18/M1_CENTERED_GATE20_${SHA}_launcher.log"
```

## Return files and acceptance

/root/autodl-tmp/outputs/CausalQ_DG/phase18/M1_CENTERED_GATE20_<sha7>/ contains
centered_report.json, train.jsonl, state.pth and SHA256SUMS, with an adjacent
.log and launcher.log. Keep checkpoint and full train trace on AutoDL.
Return centered_report.json, SHA256SUMS and final console output. To preserve
complete report bytes in a short paste, use this read-only export:

```bash
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
export CENTERED_OUT="/root/autodl-tmp/outputs/CausalQ_DG/phase18/M1_CENTERED_GATE20_${SHA}"
(cd "$CENTERED_OUT" && sha256sum -c SHA256SUMS) || exit 1
/root/autodl-tmp/envs/causalq-dg/bin/python - <<'PY'
from pathlib import Path
import os, base64, gzip, hashlib
root = Path(os.environ['CENTERED_OUT'])
for name, marker in [('centered_report.json','CENTERED_REPORT'),('SHA256SUMS','SHA256SUMS')]:
    raw = (root / name).read_bytes()
    print(marker + '_SHA256=' + hashlib.sha256(raw).hexdigest())
    print(marker + '_GZIP_BASE64=' + base64.b64encode(gzip.compress(raw,mtime=0)).decode('ascii'))
PY
```

Acceptance: centered_query_exit_code=0, ok true, clean exact commit and pinned
runtime; optimizer_updates=20; initial/final centered residual mean and query
mean-anchor max errors <=1e-6; alpha_zero_error=0; finite losses, branch
gradients nonzero by end, raw residual gradient sum <=1e-6; frozen backbone;
model/optimizer roundtrip exact, eight source prediction replays <=1e-5,
valid checksums and recorded GPU peaks. source_miou/target_miou null,
formal_training_authorized=false, target_data_accessed=false,
exact_resume_verified=false and roundtrip scope=new_head_same_frozen_backbone.
On failure return the full traceback/log and original report; do not weaken
thresholds, disable determinism, delete outputs or edit source on AutoDL.

## Local verification and research boundary

Phase17 acceptance was committed separately after 399 passed/1 skipped.
Phase18 adds 19 focused tests; full CPU suite 418 passed, 1 skipped in 35.40s.
The skip is the existing Windows unprivileged-symlink test. Local Python
3.10.20 / torch2.10.0+cu128 / pytest8.4.2, CUDA hidden, OMP threads1. These
checks are not real DINOv3/GPU acceptance. Shell syntax/diff checks pass.

Question: does the sole centered parameterization remain numerically valid,
trainable and checkpoint-compatible with the frozen DINOv3 Static-R2 source
pipeline? Algebra does not establish causal identifiability, performance gain
or anchor-only effect stability. No such hypothesis is accepted by this gate.

Stop after this bounded gate. Next local work after read-only GPU review is
the remaining Centered Query validation and, only afterward, separately scoped
anchor-only effect computation. Do not add CQE/new losses or launch formal
M1/M0 40k runs yet. Final target evaluation remains blocked.

---

# Archive: Phase17 source-development accepted — INACTIVE

Complete pilot report and original checksum manifest recovered byte-exactly.
See experiments/STRICT_DG_PHASE17/pilot500_complete_review.md and the pinned
auditor. No further Phase17 GPU run is needed. Do not rerun the 500-step pilot.
Formal training and target access remain blocked. Final target manifest and
evaluator execution are deferred to final architecture/weights freeze. The
next sequential local implementation scope is Priority1 Centered Query, with
a new bounded engineering gate prepared separately. No new loss/CQE now.

---

# Archive: completed full pilot return — INACTIVE

Updated 2026-10-17. Only this top section is active. The operator reports
source_runner_pilot_exit_code=0 at clean 482ef26. Do not rerun the 500 updates
or source evaluation. No formal training, target access or model change.

The pasted return begins in the middle of the JSON, not at the report header,
and contains checksum OK lines without the actual checksum digests. Both full
worker objects and the finish confusion matrix are present. Local read-only
analysis confirms identical worker metadata, distinct PIDs, 0–250/250–500
coverage, 500 total reported updates, complete 2,497 source holdout, exact
1,276,971,688 pixel sum and recomputed source mIoU 0.3572418460006766.
State-isolation flags are true; largest peak reserved is 1.330078125 GiB.
These are pilot diagnostics, not formal M0 or anchor/effect evidence.

Full report header, root selected-checkpoint fields and exact report checksum
are not available locally. Do not reconstruct missing fields or claim a
complete report byte audit. See pilot500_partial_review.md and preserved
pilot500_partial_return.txt. Local source code is unchanged.

## Required read-only operation

Keep the existing AutoDL checkout at producer
482ef2663f3ecf0f2f5d5545710aefd0fd2a9520; no new checkout is required.
Read the existing files and output two gzip/Base64 lines to keep the complete
return compact. This does not load datasets/weights or use GPU, train, infer,
write files, modify source or overwrite any experiment output.

```bash
OUT=/root/autodl-tmp/outputs/CausalQ_DG/phase17/M0_SOURCE_RUNNER_PILOT500_482ef26
(cd "$OUT" && sha256sum -c SHA256SUMS) || exit 1
/root/autodl-tmp/envs/causalq-dg/bin/python - <<'PY'
from pathlib import Path
import base64, gzip, hashlib, json
root = Path('/root/autodl-tmp/outputs/CausalQ_DG/phase17/M0_SOURCE_RUNNER_PILOT500_482ef26')
payload = (root / 'pilot_report.json').read_bytes()
report = json.loads(payload)
assert report['ok'] is True
assert report['git_sha'] == '482ef2663f3ecf0f2f5d5545710aefd0fd2a9520'
assert report['optimizer_updates'] == 500 and report['resumed_at'] == 250
assert report['target_data_accessed'] is False and report['formal_training_authorized'] is False
for name, marker in [('pilot_report.json', 'PILOT_REPORT'), ('SHA256SUMS', 'SHA256SUMS')]:
    raw = (root / name).read_bytes()
    print(marker + '_SHA256=' + hashlib.sha256(raw).hexdigest())
    print(marker + '_GZIP_BASE64=' + base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii'))
PY
```

Return the complete four lines beginning PILOT_REPORT_SHA256,
PILOT_REPORT_GZIP_BASE64, SHA256SUMS_SHA256 and SHA256SUMS_GZIP_BASE64,
plus checksum verification output. No checkpoint or raw data upload.

Next local step is decoding the exact report/checksum bytes, matching their
hashes and manifest, auditing full root/worker/metric contracts, then continuing
remaining Priority-0 closure work. Do not start another GPU experiment while
this returned metadata is incomplete. Current scientific conclusion is limited
to reported source-runner engineering success; source score does not test the
anchor hypothesis and is not comparable to historical target mIoU.

This return-review update changes only evidence/documentation/registry, not
model or training code. Full local CPU pytest: 393 passed, 1 skipped in 53.85s;
the skip is the existing Windows unprivileged-symlink test.

---

# Archive: completed operator-reported pilot500 — INACTIVE

Updated 2026-10-17. Only this top section is active. Prior inventory/data,
20-update smoke, full-source evaluation and corrected exact continuation are
accepted. Do not repeat them. Formal 40k training and target access stay blocked.

## Accepted return and limits

The recovered 5dc6caa resume_report.json exactly matches SHA256
`8bd864462f8119447619bbafaa947596409ac76003900183b4dfdcfed065fde9`.
Three independent workers show exact trace/model/AdamW/sampler/RNG equality
for continuous 8 versus 4+4; final checkpoint hashes also match. CUDA CE loss
reference error 0, gradient max error 6.98e-10; peak reserved 0.951172 GiB.
Raw traces/checkpoints remain remote; the operator returned all successful
remote checksum checks, not raw checkpoint bytes for local inspection. The
pinned report, pasted return and resume_review.md preserve these limits.
No source/target/effect result was measured by the continuation gate.

## Current single change

New independent tools/train_strict_source.py integrates the existing optimizer,
validated deterministic CE, source-only datasets, epoch/cursor/RNG checkpoint,
streamed training trace, immutable run metadata and state-isolated full source
validation. Shared label utilities move to label_space.py; Cityscapes public
helper exports remain compatible. The source-only runner imports neither the
Cityscapes module nor tools/train.py; a fresh poison-import CPU test enforces it.

Only the fixed 500-update engineering pilot is exposed by this CLI:

- prefix process: pretrained initialization, source updates 1 through 250;
- finish process: independently load backbone, restore prefix model/AdamW/
  sampler/all RNG, source updates 251 through 500;
- finish process: fixed-final checkpoint at 500, then full 2,497-ID source
  holdout evaluation; no best-mIoU selection or accuracy threshold;
- summary process: verify both worker reports/files/traces, fixed source order,
  40k LR sequence, complete source coverage and final selected head identity.

Total 500 new optimizer updates; no continuous-500 duplicate. No prior 8/20-step
checkpoint is reused. The checkpoint schema is now 2 to record sampler epoch,
permutation and next cursor and support epoch-boundary continuation. This
integration is not a new exact-500 comparison: exact CUDA continuation is the
accepted 8-versus-4+4 evidence, and generic multi-epoch continuation is CPU-tested.

Model/loss objective unchanged: Static R=2, alpha initially 0,
L=CE_original+CE_photometric, lambda_cf=1, AdamW 1e-4/wd .01, clip 1,
BF16, crop 512, seed 0. Absolute LR_next(t)=1e-4*(1-t/40000)^.9;
500 is the execution cap, not a shortened scheduler horizon. Single GPU,
workers 0, strict deterministic algorithms, cuDNN benchmark false, TF32 true,
CUBLAS workspace :4096:8. Workers-4 formal equivalence is not claimed. The
ordinary strict config/training entry remains blocked; no budget override.

Validation uses fixed full-frame aspect resize long edge 1024 and ignore-255
padding to 16, on resized GT. IoU=TP/(GT+Pred-TP), mIoU averages positive-union
classes; no target metric. PyTorch and NumPy integer confusion must match,
all 2,497 IDs and valid pixels must be covered, class/mIoU error <=1e-12.
CE uses the accepted deterministic backend. Model/optimizer/Python/NumPy/CPU-
CUDA RNG must remain unchanged and training mode restored after validation.
The selected finish/state.pth head must equal the evaluated head. Scores are
pilot diagnostics, not formal M0 or a basis for architecture tuning.

## Exact checkout and launch

Use Codex's latest literal full SHA:

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
mkdir -p "$CAUSALQ_OUTPUT_ROOT/phase17"
LAUNCH_LOG="$CAUSALQ_OUTPUT_ROOT/phase17/M0_SOURCE_RUNNER_PILOT500_${EXPECTED_SHA:0:7}_launcher.log"
test ! -e "$LAUNCH_LOG" || { echo existing_launcher_log; exit 1; }
nohup bash scripts/run_strict_source_pilot_phase17.sh "$EXPECTED_SHA" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "source_runner_pilot_pid=$!"
echo "launcher_log=$LAUNCH_LOG"
```

Existing runtime /root/autodl-tmp/envs/causalq-dg/bin/python; pinned torch
2.7.0+cu126 / transformers 4.56.1; offline local backbone. Synthetic CUDA CE
preflight still runs before real datasets/weights in each training process.
Progress prints every 10 updates and every 250 evaluation images. Keep the
GPU instance running until completion; terminal disconnect is safe via nohup.

```bash
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
tail -n 60 "/root/autodl-tmp/outputs/CausalQ_DG/phase17/M0_SOURCE_RUNNER_PILOT500_${SHA}_launcher.log"
```

## Return and acceptance

Output root: /root/autodl-tmp/outputs/CausalQ_DG/phase17/
M0_SOURCE_RUNNER_PILOT500_<sha7>/

- pilot_report.json with both worker summaries and full source diagnostics;
- SHA256SUMS covering all saved metadata/traces/checkpoints/logs;
- config.yaml, metadata.json and git_commit.txt;
- prefix/worker_report.json, train.jsonl, state.pth;
- finish/worker_report.json, train.jsonl, state.pth, source_eval.jsonl;
- prefix.log, finish.log, summary.log; adjacent launcher log.

Return **pilot_report.json, SHA256SUMS and final launcher output**. Preserve
all checkpoints and per-image/train traces on AutoDL; no raw datasets or
foundation weights enter Git. On failure return the failing worker report
with traceback and the full launcher log. Never modify source remotely,
remove failed outputs or continue into another budget.

Acceptance: source_runner_pilot_exit_code=0, ok true, clean exact commit,
500 total contiguous updates, resumed_at 250 in independent processes,
finite losses/gradients/parameters and fixed 40k LR/source permutation;
backbone unchanged, selected finish/state.pth identity matches evaluated head,
full source validation and exact training-state isolation, valid checksums,
recorded runtime/memory; target_data_accessed=false,
formal_training_authorized=false and pilot_is_formal_m0=false.
There is no source accuracy acceptance threshold.

## Local verification and question

New runner/auditor/launcher/pinned return tests; shared label helper extraction,
backward-compatible generic optimizer step and optional evaluator loss hook.
Old 20-step smoke budget/default native loss and legacy model/results remain
unchanged. CPU tests cover pinned acceptance/rejection, >20-step schedule,
multiple epoch boundaries with exact stochastic continuation, target-import
poisoning, label API compatibility, validation state isolation, selected
checkpoint tampering and no formal-budget override. Added 13 focused tests;
full pytest: 393 passed, 1 skipped in 34.75s. Local environment is Python
3.10.20 / torch 2.10.0+cu128 / pytest 8.4.2,
CUDA hidden, OMP threads 1. The existing Windows symlink skip remains.

Scientific/engineering question: can the independent source-only runner
perform a bounded train -> process restart -> train -> fixed-final source
validation cycle with complete provenance and no target access or state leakage?

Stop after this pilot. Review the evidence read-only locally before closing
remaining Priority-0 tasks or preparing centered Query. No formal 40k run,
new research loss or target evaluator execution now.

---

# Archive: accepted exact source continuation — INACTIVE

Active retry correction: 08c91d1 stopped in `nll_loss2d_forward_out_cuda_template`
before the first optimizer step because native CUDA NLL is unsupported under
strict determinism. The missing resume_report/SHA256SUMS are consequences of
the first worker failure, not missing successful results. Preserve the old
M0_SOURCE_RESUME_08c91d1 directory. Review is in
`experiments/STRICT_DG_PHASE17/resume_failure_08c91d1_review.md`.

The new exact commit uses FP32 log-softmax/gather/masked-sum CE only in the
continuation gate: L = -sum(valid log p_y)/count(valid). This is the unchanged
unweighted CE objective with a new reduction implementation, pinned in
checkpoint metadata. The native CE default used by old smoke/evaluation/training
is unchanged. Before loading any dataset or foundation weights, each worker
checks synthetic GPU loss/gradient against native CPU CE (1e-5 loss, 1e-6 grad)
and two exact GPU replays. Continuation state/trace equality remains exact.
Strict determinism stays on; never set warn_only or ignore deterministic errors.
Failure reports now include traceback and worker-failure checksums. Run the
same commands below with the NEW literal commit from Codex's latest handoff;
output is automatically a new SHA-named directory. No old output is overwritten.

Updated 2026-10-17. Only this top section is active. Complete source holdout
and original smoke audits are accepted; do not rerun them. No formal training
or target data access is authorized.

## Accepted evidence

The returned c7822fd report has exactly the supplied SHA256
`a70ddf23e11075b35c94e839c1cee69e2d2311d582a42992652008353ea0f29c`.
Its JSON was recovered from the complete pasted transcript, saved in Git and
independently audited. All 2,497 holdout IDs match the frozen manifest; the
1,276,971,688 valid pixels match the confusion sum. Recomputed class IoUs and
mIoU agree. Source mIoU is 0.12971660656076558, image/pixel CE is
1.3436256861/1.3548140258; all eight CE replays have error 0. Model, RNG and
original files are unchanged; peak reserved 1.054688 GiB. These are 20-update
diagnostics, not formal M0 accuracy. No target/effect metric exists. The
operator's pasted output verifies the remote trace checksum; local raw trace
bytes were not supplied. See `source_evaluation_review.md` for limits.

## Single new mechanism and fixed budget

The old compact checkpoint restores weights/optimizer but lacks sampler/RNG
continuation state. The next gate adds a separate compact state with model,
AdamW, absolute iteration/LR, fixed permutation/cursor, Python/NumPy/CPU-CUDA
RNG, config/split/backbone hashes and Git/runtime identity. Checkpoint writes
use a temporary file, flush/fsync and atomic replacement; existing outputs
are refused. No foundation weights enter the checkpoint or Git.

Three independent Python processes start the same source-only experiment:

- continuous: initialize from pretrained weights, updates 1 through 8;
- prefix: same initialization, updates 1 through 4, save continuation state;
- resume: independently initialize/load backbone, restore prefix model,
  optimizer/sampler/all RNG, updates 5 through 8.

Exactly 16 optimizer calls across the three processes. This is not continuation
of the old smoke checkpoint. Original and photometric CE sum, Static R=2,
AdamW 1e-4/wd 0.01, clip 1, BF16 and fixed 40,000-step PolyLR remain unchanged.
No sample-count/training-budget flag exists. Single GPU, workers 0, seed 0,
deterministic algorithms required, cuDNN benchmark false, TF32 true and
CUBLAS_WORKSPACE_CONFIG=:4096:8. Unsupported deterministic CUDA operators must
fail and be reported; do not disable determinism or loosen equality remotely.
No validation or target samples are read. Ordinary formal training stays blocked.

## Exact checkout and launch

Use the literal full commit from Codex's handoff:

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
LAUNCH_LOG="$BASE/phase17/M0_SOURCE_RESUME_${EXPECTED_SHA:0:7}_launcher.log"
test ! -e "$LAUNCH_LOG" || { echo existing_launcher_log; exit 1; }
nohup bash scripts/check_strict_source_resume_phase17.sh "$EXPECTED_SHA" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "source_resume_pid=$!"
echo "launcher_log=$LAUNCH_LOG"
```

Uses existing `/root/autodl-tmp/envs/causalq-dg/bin/python`, pinned torch
2.7.0+cu126 / transformers 4.56.1 and local offline DINOv3 weights. Launch log
survives terminal disconnect. Each process loads the backbone afresh.

```bash
SHA=$(git -C /root/autodl-tmp/CausalQ_DG rev-parse --short=7 HEAD)
tail -n 60 "/root/autodl-tmp/outputs/CausalQ_DG/phase17/M0_SOURCE_RESUME_${SHA}_launcher.log"
```

## Outputs and acceptance

`/root/autodl-tmp/outputs/CausalQ_DG/phase17/M0_SOURCE_RESUME_<sha7>/` contains:

- `resume_report.json` with exact comparison and all three worker summaries;
- `SHA256SUMS` covering reports, traces, checkpoints and per-process logs;
- `continuous/`, `prefix/`, `resume/`: `worker_report.json`, `train.jsonl`, `state.pth`;
- process logs including `compare.log`.

Return **resume_report.json, SHA256SUMS and final launcher output**. Keep all
checkpoints/traces on AutoDL. On failure return the failing worker report and
complete launcher log. Do not rerun into or delete existing output directories.

Acceptance requires exit 0, ok true, three distinct worker processes, exact
input/loss/LR/gradient traces (continuous == prefix + resume), exact final model,
optimizer, sampler and all RNG state, frozen backbone and pinned config/runtime.
No numerical tolerance is substituted for exact equality. Final iteration 8,
split iteration 4, total_optimizer_updates_across_processes 16,
formal_training_authorized false, target_data_accessed false and both mIoUs null.
Byte-identical torch.save files are not required; tensor/state equality is.

Scientific/engineering question: does compact source checkpoint state preserve
the stochastic optimization trajectory across a real process boundary? The
mathematical criterion is S_continuous(8) = S_resume(4+4), with S containing
parameters, AdamW moments/steps/LR, sampler cursor and every RNG state. CPU
synthetic tests already cover stochastic augmentation and dropout; GPU results
remain pending. Passing only covers this deterministic single-GPU workers-0
runtime, not the ordinary config's workers-4 trainer or arbitrary environments.

## Local changes and next boundary

Added full-source report auditor, continuation engine, AutoDL worker/comparator,
shell wrapper, focused tests, saved report/transcript and evidence review.
Updated status/protocol/result/registry docs and LF pin for the saved report.
No model, loss, existing smoke/evaluation APIs or historical results changed.
The corrected full CPU suite passes: 380 passed, 1 skipped in 30.09s; the
existing Windows unprivileged-symlink skip is unchanged. The CE correction adds
14 focused tests; the 17 earlier continuation tests also pass with the new backend.
Local environment: Python 3.10.20 / torch 2.10.0+cu128 / pytest
8.4.2, CUDA hidden, OMP threads 1. This is not DINOv3/GPU acceptance.

Stop after this bounded gate. No formal M0, centered Query or new consistency
loss. Next local work after reviewing GPU evidence is integration into the
source-only runner/checkpoint policy and final-only target-evaluation gate.

---

# Archive: completed full source holdout evaluation — INACTIVE

This section is historical. The c7822fd evaluation is accepted; do not rerun it.

## Evidence review and current scope

The initial `b16236c` attachment snapshot reports20 finite source updates,
unchanged backbone, exact model/optimizer restoration, zero prediction restore
error and1.181641GiB peak reserved memory. See Phase17
`gpu_smoke_return_review.md` for full reported digests and interpretation.
The original D:/ attachment paths became unavailable after their first read,
before a complete local raw-byte trace audit. No fabricated copy is saved.

The next AutoDL tool audits the ORIGINAL saved report and all20 trace records
against pinned SHA256 values before GPU/model/data access, then verifies the
ORIGINAL final.pth hash, provenance and restored trainable-state digest.
Only after that audit passes does it evaluate all2497 frozen GTA5 holdout IDs.
It does not construct an optimizer, update weights, select checkpoints, use
Cityscapes or change model/loss/config. Formal M0 training remains blocked.

## Inputs that must remain unchanged

```text
/root/autodl-tmp/outputs/CausalQ_DG/phase17/M0_STRICT_SOURCE_SMOKE_b16236c/
  smoke_report.json  82dbbccd90e13720103a47a265f8b60afff4b27e72f72a9b78934bb510e7c54b
  train.jsonl        42a4319b2ea02d47984faff4ad890722fc75cdb0952f62fdae0230612e118c34
  final.pth          701e0ce24d6c517baacd39d42d762b5cb1f9d2d1df5dd3465cf746642e62b383
```

Do not regenerate, rename or modify these files. The existing DINOv3-L weight
hash and runtime pins (torch2.7.0+cu126 / transformers4.56.1) remain fixed.
The tool loads local weights offline, hides no target permission behind a
generic evaluator, and has no sample-count or training-budget override.

## Exact checkout and launch

Use the literal full SHA in Codex's final response:

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
LAUNCH_LOG="$BASE/phase17/M0_SOURCE_HOLDOUT_${EXPECTED_SHA:0:7}_launcher.log"
test ! -e "$LAUNCH_LOG" || { echo existing_launcher_log; exit 1; }
nohup bash scripts/evaluate_strict_source_phase17.sh "$EXPECTED_SHA" \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &
echo "source_evaluation_pid=$!"
echo "launcher_log=$LAUNCH_LOG"
```

It runs in the existing `/root/autodl-tmp/envs/causalq-dg/bin/python` runtime
and survives terminal disconnect. Progress prints every250 source images.
It may take time to evaluate the entire source holdout. From a new terminal:

```bash
cd /root/autodl-tmp/CausalQ_DG || exit 1
SHA=$(git rev-parse --short=7 HEAD)
tail -n 40 "/root/autodl-tmp/outputs/CausalQ_DG/phase17/M0_SOURCE_HOLDOUT_${SHA}_launcher.log"
```

## Return files and acceptance

```text
/root/autodl-tmp/outputs/CausalQ_DG/phase17/
  M0_SOURCE_HOLDOUT_<sha7>/source_evaluation.json
  M0_SOURCE_HOLDOUT_<sha7>/source_eval.jsonl
  M0_SOURCE_HOLDOUT_<sha7>/SHA256SUMS
  M0_SOURCE_HOLDOUT_<sha7>.log
  M0_SOURCE_HOLDOUT_<sha7>_launcher.log
```

Return **source_evaluation.json, SHA256SUMS and final launcher output**; on
failure include the full error log. Preserve source_eval.jsonl and the original
smoke files on AutoDL; do not upload raw datasets or final.pth into Git.

Acceptance: `source_evaluation_exit_code=0`, ok true, clean exact commit and
recorded runtime; original artifact audit passes; restored iteration20 and
head/backbone digests match; exactly2497 source IDs with the frozen val hash;
finite logits/CE; eight original source CE references replay within1e-5;
PyTorch and independent NumPy int64 confusion counts agree exactly and sum
to all valid pixels; per-class/mIoU differences <=1e-12; model/RNG/original
files unchanged; valid output checksums and recorded peak GPU memory.

Metrics use the already fixed full-frame aspect resize (longest edge1024),
nearest-label resize and ignore padding to16. They refer to resized source GT.
IoU_c=TP_c/(GT_c+Pred_c-TP_c); mIoU averages classes with positive union,
absent classes are null. Also report mean-image CE and valid-pixel-weighted CE.
There is NO accuracy threshold or model-selection feedback in this gate.
The checkpoint has only20 updates: these are source diagnostics, not formal M0
or a target result. Report must retain optimizer_updates_this_run=0,
target_data_accessed=false, formal_training_authorized=false, target_miou=null,
diagnostic_only=true and exact_resume_verified=false.

Stop on completion or failure. Do not rerun optimization, alter thresholds,
repair files on AutoDL or launch40k training. Next local step is read-only
analysis of the full evaluation evidence, followed by the remaining source
runner/checkpoint/selection and final-only target-evaluation preparation.
Centered Query and new losses remain outside this Priority0 change.

## Local change and verification

New files: saved-smoke auditor, full-source metric engine/launcher, shell
wrapper, focused tests and smoke return review. Status docs updated; duplicate
LF attribute removed. Old model, losses, training entry points and experiment
results remain unchanged. Read-only evaluation replaces the previous next
action of training: the successful smoke is never repeated.

Synthetic CPU tests cover pinned byte audits, trace/protocol rejection,
known confusion/absent-class IoU, full membership/order, CE replay, invalid/
nonfinite labels/logits, restore hash/provenance/state and pre-CUDA audit order.
The full local CPU suite passes: 349 passed, 1 skipped in 14.06s. The skip is
the existing Windows unprivileged-symlink test. The exact commit is in Codex's
handoff. Environment is Python3.10.20 / torch2.10.0+cu128 / pytest8.4.2 with
CUDA hidden and OMP threads1, not GPU/runtime acceptance. No local real weights.

Scientific/engineering question: can the SAME smoke checkpoint be restored
independently and evaluated on the ENTIRE frozen source holdout with correct
streaming metrics, unchanged state and zero target access/new optimization?

---

# Archive: completed bounded source GPU smoke instructions — INACTIVE

This archived section is inactive. Do not repeat the accepted data scan or
the completed bounded GPU smoke.

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
