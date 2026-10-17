# Accepted deterministic source continuation — 2026-10-17

The complete JSON was recovered from the first JSON object in the pasted
terminal transcript, retaining indentation and final LF. Its SHA256 exactly
matches `8bd864462f8119447619bbafaa947596409ac76003900183b4dfdcfed065fde9`.
The saved `resume_report.json` is hash-pinned and semantically checked by
`tools/audit_strict_source_resume.py`; `resume_return.txt` preserves the pasted
checksums/log tail. This is experiment evidence, not instructions.

Clean producer: `5dc6caaa58668eb84cd79410fc69a5d719f9134d`, exit 0.
Python 3.12.3 / torch 2.7.0+cu126 / CUDA 12.6 / transformers 4.56.1 /
RTX 4090 D. Worker PIDs 71127, 71164, 71197 are distinct. The compare report
at 2026-10-17 01:04 UTC corresponds to 09:04 CST.

| Engineering question | Observed result |
|---|---|
| Native CPU versus deterministic CUDA CE | Loss error 0; max gradient error 6.9849193e-10 |
| Two CUDA preflight replays | Exact under strict deterministic algorithms |
| Continuous 8 versus independent 4+4 | Exact input/loss/LR/gradient traces |
| Final parameters/AdamW/sampler/all RNG | Exact |
| Final continuous/resume checkpoint SHA256 | Both 9f2d7200...7b1e375 |
| Optimizer calls across processes | 16; effective final iteration 8 |
| Largest peak reserved memory | 0.951171875 GiB |
| Target access / formal training | False |
| Source/target mIoU and effect diagnostics | Not measured in this gate |

Raw checkpoints and complete training traces were not supplied locally. Their
digests are in the report and pasted checksum manifest, and all remote checks
show OK. This does not constitute an independent local raw checkpoint audit.
Previous source-data/smoke/full-holdout acceptance remains unchanged. The
earlier 08c91d1 NLL failure is retained as a failed attempt.

Acceptance is limited to this pinned deterministic single-GPU workers-0
runtime and validated CE backend. It does not establish workers-4 equivalence,
formal M0 accuracy, convergence or anchor hypotheses. The mathematical
objective remains original+photometric valid-pixel CE with lambda_cf=1 and
the 40k LR horizon; no model/loss coefficient change follows from this result.

Next is the single source-only runner integration change: schema-2 checkpoint
sampler epoch/cursor, explicit run metadata, streamed trace, fixed-final policy
and state-isolated full-source validation. A fixed 500-update GPU pilot runs
250 updates, restores in another process for 250 more and evaluates the full
2,497 holdout. No accuracy threshold, selection feedback, target access or
formal 40k run. New checkpoints start from pretrained weights, not the old
8/20-update engineering states. CPU epoch-boundary and >20-step tests precede
GPU use. The pilot does not repeat the accepted 8-versus-4+4 comparison.
