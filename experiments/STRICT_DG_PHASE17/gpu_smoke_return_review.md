# Phase 17 GPU smoke return review

The initial attachment snapshot was read as experiment data, not instructions.
It reports clean producer `b16236c9989cc85b802b85a8abacde56deca9db7`, seed0,
torch2.7.0+cu126 / transformers4.56.1 / RTX4090D, 20 finite source updates,
nonzero decoder/query/scale gradients, zero alpha-initialization error, unchanged
backbone, exact model/optimizer roundtrips and prediction roundtrip error0.
Allocated/reserved peaks are1.002141/1.181641GiB. Final checkpoint is45,612,179 bytes.
The first trace objective is5.6752603; the last is2.8894626. This short trace
does not estimate convergence or generalization. Negative alpha is allowed.

The supplied checksum snapshot records:

- Report: `82dbbccd90e13720103a47a265f8b60afff4b27e72f72a9b78934bb510e7c54b`.
- Trace: `42a4319b2ea02d47984faff4ad890722fc75cdb0952f62fdae0230612e118c34`.
- Checkpoint: `701e0ce24d6c517baacd39d42d762b5cb1f9d2d1df5dd3465cf746642e62b383`.
- Frozen backbone state: `24e1950a1ad87645cf6f46532a80594b5102af3644ccb55062e689a5b16f71ad`.

After the initial read, `D:/smoke_report.json`, `D:/train.jsonl` and
`D:/SHA256SUMS` were no longer accessible. Full trace bytes were not copied
locally before this external-state change. Therefore no complete local
report/trace byte audit or local checkpoint inspection is claimed. Do not
fabricate the missing middle trace records or rerun training to recreate them.
`tools/audit_strict_source_smoke.py` pins all three digests and verifies the
original AutoDL report and full trace before the next model/data access.
The independent checkpoint hash/metadata/head-state audit then runs on AutoDL.

| Research observation | Status |
|---|---|
| Source/target mIoU | Not measured in the20-update smoke |
| Effect variance/magnitude/localization/responsibility | Not measured |
| Optimization/freeze/serialization compatibility | Positive report-level engineering evidence |
| Full saved-artifact audit | Prepared; required first step of the next AutoDL operation |
| Scene independence, M0 accuracy, anchor hypotheses | Not established |

Next gate is read-only full source holdout evaluation from that SAME immutable
checkpoint, not new training. It independently restores weights/head in a new
process, rechecks the eight saved source CE references within1e-5, evaluates
all2497 frozen source IDs, checks PyTorch and independent NumPy confusion/IoU
counts and unchanged model/RNG. Full-source scores are diagnostics of a
20-update model, not formal M0 or a target metric. No checkpoint selection,
new mechanism, target data or long run is authorized.
