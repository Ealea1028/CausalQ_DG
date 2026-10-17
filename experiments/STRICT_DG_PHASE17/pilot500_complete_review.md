# Complete source runner pilot acceptance — 2026-10-17

The gzip/Base64 return decodes to the original 24,277-byte pilot_report.json
and 1,194-byte checksum manifest. Hashes exactly match the supplied values:

- report: d5b4747b7200d40722728cd797da9f736dedb15d9310c3a293aa8a51b98ffd1b;
- manifest: 8ea73a530d6e43e78b1dea19795b898070fd0259f69f3dc86956730eeb3eb615.

The original manifest also pins the report. Re-serializing the complete nested
worker objects reproduces both original worker-report hashes. Worker artifact
digests match the manifest; raw checkpoints and traces were checked remotely
but were not supplied for independent local raw-file auditing.

Clean producer 482ef2663f3ecf0f2f5d5545710aefd0fd2a9520; root summary is
ok, exactly 500 updates with independent-process restart at 250, fixed-final
finish/state.pth. Runtime/config/split/backbone metadata agrees across workers.
The selected checkpoint digest is
4f78ed5f3532eb30aeeff7690781097f357257ac84231f85a3b31e775ecd211f.

The complete 2,497 source holdout has 1,276,971,688 valid pixels. Local
integer-confusion/per-class/mIoU recomputation agrees within 1e-12. Source
mIoU 0.35724184600067665, image-mean CE 0.3923538372, pixel-weighted CE
0.3940907270; largest reserved peak 1.330078125 GiB. Validation leaves model,
optimizer, all RNG unchanged and restores training mode. Target access and
formal training flags are false; pilot_is_formal_m0 is false.

This accepts the Phase17 source-development engineering path. The earlier
partial-return review remains provenance and is superseded for completeness.
No target mIoU, effect variance/magnitude, localization or responsibility was
measured, and no causal/anchor hypothesis is established. Do not interpret
the 20-versus-500-step score difference as a controlled mechanism comparison.
The filename split still cannot guarantee frame/scene independence.

No more Phase17 GPU repeats are needed. Priority1 Centered Query may be
implemented next as a separate change. Formal long runs stay blocked;
final target manifest/evaluator execution awaits architecture/weights freeze.
