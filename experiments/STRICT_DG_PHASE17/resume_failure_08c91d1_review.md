# Phase 17 deterministic CUDA CE failure — 2026-10-17

Evidence: the operator pasted the continuous worker report and launcher tail
for clean producer `08c91d10defc81ef72354f2de605814cf27849dc`. This is a
transcript-level failure record; original report/log bytes were not supplied
and no byte hash or checkpoint audit is claimed.

Runtime remains Python 3.12.3, torch 2.7.0+cu126, CUDA 12.6,
transformers 4.56.1, RTX 4090 D. The pinned source-evaluation report audit
passed. The continuous process exited 1 with:

```text
nll_loss2d_forward_out_cuda_template does not have a deterministic implementation,
but you set 'torch.use_deterministic_algorithms(True)'.
```

The native segmentation CE forward uses this CUDA NLL reduction. It failed
before the first optimizer.step() according to the pinned code ordering;
there is no successful update evidence. Prefix/resume/compare were not launched.
No resume report, successful continuation checkpoint or original SHA256SUMS
was expected: the old wrapper only generated the aggregate checksum after
all three workers and comparison succeeded. No mIoU/effect metric was measured,
and exact continuation is not accepted. Prior source-data, smoke serialization
and full-source evaluation acceptance remain unchanged.

The correction is isolated to the continuation CE backend. For valid pixels V:

L = -(1/|V|) sum_{p in V} log_softmax(Z.float(), class_dim)[y_p, p].

Use FP32 log_softmax, gather the correct class, zero ignored pixels and divide
the sum by the number of valid pixels. This is the same unweighted valid-pixel
mean CE objective and gradient, with a different reduction implementation;
bitwise equivalence to the old CUDA NLL reduction is not claimed. All three
continuation branches use the new pinned backend, which is recorded in
checkpoint metadata. The old smoke, evaluation and historical trainer keep
the native CE default. No model, label mapping, loss coefficient, schedule,
AMP, clipping or optimization budget changes. Do not disable strict
determinism or use warn_only. No new research loss is introduced.

Before any dataset or foundation-weight access, the new AutoDL worker checks
synthetic CUDA forward/backward against native CPU CE (loss <=1e-5, gradient
max error <=1e-6), and repeats the CUDA calculation with exact loss/gradient
equality. Those tolerances apply only to equivalent CE implementations;
the 8 versus 4+4 continuation trace/final-state comparison stays exact.
CPU tests cover value/analytic-gradient/native-gradient agreement, ignored
pixels, FP32/BF16/FP16, noncontiguous tensors, custom ignore IDs, extreme logits
and invalid inputs. Fresh-process stochastic CPU continuation is tested again.
Real CUDA support is still pending the next bounded AutoDL run.

Failure reports now include a traceback; the wrapper also emits checksums on
worker failure. Preserve the old failed directory and launch the new exact
commit into its own new output path. The remaining question is whether the
mathematically equivalent deterministic CE backend permits an exact stochastic
optimization continuation under the pinned single-GPU workers-0 conditions.
