# Phase20 diagnostic gate20 accepted — 2026-10-17

Complete report SHA7a158511756df69981fa36989b5e39b8e0af524f1807d58fbf62be4b9c4cc3ce
and original manifest SHAaed313bc9850190f94bafab9537f94c0728bd5c745179fe6028a60b34b26c3f6
match the gzip/base64 return byte-exactly. Clean producer9439771489c539d7e9a13b1576305ccd2e3b85fd,
same accepted Phase18 M1 checkpoint, pinned CUDA/HF runtime and deterministic
flags. All20 hash-selected IDs/seeds/replays, 40 backbone calls, zero updates,
1520 responsibility rows, 1140 effect rows and19 parameter pairs pass audit.
Parameter CSV reconstructs exactly; other CSVs remain remote and are covered
by producer hashes/operator checksum verification, not independent local raw audit.
All20 image summaries reproduce the aggregate effect means; model/RNG/original
files unchanged. Max decomposition error5.82e-11. Peak reserved2.330078125GiB.

Descriptive results on this engineering checkpoint/sample only:

| Component | Original mean absolute effect | Paired population variance | RMS-normalized pair MSE |
| --- | ---: | ---: | ---: |
| full | 0.00152048 | 6.15305e-10 | 0.00241130 |
| anchor | 0.00177651 | 8.08893e-10 | 0.00219225 |
| residual contribution | 0.000301886 | 4.47922e-11 | 0.00289978 |

Anchor raw variance is18.06x residual variance; the proposed raw-variance
inequality is not supported on this sample. RMS normalization reverses the
ordering but is a different pre-recorded diagnostic, not a replacement for
the raw hypothesis or proof of utility. Original/photo responsibility pixel
entropy0.558807/0.559010 nats (227 present image-class maps per view), versus
uniform R2 entropy ln2. This does not establish semantic specialization.

Current acceptance is diagnostic engineering only. Next remains Priority3:
full2497 source-holdout census and deterministic20-image/five-class visual
audit, preserving the same metric definitions/checkpoint. No new losses,
formal training, target access or method selection. Treat this checkpoint's
training age and filename-split correlation as explicit inference limits.
