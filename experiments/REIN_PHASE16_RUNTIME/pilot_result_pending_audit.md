# Pilot result: engineering feasibility, saved audit pending

Producer: `b4e292d8da87d186f03ac4a4769f7464a2499ef8`; operator exit 0.
Report SHA: `c90169ad57ea39a937bbd2760aff3122758399c28dc6138a6822689212fc789a`.
stderr SHA: `83ac1b99558f5fde68b18b5a7f39f5520569d083bf1a91319d472dcfa490ca43`.

The returned 50-image diagnostic shows road 95.15%, sidewalk 54.00%, building
76.04%, vegetation 81.15%, sky 71.74%, car 51.88%, pole 11.46%, wall 5.44%
IoU, with the remaining classes at zero. Multiple semantic categories are now
learned, unlike the 20-update plumbing test. It is too early to infer failure
from minority-class zeros or success against the final DINOv3 baseline. The
image count, training length, decoder and pretraining differ.

Full report fields were not returned: validate 2,000 records / 500 updates,
runtime/weight pins, optimizer LR boundaries, finite loss/norms, original target
geometry, independent confusion aggregation, checkpoint SHA and roundtrip.
The audit reads existing evidence without GPU or checkpoint deserialization;
producer-side checks remain the source for frozen-gradient/finite-logit claims.

Storage: 9.3 GiB free on data volume. Inventory cache and volume identities;
then preferentially clear disposable installation caches. Any intermediate
checkpoint cleanup must preserve referenced/final/best artifacts and failed
attempt evidence. No deletion, installation, training or CQE in this handoff.

Local verification: 175 CPU tests pass on Python 3.10.20 / torch 2.10.0+cu128,
using the existing interpreter and base Anaconda pytest. New tests use fabricated
schema fixtures, not AutoDL results. `git diff --check` passes. Saved remote
report/checkpoint integrity and real storage usage await AutoDL; no GPU is used.
