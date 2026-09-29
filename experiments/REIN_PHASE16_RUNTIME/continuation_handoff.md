# Accepted pilot and continuation boundary

Operator saved audit source: `3601ecd0ccc5425de3080d44ccca4186d72aebe7`, exit 0.
Pilot source: `b4e292d8da87d186f03ac4a4769f7464a2499ef8`.
All 2,000 records / 500 updates and 50 target metric counts pass.
Diagnostic mIoU 0.23518900978251867; first/last-20 objective means
122.30029792785645 / 57.560906982421876; zero prediction roundtrip error.
Checkpoint SHA `b9747a71b1871e201a7730d7b92ca1daf836c7fdc0bcf61b06bd2c9b40fbfda0`,
283,250,650 bytes; peak reserved 4.217 GiB. This remains optimization feasibility,
not a comparison with the final DINOv3 baseline or successful CQE transfer.

Storage: 9.3 GiB available data volume. System-overlay caches hold only 8 MB pip
cache and 154 MiB disposable conda archives. Data-volume output directory is
18 GiB; most accepted runs retain only final checkpoints. Potential larger
intermediate collections (three ViT-B and one learned-null) must be inspected
with final/best/referenced preservation before any removal. No removal here.

Long-run infrastructure requires model/buffers, optimizer moments, scheduler,
source permutation/cursor/independent generator, Python/NumPy/Torch CPU and CUDA
RNG. The new primitive/tensor-only format safely loads with weights_only=True.
CPU tests verify serialization and epoch rollover, RNG replay and exact dropout
AdamW continuation. The GPU gate replays update 21 twice from the same disk
20-update checkpoint; it is explicitly not fresh-process resume verification.
The sampler matches the accepted seed-0 source prefix. No method/protocol changes.

Local verification: 180 CPU tests pass on Python 3.10.20 / torch 2.10.0+cu128,
using existing PyTorch interpreter and base Anaconda pytest. No local GPU,
real data, REIN model or checkpoint loading. Runtime compatibility of new safe
states and GPU continuation equivalence require AutoDL. Formal 40k runner is
still deferred, and no other target datasets or CQE are introduced.
