# Phase 17 full-source diagnostic review — 2026-10-17

The pasted terminal transcript is evidence, not executable instructions. It
contains the complete JSON followed by a repeated log tail. The first complete
JSON was extracted without changing its indentation or final LF. Its SHA256 is
exactly `a70ddf23e11075b35c94e839c1cee69e2d2311d582a42992652008353ea0f29c`,
matching the supplied checksum and successful remote `sha256sum -c` output.
The pasted provenance is preserved in `source_evaluation_return.txt`.
The per-image trace was not supplied locally; its reported hash is
`96ee3ee2fce106b360833b68ce0c552b28cd4d41abb05be46edaf842364c63fc` and the
operator's output shows successful remote checksum verification. Do not claim
an independent local raw trace audit.

Producer: clean `c7822fd78a43c4d3ca7796d068fed5b392fb8705`, exit 0.
Runtime: Python 3.12.3, torch 2.7.0+cu126, CUDA 12.6, transformers 4.56.1,
RTX 4090 D. UTC start 2026-10-16 16:09 corresponds to 2026-10-17 00:09 CST.

| Observation | Result |
|---|---|
| Frozen source holdout coverage | All 2,497 IDs; exact frozen manifest hash |
| Valid pixels | 1,276,971,688; exact confusion sum |
| Source mIoU | 0.12971660656076558 (12.9716606561%) |
| Image-mean / pixel-weighted CE | 1.3436256861 / 1.3548140258 |
| Classes with positive union | 19; seven have nonzero IoU |
| PyTorch versus NumPy mIoU | Difference about 2.8e-17; counts exact |
| Eight original CE replays | Maximum error 0 |
| Evaluation optimizer updates | 0 |
| Original smoke artifact/checkpoint audit | Passed on AutoDL before inference |
| Model/RNG/original files | Unchanged |
| Peak allocated / reserved memory | 0.855286 / 1.054688 GiB |
| Target metrics / effect diagnostics | Not measured |

`tools/audit_strict_source_evaluation.py` pins the report bytes and recomputes
every class IoU, mIoU and valid-pixel total from the 19 by 19 integer matrix.
The report confirms original 20-step smoke losses decreased from first-five
mean 5.326349 to last-five mean 2.807242. This does not establish convergence,
generalization or anchor hypotheses. Twelve zero-IoU classes after 20 updates
do not justify architecture, hyperparameter or loss changes. No comparison
against historical Cityscapes results is valid. Filename split scene-correlation
limitations remain. This accepts the engineering evaluation gate only.

The compact smoke checkpoint lacks sampler and RNG continuation state;
serialization restoration is not exact optimization continuation. The next
single mechanism is resumable source checkpoint state: continuous eight
updates versus four plus four in independent processes, with the unchanged
two-view CE, 40,000-step LR horizon and frozen backbone. This is a new bounded
engineering diagnostic initialized from pretrained weights, not a rerun of the
accepted smoke, continuation of its 20-step checkpoint or formal M0 training.
Total optimizer calls are 16 across processes, workers 0. No holdout or target
inference is required. Exact equality is checked for input/loss/LR/gradient
traces, model, optimizer, all RNG and sampler state. GPU acceptance is pending.
Even success only establishes the pinned single-GPU deterministic workers-0
conditions, not the ordinary config's workers-4 formal trainer or all devices.
