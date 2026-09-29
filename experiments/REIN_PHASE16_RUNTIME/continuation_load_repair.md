# Failed continuation attempt and optimizer-load repair

Producer `a10c1f6bfd02bfa0d53267a5df2d169ee7898bcf`, exit 1.
Stage continuation_replay; optimizer scalar comparison failed.
All 80 training records / 20 optimizer updates are finite; prediction checkpoint
roundtrip error zero. Neither fact overrides the failed continuation criterion.

Report SHA `a06408898be8b84f243326099c78916c0a10e72963da7db99c850e9f36b92e40`.
stderr SHA `46bc2079c4b3a31681e3209075ffc2c3fc59656e500c51eb0faf8b23e245f010`.
Checkpoint SHA `a9787b93cef0324e4d4149c15dff664c4d8b938d1ca7db0a65bd087cc90fcd76`,
283,344,599 bytes. Existing pilot acceptance remains intact. Data volume has
9.1 GiB free; no cleanup is necessary for one bounded retry.

The initial roundtrip directly passed payload['optimizer'] to the wrapper.
Pinned [MMEngine BaseOptimWrapper source](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/optim/optimizer/base.py)
shows load_state_dict pops base_param_settings from that argument, then restores
the optimizer. The modified payload loses the base-learning-rate settings;
later replays inherit the earlier branch's base LR. This explains the observed
scalar-only failure while model/LR-group checks had already passed. The old
traceback did not identify the exact differing scalar, so the repaired tool
also records nested paths for any remaining mismatch.

All restores now use cloned optimizer/scheduler trees, require base settings
when the wrapper has them, and compare every restored field with the disk
payload before optimization. No tolerance relaxation or removed tests.
CPU regression reproduces the upstream destructive-pop behavior and verifies
two isolated reloads restore base LR and retain immutable checkpoint trees.

Local verification: 182 CPU tests pass (Python 3.10.20 / torch 2.10.0+cu128,
existing PyTorch interpreter plus base Anaconda pytest), git diff --check passes.
GPU/isolated MMEngine replay still requires AutoDL. No formal source baseline,
fresh-process resume, CQE or accuracy claim is authorized by this repair.
