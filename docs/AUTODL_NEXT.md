# Next AutoDL action

Status: waiting for the first RTX 4090D session.

## Goal

Verify the clean AutoDL image and create the independent CausalQ-DG environment. Do not download datasets or begin model training yet.

## Required instance

- GPU: RTX 4090D 24 GB
- Preferred base image: PyTorch 2.7.0, CUDA 12.8, Python 3.12
- Prefer an official base image; avoid historical DAFormer/MRM or QK Adapter community images
- System disk: at least 30 GB
- Data disk: size according to the datasets to be uploaded later

## Procedure

1. Start the instance and open its terminal.
2. Confirm `nvidia-smi` reports an RTX 4090D and note the driver version.
3. Clone this repository into `/root/autodl-tmp/CausalQ_DG` at the exact Phase 1 commit.
4. Run `bash scripts/setup_autodl.sh` from the repository root.
5. Activate the environment using the command printed by the setup script.
6. Run `python tools/check_environment.py`.
7. Save the complete output and return it to the local development task.

## Acceptance criteria

- Python is compatible with the project environment.
- Torch, Torchvision, Transformers, timm, Lightning, and TorchMetrics import successfully.
- `torch.cuda.is_available()` is true.
- Exactly one RTX 4090D is reported with approximately 24 GB VRAM.
- No NaN/Inf is produced by a small CUDA tensor operation.
- The repository remains clean after verification.

Stop after this check. Dataset preparation is Phase 3 and begins only after the environment report is reviewed locally.
