# Agent instructions

## Scope

Work on exactly one project phase at a time. Read `README.md`, this file, `docs/METHOD.md`, and `docs/AUTODL_NEXT.md` before changing the project.

## Source of truth

- Git is the only source of truth for code, configuration, documentation, and experiment metadata.
- AutoDL runs an exact commit. Do not edit source files directly on AutoDL.
- Raw datasets, pretrained weights, checkpoints, and complete run directories never enter Git.

## Local versus AutoDL

- Local work is limited to implementation, static checks, CPU smoke tests, manifests, documentation, and result analysis.
- AutoDL is required for dataset inspection at scale, DINOv3 weight loading, GPU smoke tests, training, evaluation, and memory measurements.
- Stop and update `docs/AUTODL_NEXT.md` whenever the next verification needs a GPU or remote data.

## Change discipline

- Preserve public APIs and add focused tests with each module.
- Do not copy the old DAFormer/MRM or QK Adapter projects into this repository.
- EoMT/DINOv3 is the primary implementation reference; REIN, Causal-Tune, and SoMA are protocol references.
- Do not add multiple experimental mechanisms in one change.
- Never remove a failing test merely to make the suite pass.

## Verification

Run `python -m pytest` after local changes. Record the exact Git SHA and environment for every AutoDL experiment.
