# Upstream references

No upstream source is vendored or copied into this repository during Phase 1.

| Project | Role | Repository | Pinning status |
|---|---|---|---|
| EoMT | DINOv3 and segmentation implementation reference | https://github.com/tue-mps/eomt | Resolve and record exact SHA during AutoDL setup |
| REIN | DGSS dataset/evaluation protocol reference | https://github.com/w1oves/Rein | Resolve and record exact SHA before borrowing conversion logic |
| Causal-Tune | Dataset conversion and comparison reference | https://github.com/zhangyin1996/Causal-Tune | Resolve and record exact SHA before borrowing conversion logic |
| SoMA | Strong baseline reference | https://github.com/ysj9909/SoMA | Resolve and record exact SHA before baseline comparison |

The EoMT dependency baseline inspected for Phase 1 pins Torch 2.7.0, Torchvision 0.22.0, Transformers 4.56.1, timm 1.0.15, Lightning 2.5.1.post0, and TorchMetrics 1.7.1. The exact repository SHA must be captured when the upstream repository is cloned successfully; a moving branch name is not sufficient for experiments.

