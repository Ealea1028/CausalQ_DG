# Phase 15: DINOv3-B versus DINOv3-L Static R=2

Scope: GTA5 source training to Cityscapes validation only. Both backbones are
frozen and use the same static two-query head, original/photometric views,
40,000 training iterations, and final-iteration selection. Each style-effect
comparison uses the same 500 validation images and 6,005 ground-truth-present
class maps per model. The metric is population variance across the two views
of valid-pixel-L2-normalized class-logit query-effect maps, averaged across
present class maps. Standard deviations below are **sample** standard
deviations over three training seeds; differences are paired by seed.

| Training seed | ViT-B final mIoU | ViT-L final mIoU | L − B (pp) | ViT-B effect variance | ViT-L effect variance | L variance reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.563382 | 0.647396 | 8.4014 | 0.001144741 | 0.000580858 | 49.26% |
| 1 | 0.555294 | 0.669958 | 11.4664 | 0.000998661 | 0.000620455 | 37.87% |
| 2 | 0.572957 | 0.641193 | 6.8236 | 0.000740077 | 0.000501017 | 32.30% |
| Mean ± SD | 0.563878 ± 0.008842 | 0.652849 ± 0.015138 | 8.8971 ± 2.3608 | 0.000961160 ± 0.000204922 | 0.000567443 ± 0.000060839 | 39.81% ± 8.64% |

The paired absolute variance reduction is `0.000393716 ± 0.000162966`.
ViT-L has higher final mIoU and lower normalized effect variance on all three
paired seeds. The seed-2 evaluation was run with RTX 4090 D, PyTorch
`2.7.0+cu126`, CUDA `12.6`, evaluation Git SHA
`85f1aee3c58a0b527a4455df367c17b695cdaef8`, and deterministic view
seed `20260927`; its report SHA256 is
`2761a9264593aba81712f9a5ad0ef2006c0a1b533e99a0987cd57654f3ae357b`.
The seed-2 ViT-B and ViT-L final checkpoint SHA256 values are respectively
`30049a8eb10885dd625fa1afd0204acc85713fa15ad7a239ce077030f8f38a93`
and `580bf504b85bb1d5111937017c28012a97bc39d7e3bd8198185b99002b65fdb2`.
The seed-0/1 paired report SHA256 values are respectively
`b678125478130e68d57614c30d803dc11e4f03ac5bae1d96e08349e0d02150ed`
and `2ac8b82d6c12d016511cf924cb6f6ed61b3b70c7f16d3331a6b00cfaeafda511`.

This is an association between two pretrained backbone sizes in one
source-to-target protocol, not a scaling law or causal proof. Backbone size,
representation quality, and pretrained checkpoint differ together. The
original plan's question about whether CQE gain persists with scale is **not
answered**: the isolated CQE experiment failed its mIoU retention criterion,
so CQE was not silently added to the selected static model. Evaluation on
other target datasets remains deferred at the user's request.
