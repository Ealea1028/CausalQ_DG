# Phase 14 paired zero/learned-null effect analysis

- Status: complete; learned-null rejected as the primary causal-effect
  baseline.
- Evaluation SHA: `549a9753914b9868a852a65983fb8cf0f246462b`.
- Training SHA: `d85db4e0dcf411681c1ddad09080dfeed1c419bd`.
- Coverage: all 500 Cityscapes validation images and 6,005 GT-present class
  maps.
- Zero-ablation inside/outside absolute ratio: `3.7733`; normalized contrast:
  `0.4907`.
- Learned-null inside/outside absolute ratio: `1.8451`; normalized contrast:
  `0.2052`.
- Only `4.71%` of class maps have a higher learned-null localization ratio;
  only `14.15%` have lower outside-region magnitude.
- The learned-null effect still passes the minimal inside-greater-than-outside
  check, but its outside magnitude rises from `0.6558` to `1.0805` and its
  localization is materially worse.

The model's `+0.7968`-point segmentation gain does not validate the proposed
causal baseline. Per the research decision tree, do not run learned-null seed
1/2 and do not build sufficiency or specificity losses on this baseline. Keep
the result as a negative mechanism ablation, retain Static R=2 as the supported
model, and move to external target-domain evaluation.
