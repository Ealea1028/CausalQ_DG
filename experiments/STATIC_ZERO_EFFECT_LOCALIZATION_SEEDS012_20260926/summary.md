# Static zero-effect localization audit

- Status: passed.
- Evaluation Git SHA: `cf83e7750a25b005d2c4940dcb829565dda16aae`.
- Scope: all 500 Cityscapes validation images and 6,005 GT-present class
  maps for each selected Static seed.
- Mean inside/outside absolute-effect ratios for seeds 0/1/2:
  `2.794 / 3.583 / 2.363`.
- Mean ratio across seeds: `2.913 ± 0.619`.
- Fraction of class maps with stronger inside-GT magnitude for seeds 0/1/2:
  `0.852 / 0.925 / 0.817`; across-seed mean `0.865`.
- All three seeds pass the GT-region magnitude target.
- Full report SHA-256:
  `bc7e0cdd2b6c77a5363cef167d36593defaf2983a8fa6b554e292287c2b3f561`.

The existing factual-minus-zero query effect has reproducible class-localized
semantics. This supports proceeding to a learned-null intervention baseline.
It does not by itself establish causal sufficiency or specificity.
