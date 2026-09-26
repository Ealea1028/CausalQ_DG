# GTA5 → Cityscapes Query Effect visual review (provisional)

The four user-provided PNG panels were inspected locally on 2026-09-26. They
match the seven-panel layout in project-plan §41, but the corresponding AutoDL
`report.json` and run log have not been supplied. Their checkpoint provenance,
sample IDs, and exact evaluation SHA therefore remain **unverified**. The
images are user-owned, untracked files and are not experimental records in Git.

| Class | Provisional observation |
| --- | --- |
| road | Positive signed effect covers much of the roadway and is visually similar under the shown photometric view; prediction differences from A0 are small in this example. |
| car | Vehicles receive strong positive effect in both views, but positive effect also appears on non-car regions, so the map is not class-specific evidence by itself. |
| person | The effect highlights two pedestrian-shaped regions and remains visually similar under the shown view; the image alone does not quantify segmentation improvement. |
| vegetation | Tree canopy receives positive effect and the road receives negative effect in both views; effects extend beyond precisely labeled vegetation. |

The intervention is visible but modest in these four examples. Similar-looking
heatmaps are useful diagnostic evidence, not a quantitative style-invariance
result or proof of a causal effect. Independent A0 and Static R=2 predictions
must be judged using the full Cityscapes validation metric, not cherry-picked
panels. BDD100K, Mapillary, and other targets remain deferred.

Local image SHA-256 values, for matching against AutoDL originals:

```text
road.png        5bbe0a9e270c8a4cfb7d61294d93e804257cad02e86f4cf930e721df9a00461d
car.png         6643a9a3bb6062e8c9fb66e580e734a53972eddfb942a38a6301b9f410f9162d
person.png      65a574fe3d9211f4e8cd3d8d045621ee3f73631f8902022c942169c8690972fe
vegetation.png  d651d1afd577580777e5ea7f8c170106ac0452df9cb7362de412252029dead04
```

Next gate: retrieve and check the existing AutoDL `report.json`, four original
PNG hashes, and run log. Do not rerun inference or advance to scaling until the
figure provenance has been reconciled.
