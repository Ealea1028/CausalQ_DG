# Ablation plan

| ID | Query | Style | Prediction consistency | CQE | Diversity |
|---|---:|---:|---:|---:|---:|
| A0_DINOV3L_BASE | No | No | No | No | No |
| A1_QUERY | Yes | No | No | No | No |
| A2_QUERY_STYLE | Yes | Yes | No | No | No |
| A3_PRED_CONS | Yes | Yes | Yes | No | No |
| A4_CQE | Yes | Yes | Controlled | Yes | No |
| A5_FULL | Yes | Yes | Yes | Yes | Yes |

The primary comparison is `A4_CQE`/`A5_FULL` against `A3_PRED_CONS`. Query count, interaction type, style mechanism, and effect definition are varied only after the main sequence is stable.

