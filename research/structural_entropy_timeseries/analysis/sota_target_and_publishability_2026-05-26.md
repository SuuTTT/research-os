# SOTA Target and Publishability Threshold (2026-05-26)

## 1) What is "SOTA" here
SOTA is protocol-specific, not a single number.
For this project we should use:
- legacy long-term forecasting, Look-Back-96 protocol,
- datasets ETTh1/ETTh2/ETTm1/ETTm2 (then expand to ECL/Traffic/Weather/Exchange),
- horizons 96/192/336/720,
- metrics MSE primary, MAE secondary,
- multi-seed mean/std (not best seed only).

## 2) Current project performance snapshot
From custom harness (ETTh1, horizon 24):
- baseline mean MSE: 0.054351
- SE mean MSE: 0.051532
- relative improvement: 5.19%

From standardized Phase-A starter (TSLib DLinear, horizon 96, 1 epoch smoke):
- ETTh1 MSE: 0.411866, MAE: 0.425835
- ETTh2 MSE: 0.355168, MAE: 0.405321
- commit SHA recorded from worker: 4e938a1

## 3) Publishability threshold recommendation
For a defensible paper claim:
- Strong: >= 3% relative mean MSE gain vs strong baseline under matched budget,
- Good: 1.5%-3% with consistent wins across most dataset-horizon cells,
- Weak headline-only: < 1.5% unless paired with strong variance/stability and mechanism evidence.

Minimum bar before claiming publishable main result:
- statistically significant or highly consistent gains across seeds,
- coverage across multiple datasets and horizons,
- fair matched training budget and full reproducibility metadata,
- ablations proving entropy term is causal.

## 4) Phase progression
- Phase A (in progress): benchmark-grade environment + baseline smoke path established.
- Next in Phase A: run full baseline matrix (DLinear + TimesNet) over ETTh1/ETTh2/ETTm1/ETTm2 and horizons 96/192/336/720 with >=3 seeds.
- Phase B starts only after full baseline matrix is complete.
