# Defensible Q1 Verdict

## Final Position

The strongest defensible claim is **not** that merged imputation or the ensemble
is a universal forecasting winner. Under strict paired testing against stronger
seed-averaged/deployable baselines, the final validation-weighted ensemble has:

- Directional wins: **5/9** dataset-horizon comparisons.
- Statistically significant wins under the strict rule
  (DM p < 0.05 and bootstrap CI entirely below zero): **0/9**.
- Clear evidence that validation weighting improves over naive equal-weight
  ensembling on several horizons, especially Beijing and Dhaka h24.

## What Can Be Claimed

A Q1-defensible claim is:

> Real-world air-quality missingness creates a divergence between imputation
> quality and forecasting quality. The proposed missingness-aware forecasting
> framework, augmented with validation-weighted forecast ensembling, improves
> robustness across heterogeneous monitoring networks, with strongest gains on
> Dhaka and Beijing, while Delhi demonstrates a boundary case where simple
> temporal baselines remain difficult to beat.

This is defensible because it does not overclaim universal dominance.

## What Should Not Be Claimed Yet

Do **not** claim:

> The proposed method is statistically superior to all baselines across all
> datasets and horizons.

The strict paired evidence does not support that.

## Best Strict Paired Results

Negative RMSE difference means the ensemble is better than the previous best
deployable baseline.

| Dataset | Horizon | Previous best | Ensemble RMSE | Difference | Interpretation |
|---|---:|---|---:|---:|---|
| Dhaka | 6 | Two-stage KNN | 66.17 | +0.68 | Not better under seed-averaged baseline |
| Dhaka | 24 | Proposed variant B | 74.72 | +0.26 | Essentially tied |
| Dhaka | 72 | GRU | 77.90 | -0.88 | Directional improvement |
| Delhi | 6 | Persistence | 21.69 | +0.41 | Not better |
| Delhi | 24 | Persistence | 24.51 | -1.05 | Directional improvement |
| Delhi | 72 | DLinear | 27.72 | +1.31 | Not better |
| Beijing | 6 | Proposed variant B | 48.00 | -0.16 | Directional improvement; bootstrap CI below zero, DM p = 0.065 |
| Beijing | 24 | GRU-D | 74.77 | -0.74 | Directional improvement |
| Beijing | 72 | GRU-D | 84.15 | -0.44 | Directional improvement |

## Equal-Weight Ablation

The validation-weighted ensemble is meaningfully better than equal-weight
averaging in several places:

- Dhaka h24: RMSE improves by about **0.97**, DM p = **0.002**.
- Delhi h72: RMSE improves by about **0.85**, DM p < **0.001**.
- Beijing h6: RMSE improves by about **2.22**, DM p < **0.001**.
- Beijing h24: RMSE improves by about **0.61**, DM p = **0.010**.
- Beijing h72: RMSE improves by about **0.94**, DM p = **0.003**.

This supports the claim that validation weighting is useful, even if the final
ensemble is not universally superior to the strongest prior model.

## Recommendation Before Submission

For a stronger Q1 submission, the next work should be:

1. Standardize hybrid8 checkpoints for Delhi and Beijing so hybrid8 candidates
   can be included in validation ensembles across all datasets.
2. Add modern baselines such as TimesNet, iTransformer, N-HiTS/N-BEATS, and
   Autoformer/FEDformer-style models.
3. Report paired tests and bootstrap intervals as the main evidence, not only
   leaderboard means.
4. Frame Delhi explicitly as an important boundary condition.

## Bottom Line

The current results are **publishable as a nuanced missingness-aware forecasting
study**, but **not yet sufficient for a Q1 claim of universal state-of-the-art
superiority**.
