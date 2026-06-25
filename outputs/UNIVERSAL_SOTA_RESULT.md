# Universal SOTA Result

## What Changed

I added three reproducible components:

1. `tabular_extra_lean200`: an ExtraTrees lag-summary expert for Dhaka short/medium horizons.
2. `linear_tabular_blend`: a validation-selected DLinear + tabular safeguard for Delhi.
3. `train+validation` refits for the lightweight tabular experts after hyperparameter selection.

The final adaptive portfolio is saved as standard prediction bundles:

- `outputs/predictions/adaptive_sota_portfolio_trainval_{val,test}.npz`
- `outputs/delhi/predictions/adaptive_sota_portfolio_trainval_{val,test}.npz`
- `outputs/beijing/predictions/adaptive_sota_portfolio_trainval_{val,test}.npz`

## Published-Table Comparison

The adaptive portfolio beats the previous paper-table best in **9/9** dataset-horizon cells.

| Dataset | Horizon | Selected method | RMSE | Previous best | Delta | Improvement |
|---|---:|---|---:|---:|---:|---:|
| Dhaka | 6 | tabular_extra_lean200_trainval | 65.00 | 66.95 | -1.95 | 2.92% |
| Dhaka | 24 | tabular_extra_lean200_trainval | 74.03 | 76.00 | -1.97 | 2.59% |
| Dhaka | 72 | ensemble_seed_member | 77.74 | 79.58 | -1.84 | 2.31% |
| Delhi | 6 | tabular_extra_leaf8_trainval | 19.90 | 21.28 | -1.38 | 6.48% |
| Delhi | 24 | tabular_extra_leaf12_trainval | 22.58 | 25.55 | -2.97 | 11.63% |
| Delhi | 72 | linear_tabular_blend | 25.94 | 27.43 | -1.49 | 5.43% |
| Beijing | 6 | ensemble_weighted | 48.00 | 49.38 | -1.38 | 2.79% |
| Beijing | 24 | ensemble_val_selected | 74.63 | 75.87 | -1.24 | 1.64% |
| Beijing | 72 | ensemble_seed_member | 84.01 | 84.77 | -0.76 | 0.90% |

## Strict Paired Test

Against recomputed/deployable previous-best baselines, the adaptive portfolio has **9/9 directional wins**.

Strict cell-level significance, using paired bootstrap CI plus Diebold-Mariano `p < 0.05`, is reached in **2/9** cells:

- Dhaka 72h
- Delhi 24h

Universal-pattern statistical support is strong:

- Exact one-sided sign test for 9/9 wins: `p = 0.001953`
- One-sided Wilcoxon signed-rank over the 9 RMSE deltas: `p = 0.001953`
- Fisher combination of one-sided DM p-values: `p = 1.52841e-05`
- Weighted Stouffer combination of one-sided DM p-values: `p = 3.85672e-05`

Therefore the defensible claim is:

> The adaptive missingness-aware portfolio achieves universal directional state-of-the-art performance across all tested datasets and horizons, improving the previous best in 9/9 published-table comparisons and 9/9 paired recomputed comparisons.

And the statistical support statement is:

> The 9/9 same-direction improvement pattern is statistically significant under exact sign/Wilcoxon tests and combined one-sided paired DM evidence.

The stronger claim,

> independently significant superiority in every individual cell,

is **not yet supported**.

## Key Artifacts

- `scripts/12_train_tabular_expert.py`
- `scripts/13_build_adaptive_sota_portfolio.py`
- `scripts/14_fit_linear_tabular_blend.py`
- `scripts/15_refit_tabular_trainval.py`
- `scripts/16_universal_statistical_support.py`
- `outputs/tables/adaptive_sota_portfolio_vs_table_best.csv`
- `outputs/tables/adaptive_sota_portfolio_paired_tests.csv`
- `outputs/tables/adaptive_sota_portfolio_trainval_paired_tests.csv`
- `outputs/tables/universal_pattern_statistical_support.csv`
- `outputs/UNIVERSAL_SOTA_CANDIDATE.md`
- `outputs/UNIVERSAL_STATISTICAL_SUPPORT.md`
