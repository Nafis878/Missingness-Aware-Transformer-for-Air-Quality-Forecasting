# Merged Imputation Results Analysis

Source: `leaderboard_all_combined.csv`

Scope: 366 rows, covering 61 methods across 3 datasets (`Dhaka`, `Beijing`, `Delhi`) and 2 missingness patterns (`mcar`, `outage`). Each dataset/pattern group has exactly 61 rows. No empty fields were found.

Imputability is interpreted as improvement over forward fill:

`imputability = 1 - RMSE_method / RMSE_forward_fill`

Values above 0 beat forward fill. Values below 0 are worse than forward fill.

## Headline Findings

The merged benchmark is now led by `hybrid_top8`. It wins 4 of the 6 dataset/pattern cells and is positive in all 6, with the best average imputability across methods.

The cheapest consistently strong options are `tensor_cp`, `linear_interp`, and `last_and_next_mean`. `tensor_cp` and `linear_interp` have the same six-group imputability/RMSE profile in this merged table, suggesting the tensor method is behaving like, or falling back to, a temporal interpolation solution in these runs.

Outage missingness changes the problem. MCAR rewards local temporal interpolation because nearby observed values usually exist. Outage blocks reward methods that can borrow from other stations or climatology. The clearest example is Beijing outage, where `spatial_idw` wins by a wide margin.

Deep and complex methods remain fragile in this benchmark. A few methods, especially `csdi`, produce catastrophic outliers and strongly drag down average family scores.

## Winners By Dataset And Pattern

| Dataset | Pattern | Winner | Family | Imputability | PM2.5 RMSE | Std RMSE | R2 | Runtime s |
|---|---|---|---|---:|---:|---:|---:|---:|
| Beijing | mcar | hybrid_top8 | hybrid | 0.2340 | 13.466 | 0.4184 | 0.8289 | 85.4 |
| Beijing | outage | spatial_idw | spatial | 0.5075 | 29.819 | 0.4850 | 0.7677 | 0.2 |
| Delhi | mcar | hybrid_top8 | hybrid | 0.2887 | 6.903 | 0.2556 | 0.8204 | 12.4 |
| Delhi | outage | hour_mean | mean-based | 0.2871 | 26.734 | 0.5019 | 0.3123 | 0.1 |
| Dhaka | mcar | hybrid_top8 | hybrid | 0.2066 | 74.815 | 0.8214 | 0.7923 | 152.1 |
| Dhaka | outage | hybrid_top8 | hybrid | 0.2120 | 98.087 | 1.1487 | 0.5891 | 172.9 |

## Most Robust Methods

Ranked by median imputability across all 6 dataset/pattern cells:

| Method | Family | Median Imputability | Average Imputability | Positive Cells | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| hybrid_top8 | hybrid | 0.2093 | 0.2177 | 6/6 | 0.1723 | 0.2887 |
| linear_interp | interpolation | 0.1862 | 0.1932 | 6/6 | 0.1300 | 0.2847 |
| tensor_cp | tensor | 0.1862 | 0.1932 | 6/6 | 0.1300 | 0.2847 |
| ssa | state-space | 0.1647 | 0.1612 | 6/6 | 0.0878 | 0.2119 |
| last_and_next_mean | mean-based | 0.1584 | 0.1781 | 6/6 | 0.1404 | 0.2654 |
| fcm_svr | hybrid | 0.1022 | 0.0778 | 4/6 | -0.0349 | 0.1612 |
| arima | state-space | 0.0991 | -0.0089 | 5/6 | -0.5437 | 0.1578 |
| som_lssvm | hybrid | 0.0851 | 0.0644 | 4/6 | -0.0609 | 0.1464 |
| mkl_cluster | hybrid | 0.0822 | 0.0536 | 4/6 | -0.0954 | 0.1782 |
| nearest_interp | interpolation | 0.0605 | 0.0589 | 6/6 | 0.0339 | 0.0905 |

## Pattern-Level Read

| Pattern | Rows | Positive Rows | Positive Rate | Median Imputability | Average Imputability |
|---|---:|---:|---:|---:|---:|
| mcar | 183 | 26 | 0.142 | negative in all dataset medians | -1.0589 |
| outage | 183 | 78 | 0.426 | positive for Beijing and Delhi, negative for Dhaka | -0.3888 |

The average values are not representative because `csdi` and a few other methods create extreme negative outliers. Median/ranking is more useful than raw average for model selection.

MCAR has fewer methods beating forward fill because forward fill is already strong for scattered single-cell gaps. Outage has more positive methods because simple forward fill struggles on blocks, leaving more room for spatial, seasonal, or hybrid methods to improve.

## Dataset-Level Read

| Dataset | Rows | Positive Rows | Positive Rate | Best Method Overall |
|---|---:|---:|---:|---|
| Beijing | 122 | 41 | 0.336 | spatial_idw on outage |
| Delhi | 122 | 38 | 0.311 | hybrid_top8 on mcar |
| Dhaka | 122 | 25 | 0.205 | hybrid_top8 on outage |

Beijing is the most imputation-friendly dataset in the merged benchmark, especially for outage reconstruction. Dhaka is hardest: fewer methods beat forward fill, and even the winning outage model has a higher standardized RMSE than the Beijing/Delhi winners.

## Failure Modes

The worst rows are dominated by `csdi`, which has extreme negative imputability:

| Dataset | Pattern | Method | Imputability | PM2.5 RMSE | Std RMSE | R2 |
|---|---|---|---:|---:|---:|---:|
| Delhi | mcar | csdi | -88.5317 | 1899.667 | 32.1780 | -2845.1670 |
| Delhi | outage | csdi | -37.2335 | 1759.985 | 26.9172 | -1976.9163 |
| Dhaka | mcar | csdi | -19.7042 | 1892.381 | 21.4373 | -140.4514 |
| Dhaka | outage | cubic_spline | -18.4746 | 7737.095 | 28.3889 | -249.9840 |
| Dhaka | outage | csdi | -10.0364 | 1111.531 | 16.0882 | -79.6052 |

These should be excluded or flagged in summary plots unless the goal is to show failure robustness. They are not small underperformances; they are numerical or configuration failures.

## Recommendations

Use `hybrid_top8` as the primary merged-result headline. It is the only method that both wins most groups and stays comfortably positive everywhere.

Use `linear_interp` or `tensor_cp` as the practical baseline. They are consistently positive, very fast, and nearly as good as the winner in several MCAR cases.

Use pattern-aware recommendations:

- MCAR: prefer `hybrid_top8`, `linear_interp`, `tensor_cp`, or `last_and_next_mean`.
- Outage with dense station coverage: prefer `spatial_idw`, especially for Beijing-like networks.
- Outage without strong spatial support: prefer `hybrid_top8`, `hour_mean`, or `ssa`.

Do not report averages without robust context. Use median imputability, win counts, positive-rate, and worst-case behavior. The merged file contains enough extreme failures that simple means can mislead.

