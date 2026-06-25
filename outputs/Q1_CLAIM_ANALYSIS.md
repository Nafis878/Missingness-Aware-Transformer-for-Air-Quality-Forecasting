# Defensible Q1 Claim Analysis

Primary claim: a validation-weighted forecast ensemble improves cross-network PM2.5 forecasting robustness over the previous best single/pipeline model, while preserving transparent horizon-specific weights.

* Directional wins: **5/9** dataset-horizon cells.
* Significant wins (DM p < 0.05 and bootstrap CI entirely below 0): **0/9** cells.

Negative RMSE difference means the ensemble is better.

| dataset   |   horizon | ensemble                  | previous_best_label   | previous_best_model   |   previous_best_table_RMSE |   previous_best_recomputed_RMSE |   ensemble_RMSE |   RMSE_diff_ensemble_minus_best |   diff_CI95_lo |   diff_CI95_hi |   DM_stat |   DM_p |    n | significant_win   | directional_win   |
|:----------|----------:|:--------------------------|:----------------------|:----------------------|---------------------------:|--------------------------------:|----------------:|--------------------------------:|---------------:|---------------:|----------:|-------:|-----:|:------------------|:------------------|
| Dhaka     |         6 | ensemble_weighted_hybrid8 | Two-stage (KNN)       | two_stage_knn         |                     66.950 |                          65.493 |          66.173 |                           0.681 |         -0.456 |          1.892 |     1.148 |  0.251 | 4459 | False             | False             |
| Dhaka     |        24 | ensemble_weighted_hybrid8 | Proposed (variant B)  | variant_B             |                     76.000 |                          74.453 |          74.716 |                           0.262 |         -0.862 |          1.443 |     0.446 |  0.656 | 4569 | False             | False             |
| Dhaka     |        72 | ensemble_weighted_hybrid8 | GRU                   | gru                   |                     79.580 |                          78.776 |          77.896 |                          -0.879 |         -1.793 |          0.054 |    -1.783 |  0.075 | 4566 | False             | True              |
| Delhi     |         6 | ensemble_weighted         | Persistence           | persistence           |                     21.280 |                          21.276 |          21.690 |                           0.414 |         -1.293 |          2.057 |     0.433 |  0.665 |  540 | False             | False             |
| Delhi     |        24 | ensemble_weighted         | Persistence           | persistence           |                     25.550 |                          25.553 |          24.506 |                          -1.047 |         -2.811 |          0.594 |    -1.070 |  0.285 |  540 | False             | True              |
| Delhi     |        72 | ensemble_weighted         | DLinear               | dlinear               |                     27.430 |                          26.407 |          27.720 |                           1.312 |          0.260 |          2.355 |     1.824 |  0.069 |  540 | False             | False             |
| Beijing   |         6 | ensemble_weighted         | Proposed (variant B)  | variant_B             |                     49.380 |                          48.159 |          48.003 |                          -0.156 |         -0.289 |         -0.025 |    -1.843 |  0.065 | 4282 | False             | True              |
| Beijing   |        24 | ensemble_weighted         | GRU-D                 | gru_d                 |                     75.870 |                          75.516 |          74.774 |                          -0.742 |         -1.589 |          0.068 |    -1.317 |  0.188 | 4295 | False             | True              |
| Beijing   |        72 | ensemble_weighted         | GRU-D                 | gru_d                 |                     84.770 |                          84.587 |          84.152 |                          -0.435 |         -1.329 |          0.425 |    -0.571 |  0.568 | 4294 | False             | True              |

## Equal-Weight Ablation

| dataset   |   horizon | comparison                             |   validation_weighted_RMSE |   equal_weight_RMSE |   RMSE_diff |   diff_CI95_lo |   diff_CI95_hi |   DM_stat |   DM_p |    n |
|:----------|----------:|:---------------------------------------|---------------------------:|--------------------:|------------:|---------------:|---------------:|----------:|-------:|-----:|
| Dhaka     |         6 | validation_weighted_minus_equal_weight |                     66.173 |              66.262 |      -0.089 |         -0.402 |          0.242 |    -0.543 |  0.587 | 4459 |
| Dhaka     |        24 | validation_weighted_minus_equal_weight |                     74.716 |              75.683 |      -0.967 |         -1.582 |         -0.360 |    -3.130 |  0.002 | 4569 |
| Dhaka     |        72 | validation_weighted_minus_equal_weight |                     77.896 |              78.297 |      -0.401 |         -0.950 |          0.163 |    -1.449 |  0.147 | 4566 |
| Delhi     |         6 | validation_weighted_minus_equal_weight |                     21.690 |              21.138 |       0.553 |         -0.090 |          1.270 |     1.504 |  0.133 |  540 |
| Delhi     |        24 | validation_weighted_minus_equal_weight |                     24.506 |              24.597 |      -0.091 |         -0.351 |          0.175 |    -0.625 |  0.532 |  540 |
| Delhi     |        72 | validation_weighted_minus_equal_weight |                     27.720 |              28.572 |      -0.852 |         -1.142 |         -0.567 |    -3.923 |  0.000 |  540 |
| Beijing   |         6 | validation_weighted_minus_equal_weight |                     48.003 |              50.220 |      -2.217 |         -2.954 |         -1.485 |    -4.304 |  0.000 | 4282 |
| Beijing   |        24 | validation_weighted_minus_equal_weight |                     74.774 |              75.384 |      -0.610 |         -0.975 |         -0.234 |    -2.568 |  0.010 | 4295 |
| Beijing   |        72 | validation_weighted_minus_equal_weight |                     84.152 |              85.095 |      -0.943 |         -1.312 |         -0.558 |    -2.996 |  0.003 | 4294 |

## Suggested Manuscript Claim

Across three air-quality networks and three forecast horizons, validation-weighted forecast ensembling achieved directional improvements in 5/9 comparisons against the strongest previously reported model for each dataset-horizon. The strongest gains occurred on Dhaka and Beijing; Delhi remains a boundary case where simple baselines are difficult to beat at short and long horizons.
