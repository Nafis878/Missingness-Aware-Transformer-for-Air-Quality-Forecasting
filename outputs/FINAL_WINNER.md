# Final Winning Model

The overall winner is the **validation-calibrated MAT ensemble**:

`validation_convex_intercept_stack`

It is the best model by PM2.5 test RMSE at all three forecasting horizons and
passes combined seed-level Diebold-Mariano testing with Holm correction across
all model-horizon comparisons.

## RMSE Summary

PM2.5 test RMSE (ug/m3), 3-seed mean:

| Model | 6 h | 24 h | 72 h |
|---|---:|---:|---:|
| Vanilla Transformer | 68.61 | 78.31 | 81.83 |
| Variant B dual-input ridge | 67.28 | 75.08 | 79.33 |
| **Final validation-calibrated MAT ensemble** | **65.78** | **74.23** | **77.55** |

## Statistical Verdict

- Directionally better across all three seeds.
- Combined seed-level DM + Holm significance passes **42/42** model-horizon
  comparisons.
- The stricter per-seed-only requirement is reported separately in the full
  significance tables.

## Reproducibility Artifacts

- Generator:
  [`scripts/24_validation_calibrated_ensembles.py`](../scripts/24_validation_calibrated_ensembles.py)
- Combined significance:
  [`scripts/27_combined_seed_significance.py`](../scripts/27_combined_seed_significance.py)
- Figures:
  [`scripts/28_make_final_mat_comparison_figures.py`](../scripts/28_make_final_mat_comparison_figures.py)
- Per-seed test predictions:
  [`outputs/predictions/seeds/validation_convex_intercept_stack_s42_test.npz`](predictions/seeds/validation_convex_intercept_stack_s42_test.npz),
  [`outputs/predictions/seeds/validation_convex_intercept_stack_s43_test.npz`](predictions/seeds/validation_convex_intercept_stack_s43_test.npz),
  [`outputs/predictions/seeds/validation_convex_intercept_stack_s44_test.npz`](predictions/seeds/validation_convex_intercept_stack_s44_test.npz)
- Summary tables:
  [`outputs/tables/validation_calibrated_ensemble_summary.csv`](tables/validation_calibrated_ensemble_summary.csv),
  [`outputs/tables/final_mat_ensemble_comparison_summary.csv`](tables/final_mat_ensemble_comparison_summary.csv),
  [`outputs/tables/combined_seed_significance_validation_convex_intercept_stack.csv`](tables/combined_seed_significance_validation_convex_intercept_stack.csv)
- Q1 figure panel:
  [`outputs/figures/q1_final_mat_ensemble_summary_panel.png`](figures/q1_final_mat_ensemble_summary_panel.png)
