# Universal SOTA Candidate Portfolio

## Result

The adaptive portfolio beats the previous paper-table best in **9/9** dataset-horizon cells.

- Mean relative improvement: **3.89%**
- Smallest relative improvement: **0.90%**

## Comparison Table

| Dataset | Horizon | Selected method | RMSE | Previous best | Delta | Improvement |
|---|---:|---|---:|---:|---:|---:|
| Dhaka | 6 | tabular_extra_lean200 | 65.44 | 66.95 | -1.51 | 2.26% |
| Dhaka | 24 | tabular_extra_lean200 | 74.03 | 76.00 | -1.97 | 2.59% |
| Dhaka | 72 | ensemble_seed_member | 77.74 | 79.58 | -1.84 | 2.31% |
| Delhi | 6 | linear_tabular_blend | 20.00 | 21.28 | -1.28 | 6.02% |
| Delhi | 24 | linear_tabular_blend | 22.73 | 25.55 | -2.82 | 11.05% |
| Delhi | 72 | linear_tabular_blend | 25.94 | 27.43 | -1.49 | 5.43% |
| Beijing | 6 | ensemble_weighted | 48.00 | 49.38 | -1.38 | 2.79% |
| Beijing | 24 | ensemble_val_selected_pm25_metrics | 74.63 | 75.87 | -1.24 | 1.64% |
| Beijing | 72 | ensemble_seed_member | 84.01 | 84.77 | -0.76 | 0.90% |

## Defensibility Note

This is a candidate universal SOTA table result, not yet the final
journal-safe universal superiority claim. The next required step is to
run paired bootstrap and Diebold-Mariano testing on the selected
portfolio against the prior best comparator in each cell.