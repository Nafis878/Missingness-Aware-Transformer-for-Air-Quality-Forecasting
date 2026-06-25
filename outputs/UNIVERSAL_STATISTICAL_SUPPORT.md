# Universal Statistical Support

## Layer 1: Cell-Level Tests

- Directional wins: **9/9**
- Strict cell-level wins: **2/9**

Strict cell-level significance requires the RMSE difference to be below
zero, the paired-bootstrap CI to stay below zero, and DM p < 0.05.
That threshold is intentionally conservative and remains underpowered
for several small-margin cells.

## Layer 2: Universal-Pattern Tests

- Exact one-sided sign test for 9/9 wins: **p = 0.001953**
- One-sided Wilcoxon signed-rank over the 9 RMSE deltas: **p = 0.001953**
- Fisher combination of one-sided DM p-values: **p = 1.52841e-05**
- Weighted Stouffer combination of one-sided DM p-values: **p = 3.85672e-05**

## Publication-Safe Wording

> The adaptive portfolio achieved directional improvements in all
> 9 dataset-horizon comparisons. The probability of observing 9/9
> same-direction wins under a no-improvement null is p = 0.001953 by an
> exact one-sided sign test, and combined paired DM evidence also
> rejects the global no-improvement null.

Do not state that every individual cell is independently significant;
that is not supported by the current paired tests.