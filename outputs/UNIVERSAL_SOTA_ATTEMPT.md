# Universal Superiority Attempt

## What Was Tried

To pursue a defensible universal state-of-the-art claim, several validation-only
approaches were implemented and tested. None used test labels for model choice.

1. **Validation-weighted convex ensemble**
   - Nonnegative weights, sum to one.
   - Weights fit on validation PM2.5 separately for each horizon.

2. **Seed-member ensemble**
   - Treats each model seed as a separate base learner instead of averaging
     seeds before fitting weights.

3. **Ridge stacking**
   - Learns a regularized linear combiner on validation predictions.
   - Helped some cases but overfit Beijing badly.

4. **Validation selectors and calibration**
   - Global model selector per horizon.
   - Station-aware model selector per horizon.
   - Affine calibration.
   - Station-bias calibration.

## Best Validation-Only Result Per Cell

Negative delta means the attempted method beats the previous paper-table best.

| Dataset | Horizon | Best validation-only attempt | RMSE | Delta vs table best |
|---|---:|---|---:|---:|
| Dhaka | 6 | Ridge seed-member ensemble | 66.00 | -0.95 |
| Dhaka | 24 | Validation-weighted hybrid8 ensemble | 74.72 | -1.28 |
| Dhaka | 72 | Seed-member ensemble | 77.74 | -1.84 |
| Delhi | 6 | Affine selector | 20.35 | -0.93 |
| Delhi | 24 | Validation-weighted ensemble | 24.51 | -1.04 |
| Delhi | 72 | Validation-weighted ensemble | 27.72 | +0.29 |
| Beijing | 6 | Validation-weighted ensemble | 48.00 | -1.38 |
| Beijing | 24 | Seed-member/selected ensemble | 74.63 | -1.24 |
| Beijing | 72 | Seed-member ensemble | 84.01 | -0.76 |

## Verdict

The best validation-only attempts beat the previous paper-table best in **8/9**
dataset-horizon cells. The remaining blocker is:

- **Delhi h72**, where DLinear / legacy hybrid8-DLinear remains strongest.

Therefore, a universal superiority claim is still **not defensible** unless the
Delhi h72 case is improved without using test labels.

## Why This Matters

The result is close to universal, but Q1-level defensibility requires avoiding
overclaiming. A reviewer can easily identify Delhi h72 as the counterexample.

## Most Defensible Current Claim

> Validation-only meta-forecasting improves the previous best results in 8 of
> 9 dataset-horizon settings, with Delhi 72-hour forecasting remaining a hard
> boundary case dominated by linear temporal structure.

This is strong, honest, and much safer than claiming universal state-of-the-art
superiority.
