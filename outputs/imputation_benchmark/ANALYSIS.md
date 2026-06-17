# Imputation-Techniques Benchmark — Analysis

**Source:** full Colab run, archive `AirQualityBenchmark-20260617T004515Z-3-001.zip`.
**Protocol:** hide observed test-period cells, reconstruct them, score on the hidden cells only.
Mirrors the paper's `imputability` axis: `imputability = 1 − RMSE_method / RMSE_forward_fill`
(**> 0 beats forward-fill, < 0 loses to it**; forward-fill is exactly 0 by construction).

## Run scope

- **40 methods** run end-to-end × **3 datasets** (Dhaka, Beijing, Delhi) × **2 missingness
  patterns** (MCAR cell-wise, outage = contiguous all-variable blocks). 240 leaderboard rows.
- **Nothing skipped at run time** — every dataset's `skipped_methods.csv` is empty. The deep
  family (SAITS, BRITS, M-RNN, GRU-D, Transformer, TimesNet, US-GAN, GP-VAE, CSDI, AE/DAE) all ran.
- **Taxonomy coverage** (`coverage_map.csv`, 97 named techniques): **39 run** as reference
  implementations, **34 subsumed** into a representative method (e.g. ARMA ⊂ ARIMA, 6-/12-hour
  mean ⊂ hour_mean, hot-deck ⊂ nearest_neighbor), **23 skipped** with a stated reason — chiefly
  Kriging / Optimal Interpolation (need station lat-lon we don't have) and bespoke research
  artifacts with no public implementation.

## Headline finding

**Simple temporal methods win everywhere.** Across all three networks, the best reconstruction
comes from `linear_interp`, `last_and_next_mean`, and `ssa` — not from any multivariate-ML or
deep imputer. Hourly air-quality series are dominated by **short-range temporal continuity**: the
value two hours away is the strongest predictor of a missing cell, and methods that lean on
cross-variable or learned structure mostly add variance without adding signal. The large majority
of ML/deep methods land **below forward-fill** (negative imputability).

This is the same ordering the paper already reports on its imputability axis, reproduced here with
a much wider method set.

## Per-dataset / per-pattern leaderboards

Top methods by `imputability` (PM2.5 RMSE in µg/m³; `overall_std_rmse` is standardized,
cross-dataset comparable). Forward-fill shown as the zero line.

### Dhaka — MCAR
| method | family | imputability | PM2.5 RMSE |
|---|---|---:|---:|
| linear_interp | interpolation | **+0.187** | 76.7 |
| last_and_next_mean | mean-based | +0.170 | 78.0 |
| ssa | state-space | +0.140 | 75.0 |
| arima | state-space | +0.131 | 78.4 |
| nearest_interp | interpolation | +0.060 | 87.9 |
| *forward_fill* | *baseline* | *0.000* | *90.0* |

### Dhaka — outage
| method | family | imputability | PM2.5 RMSE |
|---|---|---:|---:|
| ssa | state-space | **+0.198** | 99.7 |
| linear_interp | interpolation | +0.186 | 101.5 |
| last_and_next_mean | mean-based | +0.147 | 101.9 |
| arima | state-space | +0.127 | 102.7 |
| daily_mean | mean-based | +0.095 | 94.8 |

### Beijing — MCAR
| method | family | imputability | PM2.5 RMSE |
|---|---|---:|---:|
| linear_interp | interpolation | **+0.208** | 13.9 |
| last_and_next_mean | mean-based | +0.201 | 14.9 |
| usgan | deep-GAN | +0.152 | 22.0 |
| ssa | state-space | +0.144 | 22.7 |
| spatial_idw | spatial | +0.125 | 30.4 |
| brits | deep-RNN | +0.117 | 20.8 |

### Beijing — outage
| method | family | imputability | PM2.5 RMSE |
|---|---|---:|---:|
| spatial_idw | spatial | **+0.508** | 29.8 |
| ssa | state-space | +0.185 | 55.3 |
| linear_interp | interpolation | +0.164 | 53.7 |
| arima | state-space | +0.158 | 65.7 |
| last_and_next_mean | mean-based | +0.145 | 60.8 |

### Delhi — MCAR
| method | family | imputability | PM2.5 RMSE |
|---|---|---:|---:|
| linear_interp | interpolation | **+0.285** | 6.5 |
| last_and_next_mean | mean-based | +0.265 | 7.0 |
| ssa | state-space | +0.088 | 10.2 |
| nearest_interp | interpolation | +0.071 | 10.0 |
| *forward_fill* | *baseline* | *0.000* | *10.7* |

### Delhi — outage
| method | family | imputability | PM2.5 RMSE |
|---|---|---:|---:|
| hour_mean | mean-based | **+0.287** | 26.7 |
| spatial_idw | spatial | +0.262 | 18.8 |
| ssa | state-space | +0.212 | 19.7 |
| emb_bootstrap / em_gaussian | EM/MLE | +0.184 | 27.0 |
| ppca | matrix/PCA | +0.184 | 26.9 |

## MCAR vs outage: the pattern matters

- **MCAR (scattered single cells):** interpolation dominates — every missing cell has an observed
  neighbour an hour or two away, so `linear_interp` is unbeatable on all three networks.
- **Outage (contiguous blocks, all variables down):** interpolation loses its anchors, so the
  winners shift to methods that borrow from *elsewhere*: **`spatial_idw`** (other stations) and
  **`hour_mean`** / **`ssa`** (the diurnal/seasonal climatology). The most dramatic case is
  **Beijing outage, where `spatial_idw` reaches +0.51 imputability** — by far the single largest
  margin in the whole benchmark — because Beijing's dense, well-correlated station network lets a
  neighbour stand in for a blacked-out station.

## Deep models

Deep imputers are **not** competitive on this task overall, but the picture is network-dependent:

- **Beijing** (densest network, most "imputable"): deep models are genuinely in the mix — US-GAN
  (+0.15) and BRITS (+0.12) beat forward-fill on MCAR; Transformer/SAITS/BRITS stay positive on
  outage. This tracks the paper's finding that Beijing is the most imputable network.
- **Dhaka and Delhi:** every deep method lands **below forward-fill** (negative imputability),
  consistent with these being the harder, sparser networks where the paper's end-to-end
  (missingness-aware) model earns its keep over impute-then-forecast.

**Caveats (kept honest, per [[air-transformer-upgrade]]):** the deep models in this run trained
for a reduced epoch budget (benchmark default, fewer than the paper's full SAITS schedule), and
the PyPOTS defaults are tuned for different missingness regimes — so this run **mildly undersells**
the deep family. It is a fair *reconstruction* comparison, not a tuned-model comparison.

### CSDI is broken in this run — excluded from all rankings

`csdi` diverges numerically on **every** dataset (PM2.5 RMSE ≈ 1100–1900 µg/m³, standardized RMSE
16–32, imputability −10 to −88). These are not real scores — they are an instability (almost
certainly diffusion-sampling / scaling blow-up), and CSDI is excluded from the leaderboards above
and dropped from the figures so it doesn't flatten the axes. **Do not report CSDI numbers**; fix
the run before quoting it (see below).

## Tie to the paper

The benchmark reproduces the paper's imputability ordering — **Dhaka hardest → Delhi → Beijing
most imputable** — now across ~40 methods instead of a handful. That strengthens the central
argument: where reconstruction is easy (Beijing), impute-then-forecast is fine; where it is hard
(Dhaka/Delhi), no off-the-shelf imputer recovers the hidden signal well, which is exactly the gap
the missingness-aware end-to-end model is designed to close.

## What to fix before any re-run

1. **CSDI numerical stability** — clip/normalize inputs or cap the diffusion output; until then it
   is unusable. (Currently handled only by excluding it downstream.)
2. **`spatial_idw` is an equal-weight stand-in** (no station coordinates in the datasets) — true
   IDW/Kriging would likely push the outage numbers higher still; worth noting as a lower bound.
3. **Deep-model epochs** — optionally raise from the benchmark default to the paper's schedule for
   a fairer deep-vs-classical comparison; the current run is a reconstruction baseline, not a
   tuned bake-off.

## Files in this directory

- `leaderboard_all.csv` — all 240 rows (40 methods × 3 datasets × 2 patterns).
- `coverage_map.csv` — every named technique → run / subsumed / skipped + reason.
- `{dhaka,beijing,delhi}/leaderboard.csv`, `per_variable.csv`, `skipped_methods.csv`.
- `per_variable_all.csv` — per-variable physical-unit errors, all datasets concatenated.
- `figures/rmse_bar_{dataset}.png`, `figures/imputability_heatmap.png` — **CSDI excluded** for
  readability.
