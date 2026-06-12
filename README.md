# Missingness-Aware Transformer for Multi-Pollutant Air Quality Forecasting

Code, experiments, and paper assets for *"A Missingness-Aware Transformer for
Multi-Pollutant Air Quality Forecasting on Severely Incomplete Monitoring
Data from Bangladesh."*

An end-to-end Transformer forecasts pollutants (primary target PM2.5; also
PM10, NO2, O3, CO, SO2) **directly from incomplete sensor streams** using
learned missingness embeddings and masked attention — no imputation stage —
and is compared against the conventional two-stage pipeline (impute with
KNN/MICE, then forecast) plus statistical and RNN baselines.

**Everything runs on a desktop CPU**: small models (d_model 128, 3 layers,
~406k parameters), vanilla PyTorch, fixed seeds, deterministic flags. The
entire experimental grid (8 baselines, 25 ablation runs, two robustness
suites) trains in under a day on CPU — framed as deployability for
resource-constrained monitoring networks.

## Headline results

PM2.5 test RMSE (µg/m³) on the held-out 2024 year (observed targets only):

| Model | 6 h | 24 h | 72 h |
|---|---|---|---|
| Persistence | 93.9 | 99.0 | 105.1 |
| Seasonal-naive | 90.6 | 93.7 | 102.2 |
| SARIMA (per station) | 80.0 | 83.9 | 86.7 |
| LSTM | 68.0 | 77.9 | 79.9 |
| GRU | 67.3 | 76.3 | 79.2 |
| Two-stage (KNN → Transformer) | 67.0 | 75.6 | 80.2 |
| Two-stage (MICE → Transformer) | 67.3 | 76.0 | 79.7 |
| **Proposed (MAT)** | 66.8 | 77.1 | 80.7 |
| **Proposed + missingness dropout** | **66.2** | 77.3 | 80.5 |

- The proposed model **matches the strong two-stage pipeline on natural data**
  (Diebold–Mariano: no significant difference at 2 of 3 horizons) while
  eliminating the imputation stage entirely (1.6 ms/window inference vs
  minutes of re-imputation per data refresh).
- Under **additional synthetic missingness** (both cell-wise MCAR and
  station-outage blocks), the missingness-dropout variant is best at every
  corruption level at the 6 h horizon with the flattest degradation slope
  (+2.1 µg/m³ at +50% missingness vs +3.2 for two-stage KNN).
- **Attention analysis shows why**: in PM2.5-sparse windows, 70.6% of the
  forecast token's attention mass shifts to timesteps where only meteorology
  is observed; attention exhibits learned 24 h periodicity.

Full consolidated numbers: [`outputs/RESULTS.md`](outputs/RESULTS.md).
All tables (CSV + booktabs LaTeX) in [`outputs/tables/`](outputs/tables/),
all figures (300-dpi PNG + vector PDF) in [`outputs/figures/`](outputs/figures/).

## Data

Hourly air quality + meteorology from 16 CAMS monitoring stations across
Bangladesh, 2022–2024 (three Excel files in `data/raw/`). The raw files are
messy by construction: per-station blocks each headed by a units row, the
2024 file has PM2.5/PM10 in swapped column order, station names vary across
years, missingness ranges from ~10% to 100% per station × variable, and the
2024 PM sensors emit verbatim saturation/error codes (985.0, 999.99 —
thousands of occurrences, physically impossible in monsoon months). All
handling — unit-row stripping, plausibility bounds, sentinel codes,
stuck-sensor runs — is documented in
[`outputs/data_cleaning_report.md`](outputs/data_cleaning_report.md).

Missingness itself is a study object (a paper section): PM2.5 is 23.4%
missing overall (10–45% per station), missingness is demonstrably not MCAR
(predictable from meteorology + season, AUC 0.61), and station-level outages
dominate the long-gap mass — which motivates both the model design and the
outage-style robustness experiment.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
pip install -r requirements.txt                    # torch = CPU build
```

Python ≥ 3.10. All package versions pinned in `requirements.txt`.

## Reproducing everything

Each stage is one CLI command; `config.yaml` is the single source of truth
for every path, hyperparameter, seed, bound, and ablation switch.

```bash
python scripts/01_prepare_data.py         --config config.yaml  # xlsx -> parquet + cleaning report
python scripts/02_missingness_analysis.py --config config.yaml  # missingness tables + figures
python -m src.data.dataset                --config config.yaml  # windows, scalers, count tables
python scripts/03_train_baselines.py      --config config.yaml  # 7 baselines
python scripts/04_train_proposed.py       --config config.yaml  # the proposed model
python scripts/05_ablations.py            --config config.yaml  # 9 variants x 3 seeds + robustness
python scripts/06_interpretability.py     --config config.yaml  # attention + importance
python scripts/07_make_paper_assets.py    --config config.yaml  # regenerate ALL tables + figures
python -m src.evaluate                    --config config.yaml  # metrics + significance tests
```

Long runs checkpoint incrementally (`outputs/ablation_results.json`,
prediction bundles) and **resume automatically** — re-running script 05
skips everything already completed.

```bash
python -m pytest tests/    # 47 tests: unit-row stripping, mask/leakage
                           # contracts, scaler-on-train-only, model shapes,
                           # corruption determinism, ...
```

## Train / validation / test split

Chronological, no leakage:

| Split | Period | Windows |
|-------|--------|---------|
| train | 2022-01-01 – 2023-09-30 | 9,000 |
| val   | 2023-10-01 – 2023-12-31 | 1,420 |
| test  | 2024-01-01 – 2024-12-31 | 5,578 |

Air-quality series are strongly autocorrelated and seasonal, so random
splits leak future information through overlapping windows; a strictly
chronological split with the entire final year held out is the most honest
protocol and also tests cross-year generalization. Per-variable
standardization scalers are fit on the training period only, ignoring NaNs
(persisted in `data/processed/scalers.json`).

**Window/split assignment rule.** A window consists of 168 input hours
ending at anchor time *t*, with targets at *t*+6 h, *t*+24 h, *t*+72 h. A
window belongs to a split iff **all** of its horizon timestamps fall inside
that split's range. Inputs may reach back across the boundary
(deployment-realistic; inputs always precede targets, so nothing leaks). A
window is **usable** iff at least one (target pollutant, horizon) value is
observed; inputs may be arbitrarily incomplete — handling that is the
model's job. Loss and metrics are computed only over observed targets via
per-sample target masks.

## Model

```
x = value_proj(values) + miss_proj(1 - mask) + time_proj(time_feats)
    + station_embed + positional_encoding
```

`miss_proj` is a learned per-variable missingness embedding (so "absent" is
distinguishable from "measured zero"). The encoder is a vanilla pre-norm
`nn.TransformerEncoder` (3 layers, d_model 128, 8 heads, FFN 256). Variant B
(config switch, best in ablations at 24/72 h) additionally masks attention
to timesteps where the primary target is unobserved. Multi-horizon linear
heads; masked-MSE loss over observed targets only. The two-stage baseline
uses a size-identical vanilla Transformer on imputed inputs, isolating the
contribution of native missingness handling.

## Repository layout

```
config.yaml                  # single source of truth for ALL hyperparameters
src/data/                    # load, clean, dataset/windowing, impute
src/models/                  # proposed model + all baselines
src/train.py                 # unified training loop (early stop, checkpoints)
src/evaluate.py              # metrics, DM tests, bootstrap, figures
src/interpret.py             # attention extraction + feature importance
scripts/01..07_*.py          # one CLI per pipeline stage
tests/                       # 47 unit tests
outputs/                     # RESULTS.md, tables/, figures/, checkpoints/,
                             # predictions/, cleaning + analysis reports
data/raw/                    # the 3 source xlsx files
data/processed/              # cleaned parquet + scalers
```
