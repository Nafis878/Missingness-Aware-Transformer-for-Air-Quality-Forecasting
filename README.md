# Missingness-Aware Transformer for Multi-Pollutant Air Quality Forecasting

Code, experiments, and paper assets for *"A Missingness-Aware Transformer for
Multi-Pollutant Air Quality Forecasting on Severely Incomplete Monitoring
Data from Bangladesh."*

An end-to-end Transformer forecasts pollutants (primary target PM2.5; also
PM10, NO2, O3, CO, SO2) **directly from incomplete sensor streams** using
learned missingness embeddings and masked attention — no imputation stage —
and is compared against impute-then-forecast pipelines (KNN, MICE, and the
deep imputer **SAITS**), the missingness-native **GRU-D** RNN, modern
forecasters (**DLinear**, **PatchTST**), and statistical baselines.

**The claim is a predictive rule plus deployability, not raw accuracy:** across
**three monitoring networks** spanning the reconstructability spectrum (Dhaka,
Delhi, Beijing), the choice between end-to-end and impute-then-forecast is
governed by a **directly measured imputability** — the end-to-end advantage at
fixed severe outage declines monotonically with it and crosses zero, so a
practitioner can *measure imputability, then choose the paradigm*. On the
incomplete networks (Dhaka, Beijing), end-to-end missingness-aware forecasting
*matches* the strong impute-then-forecast pipelines — including a deep imputer —
at every horizon while removing the imputation stage. On the severely-incomplete
Dhaka network the missingness-dropout variant additionally degrades most
gracefully under realistic station outages and is best on high-pollution
episodes — an advantage that, we show honestly, is **specific to the
high-missingness, low-imputability regime** and does not carry over to the
more-imputable Delhi and Beijing networks (on complete, noisy Delhi the proposed
model is not even competitive on clean data — reported, not hidden).

**Everything runs on a desktop CPU**: small models (d_model 128, 3 layers,
~406k parameters), vanilla PyTorch, fixed seeds, deterministic flags. All
learned models are trained with **3 seeds (42/43/44)** and reported as
**mean ± std** — no single-seed number is ever reported where a multi-seed
one exists.

## Headline results

PM2.5 test RMSE (µg/m³) on the held-out 2024 year, observed targets only,
**3-seed mean ± std** for learned models (statistical baselines are
deterministic single runs):

| Model | 6 h | 24 h | 72 h |
|---|---|---|---|
| Persistence | 93.9 | 99.0 | 105.1 |
| SARIMA (per station) | 80.0 | 83.9 | 86.7 |
| LSTM | 68.20 ± 0.34 | 78.05 ± 0.08 | 80.95 ± 0.79 |
| GRU | 67.69 ± 0.47 | 76.45 ± 0.20 | **79.58 ± 0.37** |
| GRU-D | 68.89 ± 0.31 | 76.69 ± 0.49 | 80.61 ± 0.24 |
| DLinear | 70.51 ± 0.72 | 80.90 ± 1.34 | 83.55 ± 0.40 |
| PatchTST | 74.61 ± 1.07 | 84.34 ± 1.14 | 87.51 ± 1.34 |
| Two-stage (KNN → Transformer) | **66.95 ± 0.58** | 76.42 ± 0.75 | 80.95 ± 1.52 |
| Two-stage (MICE → Transformer) | 67.39 ± 0.25 | 76.81 ± 0.68 | 80.66 ± 1.13 |
| Two-stage (SAITS → Transformer) | 67.22 ± 0.82 | 76.31 ± 0.40 | 81.43 ± 1.15 |
| **Proposed (MAT)** | 67.03 ± 0.28 | 76.46 ± 0.51 | 81.54 ± 1.29 |
| **Proposed (variant B)** | 67.04 ± 0.29 | **76.00 ± 0.51** | 80.26 ± 1.36 |
| **Proposed + missingness dropout** | 67.48 ± 0.88 | 79.69 ± 1.77 | 81.54 ± 0.91 |

- **Accuracy parity, established honestly.** The proposed model, variant B,
  and all three two-stage pipelines (KNN, MICE, SAITS) sit inside each other's
  ±std at every horizon; per-seed Diebold–Mariano finds **no significant
  difference** against any of them. (A previously reported "two-stage KNN
  beats us at h24, p = 0.042" was a seed-42 artifact: across seeds
  p = 0.038 / 0.042 / 0.891. See [`outputs/RESULTS.md`](outputs/RESULTS.md).)
- **The headline: a missingness-severity crossover (two-factor, stated
  honestly).** Tracing the end-to-end *advantage* (best impute-then-forecast −
  best end-to-end RMSE) against effective input missingness shows the two
  networks behave **oppositely** under station outages. On **Dhaka** (severe,
  less-structured) end-to-end overtakes the best deep-imputer pipeline above
  **~38% effective missingness at 6 h**, and window-stratified on natural
  missingness it trails SAITS by 2.1 µg/m³ on the most complete windows but
  **leads by 2.3 µg/m³ on the most incomplete**. On **Beijing** (near-complete,
  highly periodic) the deep imputer wins under outages at *every* severity and
  its margin grows — strong diurnal structure keeps even long outage blocks
  imputable. Under cell-wise MCAR the deep imputer wins on both. So the choice
  depends on missingness severity **and series imputability**, not a single
  threshold; end-to-end forecasting helps in the high-missingness,
  low-imputability regime — the operational reality of incomplete networks.
- **A second monitoring network confirms parity.** On Beijing the proposed
  model and variant B are in fact *marginally the best* at 6 h (49.4 vs KNN
  49.7, SAITS 50.3 µg/m³), parity at the longer horizons.
- **High-pollution episodes** (Dhaka, observed PM2.5 > 150 µg/m³): proposed +
  missingness dropout is best at 6 h (130.2) and 24 h (125.3).
- **Strong baselines, not strawmen.** GRU-D (missingness-native RNN) does *not*
  beat the proposed transformer; PatchTST and DLinear are clearly behind and
  collapse under corruption — the parity result is against genuinely strong
  competitors, including a quality-gated deep imputer.
- **Deployability** is the practical edge that generalizes: the two-stage
  pipelines re-impute on every data refresh (SAITS imputer fit 5.6 min;
  re-imputation at inference); the end-to-end model runs at 1.8 ms/window with
  no imputation stage.

Full consolidated numbers and the frank robustness assessment:
[`outputs/RESULTS.md`](outputs/RESULTS.md).
All tables (CSV + booktabs LaTeX) in [`outputs/tables/`](outputs/tables/),
all figures (300-dpi PNG + vector PDF) in [`outputs/figures/`](outputs/figures/).
A second monitoring network (Beijing Multi-Site, UCI) is wired into the same
pipeline as an external-validity check; see [`COLAB.md`](COLAB.md). The
submission-ready manuscript is in [`paper/`](paper/) (build with
`latexmk -pdf main.tex`).

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

Missingness itself is a study object (a paper section): PM2.5 is 23.7%
missing overall (10–45% per station), missingness is demonstrably not MCAR
(predictable from meteorology + season, AUC 0.61), and station-level outages
dominate the long-gap mass — which motivates both the model design and the
outage-style robustness experiment.

Two further networks place the study on a **measured-imputability** axis (see
`config_beijing.yaml`, `config_delhi.yaml`): the near-complete **Beijing
Multi-Site** benchmark (2.1% missing) and the complete-but-noisy **Delhi CPCB**
network. The headline finding is a **predictive deployment rule** — the
end-to-end advantage at fixed severe outage declines monotonically with a
directly measured imputability (1 − RMSE_SAITS/RMSE_ffill on held-out observed
cells) and crosses zero: Dhaka (imputability −0.39) favors end-to-end, while the
more-imputable Delhi (−0.09) and Beijing (+0.20) favor impute-then-forecast.
*Measure a network's imputability, then choose the paradigm.* See
[`outputs/RESULTS.md`](outputs/RESULTS.md) for the full three-network results.

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
python scripts/03_train_baselines.py      --config config.yaml  # baselines x 3 seeds (--seeds 42,43,44)
python scripts/04_train_proposed.py       --config config.yaml --seeds 42,43,44          # proposed
python scripts/04_train_proposed.py       --config config.yaml --seeds 42,43,44 --variant B --name variant_B
python scripts/04_train_proposed.py       --config config.yaml --seeds 42,43,44 --miss-dropout
python scripts/05_ablations.py            --config config.yaml  # ablations + robustness suite
python scripts/05_ablations.py            --config config.yaml --export-seed-predictions  # per-seed bundles
python scripts/06_interpretability.py     --config config.yaml  # attention + importance
python scripts/07_make_paper_assets.py    --config config.yaml  # regenerate ALL tables + figures
python -m src.evaluate                    --config config.yaml  # metrics + significance tests
```

Learned models train with three seeds; per-seed prediction bundles land in
`outputs/predictions/seeds/`, the canonical seed-42 bundle stays at the top
level. Every (model, seed) run is skipped when its bundle already exists, so
all long runs **resume automatically**. The second dataset runs the identical
pipeline via `--config config_beijing.yaml` (training on Colab T4 — see
[`COLAB.md`](COLAB.md)); script 07 gains `--secondary-config config_beijing.yaml`
for the cross-dataset summary table.

```bash
python -m pytest tests/    # 82 tests: unit-row stripping, mask/leakage
                           # contracts, scaler-on-train-only, model shapes +
                           # determinism + mask-poisoning for every new model,
                           # multi-seed aggregation, per-seed significance,
                           # SAITS imputer, Beijing loader, ...
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
(config switch, significantly best at h72) additionally masks attention to
timesteps where the primary target is unobserved. Multi-horizon linear heads;
masked-MSE loss over observed targets only. The two-stage baselines use a
size-identical vanilla Transformer on inputs imputed by KNN, MICE, or SAITS
(a deep self-attention imputer trained on train-period rows only), isolating
the contribution of native missingness handling.

Baselines live in `src/models/` behind a unified `forward(batch) -> (B, T, H)`
contract and a name registry (`src/models/factory.py`): the missingness-native
**GRU-D** (Che et al. 2018; learned input/hidden decay), **DLinear** (Zeng et
al. 2023), **PatchTST** (Nie et al. 2023), the **SAITS** imputer (Du et al.
2023, minimal in-repo implementation), LSTM/GRU, and persistence /
seasonal-naive / SARIMA.

## Repository layout

```
config.yaml                  # single source of truth for ALL hyperparameters
config_beijing.yaml          # second dataset, same pipeline (outputs/beijing/)
src/data/                    # load, clean, dataset/windowing, impute, load_beijing
src/models/                  # proposed model + all baselines + factory
src/train.py                 # unified training loop (early stop, checkpoints, device)
src/evaluate.py              # metrics, multi-seed tables, per-seed DM, episode, figures
src/interpret.py             # attention extraction + feature importance
scripts/01..07_*.py          # one CLI per pipeline stage (+ 01b_prepare_beijing)
scripts/colab_run_beijing.py # one-command Beijing grid for Colab T4 (see COLAB.md)
tests/                       # 82 unit tests
outputs/                     # RESULTS.md, tables/, figures/, checkpoints/,
                             # predictions/ (+ predictions/seeds/), reports
UPGRADE_LOG.md               # change log: what ran, wall-clock, contradictions
data/raw/                    # the 3 source xlsx files (+ raw/beijing/ CSVs)
data/processed/              # cleaned parquet + scalers (per dataset)
```
