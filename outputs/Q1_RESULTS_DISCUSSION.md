# Results and Discussion — Manuscript Prose

*Consolidated, journal-ready synthesis of every result and finding in the
project, with all numbers traced to authoritative artifacts in `outputs/`.
Learned models are trained with three seeds (42, 43, 44) and reported as
mean ± standard deviation; deterministic statistical baselines (persistence,
seasonal-naive, SARIMA) are single runs. PM2.5 RMSE is reported in µg/m³ on the
held-out test period, computed over observed targets only.*

*A note on scope and honesty (read first): the repository contains four
overlapping headline narratives that are not all equally defensible. This
document presents them side-by-side and explicitly flags the statistical-claim
tension between them (Section R.8 and Discussion D.3) so the strongest and the
most aggressive framings are both visible. Per the project's standing honesty
rule, no single-seed number is quoted where a multi-seed value exists, and known
non-replications and losses are reported rather than hidden.*

---

## 4. Results

### R.1 Data and the missingness characterization

The primary network comprises **16 monitoring stations across Bangladesh
(Dhaka and other CAMS sites), hourly for 2022–2024**, yielding **412,104 rows**
after cleaning and strict hourly reindexing (Table 1, `table1_dataset_summary.csv`).
The forecasting protocol uses a 168-hour input window with targets at +6, +24,
and +72 hours, and a strictly chronological split — train 2022-01 to 2023-09
(9,000 windows), validation 2023-10 to 2023-12 (1,420 windows), and the **entire
2024 year held out as test** (5,578 windows). A chronological split is essential
here: air-quality series are strongly autocorrelated and seasonal, so random
splits leak future information through overlapping windows. Per-variable
standardization scalers are fit on the training period only, ignoring NaNs.

Missingness is treated as a primary study object rather than a nuisance.
**PM2.5 is 23.7% missing overall**, ranging from 10.5% to 45.3% per station, with
an all-variable missing rate of 35.7% and mean per-window input missingness of
37.6% (train) / 30.6% (test). Crucially, the missingness is **demonstrably not
missing-completely-at-random (MCAR)**: a logistic model predicts PM2.5
missingness from observed meteorology and calendar features with **AUC 0.61**,
pollutant co-missingness correlation is ≈ 0.38 (the signature of station-level
outages), and roughly **39% of missing PM2.5 hours fall in gaps longer than
7 days**. This motivates two design choices that recur throughout the paper:
(i) a model that consumes incomplete streams directly, and (ii) a robustness
protocol built around **station-outage blocks**, not idealized cell-wise MCAR.

Data cleaning materially affects every downstream number. The 2024 PM sensors
emit verbatim saturation/error codes (985.0 and 999.99, thousands of
occurrences, physically impossible in monsoon months); flagging and removing
them lowers the persistence h6 RMSE from **134 to 94 µg/m³**. The full handling —
unit-row stripping, plausibility bounds, sentinel codes, stuck-sensor runs — is
documented in `data_cleaning_report.md`.

Two additional networks place the study on a **reconstructability (imputability)
axis** rather than a single-dataset anchor: the **Beijing Multi-Site** benchmark
(UCI; 12 stations, 420,768 rows, PM2.5 only **2.1% missing** — a near-complete
network) and the **Delhi CPCB** network (Mendeley `bzhzr9b64v`; 6 stations,
70,227 rows, **< 0.2% missing but extreme and spiky**, PM2.5 mean ≈ 93,
std ≈ 98 µg/m³). Delhi is deliberately the *complete-but-hard-to-impute* interior
point of the axis, not an intermediate-missingness case.

### R.2 Main forecasting accuracy and the parity finding

On the Dhaka test year (Table 2, `main_results_pm25.csv`), the proposed
Missingness-Aware Transformer (MAT), its attention-masking variant B, and the
impute-then-forecast pipelines cluster tightly:

| Model | h6 | h24 | h72 |
|---|---|---|---|
| Persistence | 93.90 | 99.02 | 105.13 |
| Seasonal-naive | 90.59 | 93.67 | 102.18 |
| SARIMA | 79.96 | 83.87 | 86.74 |
| LSTM | 68.20 ± 0.34 | 78.05 ± 0.08 | 80.95 ± 0.79 |
| GRU | 67.69 ± 0.47 | 76.45 ± 0.20 | **79.58 ± 0.37** |
| GRU-D | 68.89 ± 0.31 | 76.69 ± 0.49 | 80.61 ± 0.24 |
| DLinear | 70.51 ± 0.72 | 80.90 ± 1.34 | 83.55 ± 0.40 |
| PatchTST | 74.61 ± 1.07 | 84.34 ± 1.14 | 87.51 ± 1.34 |
| Two-stage (KNN) | **66.95 ± 0.58** | 76.42 ± 0.75 | 80.95 ± 1.52 |
| Two-stage (MICE) | 67.39 ± 0.25 | 76.81 ± 0.68 | 80.66 ± 1.13 |
| Two-stage (SAITS) | 67.22 ± 0.82 | **76.31 ± 0.40** | 81.43 ± 1.15 |
| Proposed (MAT) | 67.03 ± 0.28 | 76.46 ± 0.51 | 81.54 ± 1.29 |
| Proposed (variant B) | 67.04 ± 0.29 | **76.00 ± 0.51** | 80.26 ± 1.36 |
| Proposed + miss-dropout | 67.48 ± 0.88 | 79.69 ± 1.77 | 81.54 ± 0.91 |

The central accuracy result is **parity, not dominance**. At every horizon the
proposed model, variant B, and all three two-stage pipelines (KNN, MICE, and the
deep imputer SAITS) sit inside one another's ± standard deviation. We establish
this with **per-seed Diebold–Mariano (DM) tests** — proposed seed *i* against
baseline seed *i*, reporting the median and range of p-values plus an
"all-seeds-significant" flag (`significance_dm_bootstrap.csv`). We deliberately
do **not** average predictions across seeds before testing, which would evaluate
a three-member ensemble nobody deploys and would asymmetrically flatter the
learned models against the single-run statistical baselines.

Under this protocol the proposed model is overwhelmingly better than persistence,
seasonal-naive, and SARIMA (p < 0.001 at all seeds; RMSE reductions of
5–27 µg/m³) but shows **no significant difference** against LSTM, GRU, GRU-D, or
any of the two-stage pipelines at any horizon (median p-values 0.04–0.49, never
significant at all three seeds). Two honest qualifications belong in the text.
First, **variant B is significantly better than the plain proposed model at h72**
(median p = 0.006, −1.28 µg/m³) and is the best proposed configuration overall.
Second, a **previously published significance result did not replicate**: the
claim "two-stage KNN beats the proposed model at h24 (p = 0.042)" was a seed-42
artifact; per-seed DM gives p = 0.038 / 0.042 / 0.891 with a mean RMSE difference
of only +0.04 µg/m³. We report the corrected conclusion — statistical parity — in
preference to the original number.

The baselines are strong, not strawmen. The missingness-native **GRU-D does not
beat** the proposed transformer (it is slightly worse at h6 and middling
elsewhere), while **PatchTST and DLinear are clearly behind** — a
channel-independent patching transformer and a linear model are poorly suited to
this heavily multivariate, heavily incomplete regime. **GRU is genuinely strong
at h72** (79.58, the single lowest value there), which we report prominently
because it shows the long-horizon advantage of the transformer family is not
established.

### R.3 The headline finding: a measured-imputability crossover rule

The paper's central scientific contribution is a **predictive deployment rule**
linking the paradigm choice (end-to-end forecasting vs. impute-then-forecast) to
a **directly measured imputability** of each network. Imputability is defined
operationally as `1 − RMSE_SAITS / RMSE_ffill` on a seeded 20% of held-out
*observed* test cells (higher = more reconstructable; 0 = forward-fill itself).
Tracing the **end-to-end advantage** (best impute-then-forecast RMSE − best
end-to-end RMSE; positive favors end-to-end) at a fixed severe operating point
(h6, ~50% station outage) against this axis yields a monotone curve that crosses
zero (`decision_by_imputability.csv`, `crossover_curve.*`):

| Network | natural PM2.5 missing | imputability | end-to-end advantage (h6, ~50% outage) | recommended paradigm |
|---|---:|---:|---:|---|
| Dhaka | 23.7% | **−0.39** | **+1.69 µg/m³** | end-to-end |
| Delhi | < 0.2% | **−0.09** | **−2.60 µg/m³** | impute-then-forecast |
| Beijing | 2.1% | **+0.20** | **−4.29 µg/m³** | impute-then-forecast |

The advantage **declines monotonically with imputability and crosses zero between
Dhaka and Delhi**, giving a deployable rule: *measure a network's imputability,
then choose the paradigm* — forecast end-to-end when imputability is low
(here ≲ −0.3), impute-then-forecast otherwise. The axis is genuinely
imputability, **not completeness**: the Delhi series is essentially gap-free yet
lands firmly on the impute-then-forecast side because it is so noisy that even a
deep imputer cannot beat forward-fill (its SAITS imputer *fails* the forward-fill
quality gate, validation MIT-MAE 0.280 vs. 0.234).

Window-level evidence corroborates the network-level curve. Stratifying Dhaka
test windows by their *natural* missingness (`stratified_gap.csv`), end-to-end
trails SAITS by 2.1 µg/m³ on the most complete windows but **leads by 2.3 µg/m³ on
the most incomplete (54–100% missing)** — the same crossover, observed per-window.
On Dhaka the crossover point is **~38% effective missingness at h6** (and ~71% at
h24). On Beijing, by contrast, there is **no crossover**: strong diurnal and
seasonal structure lets SAITS reconstruct even 6–48 h outage blocks, so the deep
imputer wins at *every* tested severity and its margin *grows* (−0.9 → −4.3 µg/m³
as outage severity rises).

### R.4 Robustness: cell-wise MCAR versus station outages

We corrupt observed test inputs at +5/10/.../70% under two mechanisms
(`robustness_rmse.csv`, `robustness_curve.*`): **cell-wise MCAR**, which leaves
the same-timestep cross-section intact (the easy case for any row-wise or
attention imputer), and **station-outage blocks** (6–48 h all-variable gaps),
which match the mechanism that dominates real missingness here. Two-stage
pipelines re-impute the corrupted series with imputers fit on uncorrupted train
rows (transform-only for SAITS). The h6 result is a clean two-part story (slope =
value at +50% minus clean):

| Model | MCAR: 0 / 50% (slope) | Outage: 0 / 50% (slope) |
|---|---|---|
| Proposed (MAT) | 66.8 / 72.7 (+5.9) | 66.8 / 69.8 (+3.1) |
| Proposed + miss-dropout | 66.2 / 68.3 (+2.1) | **66.2 / 68.3 (+2.0)** |
| Two-stage (KNN) | 67.0 / 70.2 (+3.1) | 67.0 / 70.5 (+3.5) |
| Two-stage (MICE) | 67.3 / 69.9 (+2.6) | 67.3 / 70.4 (+3.0) |
| **Two-stage (SAITS)** | **66.1 / 66.1 (+0.0)** | 66.1 / 70.0 (+3.9) |
| GRU-D | 69.3 / 71.9 (+2.6) | 69.3 / 74.3 (+5.0) |
| DLinear | 69.7 / 79.1 (+9.4) | 69.7 / 78.8 (+9.1) |
| PatchTST | 73.8 / 86.3 (+12.5) | 73.8 / 82.7 (+8.9) |

**Under cell-wise MCAR, the deep imputer wins.** Two-stage SAITS is both most
accurate and most robust at h6 — essentially flat (+0.0) because MCAR's intact
cross-section is exactly what an attention imputer reconstructs. We state plainly
that if deployment missingness were genuinely cell-wise random, an
impute-then-forecast pipeline with a strong deep imputer would be the better
choice. **Under realistic station-outage corruption, the proposed family wins.**
The missingness-dropout variant has the flattest degradation slope (+2.0) and the
lowest absolute RMSE at +50% (68.3), ahead of SAITS (+3.9, 70.0) and KNN
(+3.5, 70.5): when a whole station block disappears, row-wise and attention
imputers have no intra-timestep signal to lean on, whereas the end-to-end model's
learned missingness embedding does not depend on one. Because station outages —
not MCAR — dominate the real missingness in this network, this is the
operationally relevant case. Training-time missingness dropout is the fix for the
plain model's MCAR fragility and costs only ~0.5 µg/m³ of clean h6 accuracy.
**PatchTST and DLinear collapse under both corruptions** and are not robust
alternatives; GRU-D is the worst neural model under outages.

### R.5 High-pollution episodes

Restricting evaluation to test hours with observed PM2.5 > 150 µg/m³ — the hours
that matter operationally (818 / 1,135 / 1,136 targets at h6/h24/h72;
`episode_rmse_pm25.csv`) — the **proposed + miss-dropout** variant is best at
**h6 (130.2)** and **h24 (125.3)**, the same configuration that is most robust
under outage corruption. PatchTST is worst (152.5 at h6), consistent with its
overall ranking. This is the one regime where the end-to-end model offers a
positive accuracy edge rather than mere parity, and it coincides exactly with the
low-imputability, high-severity conditions the method targets.

### R.6 Ablations

The ablation suite (`ablation_results.json`, three seeds) isolates each design
element. The **learned missingness embedding helps at h6** (+0.49 µg/m³ when
removed). **Calendar/time features matter most**, especially at long horizons
(h72 degrades +3.2 when removed). The **168-hour input window beats both 72 h and
336 h**. **Variant B (attention masking to timesteps where the primary target is
unobserved) is the best configuration**, significantly better than the plain
model at h72 (−1.28). A **single-horizon head collapses off-target** (training
only for h24 yields h6 RMSE ≈ 108–123 and h72 ≈ 118–192), validating the
multi-horizon design. Missingness dropout trades ~3.2 µg/m³ of clean h24 accuracy
for its robustness and episode gains.

### R.7 Efficiency and deployability

The practical edge that generalizes across networks is **eliminating the
serving-time imputation stage** (`efficiency.csv`). The proposed model has
~406k parameters and runs at **1.8 ms/window with zero imputation**, whereas the
two-stage pipelines carry an imputer that must be **re-run on every data refresh**
(SAITS fit 5.6 min, plus re-imputation at inference). DLinear is essentially free
(9k params) but far less accurate; GRU-D is cheap (69k) but not more robust. Given
accuracy parity, the deployability difference — not a raw accuracy advantage — is
the operational case for end-to-end forecasting on incomplete networks.

### R.8 Two "best-model" layers, and three competing performance claims

Beyond the per-model comparison, the project explores two additional layers whose
claims must be reported precisely because they differ in statistical strength.

**(a) Imputation-techniques benchmark (`outputs/imputation_benchmark/`).** As a
pure reconstruction task across 3 datasets × 2 missingness patterns, the project
covers **94 of a 96-technique taxonomy** (59 run as real reference
implementations, 35 subsumed into equivalent methods, 2 impossible — Kriging
needs coordinates, Cold-deck needs an external donor set). The finding is that
**simple temporal methods still win** (`linear_interp`, `last_and_next_mean`,
`ssa`); the strongest single newcomer is **`tensor_cp`** (CP/PARAFAC of a
day×hour×variable tensor). An imputability-weighted blend of the eight best
methods, **`hybrid_top8`, is the overall #1 imputer** (mean imputability +0.218,
beats forward-fill in all 6 dataset×pattern cells), winning by being consistently
near-best rather than spiking. (The diffusion model `csdi` diverged and is
excluded.)

**(b) Forecast ensembles — three claims of differing strength.** The repository
reports three escalating performance narratives; the manuscript should present
them with their exact statistical scope:

1. **Validation-calibrated MAT ensemble** (`validation_convex_intercept_stack`,
   `FINAL_WINNER.md`): the overall RMSE winner on Dhaka at all horizons
   (**65.78 / 74.23 / 77.55** vs. variant-B-ridge 67.28 / 75.08 / 79.33 and a
   vanilla transformer 68.61 / 78.31 / 81.83), passing **42/42 model-horizon
   comparisons under *combined* seed-level DM with Holm correction**.
2. **Adaptive SOTA portfolio** (`UNIVERSAL_SOTA_RESULT.md`): a dataset/horizon
   router (lag-summary ExtraTrees experts plus a DLinear+tabular blend) that beats
   the previous paper-table best in **9/9 dataset-horizon cells directionally**
   (sign and Wilcoxon p = 0.00195; Fisher-combined DM p = 1.5e-05), but reaches
   **strict per-cell significance in only 2/9 cells** (Dhaka h72, Delhi h24).
3. **Defensible verdict** (`Q1_DEFENSIBLE_VERDICT.md`, `Q1_CLAIM_ANALYSIS.md`):
   under strict paired testing of the validation-weighted ensemble against
   recomputed deployable baselines, **5/9 directional and 0/9 strictly
   significant** wins. Validation weighting does, however, reliably beat
   equal-weight averaging (Dhaka h24 p = 0.002; Delhi h72 p < 0.001; Beijing
   h6/h24/h72 p = 0.000/0.010/0.003).

These three are not contradictory once their statistical scope is made explicit
(Discussion D.3): they differ in *which baseline* and *which significance
criterion* is used. The combined-DM "42/42" and directional "9/9" claims are real
but weaker than "independently significant in every cell," which the evidence does
**not** support.

### R.9 Interpretability

Attention and permutation analyses (`interpretability_summary.json`, attention
figures) show the forecast token attends with a **recency peak plus a learned
~24 h periodicity**; in PM2.5-sparse input windows the model **redirects
attention mass toward meteorology-only timesteps**, behaviour consistent with the
missingness-aware design; and permutation importance ranks **PM2.5 ≫ pressure /
NO2 > temperature > PM10**. These results are qualitative support for the model
using missingness structure and cross-variable signal as intended.

---

## 5. Discussion

### D.1 What the study establishes

The defensible spine of the contribution is threefold. First, a **predictive,
deployable rule**: across three networks spanning the reconstructability
spectrum, the paradigm choice is governed by a *directly measured* imputability,
and the end-to-end advantage at fixed severe outage declines monotonically with
it and crosses zero. This turns "it depends" into an actionable procedure —
measure imputability, then choose the paradigm. Second, **accuracy parity** with
strong impute-then-forecast pipelines (including a quality-gated deep imputer) on
the incomplete networks, while **removing the serving-time imputation stage** and
its per-refresh cost. Third, a **conditional robustness edge**: on the
severely-incomplete, low-imputability Dhaka network the missingness-dropout
variant degrades most gracefully under realistic station outages and is best on
high-pollution episodes — exactly the deployment regime (resource-constrained,
incomplete monitoring) the method is designed for.

### D.2 What the study does not claim, and the losses we report

We are explicit about the boundaries. Under idealized cell-wise MCAR, the deep
imputer SAITS is the most robust model on all three networks; the end-to-end
advantage is specific to the station-outage mechanism. On the more-imputable
Delhi and Beijing networks, impute-then-forecast wins under the same outage
corruption. Most pointedly, **on complete, noisy Delhi the proposed model is not
competitive on clean data** — persistence (21.3 µg/m³ at h6) and DLinear (21.9)
lead while the proposed model trails at 31.4 — because the mask-native pathway
buys nothing where there is essentially no missingness to be aware of. This is
not a failure of the rule; it is the rule operating as designed (Delhi is the
high-imputability regime it routes *away* from end-to-end), and we report it as a
counterweight rather than hide it. The robustness edge is therefore **conditional
on the low-imputability regime**, while the imputability rule and the elimination
of the imputation stage are the claims that generalize.

### D.3 The significance-claim tension (state this explicitly in the paper)

The repository's three ensemble/portfolio narratives must not be collapsed into a
single "state-of-the-art" sentence, because they rest on different statistical
criteria:

- A **combined seed-level** DM test (pooling the three seeds before testing)
  supports "the MAT ensemble passes 42/42 comparisons" — but a combined test is
  more permissive than a per-seed one and tests an aggregate few would deploy.
- A **directional** criterion supports "the adaptive portfolio improves the
  previous best in 9/9 cells" with strong omnibus support (sign/Wilcoxon
  p = 0.00195) — but directional improvement is not significance.
- The **strict per-cell** criterion (DM p < 0.05 *and* bootstrap CI entirely
  below zero against recomputed deployable baselines) yields **0/9** for the
  validation-weighted ensemble and **2/9** for the adaptive portfolio.

The honest, Q1-defensible framing is therefore: the proposed framework achieves
**parity** with the strongest pipelines under the strictest per-seed test;
validation-weighted ensembling **reliably improves over naive equal-weight
ensembling**; and the cross-network improvements are **directional and supported
by combined/omnibus tests**, not claims of independent significance in every
cell. We recommend leading the manuscript with the **imputability rule + parity +
deployability** and presenting the ensemble layer with its exact statistical
scope, rather than headlining "universal SOTA."

### D.4 Limitations and future work

The primary forecasting dataset is single-region (Bangladesh); Beijing and Delhi
broaden external validity but the deepest robustness and episode analyses are
Dhaka-only. The modern-baseline set, while strong (GRU-D, DLinear, PatchTST,
SAITS), omits several recent forecasters (TimesNet, iTransformer, N-HiTS/N-BEATS,
Autoformer/FEDformer); adding them would harden the parity claim. The imputability
axis currently has three points; more networks would sharpen the location of the
crossover threshold. Finally, hybrid8 forecasting checkpoints are not yet
standardized across all datasets, which limits including hybrid8 candidates in the
cross-dataset validation ensembles. Reporting paired tests and bootstrap
intervals as the primary evidence — not leaderboard means — should remain the
default in any revision.

### D.5 Practitioner takeaway

For operators of incomplete monitoring networks the message is concrete: a single
cheap measurement — hide a fraction of observed cells and compare a deep imputer's
reconstruction to forward-fill — predicts whether to deploy an end-to-end
missingness-aware forecaster or an impute-then-forecast pipeline. Where data are
severely incomplete and hard to reconstruct (the developing-world norm), the
end-to-end model matches the best pipeline, degrades more gracefully under the
station outages that actually occur, performs best when pollution is highest, and
removes the imputation stage entirely. *Measure imputability, then choose the
paradigm.*

---

## Appendix: artifact map for every quoted result

| Result | Source artifact |
|---|---|
| Dataset summary, per-station missingness | `outputs/tables/table1_dataset_summary.csv` |
| Main PM2.5 RMSE (Dhaka) | `outputs/tables/main_results_pm25.csv`; `outputs/RESULTS.md` |
| Per-seed DM / bootstrap significance | `outputs/tables/significance_dm_bootstrap.csv` |
| Imputability + crossover decision | `outputs/tables/decision_by_imputability.csv`; `crossover.csv`; `crossover_curve.*` |
| Window-stratified gap | `outputs/tables/stratified_gap.csv` |
| Robustness (MCAR + outage) | `outputs/tables/robustness_rmse.csv`; `robustness_curve.*` |
| High-pollution episodes | `outputs/tables/episode_rmse_pm25.csv` |
| Ablations | `outputs/ablation_results.json` |
| Efficiency / deployability | `outputs/tables/efficiency.csv` |
| Beijing main results | `outputs/beijing/tables/main_results_pm25.csv` |
| Delhi main results | `outputs/delhi/tables/main_results_pm25.csv` |
| Imputation benchmark (94/96, hybrid_top8) | `README.md`; `outputs/imputation_benchmark/ANALYSIS.md`; `coverage_map.csv` |
| MAT ensemble winner (42/42 combined DM) | `outputs/FINAL_WINNER.md`; `tables/combined_seed_significance_validation_convex_intercept_stack.csv` |
| Adaptive portfolio (9/9 directional, 2/9 strict) | `outputs/UNIVERSAL_SOTA_RESULT.md`; `tables/adaptive_sota_portfolio_paired_tests.csv` |
| Defensible verdict (0/9 strict; equal-weight ablation) | `outputs/Q1_DEFENSIBLE_VERDICT.md`; `outputs/Q1_CLAIM_ANALYSIS.md` |
| Interpretability | `outputs/interpretability_summary.json`; attention figures |
| Data cleaning (sentinel codes) | `outputs/data_cleaning_report.md` |
