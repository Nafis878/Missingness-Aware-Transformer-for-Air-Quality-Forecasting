# Consolidated Results — Missingness-Aware Transformer

Generated 2026-06-13. All numbers reproducible from `config.yaml` via scripts
01–07. Learned models are trained with **three seeds (42, 43, 44)** and
reported as **mean ± std**; the deterministic statistical baselines
(persistence, seasonal-naive, SARIMA) are single runs. PM2.5 RMSE in µg/m³ on
the held-out 2024 test year, observed targets only (4,459 / 4,569 / 4,566
targets at h6 / h24 / h72).

> **Lead claim.** On **two monitoring networks** (Dhaka, 23% missing; Beijing,
> 2% missing), end-to-end missingness-aware forecasting **matches** strong
> impute-then-forecast pipelines — including a deep imputer (SAITS) — at every
> horizon, while **eliminating the imputation stage**. On the
> severely-incomplete Dhaka network, the missingness-dropout variant
> **degrades most gracefully under realistic station-outage corruption** and
> is **best on high-pollution episodes**. We are explicit about the limits of
> the robustness claim: it is **specific to severe missingness** — on the
> near-complete Beijing network a deep imputer (SAITS) is the most robust under
> the same outage corruption — and under idealized cell-wise MCAR SAITS is most
> robust on both datasets. The missingness-native RNN (GRU-D) does not beat the
> proposed transformer. **Accuracy parity and elimination of the imputation
> stage are the claims that generalize; the robustness edge is conditional on
> the deployment regime the method targets.**

## What changed in this revision (honesty notes)

- **Every learned model now reports 3-seed mean ± std.** Earlier headline
  tables mixed a single seed-42 number for the proposed model with 3-seed
  means for ablations. The single-seed "Proposed + miss-dropout = 66.2 at h6"
  is gone; the 3-seed value is 67.48 ± 0.88.
- **A previously published significance result did not replicate.** The old
  claim "two-stage KNN is significantly better than the proposed model at h24
  (p = 0.042, +1.54 µg/m³)" was a seed-42 artifact: per-seed Diebold–Mariano
  gives p = 0.038 / 0.042 / 0.891 across seeds, mean RMSE difference +0.04.
  The corrected conclusion is **statistical parity** at every horizon against
  every learned baseline.
- **Three modern baselines and a deep imputer were added** (DLinear, PatchTST,
  GRU-D, two-stage SAITS), each with 3 seeds and in every table. The SAITS
  imputer passed a quality gate on all seeds (reconstruction MAE 0.17 vs
  forward-fill 0.19), so the two-stage SAITS pipeline is a genuine strong
  competitor.

## Data (Phases 1–2)

- 16 stations across Bangladesh, 2022–2024 hourly, 412,104 rows after
  cleaning + hourly reindexing. Chronological splits: train 2022-01–2023-09,
  val 2023-10–12, test 2024 → 9,000 / 1,420 / 5,578 windows (168 h input,
  horizons 6/24/72 h).
- Missingness after cleaning: PM2.5 23.4% overall (10.4–45.1% per station);
  Rain/VWS >83% (excluded). Mean per-window input missingness 37.6% (train) /
  30.6% (test).
- Missingness is **not MCAR**: logistic regression predicts PM2.5 missingness
  from observed meteorology + calendar with AUC 0.606; pollutant
  co-missingness corr ≈ 0.38 (station outages); ~39% of missing PM2.5 hours
  sit in gaps > 7 days. **This is why the station-outage robustness suite,
  not cell-wise MCAR, is the operationally relevant stress test.**
- **Sentinel error codes**: 985.0 / 999.99 repeat verbatim in 2024 PM2.5
  (3,310) and PM10 (1,184), flagged-and-NaN'd. Without this fix persistence
  h6 RMSE is 134 instead of 94.

## Main results (Table 2: `main_results_pm25.*`, 3-seed mean ± std)

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
| Two-stage (SAITS) | 67.22 ± 0.82 | 76.31 ± 0.40 | 81.43 ± 1.15 |
| Proposed (MAT) | 67.03 ± 0.28 | 76.46 ± 0.51 | 81.54 ± 1.29 |
| Proposed (variant B) | 67.04 ± 0.29 | **76.00 ± 0.51** | 80.26 ± 1.36 |
| Proposed + miss-dropout | 67.48 ± 0.88 | 79.69 ± 1.77 | 81.54 ± 0.91 |

(MAE and R² for all of the above are in `main_results_pm25.csv`; full per-seed
metrics for all six pollutants in `metrics_full.csv`.)

Reading the table honestly:

- **Accuracy parity.** At every horizon the proposed model, variant B, and the
  three two-stage pipelines (KNN, MICE, SAITS) sit inside each other's ±std.
  No impute-then-forecast pipeline — not even the deep imputer — beats the
  end-to-end model by a significant margin (see significance below).
- **Variant B is the best proposed configuration** (best at h24 and competitive
  at h72) and is significantly better than the plain proposed model at h72.
- **GRU is genuinely strong at h72** (79.58, the single lowest value there) —
  reported prominently because it matters: the long-horizon advantage of the
  transformer family is not established.
- **GRU-D, the canonical missingness-native RNN, does not beat the proposed
  transformer** — it is slightly worse at h6 and middling elsewhere.
- **PatchTST is the weakest learned model** and **DLinear** is weak too: a
  channel-independent patching transformer and a linear model are poorly
  suited to this heavily-missing multivariate regime.

## Significance (Table: `significance_dm_bootstrap.*`, per-seed DM)

Design decision (documented here for reviewers): we run the Diebold–Mariano
test and the paired bootstrap **per seed** — proposed seed *i* vs baseline
seed *i* — and report the median and range of the p-values plus an
"all-seeds-significant" flag. We do **not** average predictions across seeds
first: that would test a 3-member ensemble nobody trains or deploys and would
asymmetrically flatter the learned models against the single-run statistical
baselines. Per-seed pairing keeps the paired time-series structure DM
requires and surfaces fragile results.

- Proposed ≫ persistence / seasonal-naive / SARIMA at every horizon
  (p < 0.001 at all seeds; RMSE reductions 5–27 µg/m³).
- Proposed vs LSTM / GRU / GRU-D / two-stage KNN / MICE / SAITS:
  **not significant at any horizon** (median p ranges 0.04–0.49, never
  significant at all three seeds). The earlier "KNN better at h24" result was
  seed-42 only (p = 0.038 / 0.042 / 0.891).
- Variant B is **significantly better than the plain proposed model at h72**
  (p median 0.006, significant at all seeds, −1.28 µg/m³).
- Proposed vs DLinear: proposed significantly better at all horizons
  (the linear baseline is genuinely behind).

## Robustness (the operational stress test: `robustness_rmse.*`, `robustness_curve.*`)

Two corruption mechanisms applied to observed test inputs at +10/30/50%
(seed-42 checkpoints; two-stage pipelines re-impute the corrupted series with
imputers fit on uncorrupted train rows — for SAITS this is transform-only):

- **Cell-wise MCAR** leaves the same-timestep cross-section intact — the
  *easy* case for any row-wise or attention-based imputer.
- **Station-outage blocks** (6–48 h all-variable gaps) match the mechanism
  that dominates real missingness here (co-missingness 0.38, long gaps).

PM2.5 **h6** RMSE at +0 / 10 / 30 / 50% (slope = +50% − clean):

| Model | MCAR (slope) | Outage (slope) |
|---|---|---|
| Proposed (MAT) | 66.8 / 67.7 / 70.0 / 72.7 (**+5.9**) | 66.8 / 67.8 / 68.1 / 69.8 (+3.1) |
| Proposed + miss-dropout | 66.2 / 66.4 / 67.2 / 68.3 (+2.1) | **66.2 / 67.0 / 67.6 / 68.3 (+2.0)** |
| Two-stage (KNN) | 67.0 / 67.4 / 69.0 / 70.2 (+3.1) | 67.0 / 68.3 / 68.8 / 70.5 (+3.5) |
| Two-stage (MICE) | 67.3 / 67.8 / 68.7 / 69.9 (+2.6) | 67.3 / 68.4 / 68.5 / 70.4 (+3.0) |
| **Two-stage (SAITS)** | **66.1 / 65.7 / 65.3 / 66.1 (+0.0)** | 66.1 / 67.1 / 67.9 / 70.0 (+3.9) |
| GRU-D | 69.3 / 69.4 / 69.9 / 71.9 (+2.6) | 69.3 / 70.7 / 72.6 / 74.3 (+5.0) |
| DLinear | 69.7 / 71.6 / 74.4 / 79.1 (+9.4) | 69.7 / 72.1 / 74.9 / 78.8 (+9.1) |
| PatchTST | 73.8 / 75.6 / 81.5 / 86.3 (+12.5) | 73.8 / 75.9 / 78.1 / 82.7 (+8.9) |

**Honest two-part conclusion:**

1. **Under cell-wise MCAR, the deep imputer wins.** Two-stage SAITS is the most
   robust *and* most accurate model at h6 — essentially flat (+0.04), better
   than the miss-dropout variant (+2.1) and far better than the plain proposed
   model (+5.9). MCAR's intact cross-section is exactly what an attention
   imputer reconstructs well. We report this plainly: if the deployment
   missingness were genuinely cell-wise random, an impute-then-forecast
   pipeline with a strong deep imputer would be the better choice.

2. **Under realistic station-outage corruption, the proposed family wins.** The
   miss-dropout variant has the flattest degradation slope (+2.0) and the
   lowest absolute RMSE at +50% (68.3), ahead of SAITS (+3.9, 70.0) and KNN
   (+3.5, 70.5). Because station outages — not cell-wise MCAR — dominate the
   real missingness in this network, **this is the operationally relevant
   result**, and it is the case where eliminating the imputation stage pays
   off: row-wise/attention imputers have no intra-timestep signal to lean on
   when a whole station block is gone.

Plain proposed degrades fastest under MCAR because it was never trained at
those missingness levels; training-time missingness dropout is the fix and
costs only ~0.5 µg/m³ of clean h6 accuracy. **PatchTST and DLinear collapse
under both corruptions** and are not robust alternatives. GRU-D is the worst
neural model under outage corruption.

## High-pollution episodes (Table: `episode_rmse_pm25.*`, figure `episode_rmse.*`)

RMSE restricted to test hours with **observed PM2.5 > 150 µg/m³** (818 / 1,135
/ 1,136 targets at h6/h24/h72) — operationally the hours that matter. The
subset conditions on observed values, so it is identical across models.

| Model | h6 | h24 | h72 |
|---|---|---|---|
| Two-stage (SAITS) | 133.21 ± 1.20 | 126.30 ± 0.89 | 134.80 ± 0.85 |
| Proposed (MAT) | 133.61 ± 1.52 | 127.91 ± 0.84 | 134.18 ± 1.90 |
| Proposed (variant B) | 136.32 ± 1.17 | 128.07 ± 0.17 | **133.00 ± 1.21** |
| **Proposed + miss-dropout** | **130.17 ± 3.65** | **125.29 ± 2.49** | 133.00 ± 2.73 |
| Two-stage (KNN) | 134.80 ± 1.60 | 130.13 ± 0.67 | 134.06 ± 1.73 |
| PatchTST | 152.52 ± 1.68 | 147.21 ± 4.19 | 152.22 ± 3.31 |

The **proposed + miss-dropout** variant is best on the high-pollution episodes
at h6 and h24 — the same configuration that is most robust under outage
corruption. (PatchTST is worst, consistent with its overall ranking.)

## Ablations (Table 3: `table3_ablations.*`, mean ± std, 3 seeds)

| Variant | h6 | h24 | h72 |
|---|---|---|---|
| full | 67.03 ± 0.28 | 76.46 ± 0.51 | 81.54 ± 1.29 |
| no_miss_embed | 67.52 ± 0.18 | 76.35 ± 0.29 | 81.93 ± 1.89 |
| variant_B | 67.04 ± 0.29 | 76.00 ± 0.51 | 80.26 ± 1.36 |
| no_met | 67.43 ± 0.05 | 76.70 ± 1.18 | 81.46 ± 0.72 |
| no_time | 67.88 ± 0.56 | 78.13 ± 0.96 | 84.69 ± 1.06 |
| seq72 | 68.10 ± 0.52 | 77.21 ± 1.10 | 80.59 ± 0.71 |
| seq336 (1 seed, batch 32) | 68.62 | 77.99 | 80.98 |
| single_h24 | — | 77.05 ± 1.49 | — |
| miss_dropout | 67.48 ± 0.88 | 79.69 ± 1.77 | 81.54 ± 0.91 |

Reads: the missingness embedding helps at h6 (+0.49 removed); variant B is the
best configuration (significant −1.28 at h72); calendar features matter most
(h72 +3.2 removed); 168 h beats 72 h and 336 h; miss-dropout trades clean h24
accuracy (+3.2) for robustness and episode performance.

## Efficiency (Table 4: `efficiency.*`)

| Model | Params | Train (min) | Impute (min) | Latency (ms/window) |
|---|---|---|---|---|
| Proposed (MAT) | 406k | 30.2 | 0 | 1.80 |
| Proposed (variant B) | 406k | 24.6 | 0 | 1.8 |
| Two-stage (KNN) | 405k | 24.4 | 2.3 | 1.50 |
| Two-stage (MICE) | 405k | 27.9 | 0.1 | 1.59 |
| Two-stage (SAITS) | 405k | 23.3 | 5.6 | 1.41 |
| GRU-D | 69k | 18.7 | 0 | 0.53 |
| GRU | 205k | 6.2 | 0 | 0.93 |
| DLinear | 9k | 0.2 | 0 | 0.06 |
| PatchTST | 410k | 46.2 | 0 | 3.37 |

The two-stage pipelines carry an imputation stage that must be **re-run on
every data refresh** (SAITS fit 5.6 min; KNN/MICE/SAITS re-imputation at
inference); the end-to-end model needs none. This — not a raw accuracy
advantage — is the deployability case. DLinear is essentially free but far
less accurate; GRU-D is cheap but not more robust.

## Second dataset (Beijing Multi-Site, UCI id 501)

420,768 rows, 12 stations, windows 11,920 / 1,068 / 4,356; last full year
(2016-03–2017-02) held out as test, preceding 3 months as val. **Beijing's
natural PM2.5 missingness is only 2.1%** (PM10 1.6%, O3 4.2%, CO 5.3%) versus
Dhaka's 23.4% — so Beijing is an **external-validity check on a near-complete
network**, and its synthetic-corruption suite (same MCAR + outage protocol) is
where the mechanism is stressed. Models were trained on Google Colab (A100);
GPU and CPU runs at the same seed are not bit-identical.

**Beijing main results** (PM2.5 RMSE, 3-seed mean ± std; `outputs/beijing/tables/`,
copied to `main_results_beijing.*`):

| Model | h6 | h24 | h72 |
|---|---|---|---|
| GRU | 51.44 ± 0.63 | 75.90 ± 0.18 | 84.88 ± 0.37 |
| GRU-D | 50.15 ± 0.12 | **75.87 ± 0.48** | 84.77 ± 0.37 |
| DLinear | 51.55 ± 0.70 | 80.62 ± 0.33 | 89.94 ± 0.19 |
| PatchTST | 58.50 ± 0.73 | 79.71 ± 0.89 | 87.54 ± 1.59 |
| Two-stage (KNN) | 49.67 ± 0.66 | 76.83 ± 0.37 | 86.78 ± 0.99 |
| Two-stage (SAITS) | 50.28 ± 0.77 | 76.61 ± 0.40 | 87.72 ± 1.46 |
| Proposed (MAT) | 49.43 ± 0.72 | 76.79 ± 0.36 | 86.37 ± 0.45 |
| Proposed (variant B) | **49.38 ± 0.74** | 76.74 ± 0.29 | 86.46 ± 0.70 |
| Proposed + miss-dropout | 51.44 ± 0.61 | 76.37 ± 0.25 | 85.58 ± 0.24 |

**Accuracy parity holds on Beijing too** — and the proposed model / variant B
are in fact *marginally the best* at h6 (49.4 vs KNN 49.7, SAITS 50.3), with
GRU-D/GRU best at the longer horizons; all inside ±std.

**The missingness-severity crossover — a two-factor finding**
(`crossover.*`, `decision_summary.*`, `crossover_combined.*`,
`stratified_gap.*`). We trace the *advantage* (best impute-then-forecast RMSE −
best end-to-end RMSE; positive ⇒ end-to-end wins) against **effective input
missingness** so both networks share one severity axis. The two networks behave
**oppositely** under station outages, and that is the result:

| Effective input missingness (h6, outage) | Dhaka gap | Beijing gap |
|---|---|---|
| low (~3–10%) | — (Dhaka starts at 33%) | +0.7 → −0.9 |
| ~33% | −0.2 | −2.1 |
| ~50% | +0.4 | −4.3 |
| ~66% | **+1.7** | −4.0 |
| ~80% | +1.0 | (n/a) |

- **Dhaka (severe, less-structured): a clean crossover.** End-to-end overtakes
  the best impute-then-forecast pipeline above **~38% effective missingness at
  h6** (and ~71% at h24); the advantage grows with severity. Window-stratified
  on *natural* missingness (`stratified_gap.csv`), end-to-end trails SAITS by
  2.1 µg/m³ on the most complete windows but **leads by 2.3 µg/m³ on the most
  incomplete (54–100% missing)** — the same crossover, seen per-window.
- **Beijing (near-complete, highly periodic): no crossover.** The deep imputer
  wins under outages at **every** tested severity and its margin *grows*
  (−0.9 → −4.3 µg/m³). Strong diurnal/seasonal structure lets SAITS reconstruct
  even 6–48 h outage blocks, so more (well-imputed) missingness helps the
  two-stage pipeline.
- **Under cell-wise MCAR, the deep imputer wins throughout on both networks**
  (the intact same-timestep cross-section is trivially imputable).

**Honest conclusion:** the crossover is **not a single universal missingness
threshold** — it depends on missingness severity *and* series imputability.
End-to-end forecasting helps specifically in the **high-missingness,
low-imputability** regime (Dhaka), the operational reality of incomplete
networks in the developing world. What generalizes across both networks is
**accuracy parity** with the strongest pipelines (including the deep imputer)
and the **elimination of the serving-time imputation stage**.

**Making imputability measurable (`imputability.*`,
`decision_by_imputability.*`, `imputability_crossover.*`).** To turn "it depends
on imputability" from a narrative into a *predictor*, we measure imputability
directly: hide a seeded 20% of observed test cells and compare the trained SAITS
imputer's reconstruction RMSE to forward-fill, `imputability = 1 −
RMSE_SAITS/RMSE_ffill` (standardized units; higher = more reconstructable). The
two networks order as the story predicts — **Beijing +0.20** (deep imputer beats
ffill) vs **Dhaka −0.39** (deep imputer *worse* than ffill) — and the
end-to-end advantage at a fixed severe operating point (h6, +50% outage)
**declines monotonically with imputability**:

| Network | imputability | end-to-end advantage (h6, +50% outage) | winner |
|---|---|---|---|
| Dhaka | −0.39 | **+1.69 µg/m³** | end-to-end |
| Beijing | +0.20 | **−4.29 µg/m³** | impute-then-forecast |

A **third network (Delhi, CPCB; 6 sites, 2018–2019)** is being added on the same
axis to convert these two opposite anchors into a single
crossover-vs-imputability curve and a deployable rule ("measure imputability,
then choose the paradigm"). The published Delhi series is complete (gap-free, no
natural missingness) but extreme and spiky (PM2.5 mean ≈ 93, std ≈ 98
µg/m³), so it probes whether a *complete-but-noisy* network is reconstructable —
making the x-axis genuinely **imputability**, not completeness. Delhi results
are pending the GPU training run; if its imputability does not land between the
anchors, that is reported as a finding (imputability necessary but not
sufficient) rather than smoothed over.

## Interpretability (`interpretability_summary.json`, attention figures)

Unchanged from the previous analysis (seed-42 proposed checkpoint): forecast-
token attention shows a recency peak plus learned ~24 h periodicity; in
PM2.5-sparse windows 70.6% of attention mass moves to meteorology-only
timesteps; permutation importance PM2.5 ≫ PM10 > RH > BP > Temp.

## Paper narrative (revised)

1. **Parity, on two networks.** End-to-end missingness-aware forecasting
   **matches** strong impute-then-forecast pipelines — including a deep imputer
   (SAITS) — at all three horizons on both Dhaka and Beijing (no significant
   difference across seeds), while **eliminating the imputation stage** and its
   per-refresh re-imputation cost. This is the claim that generalizes.
2. **Conditional robustness, stated honestly.** On the severely-incomplete
   Dhaka network, under the realistic station-outage mechanism, the
   missingness-dropout variant **degrades most gracefully** and is **best on
   high-pollution episodes**. This advantage is **specific to severe
   missingness**: on the near-complete Beijing network a deep imputer (SAITS)
   is the most robust under the same outage corruption, and under idealized
   cell-wise MCAR SAITS is most robust on both datasets. We report these losses
   rather than hide them — the method helps most exactly where missingness is
   severe, which is the deployment regime (resource-constrained, incomplete
   monitoring networks) it is designed for.
3. **Strong baselines, not strawmen.** The **missingness-native RNN (GRU-D)
   does not beat** the proposed transformer, and **PatchTST/DLinear are poorly
   suited** to heavy multivariate missingness — establishing that parity is
   against genuinely strong competitors, including a quality-gated deep imputer.
4. **A non-replicating result, corrected.** The previously published "two-stage
   KNN beats the proposed model at h24 (p = 0.042)" was a seed-42 artifact;
   3-seed per-seed DM shows parity.
5. Cleaning matters: undocumented sensor error codes (985.0) would have
   corrupted every published Dhaka number.
