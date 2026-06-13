# Upgrade Log

Running log of the Q1-reviewer upgrade (multi-seed results, modern baselines,
Beijing dataset, metrics + framing). Every entry records what was changed,
what was run, wall-clock time, and any result that contradicts the previous
narrative.

Conventions
- Per-seed prediction bundles: `outputs/predictions/seeds/{model}_s{seed}_test.npz`
  (all seeds incl. 42); the canonical seed-42 bundle stays at the top level as
  `{model}_test.npz` so existing figure/robustness code is untouched.
- Significance: per-seed Diebold-Mariano (proposed seed *i* vs baseline seed *i*),
  reporting median p, min-max range, and an all-seeds-significant flag.
  Rationale: averaging predictions over seeds would test a 3-member ensemble
  nobody deploys and asymmetrically flatter learned models vs the single-run
  statistical baselines; per-seed pairing keeps DM's paired time-series
  structure and surfaces fragile results.
- SAITS: minimal in-repo implementation (`src/models/saits.py`), not pypots
  (pypots 1.5 pulls ~10 transitive deps incl. transformers/tsdb/benchpots into
  a pinned-requirements repo and runs its own training loop outside this
  repo's determinism/train-only-fit contracts).
- Phase 3 (Beijing) training runs on Google Colab T4 per user decision
  (~19-20 h CPU estimate exceeded the 8 h/phase limit; ~2-4 h on T4).
  GPU-trained results are not bit-identical to CPU runs at the same seed;
  noted wherever Beijing numbers are reported.

## 7→9 upgrade (crossover study + manuscript)

| date | item | result / finding |
|---|---|---|
| 2026-06-13 | Fine-grained robustness sweep (8 levels × 2 mechanisms, both datasets) + effective-missingness logging (`robustness_levels.json`) | Dhaka effective missingness spans 32.7%→79.8%, Beijing 2.9%→71.0% — one continuous severity axis ~3–80%. |
| 2026-06-13 | Crossover study (`crossover.*`, `decision_summary.*`, `crossover_combined.*`, `stratified_gap.*`) | **Key finding — the crossover is NOT a single universal threshold; it is two-factor (severity × series imputability).** Dhaka (severe, less-structured): end-to-end overtakes best deep-imputer pipeline above ~38% eff. missingness at h6 (~71% at h24); window-stratified, end-to-end trails SAITS by 2.1 µg/m³ on most-complete windows but **leads by 2.3 on most-incomplete (54–100%)**. Beijing (near-complete, highly periodic): deep imputer wins under outage at ALL severities, margin GROWS (−0.9→−4.3 µg/m³) — opposite slope. Under MCAR the imputer wins on both. **This is honest and more careful than the planned "universal crossover"; the manuscript and RESULTS are framed accordingly (no overclaim).** |
| 2026-06-13 | Manuscript `paper/main.tex` (elsarticle) + `references.bib` (23 refs) + `paper/README.md` | full draft, structurally linted (balanced envs/braces, all citations defined); built around the two-factor crossover. No local TeX toolchain — user builds on Overleaf. |
| 2026-06-13 | Beijing interpretability (`scripts/06 --config config_beijing.yaml`) | attention + importance figures (PM2.5 dominant, PRES second). |

## 8→9 upgrade (third dataset + measured imputability axis)

Goal: turn the two-factor crossover (two opposite data points: Dhaka crosses
over, Beijing does not) into a **predictive curve** by adding a third network of
intermediate completeness (Delhi, CPCB/Mendeley `bzhzr9b64v`) and a **measured
imputability axis** so the decision rule is quantitative, not post-hoc.

| date | item | result / finding |
|---|---|---|
| 2026-06-13 | Imputability metric (`src/evaluate.py: imputability_score`, `_impute_skill`) | New, measured x-axis: hide a seeded 20% of *observed* test cells, reconstruct with the trained SAITS imputer vs forward-fill, report `imputability = 1 − RMSE_SAITS/RMSE_ffill` (standardized units). **Validated on the two existing networks and it orders them correctly:** Beijing **+0.20** (deep imputer beats ffill → reconstructable), Dhaka **−0.39** (deep imputer *worse* than ffill → hard to impute). |
| 2026-06-13 | Imputability-crossover figure (`imputability_crossover_figure`, `decision_by_imputability.*`, `imputability_crossover.*`) | **The end-to-end advantage declines monotonically with imputability** (real 2-point result, h6/+50% outage): Dhaka (imputability −0.39) advantage **+1.69 µg/m³ → end-to-end wins**; Beijing (imputability +0.20) advantage **−4.29 → impute-then-forecast wins**. Delhi will supply the middle point to complete the curve. |
| 2026-06-13 | n-way refactor (`cross_dataset_table`, `combined_crossover` now take `list[dict]`; `07 --tertiary-config`) | cross-dataset summary, combined crossover, and the imputability figure now span ≥3 datasets; single/two-dataset behaviour unchanged. |
| 2026-06-13 | Delhi integration (`src/data/load_delhi.py`, `config_delhi.yaml`, `scripts/01c_prepare_delhi.py`, `scripts/colab_run_delhi.py`, `COLAB.md`) | tolerant, config-driven loader (header canonicalization + alias map + numeric wd→sin/cos). **Schema VERIFIED against the published files** and the loader/config corrected: 6 sites (AshokVihar/DCStadium/DwarkaSec8/Najafgarh/NehruNagar/Okhla), integer y/m/d/h timestamp (not "From Date"), `AT`/`Ozone`/`NOx`→`Temp`/`O3`/`NOX`. **Auto-download** of the 6 station files from content-addressed Mendeley URLs → one-command Colab run. **Prep verified locally:** 70,227 rows, gap-free hourly 2018-06-01..2019-10-01, **~0% natural missingness** — Delhi is the *complete-but-noisy* anchor (PM2.5 mean≈93/std≈98), so the x-axis is genuinely imputability, not completeness. Full grid trains on Colab GPU like Beijing (artifacts pending). |
| 2026-06-13 | `example_forecast_figure` made dataset-agnostic | skips unknown configured stations and falls back to the first stations over a test-period window (was hard-coded to Dhaka station names; needed for a new dataset whose station names are unknown at config time). |
| 2026-06-13 | Tests | +18 (9 Delhi loader, 6 imputability incl. torch-free skill core, 3 n-way crossover/imputability plumbing). **104/104 pass.** |

Honest status: the code, metric and 2-point curve are verified on real data and
already strengthen the result; the **9** depends on Delhi landing between the two
anchors on a monotone imputability curve. If Delhi's imputability does not
interpolate, that is reported (imputability would then be necessary but not
sufficient) — a stronger three-point result either way.

## Run log

| date | phase | command | wall-clock | artifacts | notes / contradictions |
|---|---|---|---|---|---|
| 2026-06-12 | 0 | pytest (after device/season/seed-helper plumbing) | 3 s | — | 47/47 pass, no behavior change on CPU |
| 2026-06-12 | 1 | `05_ablations.py --export-seed-predictions` | 1.3 min | `predictions/seeds/{proposed,variant_B,proposed_md}_s{42,43,44}_test.npz`, `variant_B_test.npz`, `variant_B_stats.json` | inference-only from existing ablation checkpoints |
| 2026-06-12 | 1 | `03_train_baselines.py` (lstm/gru/knn/mice, seeds 43+44; seed-42 artifacts auto-skipped) | (background, ~2.2 h est) | per-seed checkpoints + `seeds/` bundles | running |
| 2026-06-13 | 3 | `01b_prepare_beijing.py` (UCI download 49 MB + clean) | ~1 min | `data/processed/beijing/all_stations.parquet` (420,768 rows, 12 stations), cleaning report | Beijing natural missingness is LOW: PM2.5 2.1%, PM10 1.6%, O3 4.2%, CO 5.3% (vs Dhaka PM2.5 23.4%). On natural Beijing data, missingness-aware vs two-stage differences are expected to be small; the synthetic-corruption robustness suite is where the mechanism is exercised. Beijing = external-validity check, not a second showcase. |
| 2026-06-13 | 3 | Beijing window counts via `make_datasets(config_beijing.yaml)` | <1 min | scalers.json | train 11,920 / val 1,068 / test 4,356 windows; val count healthy for early stopping |
| 2026-06-13 | 2+4 | pytest after new models (DLinear/PatchTST/GRU-D/SAITS + factory) + multiseed/episode/Beijing tests | 10 s | — | 82/82 pass (47 original + 35 new) |
| 2026-06-13 | 1 | `03_train_baselines.py` (background) finished: lstm 43/44 (3.5/2.9 min), gru 43/44 (5.8/5.4 min), knn 43/44 (41/31 min), mice 43/44 (28/21 min) | 2.2 h total | per-seed checkpoints, stats, `seeds/` bundles | — |
| 2026-06-13 | 1 | `07_make_paper_assets.py --skip-interpretability` (interim, Phase-2 models still training) | 2 min | multi-seed `main_results_pm25`, per-seed `significance_dm_bootstrap`, `metrics_full.csv` (long, per-seed), `episode_rmse_pm25` | **CONTRADICTION: the published h24 claim "two-stage KNN significantly better than proposed (p = 0.042, +1.54)" did NOT replicate across seeds.** Per-seed DM at h24: p = 0.038 / 0.042 / 0.891, mean RMSE diff +0.04 — a seed-42 artifact. With 3 seeds, KNN h24 = 76.42 ± 0.75 vs proposed 76.46 ± 0.51: statistical parity at every horizon vs every learned baseline. Also: variant B is significantly better than plain proposed at h72 (all seeds, p median 0.006); proposed significantly better than miss-dropout at h24 on clean data (2 of 3 seeds, diff −3.2) — the robustness/accuracy tradeoff is real and must be framed as such. |
| 2026-06-13 | 2 | `03_train_baselines.py` (dlinear/gru_d/patchtst/two_stage_saits × 3 seeds) | ~5 h | per-seed checkpoints + bundles; SAITS imputer checkpoints | dlinear ~15 s/run, gru_d ~18 min/run, patchtst ~45 min/run, saits-pipeline ~30 min/run. **SAITS quality gate PASSED all 3 seeds** (val MIT-MAE 0.170–0.173 vs ffill 0.192 → the deep imputer is a legitimate strong competitor, not a strawman). |
| 2026-06-13 | 2 | `05_ablations.py --robustness` (GRU-D/DLinear/PatchTST/SAITS added) | ~2 h | 48 robustness bundles (`{model}_test_{miss,out}{10,30,50}.npz`, seed 42) | re-imputation cost: SAITS ~1.1 s/level (transform-only), KNN ~140 s, MICE ~30 s. |
| 2026-06-13 | 4 | README.md + outputs/RESULTS.md rewritten (parity-plus-deployability framing); stale-number audit (grep vs CSVs, clean); pytest | 5 s | docs | 82/82 pass; no single-seed number remains where multi-seed exists; cross-dataset table deferred until Beijing artifacts present (Colab). |
| 2026-06-13 | 3 | Beijing artifacts trained on Colab **A100** (not T4), `beijing_artifacts.zip` extracted; `07 --config config_beijing.yaml` + `07 --secondary-config` | (Colab GPU run + 2 min local regen) | `outputs/beijing/{tables,figures,predictions,checkpoints}`, `cross_dataset_summary`, `main_results_beijing`, `robustness_beijing` | 11 learned+stat models × 3 seeds on Beijing. **Accuracy parity holds on Beijing too** — proposed/variant_B are marginally BEST at h6 (49.4 vs KNN 49.7, SAITS 50.3), parity at h24/h72. **CONTRADICTION/NUANCE: the Dhaka outage-robustness advantage does NOT replicate on Beijing.** Beijing natural PM2.5 missingness is only 2.1%; under synthetic station-outage at h6, SAITS is the MOST robust (slope +13.3) and the proposed family is middle-of-pack (miss-dropout +16.8, proposed +18.5). On Dhaka (23% natural missingness) the order was reversed (miss-dropout +2.0 best, SAITS +3.9). **Honest cross-dataset conclusion: the end-to-end outage-robustness win is specific to severely-incomplete networks; on a near-complete network a deep imputer reconstructs even synthetic block gaps well.** Parity + no-imputation-stage deployability are the claims that generalize across both datasets. |
| 2026-06-13 | 2/4 | `07_make_paper_assets.py --skip-interpretability` (full, all models) | 2 min | all Dhaka tables/figures refreshed incl. `robustness_rmse`, `episode_rmse_pm25` | **DECISIVE robustness finding (seed 42, h6):** (1) under cell-wise **MCAR**, two-stage **SAITS is the most robust AND most accurate** (66.1→66.1, slope +0.04) — beats miss-dropout (slope +2.06) and plain proposed (+5.90). MCAR leaves the same-timestep cross-section intact, the easy case for an attention imputer. **Honest loss under MCAR, reported.** (2) Under realistic **station-outage** corruption (the dominant real mechanism: co-missingness 0.38, 39% of gaps >7 days), **miss-dropout has the flattest slope (+2.04) and the lowest RMSE at +50% (68.28)**, beating SAITS (+3.90, 69.97) and KNN (+3.47, 70.48). The claim survives where it matters. (3) **GRU-D does NOT beat the proposed transformer** (68.89 vs 67.03 h6 clean; worst h6 outage slope +5.01 among neural models). (4) **PatchTST is the worst learned model** and degrades catastrophically (+12.5 MCAR, +8.9 outage) — channel-independent patching is ill-suited to heavy multivariate missingness. (5) Episodes (>150 µg/m³): proposed+miss-dropout best at h6 (130.2) and h24 (125.3). |
