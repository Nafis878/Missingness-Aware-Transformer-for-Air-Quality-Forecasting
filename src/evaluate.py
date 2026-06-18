"""Phase 5 evaluation: metrics, significance tests, tables, figures.

Pure analysis over the saved prediction bundles
(``outputs/predictions/<model>_test.npz``) — no training, no model loading.
Every table is exported as CSV + booktabs LaTeX, every figure as PNG + PDF.

CLI::

    python -m src.evaluate --config config.yaml

Contents:

* RMSE / MAE / R^2 / sMAPE per model x pollutant x horizon (pooled) and per
  station for PM2.5, unscaled, observed targets only.
* Diebold-Mariano tests (proposed vs each baseline, PM2.5, per horizon) on
  anchor-time-sorted squared-error differentials with Newey-West variance
  (lag = forecast horizon in window-steps, i.e. ceil(h/24)).
* Paired bootstrap CIs (1,000 seeded resamples over windows) on RMSE
  differences.
* Seasonal (Bangladesh seasons) PM2.5 RMSE breakdown + figure.
* CPU efficiency table from the ``*_stats.json`` files.
* Example forecast trajectories with observation gaps shaded.
* Robustness curve if ``<model>_test_miss<level>.npz`` bundles exist.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting_style import apply_style, save_figure
from src.utils import (
    export_table,
    load_config,
    season_map,
    seed_everything,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Label order = display row order in the main results table.
MODEL_LABELS = {
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal-naive",
    "sarima": "SARIMA",
    "lstm": "LSTM",
    "gru": "GRU",
    "gru_d": "GRU-D",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "two_stage_knn": "Two-stage (KNN)",
    "two_stage_mice": "Two-stage (MICE)",
    "two_stage_saits": "Two-stage (SAITS)",
    "proposed": "Proposed (MAT)",
    "variant_B": "Proposed (variant B)",
    "proposed_md": "Proposed + miss-dropout",
    "hybrid8_masked_transformer": "Hybrid8 + mask (Transformer)",
    "hybrid8_masked_proposed": "Hybrid8 + mask (MAT)",
    "hybrid8_masked_variant_B": "Hybrid8 + mask (variant B)",
    "hybrid8_masked_proposed_md": "Hybrid8 + mask + miss-dropout",
}

#: Deterministic single-run baselines (no seed sensitivity by construction).
STATISTICAL_MODELS = ["persistence", "seasonal_naive", "sarima"]
#: Models trained with multiple seeds; reported as mean +/- std.
LEARNED_MODELS = [
    "lstm", "gru", "gru_d", "dlinear", "patchtst",
    "two_stage_knn", "two_stage_mice", "two_stage_saits",
    "proposed", "variant_B", "proposed_md",
    "hybrid8_masked_transformer", "hybrid8_masked_proposed",
    "hybrid8_masked_variant_B", "hybrid8_masked_proposed_md",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_bundles(cfg: dict[str, Any], suffix: str = "test") -> dict[str, dict]:
    """Load every ``<model>_<suffix>.npz`` prediction bundle."""
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    bundles = {}
    for path in sorted(pred_dir.glob(f"*_{suffix}.npz")):
        name = path.stem[: -len(suffix) - 1]
        bundles[name] = dict(np.load(path))
    logger.info("loaded %d bundles (%s): %s", len(bundles), suffix, sorted(bundles))
    return bundles


def iter_seed_bundles(
    cfg: dict[str, Any], model: str, suffix: str = "test"
) -> dict[int, dict]:
    """Per-seed prediction bundles for ``model``, keyed by seed.

    Looks for ``predictions/seeds/{model}_s{seed}_{suffix}.npz`` for every seed
    in ``ablation.seeds``; for the canonical seed (``cfg["seed"]``) falls back
    to the top-level ``{model}_{suffix}.npz``. Missing seeds are logged and
    skipped so partially trained models still appear (with honest seed counts).
    """
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    seeds = cfg.get("ablation", {}).get("seeds", [cfg["seed"]])
    out: dict[int, dict] = {}
    for seed in seeds:
        path = pred_dir / "seeds" / f"{model}_s{seed}_{suffix}.npz"
        if not path.exists() and seed == cfg["seed"]:
            path = pred_dir / f"{model}_{suffix}.npz"
        if path.exists():
            out[seed] = dict(np.load(path))
        else:
            logger.warning("no %s bundle for %s seed %d", suffix, model, seed)
    return out


def unscale(arr: np.ndarray, pollutant: str, scalers: dict) -> np.ndarray:
    mean, std = scalers[pollutant]
    return arr * std + mean


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """RMSE/MAE/R2/sMAPE over finite prediction-target pairs (unscaled)."""
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {k: np.nan for k in ("RMSE", "MAE", "R2", "sMAPE", "n")}
    err = p - y
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    smape = float(np.mean(2 * np.abs(err) / np.clip(np.abs(p) + np.abs(y), 1e-6, None)) * 100)
    return {
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE": float(np.abs(err).mean()),
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "sMAPE": smape,
        "n": int(len(p)),
    }


def metrics_table(
    bundles: dict[str, dict], cfg: dict, scalers: dict
) -> pd.DataFrame:
    """Long-format metrics: model x pollutant x horizon (pooled stations)."""
    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    rows = []
    for name, b in bundles.items():
        for ti, pol in enumerate(targets):
            for hi, h in enumerate(horizons):
                m = b["target_mask"][:, ti, hi] > 0
                p = unscale(b["predictions"][m, ti, hi], pol, scalers)
                y = unscale(b["targets"][m, ti, hi], pol, scalers)
                rows.append({"model": name, "pollutant": pol, "horizon": h,
                             **_metrics(p, y)})
    return pd.DataFrame(rows)


def pm25_station_table(
    bundles: dict[str, dict], cfg: dict, scalers: dict, stations: list[str]
) -> pd.DataFrame:
    """PM2.5 RMSE at h24 per station x model."""
    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    ti = targets.index(cfg["dataset"]["primary_target"])
    hi = horizons.index(24)
    rows = {}
    for name, b in bundles.items():
        col = {}
        for sid, st in enumerate(stations):
            m = (b["target_mask"][:, ti, hi] > 0) & (b["station_id"] == sid)
            p = unscale(b["predictions"][m, ti, hi], "PM2.5", scalers)
            y = unscale(b["targets"][m, ti, hi], "PM2.5", scalers)
            col[st] = _metrics(p, y)["RMSE"]
        rows[MODEL_LABELS.get(name, name)] = col
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def diebold_mariano(
    e1_sq: np.ndarray, e2_sq: np.ndarray, order: np.ndarray, nw_lag: int
) -> tuple[float, float]:
    """DM statistic and two-sided p-value for squared-error differentials.

    ``d_t = e1_t^2 - e2_t^2`` sorted by ``order`` (anchor time); long-run
    variance estimated with a Newey-West (Bartlett) kernel of lag ``nw_lag``.
    Negative statistic => model 1 (proposed) has lower loss.
    """
    from scipy import stats

    d = (e1_sq - e2_sq)[np.argsort(order, kind="stable")]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float((dc @ dc) / n)
    lrv = gamma0
    for k in range(1, nw_lag + 1):
        gk = float((dc[k:] @ dc[:-k]) / n)
        lrv += 2 * (1 - k / (nw_lag + 1)) * gk
    if lrv <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(lrv / n)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))  # HLN small-sample t-dist
    return float(dm), float(p)


def paired_bootstrap_rmse_diff(
    e1_sq: np.ndarray, e2_sq: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Bootstrap CI (2.5/97.5%) for RMSE(model1) - RMSE(model2)."""
    rng = np.random.default_rng(seed)
    n = len(e1_sq)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = np.sqrt(e1_sq[idx].mean()) - np.sqrt(e2_sq[idx].mean())
    point = float(np.sqrt(e1_sq.mean()) - np.sqrt(e2_sq.mean()))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def significance_table(
    bundles: dict[str, dict], cfg: dict, scalers: dict, reference: str = "proposed"
) -> pd.DataFrame:
    """DM + bootstrap for the reference model vs every baseline (PM2.5)."""
    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    ti = targets.index(cfg["dataset"]["primary_target"])
    ref = bundles[reference]
    rows = []
    for name, b in bundles.items():
        if name == reference:
            continue
        for hi, h in enumerate(horizons):
            m = (ref["target_mask"][:, ti, hi] > 0) & (b["target_mask"][:, ti, hi] > 0)
            m &= np.isfinite(b["predictions"][:, ti, hi])
            y = unscale(ref["targets"][m, ti, hi], "PM2.5", scalers)
            p_ref = unscale(ref["predictions"][m, ti, hi], "PM2.5", scalers)
            p_b = unscale(b["predictions"][m, ti, hi], "PM2.5", scalers)
            e_ref, e_b = (p_ref - y) ** 2, (p_b - y) ** 2
            order = ref["anchor_time"][m]
            dm, pval = diebold_mariano(e_ref, e_b, order, nw_lag=max(1, -(-h // 24)))
            diff, lo, hi_ci = paired_bootstrap_rmse_diff(e_ref, e_b, 1000, cfg["seed"])
            rows.append({
                "baseline": MODEL_LABELS.get(name, name), "horizon": h, "n": int(m.sum()),
                "DM_stat": dm, "DM_p": pval,
                "RMSE_diff": diff, "CI_lo": lo, "CI_hi": hi_ci,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def seed_metrics_long(cfg: dict, scalers: dict) -> pd.DataFrame:
    """Per-seed metrics, long format: model x seed x pollutant x horizon.

    Learned models get one row per (seed, pollutant, horizon); deterministic
    statistical baselines a single row with ``seed = NaN``. This is the
    feedstock for the headline table and is written out as
    ``metrics_full.csv``.
    """
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    frames = []
    for model in STATISTICAL_MODELS:
        path = pred_dir / f"{model}_test.npz"
        if not path.exists():
            continue
        df = metrics_table({model: dict(np.load(path))}, cfg, scalers)
        df.insert(1, "seed", np.nan)
        frames.append(df)
    for model in LEARNED_MODELS:
        for seed, b in iter_seed_bundles(cfg, model).items():
            df = metrics_table({model: b}, cfg, scalers)
            df.insert(1, "seed", seed)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_main_results(long_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Headline primary-target table from the per-seed long frame.

    RMSE cells are "mean ± std" strings over training seeds for learned
    models (population std, matching the ablation table); statistical
    baselines show their single deterministic value. MAE and R^2 columns are
    seed means. Never reports a single-seed number where multi-seed exists.
    """
    pol = cfg["dataset"]["primary_target"]
    horizons = cfg["dataset"]["horizons"]
    pm = long_df[long_df["pollutant"] == pol]
    rows: dict[str, dict] = {}
    for model in [m for m in MODEL_LABELS if m in set(pm["model"])]:
        grp = pm[pm["model"] == model]
        row: dict = {}
        for h in horizons:
            hg = grp[grp["horizon"] == h]
            if hg.empty:
                continue
            r = hg["RMSE"].to_numpy()
            if model in STATISTICAL_MODELS or len(r) == 1:
                row[("RMSE", f"h{h}")] = f"{r[0]:.2f}"
            else:
                row[("RMSE", f"h{h}")] = f"{r.mean():.2f} ± {r.std():.2f}"
            row[("MAE", f"h{h}")] = f"{hg['MAE'].mean():.2f}"
            row[("R2", f"h{h}")] = f"{hg['R2'].mean():.3f}"
        row[("", "seeds")] = (1 if model in STATISTICAL_MODELS
                              else int(grp["seed"].nunique()))
        rows[MODEL_LABELS.get(model, model)] = row
    tbl = pd.DataFrame(rows).T
    tbl.columns = pd.MultiIndex.from_tuples(tbl.columns)
    return tbl


def significance_table_multiseed(
    cfg: dict, scalers: dict, reference: str = "proposed"
) -> pd.DataFrame:
    """Per-seed Diebold-Mariano + bootstrap vs every baseline (primary target).

    Design (documented in RESULTS.md): the reference model's seed *i* run is
    paired with the baseline's seed *i* run (statistical baselines reuse their
    single deterministic bundle for every pairing). Each pairing yields one DM
    test on the anchor-time-sorted squared-error differential; the table
    reports the median and range of the per-seed p-values plus an
    all-seeds-significant flag. Averaging predictions over seeds first would
    test a 3-member ensemble nobody deploys and flatter learned models
    against the single-run statistical baselines.
    """
    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    pol = cfg["dataset"]["primary_target"]
    ti = targets.index(pol)
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    ref_bundles = iter_seed_bundles(cfg, reference)
    if not ref_bundles:
        logger.warning("significance: no %s bundles, skipping", reference)
        return pd.DataFrame()
    rows = []
    for name in STATISTICAL_MODELS + LEARNED_MODELS:
        if name == reference:
            continue
        if name in STATISTICAL_MODELS:
            path = pred_dir / f"{name}_test.npz"
            if not path.exists():
                continue
            single = dict(np.load(path))
            base = {seed: single for seed in ref_bundles}
        else:
            base = iter_seed_bundles(cfg, name)
        for hi, h in enumerate(horizons):
            per_seed = []
            for seed, ref in ref_bundles.items():
                b = base.get(seed)
                if b is None:
                    continue
                m = (ref["target_mask"][:, ti, hi] > 0) & (b["target_mask"][:, ti, hi] > 0)
                m &= np.isfinite(b["predictions"][:, ti, hi])
                if m.sum() < 10:
                    continue
                y = unscale(ref["targets"][m, ti, hi], pol, scalers)
                e_ref = (unscale(ref["predictions"][m, ti, hi], pol, scalers) - y) ** 2
                e_b = (unscale(b["predictions"][m, ti, hi], pol, scalers) - y) ** 2
                dm, pval = diebold_mariano(e_ref, e_b, ref["anchor_time"][m],
                                           nw_lag=max(1, -(-h // 24)))
                diff, lo, hi_ci = paired_bootstrap_rmse_diff(e_ref, e_b, 1000, seed)
                per_seed.append({"p": pval, "diff": diff, "lo": lo, "hi": hi_ci,
                                 "n": int(m.sum())})
            if not per_seed:
                continue
            ps = np.array([r["p"] for r in per_seed])
            rows.append({
                "baseline": MODEL_LABELS.get(name, name), "horizon": h,
                "seeds": len(per_seed), "n": per_seed[0]["n"],
                "DM_p_median": float(np.median(ps)),
                "DM_p_min": float(ps.min()),
                "DM_p_max": float(ps.max()),
                "sig_all_seeds": bool((ps < 0.05).all()),
                "RMSE_diff_mean": float(np.mean([r["diff"] for r in per_seed])),
                "CI_lo_min": float(min(r["lo"] for r in per_seed)),
                "CI_hi_max": float(max(r["hi"] for r in per_seed)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# High-pollution-episode metric
# ---------------------------------------------------------------------------

def episode_table(cfg: dict, scalers: dict) -> pd.DataFrame:
    """RMSE restricted to high-pollution hours (observed PM2.5 > threshold).

    Operationally these are the hours that matter most. The subset conditions
    on the OBSERVED target, so it is identical for every model — no
    model-dependent selection. Learned models report mean +/- std over seeds;
    statistical baselines a single value. ``n`` is the episode target count
    at each horizon.
    """
    thr = float(cfg.get("evaluation", {}).get("episode_threshold", 150))
    pol = cfg["dataset"]["primary_target"]
    ti = cfg["dataset"]["target_pollutants"].index(pol)
    horizons = cfg["dataset"]["horizons"]
    pred_dir = Path(cfg["paths"]["predictions_dir"])

    def episode_rmse(b: dict) -> dict[int, tuple[float, int]]:
        out = {}
        for hi, h in enumerate(horizons):
            y_all = unscale(b["targets"][:, ti, hi], pol, scalers)
            m = (b["target_mask"][:, ti, hi] > 0) & (y_all > thr)
            m &= np.isfinite(b["predictions"][:, ti, hi])
            p = unscale(b["predictions"][m, ti, hi], pol, scalers)
            out[h] = (_metrics(p, y_all[m])["RMSE"], int(m.sum()))
        return out

    rows: dict[str, dict] = {}
    n_row: dict = {}
    for model in STATISTICAL_MODELS:
        path = pred_dir / f"{model}_test.npz"
        if not path.exists():
            continue
        per_h = episode_rmse(dict(np.load(path)))
        rows[MODEL_LABELS[model]] = {f"h{h}": f"{r:.2f}"
                                     for h, (r, _) in per_h.items()}
        n_row = {f"h{h}": n for h, (_, n) in per_h.items()}
    for model in LEARNED_MODELS:
        per_seed = [episode_rmse(b)
                    for b in iter_seed_bundles(cfg, model).values()]
        if not per_seed:
            continue
        cells = {}
        for h in horizons:
            r = np.array([s[h][0] for s in per_seed])
            cells[f"h{h}"] = (f"{r[0]:.2f}" if len(r) == 1
                              else f"{r.mean():.2f} ± {r.std():.2f}")
        rows[MODEL_LABELS[model]] = cells
        n_row = {f"h{h}": per_seed[0][h][1] for h in horizons}
    if not rows:
        return pd.DataFrame()
    tbl = pd.DataFrame(rows).T
    tbl = tbl.reindex([lbl for lbl in MODEL_LABELS.values() if lbl in tbl.index])
    tbl.loc["n (episode targets)"] = {k: str(v) for k, v in n_row.items()}
    return tbl


def episode_figure(tbl: pd.DataFrame, cfg: dict, figures_dir: Path) -> None:
    """Grouped bars: episode RMSE per model x horizon (means over seeds)."""
    thr = float(cfg.get("evaluation", {}).get("episode_threshold", 150))
    data = tbl.drop(index=["n (episode targets)"], errors="ignore")
    means = data.map(lambda s: float(str(s).split(" ±")[0]))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    means.plot.bar(ax=ax, width=0.85)
    ax.set_ylabel(f"PM2.5 RMSE (µg/m³), observed > {thr:.0f} µg/m³")
    ax.set_xlabel("")
    ax.legend(title="horizon", fontsize=8)
    ax.set_title("High-pollution-episode forecast error (test period)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures_dir, "episode_rmse")


# ---------------------------------------------------------------------------
# Cross-dataset summary
# ---------------------------------------------------------------------------

def _load_scalers_for(cfg: dict) -> dict:
    return json.loads(
        (Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text()
    )


def _outage_slope_h6(cfg: dict, scalers: dict, model: str) -> float | None:
    """RMSE(+50% outage) - RMSE(clean) at h6, canonical-seed bundles."""
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    clean_p = pred_dir / f"{model}_test.npz"
    corr_p = pred_dir / f"{model}_test_out50.npz"
    if not (clean_p.exists() and corr_p.exists()):
        return None
    pol = cfg["dataset"]["primary_target"]
    ti = cfg["dataset"]["target_pollutants"].index(pol)
    hi = cfg["dataset"]["horizons"].index(6)
    vals = []
    for p in (clean_p, corr_p):
        b = dict(np.load(p))
        m = b["target_mask"][:, ti, hi] > 0
        pr = unscale(b["predictions"][m, ti, hi], pol, scalers)
        y = unscale(b["targets"][m, ti, hi], pol, scalers)
        vals.append(_metrics(pr, y)["RMSE"])
    return vals[1] - vals[0]


def cross_dataset_table(configs: list[dict]) -> tuple[pd.DataFrame, str]:
    """Models present on every dataset: h24 RMSE + h6 outage-degradation.

    ``configs`` is the ordered list of dataset configs (two or more). Returns
    (table, dataset-stats sentence for the caption). RMSE cells are mean +/-
    std over seeds for learned models (single value for statistical baselines);
    the robustness column is the canonical-seed RMSE increase from clean to
    +50% station-outage corruption at h6.
    """
    blocks = {}
    stats_bits = []
    for cfg in configs:
        ds_name = str(cfg.get("dataset_name", "dhaka")).capitalize()
        scalers = _load_scalers_for(cfg)
        long_df = seed_metrics_long(cfg, scalers)
        pol = cfg["dataset"]["primary_target"]
        col_rmse, col_slope = {}, {}
        for model in MODEL_LABELS:
            grp = long_df[(long_df["model"] == model)
                          & (long_df["pollutant"] == pol)
                          & (long_df["horizon"] == 24)] if not long_df.empty \
                else pd.DataFrame()
            if grp.empty:
                continue
            r = grp["RMSE"].to_numpy()
            label = MODEL_LABELS[model]
            col_rmse[label] = (f"{r[0]:.2f}" if model in STATISTICAL_MODELS
                               or len(r) == 1
                               else f"{r.mean():.2f} ± {r.std():.2f}")
            slope = _outage_slope_h6(cfg, scalers, model)
            col_slope[label] = "—" if slope is None else f"+{slope:.2f}"
        blocks[(ds_name, "RMSE h24")] = col_rmse
        blocks[(ds_name, "ΔRMSE h6 @+50% outage")] = col_slope

        df = pd.read_parquet(
            Path(cfg["paths"]["processed_dir"]) / "all_stations.parquet"
        )
        stats_bits.append(
            f"{ds_name}: {df['station'].nunique()} stations, {len(df):,} "
            f"station-hours, natural PM2.5 missingness "
            f"{df['PM2.5'].isna().mean() * 100:.1f}\\%"
        )
    tbl = pd.DataFrame(blocks)
    tbl.columns = pd.MultiIndex.from_tuples(tbl.columns)
    order = [lbl for lbl in MODEL_LABELS.values() if lbl in tbl.index]
    return tbl.reindex(order).dropna(how="all"), "; ".join(stats_bits)


# ---------------------------------------------------------------------------
# Seasonal breakdown
# ---------------------------------------------------------------------------

def seasonal_table(
    bundles: dict[str, dict], cfg: dict, scalers: dict
) -> pd.DataFrame:
    """PM2.5 RMSE at h24 per season x model."""
    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    ti = targets.index(cfg["dataset"]["primary_target"])
    hi = horizons.index(24)
    month_season, season_order = season_map(cfg)
    rows = {}
    for name, b in bundles.items():
        months = pd.to_datetime(b["anchor_time"], unit="s").month
        seasons = pd.Series(months).map(month_season).to_numpy()
        col = {}
        for season in season_order:
            m = (b["target_mask"][:, ti, hi] > 0) & (seasons == season)
            m &= np.isfinite(b["predictions"][:, ti, hi])
            p = unscale(b["predictions"][m, ti, hi], "PM2.5", scalers)
            y = unscale(b["targets"][m, ti, hi], "PM2.5", scalers)
            col[season] = _metrics(p, y)["RMSE"]
        rows[MODEL_LABELS.get(name, name)] = col
    return pd.DataFrame(rows).reindex(season_order)


def seasonal_figure(tbl: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    tbl.plot.bar(ax=ax, width=0.85)
    ax.set_ylabel("PM2.5 RMSE at 24 h (µg/m³)")
    ax.set_xlabel("")
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Seasonal forecast performance (test period)")
    plt.setp(ax.get_xticklabels(), rotation=0)
    save_figure(fig, figures_dir, "seasonal_performance")


# ---------------------------------------------------------------------------
# Efficiency table
# ---------------------------------------------------------------------------

def efficiency_table(cfg: dict) -> pd.DataFrame:
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    rows = {}
    for path in sorted(ckpt_dir.glob("*_stats.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        name = s["name"]
        if name not in MODEL_LABELS:
            continue
        rows[MODEL_LABELS[name]] = {
            "Parameters": s.get("n_parameters", 0),
            "Train time (min)": round(s.get("train_time_s", 0.0) / 60, 1),
            "Impute time (min)": round(s.get("impute_time_s", 0.0) / 60, 1),
            "Latency (ms/window)": round(float(s.get("latency_ms_per_window", np.nan)), 2),
            "Epochs": s.get("epochs_run", "-"),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Example forecasts
# ---------------------------------------------------------------------------

def example_forecast_figure(
    bundles: dict[str, dict], cfg: dict, scalers: dict,
    df: pd.DataFrame, stations: list[str], figures_dir: Path,
    picks: list[tuple[str, str, str]] | None = None,
) -> None:
    """Actual PM2.5 with gaps shaded + h24 predictions across anchors.

    ``picks``: list of (station, start, end) periods; defaults to
    ``cfg["evaluation"]["example_picks"]``, falling back to two Dhaka periods
    chosen to show one high-pollution winter period and one gappy monsoon
    period.
    """
    picks = picks or [
        tuple(p) for p in cfg.get("evaluation", {}).get("example_picks", [])
    ]
    # keep only configured stations that actually exist (configs can drift /
    # a new dataset's station names may be unknown at config-writing time)
    picks = [p for p in picks if p[0] in stations]
    if not picks and stations:
        # dataset-agnostic fallback: first stations over a 30-day test window
        v_end = pd.Timestamp(cfg["splits"]["val_end"])
        start = (v_end + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
        end = (v_end + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        picks = [(s, start, end) for s in stations[:2]]
    if not picks:
        logger.warning("example forecasts: no usable stations, skipping")
        return
    show = [m for m in ("proposed", "two_stage_knn", "gru") if m in bundles]
    targets = cfg["dataset"]["target_pollutants"]
    ti = targets.index("PM2.5")
    hi = cfg["dataset"]["horizons"].index(24)

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3.2 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, (station, start, end) in zip(axes, picks):
        sid = stations.index(station)
        sub = df[(df["station"] == station) & (df["datetime"] >= start)
                 & (df["datetime"] <= end)].set_index("datetime")
        ax.plot(sub.index, sub["PM2.5"], color="black", linewidth=1.0,
                label="Observed PM2.5")
        # shade observation gaps
        isna = sub["PM2.5"].isna().to_numpy()
        if isna.any():
            edges = np.flatnonzero(np.diff(np.concatenate([[0], isna.astype(int), [0]])))
            for lo_i, hi_i in edges.reshape(-1, 2):
                ax.axvspan(sub.index[lo_i], sub.index[min(hi_i, len(sub) - 1)],
                           color="0.85", zorder=0)
        for name in show:
            b = bundles[name]
            t_target = pd.to_datetime(b["anchor_time"], unit="s") + pd.Timedelta(hours=24)
            m = ((b["station_id"] == sid) & (t_target >= pd.Timestamp(start))
                 & (t_target <= pd.Timestamp(end))
                 & np.isfinite(b["predictions"][:, ti, hi]))
            p = unscale(b["predictions"][m, ti, hi], "PM2.5", scalers)
            ax.plot(t_target[m], p, marker="o", markersize=2.5, linewidth=0.9,
                    label=f"{MODEL_LABELS.get(name, name)} (24 h ahead)")
        ax.set_title(f"{station}, {start} to {end} (grey = PM2.5 unobserved)",
                     fontsize=9)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, figures_dir, "example_forecasts")


# ---------------------------------------------------------------------------
# Robustness curve
# ---------------------------------------------------------------------------

ROBUST_MODES = {"miss": "Cell-wise MCAR", "out": "Station-outage blocks"}


def robustness_long(cfg: dict, scalers: dict) -> pd.DataFrame | None:
    """Long-format PM2.5 RMSE per (mode, model, level, horizon).

    Shared by the robustness figure and the crossover study. ``level`` is the
    nominal corruption percentage (0 = clean). Returns None if the proposed
    model's robustness bundles are incomplete.
    """
    levels = [0.0] + list(cfg["dataset"]["synthetic_missingness"])
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    pred_dir = Path(cfg["paths"]["predictions_dir"])

    def has_all_bundles(name: str) -> bool:
        suffixes = ["test"] + [f"test_{m}{int(lv * 100)}"
                               for m in ROBUST_MODES for lv in levels if lv > 0]
        return all((pred_dir / f"{name}_{s}.npz").exists() for s in suffixes)

    rcfg = cfg.get("robustness", {})
    candidates = list(rcfg.get("direct", ["proposed", "proposed_md"])) + \
        list(rcfg.get("two_stage", ["two_stage_knn", "two_stage_mice"]))
    models = [m for m in candidates if has_all_bundles(m)]
    if "proposed" not in models:
        logger.warning("robustness: proposed bundles incomplete, skipping")
        return None
    rows = []
    for mode in ROBUST_MODES:
        for name in models:
            for level in levels:
                suffix = "test" if level == 0 else f"test_{mode}{int(level * 100)}"
                b = dict(np.load(pred_dir / f"{name}_{suffix}.npz"))
                for hi, h in enumerate(cfg["dataset"]["horizons"]):
                    m = b["target_mask"][:, ti, hi] > 0
                    p = unscale(b["predictions"][m, ti, hi], "PM2.5", scalers)
                    y = unscale(b["targets"][m, ti, hi], "PM2.5", scalers)
                    rows.append({"mode": mode, "name": name,
                                 "model": MODEL_LABELS.get(name, name),
                                 "level": int(level * 100), "horizon": h,
                                 "RMSE": _metrics(p, y)["RMSE"]})
    return pd.DataFrame(rows)


def robustness_figure(cfg: dict, scalers: dict, figures_dir: Path,
                      tables_dir: Path) -> pd.DataFrame | None:
    """PM2.5 RMSE vs synthetic extra-missingness (the money figure)."""
    tbl = robustness_long(cfg, scalers)
    if tbl is None:
        return None
    export_table(
        tbl.pivot_table(index=["mode", "model", "horizon"], columns="level",
                        values="RMSE").round(2),
        tables_dir, "robustness_rmse",
        "PM2.5 RMSE (\\si{\\micro\\gram\\per\\cubic\\metre}) under additional "
        "synthetic input missingness: cell-wise MCAR vs station-outage blocks. "
        "Two-stage pipelines re-impute the corrupted series with imputers fit "
        "on (uncorrupted) train rows; for SAITS this means reusing the trained "
        "imputer, transform-only.",
        "tab:robustness", float_format="%.2f",
    )
    xticks = sorted(tbl["level"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)
    for row_i, (mode, mode_label) in enumerate(ROBUST_MODES.items()):
        for col_i, h in enumerate(cfg["dataset"]["horizons"]):
            ax = axes[row_i, col_i]
            sub = tbl[(tbl["horizon"] == h) & (tbl["mode"] == mode)]
            for model, grp in sub.groupby("model", sort=False):
                ax.plot(grp["level"], grp["RMSE"], marker="o", markersize=3,
                        label=model)
            if row_i == 0:
                ax.set_title(f"{h} h ahead")
            if row_i == 1:
                ax.set_xlabel("Additional missingness (%)")
            if col_i == 0:
                ax.set_ylabel(f"{mode_label}\nPM2.5 RMSE (µg/m³)")
            ax.set_xticks(xticks)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Robustness to additional input missingness (test period)")
    fig.tight_layout()
    save_figure(fig, figures_dir, "robustness_curve")
    return tbl


# ---------------------------------------------------------------------------
# Missingness-severity crossover study (impute-then-forecast vs end-to-end)
# ---------------------------------------------------------------------------

#: end-to-end (mask-native) model names and impute-then-forecast names.
END_TO_END = ["proposed", "proposed_md", "variant_B"]
TWO_STAGE = ["two_stage_knn", "two_stage_mice", "two_stage_saits"]


def _levels_map(cfg: dict) -> dict:
    """Effective test-input missingness per corruption key (from script 05)."""
    p = Path(cfg["paths"]["outputs_dir"]) / "robustness_levels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def crossover_long(cfg: dict, rob_tbl: pd.DataFrame) -> pd.DataFrame | None:
    """Per (mode, horizon, level): best two-stage − best end-to-end RMSE.

    The gap is positive when end-to-end forecasting wins. The x-axis is the
    *effective* mean test-input missingness (natural + injected), read from
    ``robustness_levels.json`` so curves are comparable across datasets.
    """
    lm = _levels_map(cfg)
    if rob_tbl is None or not lm:
        return None
    have = set(rob_tbl["name"])
    e2e = [m for m in END_TO_END if m in have]
    ts = [m for m in TWO_STAGE if m in have]
    if not e2e or not ts:
        return None

    def eff(mode: str, level: int) -> float | None:
        return lm.get("clean") if level == 0 else lm.get(f"{mode}{level}")

    rows = []
    for mode in rob_tbl["mode"].unique():
        for h in cfg["dataset"]["horizons"]:
            sub = rob_tbl[(rob_tbl["mode"] == mode) & (rob_tbl["horizon"] == h)]
            for level in sorted(sub["level"].unique()):
                em = eff(mode, int(level))
                if em is None:
                    continue
                s = sub[sub["level"] == level]
                best_e2e = s[s["name"].isin(e2e)]["RMSE"].min()
                best_ts = s[s["name"].isin(ts)]["RMSE"].min()
                if not (np.isfinite(best_e2e) and np.isfinite(best_ts)):
                    continue
                rows.append({
                    "mode": mode, "horizon": h, "level": int(level),
                    "eff_missing_pct": round(em * 100, 1),
                    "best_two_stage": round(best_ts, 2),
                    "best_end_to_end": round(best_e2e, 2),
                    "gap": round(best_ts - best_e2e, 2),  # >0 => end-to-end wins
                })
    return pd.DataFrame(rows)


def crossover_points(cross_df: pd.DataFrame) -> pd.DataFrame:
    """Effective missingness at which end-to-end overtakes impute-then-forecast.

    Linear interpolation of the first negative→positive zero crossing of the
    gap vs effective-missingness curve, per (mode, horizon). Reports a decision
    recommendation. ``<min`` = end-to-end already wins at the lowest tested
    severity; ``>max`` = impute-then-forecast wins throughout.
    """
    rows = []
    for (mode, h), g in cross_df.groupby(["mode", "horizon"]):
        g = g.sort_values("eff_missing_pct")
        x, y = g["eff_missing_pct"].to_numpy(), g["gap"].to_numpy()
        cp: float | str
        if (y > 0).all():
            cp = "<min"
        elif (y <= 0).all():
            cp = ">max"
        else:
            cp = ">max"
            for i in range(1, len(x)):
                if y[i - 1] < 0 <= y[i]:
                    cp = round(x[i - 1] + (-y[i - 1]) * (x[i] - x[i - 1])
                               / (y[i] - y[i - 1]), 1)
                    break
        rec = ("end-to-end at all tested severities" if cp == "<min"
               else "impute-then-forecast at all tested severities" if cp == ">max"
               else f"end-to-end above ~{cp}% input missingness")
        rows.append({"mode": mode, "horizon": h,
                     "crossover_missing_pct": cp, "recommendation": rec})
    return pd.DataFrame(rows)


def crossover_figure(cross_df: pd.DataFrame, cfg: dict, figures_dir: Path,
                     name: str = "crossover_curve") -> None:
    """Gap (best two-stage − best end-to-end) vs effective input missingness."""
    horizons = cfg["dataset"]["horizons"]
    fig, axes = plt.subplots(1, len(ROBUST_MODES),
                             figsize=(5.2 * len(ROBUST_MODES), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (mode, mlabel) in zip(axes, ROBUST_MODES.items()):
        for h in horizons:
            g = cross_df[(cross_df["mode"] == mode)
                         & (cross_df["horizon"] == h)].sort_values("eff_missing_pct")
            if g.empty:
                continue
            ax.plot(g["eff_missing_pct"], g["gap"], marker="o", markersize=3,
                    label=f"{h} h")
        ax.axhline(0, color="0.4", lw=1, ls="--")
        ax.set_title(mlabel)
        ax.set_xlabel("Effective input missingness (%)")
    axes[0].set_ylabel("RMSE gap (µg/m³)\n← impute-then-forecast  |  end-to-end →")
    axes[0].legend(title="horizon", fontsize=8)
    fig.suptitle("When does end-to-end beat impute-then-forecast? "
                 "(gap > 0 ⇒ end-to-end wins)")
    fig.tight_layout()
    save_figure(fig, figures_dir, name)


def run_crossover(cfg: dict, scalers: dict, rob_tbl: pd.DataFrame | None,
                  tables_dir: Path, figures_dir: Path) -> pd.DataFrame | None:
    """Build and export the crossover table, decision summary, and figure."""
    if rob_tbl is None:
        return None
    cross = crossover_long(cfg, rob_tbl)
    if cross is None or cross.empty:
        logger.warning("crossover: insufficient bundles or robustness_levels.json")
        return None
    export_table(
        cross.set_index(["mode", "horizon", "level"]),
        tables_dir, "crossover",
        "Missingness-severity crossover (PM2.5): best impute-then-forecast "
        "minus best end-to-end RMSE (\\si{\\micro\\gram\\per\\cubic\\metre}) "
        "vs effective test-input missingness. Positive gap = end-to-end wins.",
        "tab:crossover", "%.2f")
    pts = crossover_points(cross)
    export_table(
        pts.set_index(["mode", "horizon"]), tables_dir, "decision_summary",
        "Decision rule: effective input missingness at which end-to-end "
        "forecasting overtakes impute-then-forecast, per mechanism and horizon.",
        "tab:decision", "%s")
    crossover_figure(cross, cfg, figures_dir)
    return cross


def combined_crossover(configs: list[dict],
                       figures_dir: Path, tables_dir: Path) -> None:
    """Overlay every dataset's crossover curve on one effective-missingness axis.

    This is the unifying result: by plotting the end-to-end advantage against
    *effective input missingness* (not the nominal corruption level), networks
    of differing completeness fall on a single severity axis and the crossover
    is dataset-agnostic. ``configs`` is the ordered list of dataset configs.
    """
    frames = []
    for cfg in configs:
        rob = robustness_long(cfg, _load_scalers_for(cfg))
        cross = crossover_long(cfg, rob) if rob is not None else None
        if cross is None or cross.empty:
            continue
        cross = cross.copy()
        cross["dataset"] = str(cfg.get("dataset_name", "dhaka")).capitalize()
        frames.append(cross)
    if not frames:
        return
    allc = pd.concat(frames, ignore_index=True)
    allc.to_csv(tables_dir / "crossover_combined.csv", index=False)

    horizons = configs[0]["dataset"]["horizons"]
    h = 6 if 6 in horizons else horizons[0]
    fig, axes = plt.subplots(1, len(ROBUST_MODES),
                             figsize=(5.2 * len(ROBUST_MODES), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (mode, mlabel) in zip(axes, ROBUST_MODES.items()):
        for ds_name, g in allc[(allc["mode"] == mode)
                               & (allc["horizon"] == h)].groupby("dataset"):
            g = g.sort_values("eff_missing_pct")
            ax.plot(g["eff_missing_pct"], g["gap"], marker="o", markersize=4,
                    label=ds_name)
        ax.axhline(0, color="0.4", lw=1, ls="--")
        ax.set_title(mlabel)
        ax.set_xlabel("Effective input missingness (%)")
    axes[0].set_ylabel(f"RMSE gap at {h} h (µg/m³)\n"
                       "← impute-then-forecast  |  end-to-end →")
    axes[0].legend(title="network", fontsize=9)
    n_ds = allc["dataset"].nunique()
    fig.suptitle(f"Missingness-severity crossover across {n_ds} networks "
                 "(gap > 0 ⇒ end-to-end wins)")
    fig.tight_layout()
    save_figure(fig, figures_dir, "crossover_combined")
    logger.info("wrote combined crossover figure (%d datasets)", n_ds)


# ---------------------------------------------------------------------------
# Series imputability: how reconstructable is a network's missingness?
# (the measured x-axis that explains *why* a dataset crosses over or not)
# ---------------------------------------------------------------------------

def _impute_skill(
    station_slices: list[tuple[np.ndarray, np.ndarray]],
    model_impute,
    ffill_impute,
    holdout_rate: float,
    seed: int,
) -> dict[str, float]:
    """Reconstruction skill on artificially-hidden *observed* cells.

    For each ``(values, mask)`` slice (scaled units, missing cells already 0),
    a seeded fraction ``holdout_rate`` of observed cells is hidden, both
    imputers reconstruct the full array from the reduced input, and RMSE is
    measured at the hidden cells. ``imputability = 1 - RMSE_model/RMSE_ffill``
    (1 = perfect, 0 = no better than forward-fill, <0 = worse). Computed in
    standardized units so it is comparable across datasets and features.

    ``model_impute`` / ``ffill_impute`` are ``fn(x_zerofilled, m_in) -> recon``
    (both (L, V)); injectable so the bookkeeping is testable without torch.
    """
    rng = np.random.default_rng(seed)
    sq_m = sq_f = 0.0
    n = 0
    for vals, mask in station_slices:
        obs = mask > 0
        art = (rng.random(mask.shape) < holdout_rate) & obs
        if not art.any():
            continue
        m_in = (obs & ~art).astype(np.float32)
        x_in = (vals * m_in).astype(np.float32)
        rec_m = np.asarray(model_impute(x_in, m_in), dtype=np.float64)
        rec_f = np.asarray(ffill_impute(x_in, m_in), dtype=np.float64)
        tgt = vals[art].astype(np.float64)
        sq_m += float(((rec_m[art] - tgt) ** 2).sum())
        sq_f += float(((rec_f[art] - tgt) ** 2).sum())
        n += int(art.sum())
    if n == 0:
        return {"rmse_model": float("nan"), "rmse_ffill": float("nan"),
                "imputability": float("nan"), "n_cells": 0}
    rmse_m = float(np.sqrt(sq_m / n))
    rmse_f = float(np.sqrt(sq_f / n))
    return {
        "rmse_model": round(rmse_m, 4),
        "rmse_ffill": round(rmse_f, 4),
        "imputability": round(1.0 - rmse_m / rmse_f, 4) if rmse_f > 0 else float("nan"),
        "n_cells": n,
    }


def _saits_impute_array(model, x_in: np.ndarray, m_in: np.ndarray,
                        seg_len: int) -> np.ndarray:
    """Segment-wise SAITS reconstruction of one (L, V) station array.

    Mirrors :func:`src.models.saits.impute_full_series_saits` (non-overlapping
    windows + a right-aligned tail), but on an already-corrupted single array.
    Hidden cells (mask 0) are reconstructed; observed cells are preserved.
    """
    import torch

    n = len(x_in)
    length = min(seg_len, n)
    starts = list(range(0, n - length + 1, length)) or [0]
    if starts[-1] + length < n:
        starts.append(n - length)
    out = np.empty_like(x_in)
    filled = np.zeros(n, dtype=bool)
    x = torch.from_numpy(np.stack([x_in[s: s + length] for s in starts]))
    m = torch.from_numpy(np.stack([m_in[s: s + length] for s in starts]))
    seg = model.impute(x, m).cpu().numpy()
    for j, s in enumerate(starts):
        rows = ~filled[s: s + length]
        out[s: s + length][rows] = seg[j][rows]
        filled[s: s + length] = True
    return out


def imputability_score(cfg: dict, scalers: dict, *, model_impute=None,
                       ffill_impute=None, holdout_rate: float = 0.2,
                       seed: int | None = None) -> pd.DataFrame | None:
    """Dataset-level imputability from deep-imputer vs forward-fill skill.

    Builds the test-period station arrays, hides a seeded sample of observed
    cells, and compares SAITS reconstruction RMSE to forward-fill RMSE on those
    cells (standardized units). Returns a one-row frame
    ``[dataset, natural_PM25_missing_pct, imputability, rmse_model, rmse_ffill,
    n_cells]`` or ``None`` when the SAITS checkpoint is absent. The imputers are
    injectable so the metric is testable without a trained model (torch is only
    imported on the default SAITS path).
    """
    from src.data.dataset import build_station_arrays, split_ranges

    seed = cfg["seed"] if seed is None else seed
    df = pd.read_parquet(
        Path(cfg["paths"]["processed_dir"]) / "all_stations.parquet"
    )
    stations = build_station_arrays(df, cfg, scalers)
    test_lo = np.datetime64(split_ranges(cfg)["test"][0])
    slices = []
    for st in stations:
        sel = st.times >= test_lo
        if sel.sum() == 0:
            continue
        slices.append((st.values[sel], st.mask[sel]))
    if not slices:
        return None

    if ffill_impute is None:
        from src.data.impute import ffill_mean_impute
        ffill_impute = ffill_mean_impute

    if model_impute is None:
        ckpt = (Path(cfg["paths"]["checkpoints_dir"])
                / f"saits_imputer_seed{seed}.pt")
        if not ckpt.exists():
            logger.warning("imputability: no SAITS checkpoint %s — skipping", ckpt)
            return None
        import torch
        from src.models.saits import SAITS
        model = SAITS(stations[0].values.shape[1], cfg)
        model.load_state_dict(
            torch.load(ckpt, weights_only=False)["model_state"]
        )
        model.eval()
        seg_len = int(cfg["baselines"]["saits"]["segment_len"])
        model_impute = lambda x, m: _saits_impute_array(model, x, m, seg_len)  # noqa: E731

    skill = _impute_skill(slices, model_impute, ffill_impute, holdout_rate, seed)
    pol = cfg["dataset"]["primary_target"]
    row = {
        "dataset": str(cfg.get("dataset_name", "dhaka")).capitalize(),
        "natural_PM25_missing_pct": round(float(df[pol].isna().mean() * 100), 1),
        **skill,
    }
    return pd.DataFrame([row])


def imputability_crossover_figure(configs: list[dict], figures_dir: Path,
                                  tables_dir: Path, *, mode: str = "out",
                                  horizon: int = 6, level: int = 50
                                  ) -> pd.DataFrame | None:
    """The headline: end-to-end advantage vs *measured imputability* per dataset.

    For each config, plots the end-to-end advantage at a fixed severe operating
    point (default station-outage, 6 h, nearest level to +50%) against the
    dataset's imputability score. The expected monotone decline through zero
    turns the severity x imputability story into a single predictive curve.
    Exports ``decision_by_imputability.{csv,tex}`` and
    ``imputability_crossover.{png,pdf}``.
    """
    rows = []
    for cfg in configs:
        scalers = _load_scalers_for(cfg)
        imp = imputability_score(cfg, scalers)
        if imp is None or imp.empty:
            continue
        rob = robustness_long(cfg, scalers)
        cross = crossover_long(cfg, rob) if rob is not None else None
        adv = eff = float("nan")
        if cross is not None and not cross.empty:
            sub = cross[(cross["mode"] == mode) & (cross["horizon"] == horizon)]
            if not sub.empty:
                sub = sub.assign(_d=(sub["level"] - level).abs())
                pick = sub.sort_values("_d").iloc[0]
                adv, eff = float(pick["gap"]), float(pick["eff_missing_pct"])
        r = imp.iloc[0].to_dict()
        r["adv_h%d_%s%d" % (horizon, mode, level)] = (
            round(adv, 2) if np.isfinite(adv) else float("nan"))
        r["eff_missing_pct"] = eff
        r["recommendation"] = (
            "end-to-end" if np.isfinite(adv) and adv > 0
            else "impute-then-forecast" if np.isfinite(adv) else "—")
        rows.append(r)
    if not rows:
        return None
    tbl = pd.DataFrame(rows).sort_values("imputability")
    tbl.to_csv(tables_dir / "decision_by_imputability.csv", index=False)
    export_table(
        tbl.set_index("dataset"), tables_dir, "decision_by_imputability",
        "Decision by measured imputability: per network, series imputability "
        "(1 $-$ SAITS/ffill reconstruction RMSE on held-out observed cells) "
        "and the end-to-end advantage at "
        f"{horizon}\\,h under +{level}\\% station-outage corruption "
        "(\\si{\\micro\\gram\\per\\cubic\\metre}; $>0$ favours end-to-end).",
        "tab:decision_imputability", "%s")

    adv_col = "adv_h%d_%s%d" % (horizon, mode, level)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(tbl["imputability"], tbl[adv_col], s=60, color="#0072B2", zorder=3)
    if tbl[adv_col].notna().sum() >= 2:
        g = tbl.dropna(subset=[adv_col]).sort_values("imputability")
        ax.plot(g["imputability"], g[adv_col], color="#0072B2", lw=1, zorder=2)
    for _, r in tbl.iterrows():
        if np.isfinite(r[adv_col]):
            ax.annotate(r["dataset"], (r["imputability"], r[adv_col]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set_xlabel("Series imputability  (1 − SAITS/ffill reconstruction RMSE)")
    ax.set_ylabel(f"End-to-end advantage at {horizon} h, +{level}% outage "
                  "(µg/m³)\n← impute-then-forecast  |  end-to-end →")
    ax.set_title("End-to-end advantage declines with imputability")
    fig.tight_layout()
    save_figure(fig, figures_dir, "imputability_crossover")
    logger.info("wrote imputability-crossover figure (%d datasets)", len(tbl))
    return tbl


# ---------------------------------------------------------------------------
# Window-stratified mechanism: does the end-to-end advantage track per-window
# input missingness?
# ---------------------------------------------------------------------------

def window_input_missingness(cfg: dict) -> dict[tuple[int, int], float]:
    """Per-window mean input missingness on the test set, keyed by
    ``(station_id, anchor_time)`` so it aligns to any saved bundle.

    Recomputed offline from the windowed dataset (no retraining / no
    re-inference). Imported lazily so :mod:`src.evaluate` stays torch-free for
    pure-table unit tests.
    """
    from src.data.dataset import make_datasets

    datasets, _, _ = make_datasets(cfg)
    ds = datasets["test"]
    out: dict[tuple[int, int], float] = {}
    for i in range(len(ds)):
        s = ds[i]
        key = (int(s["station_id"]), int(s["anchor_time"]))
        out[key] = float((s["mask"].numpy() == 0).mean())
    return out


def stratified_gap_table(cfg: dict, scalers: dict, n_bins: int = 5,
                         horizon: int | None = None,
                         wim: dict[tuple[int, int], float] | None = None,
                         ) -> pd.DataFrame | None:
    """Clean-test PM2.5 RMSE of end-to-end vs best two-stage, stratified by the
    window's input missingness. The gap is expected to widen on the most
    incomplete windows — mechanistic evidence rather than an aggregate.

    ``wim`` (``(station_id, anchor_time) -> missingness``) is computed via
    :func:`window_input_missingness` when not supplied (injectable for tests).
    """
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    e2e = "proposed_md" if (pred_dir / "proposed_md_test.npz").exists() else "proposed"
    ts = next((t for t in ("two_stage_saits", "two_stage_knn", "two_stage_mice")
               if (pred_dir / f"{t}_test.npz").exists()), None)
    if ts is None or not (pred_dir / f"{e2e}_test.npz").exists():
        return None
    horizon = horizon or (6 if 6 in cfg["dataset"]["horizons"]
                          else cfg["dataset"]["horizons"][0])
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    hi = cfg["dataset"]["horizons"].index(horizon)
    if wim is None:
        wim = window_input_missingness(cfg)

    def load(name):
        b = dict(np.load(pred_dir / f"{name}_test.npz"))
        miss = np.array([wim.get((int(s), int(a)), np.nan)
                         for s, a in zip(b["station_id"], b["anchor_time"])])
        return b, miss

    be, miss = load(e2e)
    bt, _ = load(ts)
    valid = np.isfinite(miss)
    edges = np.quantile(miss[valid], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    e2e_lbl, ts_lbl = MODEL_LABELS.get(e2e, e2e), MODEL_LABELS.get(ts, ts)
    rows = []
    for bi in range(n_bins):
        sel = valid & (miss >= edges[bi]) & (miss < edges[bi + 1])
        m = sel & (be["target_mask"][:, ti, hi] > 0) & (bt["target_mask"][:, ti, hi] > 0)
        m &= np.isfinite(be["predictions"][:, ti, hi]) & np.isfinite(bt["predictions"][:, ti, hi])
        y = unscale(be["targets"][m, ti, hi], "PM2.5", scalers)
        re = _metrics(unscale(be["predictions"][m, ti, hi], "PM2.5", scalers), y)["RMSE"]
        rt = _metrics(unscale(bt["predictions"][m, ti, hi], "PM2.5", scalers), y)["RMSE"]
        rows.append({
            "missingness_bin": f"{edges[bi] * 100:.0f}-{edges[bi + 1] * 100:.0f}%",
            "mid_missing_pct": round((edges[bi] + edges[bi + 1]) / 2 * 100, 1),
            "n": int(m.sum()), e2e_lbl: round(re, 2), ts_lbl: round(rt, 2),
            "gap (two-stage − end-to-end)": round(rt - re, 2),
        })
    return pd.DataFrame(rows)


def stratified_gap_figure(tbl: pd.DataFrame, cfg: dict, figures_dir: Path) -> None:
    horizon = 6 if 6 in cfg["dataset"]["horizons"] else cfg["dataset"]["horizons"][0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(tbl["missingness_bin"], tbl["gap (two-stage − end-to-end)"],
           color="#0072B2", width=0.7)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xlabel("Per-window input missingness")
    ax.set_ylabel(f"RMSE gap at {horizon} h (µg/m³)\ntwo-stage − end-to-end")
    ax.set_title("End-to-end advantage by window incompleteness "
                 "(positive ⇒ end-to-end better)")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures_dir, "stratified_gap")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_evaluation(cfg: dict[str, Any]) -> None:
    """Generate all Phase 5 tables and figures from saved artifacts."""
    apply_style()
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    scalers = json.loads(
        (Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text()
    )
    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "all_stations.parquet")
    stations = sorted(df["station"].unique())
    bundles = load_bundles(cfg)

    # Table 2 feedstock: per-seed metrics (long format, seed column)
    long_df = seed_metrics_long(cfg, scalers)
    if long_df.empty:
        logger.warning("no prediction bundles found, skipping main tables")
    else:
        long_df.to_csv(tables_dir / "metrics_full.csv", index=False)
        main_tbl = build_main_results(long_df, cfg)
        export_table(
            main_tbl, tables_dir, "main_results_pm25",
            "PM2.5 forecasting performance on the held-out test period "
            "(observed targets only). RMSE cells are mean $\\pm$ std over "
            "training seeds for learned models; persistence, seasonal-naive "
            "and SARIMA are deterministic single runs. MAE and R$^2$ are "
            "seed means.", "tab:main_results", "%s")

    export_table(pm25_station_table(bundles, cfg, scalers, stations).round(1),
                 tables_dir, "pm25_rmse_by_station",
                 "PM2.5 RMSE at 24 h per station (canonical seed).",
                 "tab:station_rmse")

    sig = significance_table_multiseed(cfg, scalers)
    if not sig.empty:
        export_table(
            sig.set_index(["baseline", "horizon"]).round(4),
            tables_dir, "significance_dm_bootstrap",
            "Per-seed Diebold--Mariano tests and paired-bootstrap "
            "RMSE-difference CIs: proposed (seed $i$) vs each baseline "
            "(seed $i$; statistical baselines are deterministic). Median and "
            "range of $p$ over seeds; negative RMSE difference = proposed "
            "better.", "tab:significance", "%.4f")

    season_tbl = seasonal_table(bundles, cfg, scalers)
    export_table(season_tbl.round(1), tables_dir, "seasonal_rmse_pm25",
                 "PM2.5 RMSE at 24 h per season.", "tab:seasonal")
    seasonal_figure(season_tbl, figures_dir)

    ep_tbl = episode_table(cfg, scalers)
    if not ep_tbl.empty:
        thr = float(cfg.get("evaluation", {}).get("episode_threshold", 150))
        export_table(
            ep_tbl, tables_dir, "episode_rmse_pm25",
            f"High-pollution-episode PM2.5 RMSE "
            f"(\\si{{\\micro\\gram\\per\\cubic\\metre}}): test targets with "
            f"observed PM2.5 $>$ {thr:.0f}. Mean $\\pm$ std over seeds for "
            "learned models; the subset conditions on observed values, so it "
            "is identical across models.", "tab:episode", "%s")
        episode_figure(ep_tbl, cfg, figures_dir)

    export_table(efficiency_table(cfg), tables_dir, "efficiency",
                 "CPU efficiency: parameters, wall-clock training time, and "
                 "single-window inference latency on a desktop CPU.",
                 "tab:efficiency", "%.2f")

    example_forecast_figure(bundles, cfg, scalers, df, stations, figures_dir)
    rob_tbl = robustness_figure(cfg, scalers, figures_dir, tables_dir)

    # Missingness-severity crossover study + window-stratified mechanism
    run_crossover(cfg, scalers, rob_tbl, tables_dir, figures_dir)

    # Series imputability (the measured axis behind the crossover)
    imp = imputability_score(cfg, scalers)
    if imp is not None and not imp.empty:
        export_table(
            imp.set_index("dataset"), tables_dir, "imputability",
            "Series imputability: deep-imputer (SAITS) vs forward-fill "
            "reconstruction RMSE (standardized units) on held-out observed "
            "test cells. imputability $= 1 - $ RMSE\\textsubscript{SAITS}/"
            "RMSE\\textsubscript{ffill} (higher = more reconstructable).",
            "tab:imputability", "%s")

    strat = stratified_gap_table(cfg, scalers)
    if strat is not None:
        export_table(
            strat.set_index("missingness_bin"), tables_dir, "stratified_gap",
            "Clean-test PM2.5 RMSE of end-to-end vs best two-stage, stratified "
            "by per-window input missingness; positive gap = end-to-end better.",
            "tab:stratified", "%s")
        stratified_gap_figure(strat, cfg, figures_dir)
    logger.info("evaluation complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging("evaluate", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
