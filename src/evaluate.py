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

from src.plotting_style import SEASON_ORDER, apply_style, save_figure
from src.utils import export_table, load_config, seed_everything, setup_logging

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal-naive",
    "sarima": "SARIMA",
    "lstm": "LSTM",
    "gru": "GRU",
    "two_stage_knn": "Two-stage (KNN)",
    "two_stage_mice": "Two-stage (MICE)",
    "proposed": "Proposed (MAT)",
    "proposed_md": "Proposed + miss-dropout",
}

SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}


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
    rows = {}
    for name, b in bundles.items():
        months = pd.to_datetime(b["anchor_time"], unit="s").month
        seasons = pd.Series(months).map(SEASON_OF_MONTH).to_numpy()
        col = {}
        for season in SEASON_ORDER:
            m = (b["target_mask"][:, ti, hi] > 0) & (seasons == season)
            m &= np.isfinite(b["predictions"][:, ti, hi])
            p = unscale(b["predictions"][m, ti, hi], "PM2.5", scalers)
            y = unscale(b["targets"][m, ti, hi], "PM2.5", scalers)
            col[season] = _metrics(p, y)["RMSE"]
        rows[MODEL_LABELS.get(name, name)] = col
    return pd.DataFrame(rows).reindex(SEASON_ORDER)


def seasonal_figure(tbl: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    tbl.plot.bar(ax=ax, width=0.85)
    ax.set_ylabel("PM2.5 RMSE at 24 h (µg/m³)")
    ax.set_xlabel("")
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Seasonal forecast performance (test year 2024)")
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

    ``picks``: list of (station, start, end) periods; defaults chosen to show
    one high-pollution winter period and one gappy monsoon period.
    """
    picks = picks or [
        ("Darussalam", "2024-01-05", "2024-02-05"),
        ("Barishal", "2024-07-01", "2024-08-01"),
    ]
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

def robustness_figure(cfg: dict, scalers: dict, figures_dir: Path,
                      tables_dir: Path) -> pd.DataFrame | None:
    """PM2.5 RMSE vs synthetic extra-missingness (the money figure).

    Two corruption mechanisms (rows of panels): cell-wise MCAR (``miss``) and
    station-outage blocks (``out``), the latter matching the dominant
    real-world mechanism found in the Phase 1 missingness analysis.
    """
    levels = [0.0] + list(cfg["dataset"]["synthetic_missingness"])
    modes = {"miss": "Cell-wise MCAR", "out": "Station-outage blocks"}
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    pred_dir = Path(cfg["paths"]["predictions_dir"])

    def has_all_bundles(name: str) -> bool:
        suffixes = ["test"] + [f"test_{m}{int(lv * 100)}"
                               for m in modes for lv in levels if lv > 0]
        return all((pred_dir / f"{name}_{s}.npz").exists() for s in suffixes)

    candidates = ["proposed", "proposed_md", "two_stage_knn", "two_stage_mice"]
    models = [m for m in candidates if has_all_bundles(m)]
    if "proposed" not in models:
        logger.warning("robustness: proposed bundles incomplete, skipping figure")
        return None
    rows = []
    for mode in modes:
        for name in models:
            for level in levels:
                suffix = "test" if level == 0 else f"test_{mode}{int(level * 100)}"
                b = dict(np.load(pred_dir / f"{name}_{suffix}.npz"))
                for hi, h in enumerate(cfg["dataset"]["horizons"]):
                    m = b["target_mask"][:, ti, hi] > 0
                    p = unscale(b["predictions"][m, ti, hi], "PM2.5", scalers)
                    y = unscale(b["targets"][m, ti, hi], "PM2.5", scalers)
                    rows.append({"mode": mode, "model": MODEL_LABELS.get(name, name),
                                 "level": int(level * 100), "horizon": h,
                                 "RMSE": _metrics(p, y)["RMSE"]})
    tbl = pd.DataFrame(rows)
    export_table(
        tbl.pivot_table(index=["mode", "model", "horizon"], columns="level",
                        values="RMSE").round(2),
        tables_dir, "robustness_rmse",
        "PM2.5 RMSE (\\si{\\micro\\gram\\per\\cubic\\metre}) under additional "
        "synthetic input missingness: cell-wise MCAR vs station-outage blocks.",
        "tab:robustness", float_format="%.2f",
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)
    for row_i, (mode, mode_label) in enumerate(modes.items()):
        for col_i, h in enumerate(cfg["dataset"]["horizons"]):
            ax = axes[row_i, col_i]
            sub = tbl[(tbl["horizon"] == h) & (tbl["mode"] == mode)]
            for model, grp in sub.groupby("model", sort=False):
                ax.plot(grp["level"], grp["RMSE"], marker="o", label=model)
            if row_i == 0:
                ax.set_title(f"{h} h ahead")
            if row_i == 1:
                ax.set_xlabel("Additional missingness (%)")
            if col_i == 0:
                ax.set_ylabel(f"{mode_label}\nPM2.5 RMSE (µg/m³)")
            ax.set_xticks([0, 10, 30, 50])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Robustness to additional input missingness (test 2024)")
    fig.tight_layout()
    save_figure(fig, figures_dir, "robustness_curve")
    return tbl


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

    # Table 2 feedstock: full metrics
    full = metrics_table(bundles, cfg, scalers)
    full.to_csv(tables_dir / "metrics_full.csv", index=False)
    pm25 = full[full["pollutant"] == "PM2.5"].copy()
    pm25["model"] = pm25["model"].map(lambda m: MODEL_LABELS.get(m, m))
    main_tbl = pm25.pivot_table(index="model", columns="horizon",
                                values=["RMSE", "MAE", "R2", "sMAPE"]).round(2)
    main_tbl = main_tbl.reindex([MODEL_LABELS[m] for m in MODEL_LABELS
                                 if MODEL_LABELS[m] in main_tbl.index])
    export_table(main_tbl, tables_dir, "main_results_pm25",
                 "PM2.5 forecasting performance on the 2024 test year "
                 "(observed targets only).", "tab:main_results", "%.2f")

    export_table(pm25_station_table(bundles, cfg, scalers, stations).round(1),
                 tables_dir, "pm25_rmse_by_station",
                 "PM2.5 RMSE at 24 h per station.", "tab:station_rmse")

    if "proposed" in bundles:
        sig = significance_table(bundles, cfg, scalers)
        export_table(sig.set_index(["baseline", "horizon"]).round(4),
                     tables_dir, "significance_dm_bootstrap",
                     "Diebold--Mariano tests and paired-bootstrap RMSE-difference "
                     "CIs: proposed vs each baseline (PM2.5; negative = proposed "
                     "better).", "tab:significance", "%.4f")

    season_tbl = seasonal_table(bundles, cfg, scalers)
    export_table(season_tbl.round(1), tables_dir, "seasonal_rmse_pm25",
                 "PM2.5 RMSE at 24 h per Bangladesh season.", "tab:seasonal")
    seasonal_figure(season_tbl, figures_dir)

    export_table(efficiency_table(cfg), tables_dir, "efficiency",
                 "CPU efficiency: parameters, wall-clock training time, and "
                 "single-window inference latency on a desktop CPU.",
                 "tab:efficiency", "%.2f")

    example_forecast_figure(bundles, cfg, scalers, df, stations, figures_dir)
    robustness_figure(cfg, scalers, figures_dir, tables_dir)
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
