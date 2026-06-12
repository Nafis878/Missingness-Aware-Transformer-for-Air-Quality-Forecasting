"""Phase 1, step 2: full missingness analysis (a paper section in itself).

Usage::

    python scripts/02_missingness_analysis.py --config config.yaml

Produces (every table as CSV + booktabs LaTeX, every figure as PNG 300dpi + PDF):

* Per-station x per-variable missingness rate table.
* Missingness heatmaps (time x variable, one panel per station).
* Per-season and per-month missingness breakdown.
* Gap-length distribution histograms.
* Co-missingness correlation matrix.
* MAR-vs-MCAR evidence: logistic regression predicting PM2.5 missingness
  from observed meteorology + calendar features (Little's MCAR test is
  infeasible here: with 16 variables and arbitrary missingness patterns the
  number of distinct patterns is in the thousands, and the test's chi-square
  approximation breaks down; the regression test answers the question that
  actually matters for modeling - is missingness predictable?).
* A machine-readable ``missingness_summary.json`` consumed by the Phase 1 report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting_style import SEASON_ORDER, apply_style, save_figure
from src.utils import export_table as _export_table
from src.utils import load_config, seed_everything, setup_logging

import logging

logger = logging.getLogger("02_missingness_analysis")

SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}


export_table = _export_table


def missingness_rate_table(df: pd.DataFrame, meas: list[str], tables_dir: Path) -> pd.DataFrame:
    """Per-station x per-variable missingness rates (%)."""
    tbl = df.groupby("station")[meas].apply(lambda g: g.isna().mean() * 100).round(1)
    tbl.loc["ALL"] = (df[meas].isna().mean() * 100).round(1)
    export_table(
        tbl, tables_dir, "missingness_by_station_variable",
        "Missingness rate (\\%) per station and variable after cleaning and hourly reindexing.",
        "tab:missingness_station_variable",
    )
    return tbl


def missingness_heatmaps(df: pd.DataFrame, meas: list[str], figures_dir: Path) -> None:
    """One time-x-variable binary missingness heatmap per station (4x4 grid)."""
    stations = sorted(df["station"].unique())
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=False)
    for ax, st in zip(axes.flat, stations):
        sub = df[df["station"] == st].set_index("datetime").sort_index()
        # daily missing fraction per variable -> manageable image size
        daily = sub[meas].isna().resample("D").mean()
        im = ax.imshow(
            daily.T.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1,
            interpolation="nearest",
            extent=[0, len(daily), len(meas) - 0.5, -0.5],
        )
        ax.set_title(st, fontsize=9)
        ax.set_yticks(range(len(meas)))
        ax.set_yticklabels(meas, fontsize=5)
        # x ticks at year boundaries
        years = daily.index.year.unique()
        ticks = [int(np.searchsorted(daily.index, pd.Timestamp(f"{y}-01-01"))) for y in years]
        ax.set_xticks(ticks)
        ax.set_xticklabels(years, fontsize=7)
        ax.grid(False)
    for ax in axes.flat[len(stations):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.5, label="daily fraction missing")
    fig.suptitle("Missingness structure per station (daily fraction missing)", y=1.0)
    save_figure(fig, figures_dir, "missingness_heatmap_all_stations")


def seasonal_breakdown(df: pd.DataFrame, meas: list[str], tables_dir: Path,
                       figures_dir: Path) -> pd.DataFrame:
    """Per-season and per-month missingness tables + figure."""
    df = df.copy()
    df["season"] = df["datetime"].dt.month.map(SEASON_OF_MONTH)
    df["month"] = df["datetime"].dt.month

    season_tbl = (
        df.groupby("season")[meas].apply(lambda g: g.isna().mean() * 100)
        .reindex(SEASON_ORDER).round(1)
    )
    export_table(
        season_tbl, tables_dir, "missingness_by_season",
        "Missingness rate (\\%) per Bangladesh season "
        "(winter Dec--Feb, pre-monsoon Mar--May, monsoon Jun--Sep, post-monsoon Oct--Nov).",
        "tab:missingness_season",
    )
    month_tbl = (
        df.groupby("month")[meas].apply(lambda g: g.isna().mean() * 100).round(1)
    )
    export_table(
        month_tbl, tables_dir, "missingness_by_month",
        "Missingness rate (\\%) per calendar month.", "tab:missingness_month",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    key_vars = ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2", "Temp", "RH"]
    month_tbl[key_vars].plot(ax=ax, marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Missing (%)")
    ax.set_xticks(range(1, 13))
    ax.legend(ncol=4, fontsize=8)
    ax.set_title("Monthly missingness by variable (all stations pooled)")
    save_figure(fig, figures_dir, "missingness_by_month")
    return season_tbl


def gap_lengths(series: pd.Series) -> np.ndarray:
    """Lengths (hours) of consecutive-NaN runs in an hourly series."""
    isna = series.isna().to_numpy()
    if not isna.any():
        return np.array([], dtype=int)
    change = np.diff(np.concatenate([[0], isna.astype(int), [0]]))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return ends - starts


def gap_length_analysis(df: pd.DataFrame, meas: list[str], tables_dir: Path,
                        figures_dir: Path) -> dict[str, Any]:
    """Gap-length histograms + summary stats; returns block-structure metrics."""
    key_vars = ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2"]
    all_gaps: dict[str, np.ndarray] = {}
    for var in key_vars:
        gaps = [gap_lengths(g.set_index("datetime")[var])
                for _, g in df.groupby("station")]
        all_gaps[var] = np.concatenate(gaps) if gaps else np.array([])

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    bins = np.logspace(0, np.log10(24 * 365), 40)
    for ax, var in zip(axes.flat, key_vars):
        g = all_gaps[var]
        ax.hist(g, bins=bins, color="#0072B2", edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(24, color="#D55E00", linestyle="--", linewidth=0.8, label="1 day")
        ax.axvline(24 * 30, color="#009E73", linestyle="--", linewidth=0.8, label="30 days")
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("gap length (h)")
        ax.set_ylabel("count")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Distribution of consecutive-missing run lengths (all stations)")
    fig.tight_layout()
    save_figure(fig, figures_dir, "gap_length_distributions")

    rows = []
    block_metrics: dict[str, Any] = {}
    for var, g in all_gaps.items():
        if len(g) == 0:
            continue
        total_missing = g.sum()
        rows.append({
            "variable": var,
            "n_gaps": len(g),
            "median_h": float(np.median(g)),
            "p90_h": float(np.percentile(g, 90)),
            "max_h": int(g.max()),
            "share_missing_in_gaps_gt_7d": float(g[g > 168].sum() / total_missing),
            "share_missing_in_gaps_gt_30d": float(g[g > 720].sum() / total_missing),
        })
    gap_tbl = pd.DataFrame(rows).set_index("variable")
    export_table(
        gap_tbl, tables_dir, "gap_length_summary",
        "Summary of consecutive-missing run lengths per pollutant (hours).",
        "tab:gap_lengths", float_format="%.2f",
    )
    block_metrics["pm25_share_gt_7d"] = float(gap_tbl.loc["PM2.5", "share_missing_in_gaps_gt_7d"])
    block_metrics["pm25_share_gt_30d"] = float(gap_tbl.loc["PM2.5", "share_missing_in_gaps_gt_30d"])
    block_metrics["pm25_median_gap_h"] = float(gap_tbl.loc["PM2.5", "median_h"])
    return block_metrics


def co_missingness(df: pd.DataFrame, meas: list[str], tables_dir: Path,
                   figures_dir: Path) -> pd.DataFrame:
    """Correlation matrix of missingness indicators across variables."""
    ind = df[meas].isna().astype(float)
    corr = ind.corr()
    export_table(
        corr.round(2), tables_dir, "co_missingness_correlation",
        "Pearson correlation between missingness indicators of all variables.",
        "tab:co_missingness", float_format="%.2f",
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(meas)))
    ax.set_xticklabels(meas, rotation=90, fontsize=8)
    ax.set_yticks(range(len(meas)))
    ax.set_yticklabels(meas, fontsize=8)
    for i in range(len(meas)):
        for j in range(len(meas)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation of missingness")
    ax.set_title("Co-missingness correlation matrix")
    ax.grid(False)
    save_figure(fig, figures_dir, "co_missingness_matrix")
    return corr


def mar_evidence(df: pd.DataFrame, tables_dir: Path, seed: int) -> dict[str, Any]:
    """Logistic regression: is PM2.5 missingness predictable from observed data?

    Uses rows where the meteorological predictors are observed, predicting the
    PM2.5 missingness indicator from standardized meteorology + cyclic calendar
    features. AUC well above 0.5 (and significant coefficients) is evidence
    against MCAR and consistent with MAR/MNAR structure.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    met = ["Temp", "RH", "WS", "BP", "SR"]
    sub = df.dropna(subset=met).copy()
    sub["y"] = sub["PM2.5"].isna().astype(int)
    sub["hour_sin"] = np.sin(2 * np.pi * sub["datetime"].dt.hour / 24)
    sub["hour_cos"] = np.cos(2 * np.pi * sub["datetime"].dt.hour / 24)
    sub["month_sin"] = np.sin(2 * np.pi * sub["datetime"].dt.month / 12)
    sub["month_cos"] = np.cos(2 * np.pi * sub["datetime"].dt.month / 12)
    feats = met + ["hour_sin", "hour_cos", "month_sin", "month_cos"]

    X = StandardScaler().fit_transform(sub[feats])
    y = sub["y"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))

    coef_tbl = pd.DataFrame(
        {"coefficient": clf.coef_[0]}, index=pd.Index(feats, name="feature")
    ).sort_values("coefficient", key=np.abs, ascending=False)
    export_table(
        coef_tbl, tables_dir, "mar_logistic_coefficients",
        f"Standardized logistic-regression coefficients predicting PM2.5 missingness "
        f"from observed meteorology and calendar features (held-out AUC = {auc:.3f}).",
        "tab:mar_logistic", float_format="%.3f",
    )
    logger.info("MAR test: n=%d, base rate=%.3f, held-out AUC=%.3f",
                len(sub), y.mean(), auc)
    return {"n": int(len(sub)), "base_rate": float(y.mean()), "auc": auc,
            "top_coefficients": coef_tbl.head(5)["coefficient"].round(3).to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("02_missingness_analysis", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))
    apply_style()

    meas = cfg["data"]["measurement_cols"]
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])

    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "all_stations.parquet")
    logger.info("loaded %d rows, %d stations", len(df), df["station"].nunique())

    summary: dict[str, Any] = {}
    rate_tbl = missingness_rate_table(df, meas, tables_dir)
    summary["overall_missing_pct"] = rate_tbl.loc["ALL"].to_dict()
    summary["pm25_station_min_max"] = [
        float(rate_tbl["PM2.5"].drop("ALL").min()),
        float(rate_tbl["PM2.5"].drop("ALL").max()),
    ]

    missingness_heatmaps(df, meas, figures_dir)
    season_tbl = seasonal_breakdown(df, meas, tables_dir, figures_dir)
    summary["pm25_by_season"] = season_tbl["PM2.5"].to_dict()
    summary["gap_structure"] = gap_length_analysis(df, meas, tables_dir, figures_dir)

    corr = co_missingness(df, meas, tables_dir, figures_dir)
    pollutants = ["SO2", "NO", "NO2", "NOX", "CO", "O3", "PM10", "PM2.5"]
    pol_corr = corr.loc[pollutants, pollutants]
    off_diag = pol_corr.to_numpy()[~np.eye(len(pollutants), dtype=bool)]
    summary["mean_pollutant_co_missingness_corr"] = float(off_diag.mean())

    summary["mar_logistic"] = mar_evidence(df, tables_dir, cfg["seed"])

    out = Path(cfg["paths"]["outputs_dir"]) / "missingness_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
