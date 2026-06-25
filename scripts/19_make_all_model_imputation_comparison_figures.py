"""Model-level before/after imputation forecasting comparison figures.

This script is deliberately stricter than the portfolio figures: it compares
each model only when a matching before/after-imputation counterpart exists in
the artifacts.  It writes a traceable source CSV plus journal-style PNG/PDF
figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting_style import PALETTE, apply_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs/figures"
TABLE_DIR = ROOT / "outputs/tables"

DATASETS = {
    "Dhaka": ("config.yaml", ROOT / "outputs"),
    "Delhi": ("config_delhi.yaml", ROOT / "outputs/delhi"),
    "Beijing": ("config_beijing.yaml", ROOT / "outputs/beijing"),
}

DHAKA_LOCAL_PAIRS = [
    ("Persistence", "persistence", "hybrid8_persistence"),
    ("Seasonal naive", "seasonal_naive", "hybrid8_seasonal_naive"),
    ("SARIMA", "sarima", "hybrid8_sarima"),
    ("LSTM", "lstm", "hybrid8_lstm"),
    ("GRU", "gru", "hybrid8_gru"),
    ("GRU-D", "gru_d", "hybrid8_gru_d"),
    ("DLinear", "dlinear", "hybrid8_dlinear"),
    ("MAT", "proposed", "hybrid8_masked_proposed"),
    ("MAT variant B", "variant_B", "hybrid8_masked_variant_B"),
    ("MAT + miss-dropout", "proposed_md", "hybrid8_masked_proposed_md"),
]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))


def rmse_from_bundle(bundle: dict[str, np.ndarray], cfg: dict[str, Any],
                     horizon_idx: int) -> float:
    primary = cfg["dataset"]["primary_target"]
    target_idx = cfg["dataset"]["target_pollutants"].index(primary)
    scalers = json.loads((Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text())
    mean, std = scalers[primary]
    mask = bundle["target_mask"][:, target_idx, horizon_idx] > 0
    pred = bundle["predictions"][mask, target_idx, horizon_idx] * std + mean
    target = bundle["targets"][mask, target_idx, horizon_idx] * std + mean
    ok = np.isfinite(pred) & np.isfinite(target)
    return float(np.sqrt(np.mean((pred[ok] - target[ok]) ** 2)))


def collect_dhaka_local_rows() -> list[dict[str, Any]]:
    cfg_path, out_dir = DATASETS["Dhaka"]
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    pred_dir = out_dir / "predictions"
    rows = []
    for label, before_name, after_name in DHAKA_LOCAL_PAIRS:
        before_path = pred_dir / f"{before_name}_test.npz"
        after_path = pred_dir / f"{after_name}_test.npz"
        if not before_path.exists() or not after_path.exists():
            continue
        before = load_npz(before_path)
        after = load_npz(after_path)
        for hi, horizon in enumerate(cfg["dataset"]["horizons"]):
            before_rmse = rmse_from_bundle(before, cfg, hi)
            after_rmse = rmse_from_bundle(after, cfg, hi)
            rows.append({
                "dataset": "Dhaka",
                "model": label,
                "before_model": before_name,
                "after_model": after_name,
                "horizon": int(horizon),
                "before_RMSE": before_rmse,
                "after_RMSE": after_rmse,
                "delta_after_minus_before": after_rmse - before_rmse,
                "improvement_pct": (before_rmse - after_rmse) / before_rmse * 100.0,
                "source": "local_prediction_bundles",
            })
    return rows


def collect_zip_rows() -> list[dict[str, Any]]:
    path = TABLE_DIR / "hybrid8_zip_vs_standard_pm25.csv"
    if not path.exists():
        return []
    raw = pd.read_csv(path)
    rows = []
    label_map = {
        "Hybrid8 + mask + MAT": "MAT",
        "Hybrid8 + mask + MAT variant B": "MAT variant B",
        "Hybrid8 + mask + MAT + miss-dropout": "MAT + miss-dropout",
    }
    for row in raw.itertuples(index=False):
        for horizon in [6, 24, 72]:
            before = float(getattr(row, f"h{horizon}_standard"))
            after = float(getattr(row, f"h{horizon}_hybrid"))
            rows.append({
                "dataset": row.dataset,
                "model": label_map.get(row.model, row.model),
                "before_model": row.standard_equiv,
                "after_model": row.model,
                "horizon": horizon,
                "before_RMSE": before,
                "after_RMSE": after,
                "delta_after_minus_before": after - before,
                "improvement_pct": (before - after) / before * 100.0,
                "source": "hybrid8_zip_summary",
            })
    return rows


def collect_rows() -> pd.DataFrame:
    rows = collect_dhaka_local_rows()
    zip_rows = collect_zip_rows()
    # Keep local Dhaka rows for overlapping MAT models, and use the zip rows
    # mainly to add Delhi/Beijing coverage.
    rows.extend([r for r in zip_rows if r["dataset"] != "Dhaka"])
    df = pd.DataFrame(rows)
    model_order = [p[0] for p in DHAKA_LOCAL_PAIRS]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["dataset", "model", "horizon"])
    out_path = TABLE_DIR / "all_model_before_after_imputation_forecasting.csv"
    df.to_csv(out_path, index=False)
    return df


def figure_dhaka_grouped_bars(df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == "Dhaka"].copy()
    model_order = [m for m in [p[0] for p in DHAKA_LOCAL_PAIRS] if m in set(sub["model"].astype(str))]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2), sharex=True)
    for ax, horizon in zip(axes, [6, 24, 72]):
        hdf = sub[sub["horizon"] == horizon].set_index("model").loc[model_order].reset_index()
        x = np.arange(len(hdf))
        width = 0.38
        ax.bar(x - width / 2, hdf["before_RMSE"], width, color="#8A8A8A", label="Before")
        ax.bar(x + width / 2, hdf["after_RMSE"], width, color=PALETTE[0], label="After hybrid_top8")
        for i, row in enumerate(hdf.itertuples(index=False)):
            color = PALETTE[2] if row.improvement_pct > 0 else PALETTE[1]
            ax.text(
                i, max(row.before_RMSE, row.after_RMSE) + 1.2,
                f"{row.improvement_pct:+.1f}%",
                ha="center", va="bottom", fontsize=7, color=color,
            )
        ax.set_ylabel(f"{horizon}h RMSE")
        ax.set_title(f"Dhaka model-level before/after imputation ({horizon}h)")
        ax.margins(x=0.01)
    axes[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.38))
    axes[-1].set_xticks(np.arange(len(model_order)))
    axes[-1].set_xticklabels(model_order, rotation=35, ha="right")
    fig.tight_layout()
    save_figure(fig, FIG_DIR, "all_models_dhaka_before_after_imputation_rmse")


def figure_dhaka_heatmap(df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == "Dhaka"].copy()
    model_order = [m for m in [p[0] for p in DHAKA_LOCAL_PAIRS] if m in set(sub["model"].astype(str))]
    horizons = [6, 24, 72]
    mat = np.array([
        [sub[(sub["model"].astype(str) == model) & (sub["horizon"] == h)]["improvement_pct"].iloc[0]
         for h in horizons]
        for model in model_order
    ])
    vmax = max(abs(float(np.nanmin(mat))), abs(float(np.nanmax(mat))), 1.0)
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    im = ax.imshow(mat, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order)
    ax.set_title("Dhaka: model-level RMSE change after hybrid_top8 imputation")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.1f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE improvement (%)")
    save_figure(fig, FIG_DIR, "all_models_dhaka_imputation_improvement_heatmap")


def figure_all_available_heatmap(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["row_label"] = plot_df["dataset"] + " | " + plot_df["model"].astype(str)
    row_order = list(dict.fromkeys(plot_df["row_label"]))
    horizons = [6, 24, 72]
    mat = np.array([
        [plot_df[(plot_df["row_label"] == row) & (plot_df["horizon"] == h)]["improvement_pct"].iloc[0]
         if len(plot_df[(plot_df["row_label"] == row) & (plot_df["horizon"] == h)]) else np.nan
         for h in horizons]
        for row in row_order
    ])
    vmax = max(abs(float(np.nanmin(mat))), abs(float(np.nanmax(mat))), 1.0)
    fig_h = max(5.0, 0.34 * len(row_order) + 1.2)
    fig, ax = plt.subplots(figsize=(6.2, fig_h))
    im = ax.imshow(mat, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_title("All available model pairs: before vs after imputation")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.1f}%", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("RMSE improvement (%)")
    save_figure(fig, FIG_DIR, "all_available_models_imputation_improvement_heatmap")


def figure_mat_cross_dataset_bars(df: pd.DataFrame) -> None:
    sub = df[df["model"].astype(str).isin(["MAT", "MAT variant B", "MAT + miss-dropout"])].copy()
    sub = sub.sort_values(["dataset", "model", "horizon"])
    labels = sub["dataset"] + "\n" + sub["model"].astype(str) + "\n" + sub["horizon"].astype(str) + "h"
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    width = 0.38
    ax.bar(x - width / 2, sub["before_RMSE"], width, color="#8A8A8A", label="Before")
    ax.bar(x + width / 2, sub["after_RMSE"], width, color=PALETTE[0], label="After hybrid_top8")
    for i, row in enumerate(sub.itertuples(index=False)):
        color = PALETTE[2] if row.improvement_pct > 0 else PALETTE[1]
        ax.text(i, max(row.before_RMSE, row.after_RMSE) + 1.0, f"{row.improvement_pct:+.1f}%",
                ha="center", va="bottom", fontsize=6.5, color=color)
    ax.set_ylabel("PM2.5 RMSE")
    ax.set_title("MAT-family models before and after hybrid_top8 imputation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.margins(x=0.005)
    save_figure(fig, FIG_DIR, "mat_family_cross_dataset_before_after_imputation_rmse")


def main() -> None:
    apply_style()
    df = collect_rows()
    figure_dhaka_grouped_bars(df)
    figure_dhaka_heatmap(df)
    figure_all_available_heatmap(df)
    figure_mat_cross_dataset_bars(df)
    print(f"wrote source table: {TABLE_DIR / 'all_model_before_after_imputation_forecasting.csv'}")
    print(f"wrote figures to {FIG_DIR}")
    for name in [
        "all_models_dhaka_before_after_imputation_rmse",
        "all_models_dhaka_imputation_improvement_heatmap",
        "all_available_models_imputation_improvement_heatmap",
        "mat_family_cross_dataset_before_after_imputation_rmse",
    ]:
        print(FIG_DIR / f"{name}.png")
        print(FIG_DIR / f"{name}.pdf")


if __name__ == "__main__":
    main()
