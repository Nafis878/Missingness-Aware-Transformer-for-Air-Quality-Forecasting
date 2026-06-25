"""Make before/after merged-imputation forecasting comparison figures.

The figures compare the previous best forecasting result ("before") with the
final merged-imputation/adaptive portfolio ("after") across all datasets and
horizons.  They are intended for the paper results section, so each figure is
saved as both 300-dpi PNG and vector PDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting_style import PALETTE, apply_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs/figures"
TABLE_DIR = ROOT / "outputs/tables"
SOURCE = TABLE_DIR / "adaptive_sota_portfolio_trainval_vs_table_best.csv"


DISPLAY_METHOD = {
    "tabular_extra_lean200_trainval": "Tabular refit",
    "tabular_extra_leaf8_trainval": "Tabular refit",
    "tabular_extra_leaf12_trainval": "Tabular refit",
    "linear_tabular_blend": "Linear-tabular blend",
    "ensemble_weighted": "Weighted ensemble",
    "ensemble_val_selected_pm25_metrics": "Validation-selected ensemble",
    "ensemble_seed_member": "Seed-member ensemble",
}


def load_table() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    df = df.rename(
        columns={
            "previous_table_best_RMSE": "before_RMSE",
            "RMSE": "after_RMSE",
            "relative_improvement_pct": "improvement_pct",
            "adaptive_portfolio_method": "after_method",
        }
    )
    df["after_method_label"] = df["after_method"].map(DISPLAY_METHOD).fillna(df["after_method"])
    df["cell"] = df["dataset"] + " " + df["horizon"].astype(str) + "h"
    df["cell_multiline"] = df["dataset"] + "\n" + df["horizon"].astype(str) + "h"
    df["absolute_reduction"] = df["before_RMSE"] - df["after_RMSE"]
    df = df.sort_values(
        ["dataset", "horizon"],
        key=lambda s: s.map({"Dhaka": 0, "Delhi": 1, "Beijing": 2}).fillna(s),
    )
    out = TABLE_DIR / "before_after_imputation_forecasting.csv"
    df.to_csv(out, index=False)
    return df


def annotate_bars(ax, bars, fmt="{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(0.5, height * 0.015),
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def figure_before_after_bars(df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    b1 = ax.bar(
        x - width / 2,
        df["before_RMSE"],
        width,
        color="#8A8A8A",
        label="Before imputation-aware portfolio",
    )
    b2 = ax.bar(
        x + width / 2,
        df["after_RMSE"],
        width,
        color=PALETTE[0],
        label="After merged-imputation portfolio",
    )
    for i, row in enumerate(df.itertuples(index=False)):
        ax.plot(
            [i - width / 2, i + width / 2],
            [row.before_RMSE, row.after_RMSE],
            color=PALETTE[1],
            lw=1.0,
            alpha=0.8,
        )
        ax.text(
            i,
            max(row.before_RMSE, row.after_RMSE) + 2.0,
            f"{row.improvement_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE[1],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["cell_multiline"], rotation=0)
    ax.set_ylabel("PM2.5 RMSE")
    ax.set_title("Forecasting improves after merged imputation")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.margins(x=0.01)
    save_figure(fig, FIG_DIR, "imputation_before_after_forecasting_rmse")


def figure_improvement_heatmap(df: pd.DataFrame) -> None:
    datasets = ["Dhaka", "Delhi", "Beijing"]
    horizons = [6, 24, 72]
    mat = np.array(
        [
            [
                df[(df["dataset"] == dataset) & (df["horizon"] == horizon)]["improvement_pct"].iloc[0]
                for horizon in horizons
            ]
            for dataset in datasets
        ]
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(12, mat.max()))
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("Forecasting RMSE reduction after imputation")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE reduction (%)")
    save_figure(fig, FIG_DIR, "imputation_forecasting_improvement_heatmap")


def figure_lollipop(df: pd.DataFrame) -> None:
    df = df.copy().sort_values("absolute_reduction")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hlines(y, 0, df["absolute_reduction"], color=PALETTE[0], lw=2)
    ax.plot(df["absolute_reduction"], y, "o", color=PALETTE[1], ms=6)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["cell"])
    ax.set_xlabel("Absolute RMSE reduction")
    ax.set_title("Every dataset-horizon cell improves after imputation")
    for yi, row in zip(y, df.itertuples(index=False)):
        ax.text(
            row.absolute_reduction + 0.08,
            yi,
            f"{row.absolute_reduction:.2f}",
            va="center",
            fontsize=8,
        )
    save_figure(fig, FIG_DIR, "imputation_forecasting_absolute_reduction")


def figure_method_contribution(df: pd.DataFrame) -> None:
    method_order = [
        "Tabular refit",
        "Linear-tabular blend",
        "Weighted ensemble",
        "Validation-selected ensemble",
        "Seed-member ensemble",
    ]
    summary = (
        df.groupby("after_method_label", as_index=False)
        .agg(cells=("cell", "count"), mean_improvement=("improvement_pct", "mean"))
    )
    summary["after_method_label"] = pd.Categorical(
        summary["after_method_label"], categories=method_order, ordered=True
    )
    summary = summary.sort_values("after_method_label")

    fig, ax1 = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(summary))
    bars = ax1.bar(x, summary["mean_improvement"], color=PALETTE[0], width=0.58)
    ax1.set_ylabel("Mean RMSE reduction (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["after_method_label"], rotation=20, ha="right")
    ax1.set_title("After-imputation forecasting components")
    annotate_bars(ax1, bars, "{:.1f}%")

    ax2 = ax1.twinx()
    ax2.plot(x, summary["cells"], color=PALETTE[1], marker="o", lw=1.8)
    ax2.set_ylabel("Cells selected")
    ax2.set_ylim(0, max(4, int(summary["cells"].max()) + 1))
    ax2.grid(False)
    save_figure(fig, FIG_DIR, "imputation_forecasting_method_components")


def main() -> None:
    apply_style()
    df = load_table()
    figure_before_after_bars(df)
    figure_improvement_heatmap(df)
    figure_lollipop(df)
    figure_method_contribution(df)
    print(f"wrote before/after imputation figures to {FIG_DIR}")
    for name in [
        "imputation_before_after_forecasting_rmse",
        "imputation_forecasting_improvement_heatmap",
        "imputation_forecasting_absolute_reduction",
        "imputation_forecasting_method_components",
    ]:
        print(FIG_DIR / f"{name}.png")
        print(FIG_DIR / f"{name}.pdf")
    print(TABLE_DIR / "before_after_imputation_forecasting.csv")


if __name__ == "__main__":
    main()
