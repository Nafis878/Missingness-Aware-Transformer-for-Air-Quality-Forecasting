"""Create Q1-style figures for the adaptive universal SOTA result."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import binomtest, combine_pvalues, norm, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting_style import PALETTE, apply_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs/figures"
TABLE = ROOT / "outputs/tables/adaptive_sota_portfolio_trainval_vs_table_best.csv"
PAIRED = ROOT / "outputs/tables/adaptive_sota_portfolio_trainval_paired_tests.csv"


def short_method(name: str) -> str:
    mapping = {
        "tabular_extra_lean200_trainval": "Tabular\nrefit",
        "tabular_extra_leaf8_trainval": "Tabular\nrefit",
        "tabular_extra_leaf12_trainval": "Tabular\nrefit",
        "linear_tabular_blend": "Linear-\ntabular",
        "ensemble_weighted": "Weighted\nensemble",
        "ensemble_val_selected_pm25_metrics": "Selected\nensemble",
        "ensemble_seed_member": "Seed\nensemble",
    }
    return mapping.get(name, name.replace("_", "\n"))


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08, 1.04, label, transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="right",
    )


def figure_rmse_comparison(df: pd.DataFrame) -> None:
    order = [(d, h) for d in ["Dhaka", "Delhi", "Beijing"] for h in [6, 24, 72]]
    x = np.arange(len(order))
    prev = np.array([
        df[(df.dataset == d) & (df.horizon == h)]["previous_table_best_RMSE"].iloc[0]
        for d, h in order
    ])
    new = np.array([
        df[(df.dataset == d) & (df.horizon == h)]["RMSE"].iloc[0]
        for d, h in order
    ])
    labels = [f"{d}\n{h}h" for d, h in order]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    width = 0.36
    ax.bar(x - width / 2, prev, width, label="Previous best", color="#8A8A8A")
    ax.bar(x + width / 2, new, width, label="Adaptive portfolio", color=PALETTE[0])
    for i, (p, n) in enumerate(zip(prev, new)):
        ax.plot([i - width / 2, i + width / 2], [p, n], color=PALETTE[1], lw=1.0, alpha=0.75)
        ax.text(i, max(p, n) + 2.0, f"{(p - n) / p * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PM2.5 RMSE")
    ax.set_title("Adaptive portfolio improves every published benchmark cell")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.margins(x=0.01)
    save_figure(fig, FIG_DIR, "q1_universal_rmse_comparison")


def figure_improvement_heatmap(df: pd.DataFrame) -> None:
    datasets = ["Dhaka", "Delhi", "Beijing"]
    horizons = [6, 24, 72]
    mat = np.array([
        [df[(df.dataset == d) & (df.horizon == h)]["relative_improvement_pct"].iloc[0] for h in horizons]
        for d in datasets
    ])
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(12, mat.max()))
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("Relative improvement over previous best")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Improvement (%)")
    save_figure(fig, FIG_DIR, "q1_relative_improvement_heatmap")


def figure_paired_forest(paired: pd.DataFrame) -> None:
    paired = paired.copy()
    paired["label"] = paired["dataset"] + " " + paired["horizon"].astype(str) + "h"
    paired = paired.sort_values(["dataset", "horizon"], key=lambda s: s.map({"Dhaka": 0, "Delhi": 1, "Beijing": 2}).fillna(s))
    y = np.arange(len(paired))[::-1]
    diff = paired["RMSE_diff_ensemble_minus_best"].to_numpy()
    lo = paired["diff_CI95_lo"].to_numpy()
    hi = paired["diff_CI95_hi"].to_numpy()
    colors = [PALETTE[2] if sig else PALETTE[0] for sig in paired["significant_win"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for yi, d, l, h, c in zip(y, diff, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=2)
        ax.plot(d, yi, "o", color=c, ms=5)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(paired["label"])
    ax.set_xlabel("RMSE difference vs recomputed previous best\n(negative favors adaptive portfolio)")
    ax.set_title("Paired RMSE differences with bootstrap 95% intervals")
    legend_items = [
        Line2D([0], [0], color=PALETTE[2], marker="o", lw=2, label="Strict cell-level win"),
        Line2D([0], [0], color=PALETTE[0], marker="o", lw=2, label="Directional win"),
    ]
    ax.legend(handles=legend_items, loc="lower left")
    save_figure(fig, FIG_DIR, "q1_paired_delta_forest")


def figure_statistical_support(paired: pd.DataFrame) -> None:
    diffs = paired["RMSE_diff_ensemble_minus_best"].to_numpy()
    wins = int((diffs < 0).sum())
    n = len(diffs)
    sign_p = binomtest(wins, n, p=0.5, alternative="greater").pvalue
    wilcox_p = wilcoxon(diffs, alternative="less", zero_method="wilcox").pvalue
    one_sided_dm = []
    for row in paired.itertuples(index=False):
        p = float(row.DM_p)
        one_sided_dm.append(p / 2.0 if row.RMSE_diff_ensemble_minus_best < 0 else 1.0 - p / 2.0)
    fisher_p = combine_pvalues(one_sided_dm, method="fisher").pvalue
    z = norm.isf(np.clip(one_sided_dm, 1e-15, 1.0))
    weights = np.sqrt(paired["n"].to_numpy(dtype=float))
    stouffer_p = norm.sf(np.sum(weights * z) / np.sqrt(np.sum(weights * weights)))

    names = ["Sign test", "Wilcoxon", "Fisher DM", "Stouffer DM"]
    pvals = np.array([sign_p, wilcox_p, fisher_p, stouffer_p])
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bars = ax.bar(names, -np.log10(pvals), color=[PALETTE[0], PALETTE[0], PALETTE[2], PALETTE[2]])
    ax.axhline(-np.log10(0.05), color=PALETTE[1], lw=1.2, ls="--", label="p = 0.05")
    for bar, p in zip(bars, pvals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"p={p:.2g}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title("Universal 9/9 improvement pattern is statistically supported")
    ax.legend(loc="upper left")
    save_figure(fig, FIG_DIR, "q1_universal_statistical_support")


def figure_summary_panel(df: pd.DataFrame, paired: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 6.6))
    ax = axs[0, 0]
    datasets = ["Dhaka", "Delhi", "Beijing"]
    horizons = [6, 24, 72]
    mat = np.array([
        [df[(df.dataset == d) & (df.horizon == h)]["relative_improvement_pct"].iloc[0] for h in horizons]
        for d in datasets
    ])
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(12, mat.max()))
    ax.set_xticks(np.arange(3)); ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_yticks(np.arange(3)); ax.set_yticklabels(datasets)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    ax.set_title("Improvement")
    add_panel_label(ax, "A")

    ax = axs[0, 1]
    paired = paired.copy()
    prefix = {"Dhaka": "Dh", "Delhi": "De", "Beijing": "Bj"}
    paired["label"] = paired["dataset"].map(prefix) + paired["horizon"].astype(str)
    y = np.arange(len(paired))[::-1]
    for yi, row in zip(y, paired.itertuples(index=False)):
        c = PALETTE[2] if row.significant_win else PALETTE[0]
        ax.plot([row.diff_CI95_lo, row.diff_CI95_hi], [yi, yi], color=c, lw=1.8)
        ax.plot(row.RMSE_diff_ensemble_minus_best, yi, "o", color=c, ms=4)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(paired["label"])
    ax.set_title("Paired deltas")
    ax.set_xlabel("RMSE diff")
    add_panel_label(ax, "B")

    ax = axs[1, 0]
    prev = df["previous_table_best_RMSE"].to_numpy()
    new = df["RMSE"].to_numpy()
    ax.scatter(prev, new, color=PALETTE[0], s=32)
    lim = [min(prev.min(), new.min()) - 3, max(prev.max(), new.max()) + 3]
    ax.plot(lim, lim, color="black", lw=1, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Previous best RMSE")
    ax.set_ylabel("Adaptive RMSE")
    ax.set_title("All points below parity")
    add_panel_label(ax, "C")

    ax = axs[1, 1]
    diffs = paired["RMSE_diff_ensemble_minus_best"].to_numpy()
    wins = int((diffs < 0).sum())
    sign_p = binomtest(wins, len(diffs), p=0.5, alternative="greater").pvalue
    wilcox_p = wilcoxon(diffs, alternative="less", zero_method="wilcox").pvalue
    one_sided_dm = [
        row.DM_p / 2.0 if row.RMSE_diff_ensemble_minus_best < 0 else 1.0 - row.DM_p / 2.0
        for row in paired.itertuples(index=False)
    ]
    pvals = np.array([sign_p, wilcox_p, combine_pvalues(one_sided_dm, method="fisher").pvalue])
    labels = ["Sign", "Wilcoxon", "Fisher DM"]
    ax.bar(labels, -np.log10(pvals), color=[PALETTE[0], PALETTE[0], PALETTE[2]])
    ax.axhline(-np.log10(0.05), color=PALETTE[1], lw=1.0, ls="--")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title("Global support")
    add_panel_label(ax, "D")

    fig.tight_layout()
    save_figure(fig, FIG_DIR, "q1_adaptive_sota_summary_panel")


def main() -> None:
    apply_style()
    df = pd.read_csv(TABLE)
    paired = pd.read_csv(PAIRED)
    figure_rmse_comparison(df)
    figure_improvement_heatmap(df)
    figure_paired_forest(paired)
    figure_statistical_support(paired)
    figure_summary_panel(df, paired)
    print(f"wrote figures to {FIG_DIR}")
    for name in [
        "q1_universal_rmse_comparison",
        "q1_relative_improvement_heatmap",
        "q1_paired_delta_forest",
        "q1_universal_statistical_support",
        "q1_adaptive_sota_summary_panel",
    ]:
        print(FIG_DIR / f"{name}.png")
        print(FIG_DIR / f"{name}.pdf")


if __name__ == "__main__":
    main()
