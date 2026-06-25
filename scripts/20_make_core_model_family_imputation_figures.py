"""Focused before/after imputation figures for core implemented model families.

Includes the model families explicitly used in the experiments:
ARIMA/SARIMA, DLinear, GRU/GRU-D, and Transformer/MAT variants.  The source
CSV records which dataset/model pairs are available in the saved artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting_style import PALETTE, apply_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs/figures"
TABLE_DIR = ROOT / "outputs/tables"
SOURCE = TABLE_DIR / "all_model_before_after_imputation_forecasting.csv"

CORE_MODELS = [
    "LSTM",
    "SARIMA",
    "DLinear",
    "PatchTST",
    "GRU",
    "GRU-D",
    "MAT",
    "MAT variant B",
    "MAT + miss-dropout",
]

DISPLAY = {
    "LSTM": "LSTM",
    "SARIMA": "ARIMA/SARIMA",
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "GRU": "GRU",
    "GRU-D": "GRU-D",
    "MAT": "Transformer/MAT",
    "MAT variant B": "Transformer/MAT-B",
    "MAT + miss-dropout": "Transformer/MAT-MD",
}


def load_core() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    df = df[df["model"].isin(CORE_MODELS)].copy()
    df["model_family"] = df["model"].map(DISPLAY)
    df["row_label"] = df["dataset"] + " | " + df["model_family"]
    df["available_pair"] = True

    # Explicit availability grid for the implemented model families.  This is
    # useful because Delhi/Beijing local hybrid8 counterparts are not available
    # for DLinear/GRU in the current artifact set.
    availability_rows = []
    for dataset in ["Dhaka", "Delhi", "Beijing"]:
        for model in CORE_MODELS:
            has_pair = bool(((df["dataset"] == dataset) & (df["model"] == model)).any())
            availability_rows.append({
                "dataset": dataset,
                "model": model,
                "model_family": DISPLAY[model],
                "before_after_pair_available": has_pair,
            })
    pd.DataFrame(availability_rows).to_csv(
        TABLE_DIR / "core_model_before_after_imputation_availability.csv",
        index=False,
    )
    df.to_csv(TABLE_DIR / "core_model_before_after_imputation_forecasting.csv", index=False)
    return df


def availability_figure() -> None:
    avail = pd.read_csv(TABLE_DIR / "core_model_before_after_imputation_availability.csv")
    datasets = ["Dhaka", "Delhi", "Beijing"]
    models = [DISPLAY[m] for m in CORE_MODELS]
    mat = np.array([
        [
            int(avail[(avail["dataset"] == dataset)
                      & (avail["model_family"] == model)]["before_after_pair_available"].iloc[0])
            for dataset in datasets
        ]
        for model in models
    ])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Implemented model before/after pair availability")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "yes" if mat[i, j] else "N/A",
                    ha="center", va="center", fontsize=9,
                    color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pair available")
    save_figure(fig, FIG_DIR, "implemented_model_imputation_pair_availability")


def core_heatmap(df: pd.DataFrame) -> None:
    model_order = [DISPLAY[m] for m in CORE_MODELS]
    row_order = []
    for dataset in ["Dhaka", "Delhi", "Beijing"]:
        for model in model_order:
            label = f"{dataset} | {model}"
            if label in set(df["row_label"]):
                row_order.append(label)
    horizons = [6, 24, 72]
    mat = np.array([
        [
            df[(df["row_label"] == row) & (df["horizon"] == h)]["improvement_pct"].iloc[0]
            if len(df[(df["row_label"] == row) & (df["horizon"] == h)]) else np.nan
            for h in horizons
        ]
        for row in row_order
    ])
    vmax = max(abs(float(np.nanmin(mat))), abs(float(np.nanmax(mat))), 1.0)
    fig_h = max(5.0, 0.42 * len(row_order) + 1.2)
    fig, ax = plt.subplots(figsize=(6.6, fig_h))
    im = ax.imshow(
        mat,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_title("Core models before vs after hybrid_top8 imputation")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.1f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("RMSE improvement (%)")
    save_figure(fig, FIG_DIR, "core_models_before_after_imputation_heatmap")


def dhaka_core_bars(df: pd.DataFrame) -> None:
    sub = df[df["dataset"] == "Dhaka"].copy()
    model_order = [DISPLAY[m] for m in CORE_MODELS if DISPLAY[m] in set(sub["model_family"])]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.6), sharex=True)
    for ax, horizon in zip(axes, [6, 24, 72]):
        hdf = sub[sub["horizon"] == horizon].set_index("model_family").loc[model_order].reset_index()
        x = np.arange(len(hdf))
        width = 0.38
        ax.bar(x - width / 2, hdf["before_RMSE"], width, color="#8A8A8A", label="Before")
        ax.bar(x + width / 2, hdf["after_RMSE"], width, color=PALETTE[0], label="After hybrid_top8")
        for i, row in enumerate(hdf.itertuples(index=False)):
            color = PALETTE[2] if row.improvement_pct > 0 else PALETTE[1]
            ax.text(
                i,
                max(row.before_RMSE, row.after_RMSE) + 1.1,
                f"{row.improvement_pct:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
        ax.set_ylabel(f"{horizon}h RMSE")
        ax.set_title(f"Dhaka core models ({horizon}h)")
    axes[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.42))
    axes[-1].set_xticks(np.arange(len(model_order)))
    axes[-1].set_xticklabels(model_order, rotation=25, ha="right")
    fig.tight_layout()
    save_figure(fig, FIG_DIR, "core_models_dhaka_before_after_imputation_rmse")


def mat_cross_dataset(df: pd.DataFrame) -> None:
    sub = df[df["model"].isin(["MAT", "MAT variant B", "MAT + miss-dropout"])].copy()
    sub = sub.sort_values(["dataset", "model", "horizon"])
    labels = sub["dataset"] + "\n" + sub["model_family"] + "\n" + sub["horizon"].astype(str) + "h"
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
    ax.set_title("Transformer/MAT family before and after imputation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.margins(x=0.005)
    save_figure(fig, FIG_DIR, "core_transformer_family_cross_dataset_imputation_rmse")


def main() -> None:
    apply_style()
    df = load_core()
    core_heatmap(df)
    dhaka_core_bars(df)
    mat_cross_dataset(df)
    availability_figure()
    print(TABLE_DIR / "core_model_before_after_imputation_forecasting.csv")
    print(TABLE_DIR / "core_model_before_after_imputation_availability.csv")
    for name in [
        "core_models_before_after_imputation_heatmap",
        "core_models_dhaka_before_after_imputation_rmse",
        "core_transformer_family_cross_dataset_imputation_rmse",
        "implemented_model_imputation_pair_availability",
    ]:
        print(FIG_DIR / f"{name}.png")
        print(FIG_DIR / f"{name}.pdf")


if __name__ == "__main__":
    main()
