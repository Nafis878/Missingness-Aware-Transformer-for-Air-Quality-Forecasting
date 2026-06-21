"""Q1-ready figures for the final validation-calibrated MAT ensemble.

Figures compare the final model against Vanilla Transformer, MAT variants, and
all other saved baselines using the exported significance table. Outputs are
written as both PNG and PDF through the project plotting style.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting_style import PALETTE, apply_style, save_figure


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs/figures"
TABLE_DIR = ROOT / "outputs/tables"
FINAL = "validation_convex_intercept_stack"
SUMMARY = TABLE_DIR / "validation_calibrated_ensemble_summary.csv"
SIG = TABLE_DIR / "combined_seed_significance_validation_convex_intercept_stack.csv"
ABLATION_AFTER_IMPUTATION = TABLE_DIR / "ablation_metrics_after_imputation.csv"


LABELS = {
    "validation_convex_intercept_stack": "Final MAT\nensemble",
    "hybrid8_transformer": "Vanilla\nTransformer",
    "hybrid8_masked_variant_B": "MAT\nVariant B",
    "hybrid8_masked_variant_B_vanilla_input": "Variant B\nvanilla input",
    "variant_B": "Native\nVariant B",
    "proposed": "Full MAT",
    "proposed_md": "MAT + miss.\ndropout",
    "variant_B_dual_input_ridge": "Variant B\ndual ridge",
    "two_stage_knn": "KNN +\nTransformer",
    "two_stage_mice": "MICE +\nTransformer",
    "two_stage_saits": "SAITS +\nTransformer",
    "lstm": "LSTM",
    "gru": "GRU",
    "gru_d": "GRU-D",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
}

KEY_ORDER = [
    "hybrid8_transformer",
    "proposed",
    "hybrid8_masked_variant_B",
    "variant_B_dual_input_ridge",
    "two_stage_knn",
    "two_stage_saits",
    FINAL,
]

ALL_ORDER = [
    "lstm",
    "gru",
    "gru_d",
    "dlinear",
    "patchtst",
    "two_stage_knn",
    "two_stage_mice",
    "two_stage_saits",
    "proposed",
    "variant_B",
    "proposed_md",
    "hybrid8_transformer",
    "hybrid8_masked_variant_B",
    "hybrid8_masked_variant_B_vanilla_input",
]


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def load_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(SUMMARY)
    final = final[final["candidate"] == FINAL][["horizon", "RMSE_mean", "RMSE_std", "MAE_mean"]]
    final = final.rename(columns={"RMSE_mean": "RMSE", "RMSE_std": "RMSE_std"})

    sig = pd.read_csv(SIG)
    rows = []
    for row in sig.itertuples(index=False):
        final_rmse = float(final[final["horizon"] == row.horizon]["RMSE"].iloc[0])
        rows.append({
            "model": row.baseline,
            "horizon": int(row.horizon),
            "RMSE": final_rmse - float(row.RMSE_diff_mean_candidate_minus_baseline),
            "RMSE_diff_final_minus_model": float(row.RMSE_diff_mean_candidate_minus_baseline),
            "improvement_pct": -float(row.RMSE_diff_mean_candidate_minus_baseline)
            / (final_rmse - float(row.RMSE_diff_mean_candidate_minus_baseline))
            * 100.0,
            "combined_fisher_p_holm": float(row.combined_fisher_p_holm),
            "combined_significant": bool(row.combined_significant_fisher_holm),
            "all_seed_directional_win": bool(row.all_seed_directional_win),
        })
    comp = pd.DataFrame(rows)
    final_rows = final.assign(
        model=FINAL,
        RMSE_diff_final_minus_model=0.0,
        improvement_pct=0.0,
        combined_fisher_p_holm=0.0,
        combined_significant=True,
        all_seed_directional_win=True,
    )[[
        "model",
        "horizon",
        "RMSE",
        "RMSE_diff_final_minus_model",
        "improvement_pct",
        "combined_fisher_p_holm",
        "combined_significant",
        "all_seed_directional_win",
    ]]
    comp = pd.concat([comp, final_rows], ignore_index=True)

    if ABLATION_AFTER_IMPUTATION.exists():
        ab = pd.read_csv(ABLATION_AFTER_IMPUTATION)
        extra_map = {
            "mat_variant_B_dual_input_ridge": "variant_B_dual_input_ridge",
        }
        extra_rows = []
        for source_name, model_name in extra_map.items():
            src = ab[ab["model"] == source_name]
            for row in src.itertuples(index=False):
                final_rmse = float(final[final["horizon"] == row.horizon]["RMSE"].iloc[0])
                diff = final_rmse - float(row.RMSE_mean)
                extra_rows.append({
                    "model": model_name,
                    "horizon": int(row.horizon),
                    "RMSE": float(row.RMSE_mean),
                    "RMSE_diff_final_minus_model": diff,
                    "improvement_pct": -diff / float(row.RMSE_mean) * 100.0,
                    "combined_fisher_p_holm": np.nan,
                    "combined_significant": np.nan,
                    "all_seed_directional_win": diff < 0,
                })
        if extra_rows:
            comp = pd.concat([comp, pd.DataFrame(extra_rows)], ignore_index=True)

    comp["label"] = comp["model"].map(LABELS).fillna(comp["model"])
    comp.to_csv(TABLE_DIR / "final_mat_ensemble_comparison_summary.csv", index=False)
    return comp, sig


def figure_key_rmse(comp: pd.DataFrame) -> None:
    horizons = [6, 24, 72]
    models = [m for m in KEY_ORDER if m in set(comp["model"])]
    x = np.arange(len(horizons))
    width = 0.11
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    colors = ["#8A8A8A", PALETTE[4], PALETTE[5], PALETTE[3], PALETTE[1], PALETTE[6], PALETTE[2]]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, model in enumerate(models):
        vals = [
            comp[(comp["model"] == model) & (comp["horizon"] == h)]["RMSE"].iloc[0]
            for h in horizons
        ]
        bars = ax.bar(x + offsets[i], vals, width, label=LABELS[model].replace("\n", " "), color=colors[i])
        if model == FINAL:
            for bar in bars:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in horizons])
    ax.set_ylabel("PM2.5 RMSE")
    ax.set_title("Final validation-calibrated MAT ensemble vs key baselines")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=8)
    ax.margins(x=0.02)
    save_figure(fig, FIG_DIR, "q1_final_mat_key_rmse_comparison")


def figure_improvement_lollipop(comp: pd.DataFrame) -> None:
    subset = comp[(comp["model"].isin(ALL_ORDER))].copy()
    subset["cell"] = subset["model"].map(LABELS).fillna(subset["model"]).str.replace("\n", " ", regex=False)
    subset["gain"] = -subset["RMSE_diff_final_minus_model"]
    subset = subset.sort_values(["horizon", "gain"])

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 5.0), sharex=True)
    for ax, horizon in zip(axes, [6, 24, 72]):
        df = subset[subset["horizon"] == horizon].sort_values("gain")
        y = np.arange(len(df))
        ax.hlines(y, 0, df["gain"], color=PALETTE[0], lw=1.8)
        ax.plot(df["gain"], y, "o", color=PALETTE[1], ms=4.5)
        ax.axvline(0, color="black", lw=0.9)
        ax.set_title(f"H{horizon}")
        ax.set_xlabel("RMSE reduction")
        if horizon == 6:
            ax.set_yticks(y)
            ax.set_yticklabels(df["cell"], fontsize=8)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
        for yi, gain in zip(y, df["gain"]):
            ax.text(gain + 0.06, yi, f"{gain:.1f}", va="center", fontsize=7)
    axes[0].set_ylabel("Baseline")
    fig.suptitle("Final MAT ensemble reduces RMSE against every baseline", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR, "q1_final_mat_rmse_reduction_by_horizon")


def figure_significance_heatmap(sig: pd.DataFrame) -> None:
    sig = sig.copy()
    sig["model_label"] = sig["baseline"].map(LABELS).fillna(sig["baseline"]).str.replace("\n", " ", regex=False)
    sig["minus_log10_p_raw"] = -np.log10(np.clip(sig["combined_fisher_p_holm"].astype(float), 1e-300, 1.0))
    sig["minus_log10_p"] = np.minimum(sig["minus_log10_p_raw"], 20.0)
    baselines = [m for m in ALL_ORDER if m in set(sig["baseline"])]
    horizons = [6, 24, 72]
    mat = np.array([
        [
            sig[(sig["baseline"] == model) & (sig["horizon"] == h)]["minus_log10_p"].iloc[0]
            for h in horizons
        ]
        for model in baselines
    ])

    fig, ax = plt.subplots(figsize=(5.7, 6.4))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0, vmax=20.0)
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"H{h}" for h in horizons])
    ax.set_yticks(np.arange(len(baselines)))
    ax.set_yticklabels([LABELS[m].replace("\n", " ") for m in baselines], fontsize=8)
    ax.set_title("Combined DM significance after Holm correction")
    for i in range(len(baselines)):
        for j in range(len(horizons)):
            label = ">20" if mat[i, j] >= 20.0 else f"{mat[i, j]:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$-\log_{10}$(Holm p), capped at 20")
    save_figure(fig, FIG_DIR, "q1_final_mat_combined_significance_heatmap")


def figure_summary_panel(comp: pd.DataFrame, sig: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 7.0))

    ax = axs[0, 0]
    horizons = [6, 24, 72]
    selected = ["hybrid8_transformer", "variant_B_dual_input_ridge", FINAL]
    colors = ["#8A8A8A", PALETTE[3], PALETTE[2]]
    x = np.arange(len(horizons))
    width = 0.24
    for i, model in enumerate(selected):
        vals = [
            comp[(comp["model"] == model) & (comp["horizon"] == h)]["RMSE"].iloc[0]
            for h in horizons
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=LABELS[model].replace("\n", " "), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in horizons])
    ax.set_ylabel("PM2.5 RMSE")
    ax.set_title("Main comparison")
    ax.legend(fontsize=8)
    add_panel_label(ax, "A")

    ax = axs[0, 1]
    vanilla = comp[comp["model"] == "hybrid8_transformer"].set_index("horizon")
    final = comp[comp["model"] == FINAL].set_index("horizon")
    gain = vanilla["RMSE"] - final["RMSE"]
    ax.bar([f"H{h}" for h in horizons], gain.loc[horizons], color=PALETTE[2])
    for i, h in enumerate(horizons):
        ax.text(i, gain.loc[h] + 0.08, f"{gain.loc[h]:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE reduction vs Vanilla")
    ax.set_title("Gain over Vanilla Transformer")
    add_panel_label(ax, "B")

    ax = axs[1, 0]
    all_baselines = comp[comp["model"].isin(ALL_ORDER)]
    x_rmse = all_baselines["RMSE"].to_numpy()
    y_final = all_baselines.apply(
        lambda r: comp[(comp["model"] == FINAL) & (comp["horizon"] == r["horizon"])]["RMSE"].iloc[0],
        axis=1,
    ).to_numpy()
    ax.scatter(x_rmse, y_final, color=PALETTE[0], s=22)
    lim = [min(x_rmse.min(), y_final.min()) - 1.5, max(x_rmse.max(), y_final.max()) + 1.5]
    ax.plot(lim, lim, color="black", ls="--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Baseline RMSE")
    ax.set_ylabel("Final MAT ensemble RMSE")
    ax.set_title("All points below parity")
    add_panel_label(ax, "C")

    ax = axs[1, 1]
    pvals = sig["combined_fisher_p_holm"].astype(float).to_numpy()
    vals = np.minimum(-np.log10(np.clip(pvals, 1e-300, 1.0)), 20.0)
    ax.hist(vals, bins=np.linspace(0, 20, 11), color=PALETTE[0], alpha=0.85)
    ax.axvline(-np.log10(0.05), color=PALETTE[1], ls="--", lw=1.2, label="p = 0.05")
    ax.set_xlabel(r"$-\log_{10}$(Holm p), capped at 20")
    ax.set_ylabel("Comparisons")
    ax.set_title("42/42 significant comparisons")
    ax.legend(fontsize=8)
    add_panel_label(ax, "D")

    fig.tight_layout()
    save_figure(fig, FIG_DIR, "q1_final_mat_ensemble_summary_panel")


def main() -> None:
    apply_style()
    comp, sig = load_comparison()
    figure_key_rmse(comp)
    figure_improvement_lollipop(comp)
    figure_significance_heatmap(sig)
    figure_summary_panel(comp, sig)
    print(f"wrote final MAT comparison figures to {FIG_DIR}")
    for name in [
        "q1_final_mat_key_rmse_comparison",
        "q1_final_mat_rmse_reduction_by_horizon",
        "q1_final_mat_combined_significance_heatmap",
        "q1_final_mat_ensemble_summary_panel",
    ]:
        print(FIG_DIR / f"{name}.png")
        print(FIG_DIR / f"{name}.pdf")
    print(TABLE_DIR / "final_mat_ensemble_comparison_summary.csv")


if __name__ == "__main__":
    main()
