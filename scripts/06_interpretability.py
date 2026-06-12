"""Phase 7: interpretability figures from the trained proposed model.

Usage::

    python scripts/06_interpretability.py --config config.yaml

Loads the seed-42 proposed checkpoint, extracts attention over a
deterministic sample of test windows, and produces:

* ``attention_by_lag``        average forecast-token attention vs lag, with
                              24 h grid lines (learned periodicity evidence)
* ``attention_missingness``   mean attention maps, high- vs low-missingness
                              windows + PM2.5-sparse attention-mass split
* ``head_specialization``     peak lag + entropy per layer/head (table+figure)
* ``attention_seasonal``      monsoon vs winter lag profiles
* ``feature_importance``      permutation importance vs gradient saliency
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting_style import apply_style, save_figure
from src.utils import export_table, load_config, seed_everything, setup_logging

logger = logging.getLogger("06_interpretability")

SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-windows", type=int, default=1024)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("06_interpretability", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))
    apply_style()
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])

    import torch
    from torch.utils.data import Subset

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    builder = __import__("04_train_proposed")

    from src.data.dataset import feature_columns, make_datasets
    from src.interpret import (
        AttentionAggregator,
        gradient_saliency,
        layer_attention,
        permutation_importance,
    )
    from src.train import make_loader

    datasets, stations, _ = make_datasets(cfg)
    feats = feature_columns(cfg)
    pm_idx = feats.index("PM2.5")
    met_idx = [feats.index(v) for v in ("Temp", "RH", "WS") if v in feats]
    L = cfg["dataset"]["input_length"]
    mcfg = cfg["model"]

    model = builder.build_proposed(cfg, n_stations=len(stations))
    ckpt = Path(cfg["paths"]["checkpoints_dir"]) / f"proposed_seed{cfg['seed']}.pt"
    model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"])
    model.eval()
    logger.info("loaded %s", ckpt)

    # deterministic stratified sample of test windows
    test = datasets["test"]
    rng = np.random.default_rng(cfg["seed"])
    sample_idx = np.sort(rng.choice(len(test), size=min(args.max_windows, len(test)),
                                    replace=False))
    loader = make_loader(Subset(test, sample_idx.tolist()), cfg, shuffle=False)

    agg = AttentionAggregator(int(mcfg["n_layers"]), int(mcfg["n_heads"]), L)
    for batch in loader:
        seasons = pd.to_datetime(batch["anchor_time"].numpy(), unit="s").month.map(
            SEASON_OF_MONTH).to_numpy()
        maps = layer_attention(model, batch)
        agg.update(maps, batch, pm_idx, met_idx, seasons)
    logger.info("aggregated attention over %d windows (high=%d, low=%d miss groups)",
                agg.lag_n, agg.group_n["high"], agg.group_n["low"])

    # (a) attention by lag -----------------------------------------------
    prof = agg.lag_profile().mean(axis=(0, 1))           # (L,)
    lags = np.arange(L - 1, -1, -1)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(lags, prof, linewidth=1.0)
    for d in range(0, L, 24):
        ax.axvline(d, color="0.8", linewidth=0.6, zorder=0)
    ax.set_xlabel("Lag behind forecast token (h)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Forecast-token attention by lag (mean over layers/heads/windows)")
    ax.invert_xaxis()
    save_figure(fig, figures_dir, "attention_by_lag")

    # (b) high vs low missingness maps + mass split ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, group in zip(axes, ("low", "high")):
        n = max(agg.group_n[group], 1)
        im = ax.imshow(agg.group_maps[group] / n, aspect="auto", cmap="viridis",
                       origin="lower")
        ax.set_title(f"{group}-missingness windows (n={agg.group_n[group]})")
        ax.set_xlabel("Key timestep")
        ax.set_ylabel("Query timestep")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="attention")
    fig.suptitle("Attention maps by input missingness")
    save_figure(fig, figures_dir, "attention_missingness")

    mass = {
        "n_pm25_sparse_windows": len(agg.mass_pm),
        "mean_attention_mass_on_pm25_observed": float(np.mean(agg.mass_pm)) if agg.mass_pm else None,
        "mean_attention_mass_on_met_only": float(np.mean(agg.mass_met)) if agg.mass_met else None,
    }
    logger.info("PM2.5-sparse attention mass: %s", mass)

    # (c) head specialization ----------------------------------------------
    spec = pd.DataFrame(agg.head_specialization())
    export_table(spec.set_index(["layer", "head"]), tables_dir,
                 "head_specialization",
                 "Per-head attention specialization: peak lag of the "
                 "forecast-token attention profile and its entropy.",
                 "tab:heads", "%.3f")
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(spec["peak_lag_h"], spec["entropy_nats"],
                    c=spec["layer"], cmap="viridis", s=40)
    for _, r in spec.iterrows():
        ax.annotate(f"L{r['layer']}H{r['head']}", (r["peak_lag_h"], r["entropy_nats"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="layer")
    ax.set_xlabel("Peak attention lag (h)")
    ax.set_ylabel("Attention entropy (nats)")
    ax.set_title("Per-head specialization")
    save_figure(fig, figures_dir, "head_specialization")

    # (d) seasonal ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for season in ("Winter", "Monsoon"):
        n = max(agg.season_n[season], 1)
        ax.plot(lags, agg.season_sum[season] / n, linewidth=1.0,
                label=f"{season} (n={agg.season_n[season]})")
    for d in range(0, L, 24):
        ax.axvline(d, color="0.85", linewidth=0.6, zorder=0)
    ax.set_xlabel("Lag behind forecast token (h)")
    ax.set_ylabel("Mean attention weight")
    ax.invert_xaxis()
    ax.legend()
    ax.set_title("Seasonal attention profiles")
    save_figure(fig, figures_dir, "attention_seasonal")

    # feature importance ------------------------------------------------------
    big_idx = rng.choice(len(test), size=min(512, len(test)), replace=False)
    from torch.utils.data import default_collate
    big_batch = default_collate([test[int(i)] for i in big_idx])
    h_idx = cfg["dataset"]["horizons"].index(24)
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    perm = permutation_importance(model, big_batch, feats, ti, h_idx, cfg["seed"])
    sal = gradient_saliency(model, big_batch, feats, ti, h_idx)

    imp = pd.DataFrame({"permutation_dRMSE_scaled": perm,
                        "gradient_saliency": sal}).sort_values(
        "permutation_dRMSE_scaled", ascending=False)
    export_table(imp.round(5), tables_dir, "feature_importance",
                 "Variable importance for 24 h PM2.5 forecasts: permutation "
                 "importance (scaled $\\Delta$RMSE) vs mean absolute gradient "
                 "saliency over observed cells.", "tab:importance", "%.5f")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    imp["permutation_dRMSE_scaled"].plot.barh(ax=axes[0], color="#0072B2")
    axes[0].set_title("Permutation importance (ΔRMSE, scaled)")
    imp["gradient_saliency"].plot.barh(ax=axes[1], color="#D55E00")
    axes[1].set_title("Gradient saliency")
    axes[0].invert_yaxis()
    fig.suptitle("Variable importance for 24 h PM2.5 forecasts")
    fig.tight_layout()
    save_figure(fig, figures_dir, "feature_importance")

    summary = {
        "lag_profile_peaks_h": [int((L - 1) - i) for i in
                                np.argsort(prof)[::-1][:5]],
        "pm25_sparse_attention_mass": mass,
        "top_permutation_features": imp.head(5)["permutation_dRMSE_scaled"].round(4).to_dict(),
    }
    out = Path(cfg["paths"]["outputs_dir"]) / "interpretability_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
