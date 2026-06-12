"""Phase 8: regenerate EVERY paper figure and table in one run.

Usage::

    python scripts/07_make_paper_assets.py --config config.yaml

Pure regeneration from saved artifacts (processed parquet, prediction
bundles, stats jsons, ablation results) — no training. Produces:

* Table 1  dataset + missingness summary        (table1_dataset_summary)
* Table 2  main results, PM2.5 all models        (main_results_pm25 + significance)
* Table 3  ablations, mean ± std over 3 seeds    (table3_ablations)
* Table 4  CPU efficiency                        (efficiency)
* Figures  missingness heatmap & monthly profile, robustness curve,
           seasonal performance, example forecasts, attention analyses,
           feature importance.

Missingness figures come from re-invoking script 02's functions; evaluation
tables/figures from src.evaluate; attention figures from script 06 artifacts
(re-run script 06 to refresh those — it needs the trained checkpoint).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import export_table, load_config, seed_everything, setup_logging

logger = logging.getLogger("07_make_paper_assets")


def _import_script(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table1_dataset_summary(cfg: dict) -> None:
    """Table 1: stations, coverage, observations, missingness for key vars."""
    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "all_stations.parquet")
    key_vars = ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2", "Temp", "RH"]
    rows = []
    for st, grp in df.groupby("station"):
        rows.append({
            "Station": st,
            "First hour": str(grp["datetime"].min())[:10],
            "Hours": len(grp),
            "PM2.5 obs": int(grp["PM2.5"].notna().sum()),
            "PM2.5 miss (%)": round(grp["PM2.5"].isna().mean() * 100, 1),
            "All-vars miss (%)": round(grp[key_vars].isna().to_numpy().mean() * 100, 1),
        })
    tbl = pd.DataFrame(rows).set_index("Station")
    tbl.loc["TOTAL"] = ["-", tbl["Hours"].sum(), tbl["PM2.5 obs"].sum(),
                        round(df["PM2.5"].isna().mean() * 100, 1),
                        round(df[key_vars].isna().to_numpy().mean() * 100, 1)]
    export_table(tbl, Path(cfg["paths"]["tables_dir"]), "table1_dataset_summary",
                 "Dataset summary: per-station hourly coverage and missingness "
                 "after cleaning (2022--2024, 16 stations).", "tab:dataset", "%.1f")


def table3_ablations(cfg: dict) -> None:
    """Table 3: ablation PM2.5 RMSE mean ± std over seeds."""
    path = Path(cfg["paths"]["outputs_dir"]) / "ablation_results.json"
    if not path.exists():
        logger.warning("no ablation_results.json yet — skipping Table 3")
        return
    results = json.loads(path.read_text(encoding="utf-8"))
    horizons = [f"h{h}" for h in cfg["dataset"]["horizons"]]
    rows = {}
    for variant, seeds in results.items():
        vals = {h: [s["pm25_rmse"][h] for s in seeds.values()] for h in horizons}
        rows[variant] = {
            h: f"{np.mean(v):.2f} ± {np.std(v):.2f}" for h, v in vals.items()
        }
        if variant == "single_h24":
            # heads for h6/h72 carry zero loss weight -> untrained, not comparable
            for h in horizons:
                if h != "h24":
                    rows[variant][h] = "—"
        rows[variant]["seeds"] = len(seeds)
    tbl = pd.DataFrame(rows).T
    order = [v for v in ("full", "no_miss_embed", "variant_B", "no_met", "no_time",
                         "seq72", "seq336", "single_h24", "miss_dropout")
             if v in tbl.index]
    tbl = tbl.loc[order]
    export_table(tbl, Path(cfg["paths"]["tables_dir"]), "table3_ablations",
                 "Ablation study: PM2.5 test RMSE "
                 "(\\si{\\micro\\gram\\per\\cubic\\metre}), mean $\\pm$ std over "
                 "3 seeds.", "tab:ablations", "%s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-interpretability", action="store_true",
                        help="skip re-running attention extraction (slow-ish)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("07_make_paper_assets", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))

    from src.plotting_style import apply_style

    apply_style()

    # Tables 1 & 3 (local builders)
    table1_dataset_summary(cfg)
    table3_ablations(cfg)

    # Missingness figures + tables (script 02 functions on the processed parquet)
    m02 = _import_script("02_missingness_analysis")
    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "all_stations.parquet")
    meas = cfg["data"]["measurement_cols"]
    m02.missingness_rate_table(df, meas, Path(cfg["paths"]["tables_dir"]))
    m02.missingness_heatmaps(df, meas, Path(cfg["paths"]["figures_dir"]))
    m02.seasonal_breakdown(df, meas, Path(cfg["paths"]["tables_dir"]),
                           Path(cfg["paths"]["figures_dir"]))

    # Table 2, 4 + evaluation figures (incl. robustness + example forecasts)
    from src.evaluate import run_evaluation

    run_evaluation(cfg)

    # Attention/importance figures
    if not args.skip_interpretability:
        m06 = _import_script("06_interpretability")
        sys.argv = ["06_interpretability"] + (
            ["--config", args.config] if args.config else []
        )
        m06.main()

    fig_dir = Path(cfg["paths"]["figures_dir"])
    tab_dir = Path(cfg["paths"]["tables_dir"])
    logger.info("DONE: %d figure files, %d table files",
                len(list(fig_dir.glob("*"))), len(list(tab_dir.glob("*"))))


if __name__ == "__main__":
    main()
