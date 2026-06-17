"""Full CPU run of the extended imputation methods, then merge with the base run.

Writes per-method checkpoints under ``outputs/imputation_benchmark_extended/`` (resumable),
then builds the combined 60-method leaderboard, the updated coverage map, and figures.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src import imputation_benchmark_extended as ext          # noqa: E402
from src.imputation_benchmark import aggregate_all, coverage_map  # noqa: E402

EXT_DIR = "outputs/imputation_benchmark_extended"
BASE = "outputs/imputation_benchmark/leaderboard_all.csv"
DATASETS = [("dhaka", "config.yaml"),
            ("beijing", "config_beijing.yaml"),
            ("delhi", "config_delhi.yaml")]


def log(*a):
    print(*a, flush=True)


def main():
    os.makedirs(EXT_DIR, exist_ok=True)
    grand = time.perf_counter()
    for name, cfgp in DATASETS:
        if not os.path.exists(cfgp):
            log(f"[{name}] {cfgp} missing -> skip"); continue
        cfg = yaml.safe_load(open(cfgp)); cfg["dataset_name"] = name
        proc = cfg["paths"]["processed_dir"]
        parq = os.path.join(proc, "all_stations.parquet")
        if not os.path.exists(parq):
            log(f"[{name}] {parq} missing -> skip"); continue
        sc = json.load(open(os.path.join(proc, "scalers.json")))
        log(f"\n===== {name.upper()} =====")
        t0 = time.perf_counter()
        ext.run_extended_resumable(cfg, sc, EXT_DIR, seed=42, include_deep=True,
                                   device="cpu", log=log)
        log(f"[{name}] dataset done in {(time.perf_counter()-t0)/60:.1f} min")

    log("\n--- aggregate + merge ---")
    ext_board = aggregate_all(EXT_DIR)
    ext_board.to_csv(os.path.join(EXT_DIR, "leaderboard_extended.csv"), index=False)
    log(f"extended methods: {ext_board.method.nunique()} | rows: {len(ext_board)}")

    if os.path.exists(BASE):
        base = pd.read_csv(BASE)
        combined = (pd.concat([base, ext_board], ignore_index=True)
                      .drop_duplicates(["dataset", "method", "pattern"], keep="last"))
        combined.to_csv(os.path.join(EXT_DIR, "leaderboard_all_combined.csv"), index=False)
        log(f"combined methods: {combined.method.nunique()} | rows: {len(combined)}")
    else:
        combined = ext_board

    cov = coverage_map().set_index("technique")
    cov.update(ext.extended_coverage_map().set_index("technique"))
    cov = cov.reset_index()
    cov.to_csv(os.path.join(EXT_DIR, "coverage_map_updated.csv"), index=False)
    log("coverage: " + str(cov["status"].value_counts().to_dict()))

    try:
        sys.path.insert(0, "scripts")
        from run_imputation_benchmark import _figures
        clean = combined[(combined.method != "csdi")
                         & (combined.overall_std_rmse.abs() <= 5)].copy()
        _figures(clean, EXT_DIR)
        log("figures written")
    except Exception as exc:
        log(f"figures skipped: {exc}")

    log(f"\n=== ALL DONE in {(time.perf_counter()-grand)/60:.1f} min ===")
    # quick top-3 preview per dataset/pattern (extended methods only)
    for ds in ext_board.dataset.unique():
        for pat in sorted(ext_board.pattern.unique()):
            sub = ext_board[(ext_board.dataset == ds) & (ext_board.pattern == pat)]
            if sub.empty:
                continue
            top = sub.sort_values("imputability", ascending=False).head(3)
            log(f"  {ds}/{pat}: " + ", ".join(
                f"{m}={i:+.3f}" for m, i in zip(top.method, top.imputability)))


if __name__ == "__main__":
    main()
