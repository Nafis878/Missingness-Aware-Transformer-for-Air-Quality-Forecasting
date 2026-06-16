"""CLI runner for the imputation-techniques benchmark (CPU- or GPU-local).

Mirrors ``notebooks/imputation_benchmark.ipynb`` without the Colab upload/download
plumbing, so the same benchmark can be run on a workstation. Writes per-dataset
leaderboards, a combined table, the coverage map, figures, and the artifacts zip
to ``outputs/imputation_benchmark/``.

Examples::

    # classical/ML only, all three networks (the ~1 h CPU run)
    python scripts/run_imputation_benchmark.py --no-deep

    # include the deep family (use a GPU; impractically slow on CPU)
    python scripts/run_imputation_benchmark.py --deep --device cuda

    # quick wiring check
    python scripts/run_imputation_benchmark.py --smoke
"""

from __future__ import annotations

import argparse
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

DATASETS = [("dhaka", "config.yaml"),
            ("beijing", "config_beijing.yaml"),
            ("delhi", "config_delhi.yaml")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", dest="deep", action="store_true",
                    help="include the deep family (needs a GPU to be practical)")
    ap.add_argument("--no-deep", dest="deep", action="store_false")
    ap.set_defaults(deep=False)
    ap.add_argument("--no-slow", dest="slow", action="store_false",
                    help="skip ARIMA/Kalman/SSA/MissForest/SVR/SOM/fuzzy")
    ap.set_defaults(slow=True)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cpu | cuda (auto if omitted)")
    ap.add_argument("--patterns", default="mcar,outage")
    ap.add_argument("--datasets", default="dhaka,beijing,delhi")
    ap.add_argument("--smoke", action="store_true",
                    help="2 stations, 5 fast methods, short slices (Dhaka only)")
    args = ap.parse_args()

    from src import imputation_benchmark as ib
    from src.data.dataset import build_station_arrays

    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"device={device} deep={args.deep} slow={args.slow} smoke={args.smoke}")

    patterns = tuple(p.strip() for p in args.patterns.split(","))
    wanted = {d.strip() for d in args.datasets.split(",")}
    ds_list = [(n, c) for n, c in DATASETS if n in wanted]
    if args.smoke:
        ds_list = ds_list[:1]
    only = {"mean", "forward_fill", "linear_interp", "hour_mean", "knn"} if args.smoke else None

    cov = ib.coverage_map()
    all_board, all_long = [], []
    for name, cfgpath in ds_list:
        if not os.path.exists(cfgpath):
            print(f"[{name}] {cfgpath} missing — skipping")
            continue
        t0 = time.perf_counter()
        cfg = yaml.safe_load(open(cfgpath))
        proc = cfg["paths"]["processed_dir"]
        parquet = os.path.join(proc, "all_stations.parquet")
        if not os.path.exists(parquet):
            print(f"[{name}] {parquet} missing — run scripts/01* first; skipping")
            continue
        sc = json.load(open(os.path.join(proc, "scalers.json")))
        df = pd.read_parquet(parquet)
        stations = build_station_arrays(df, cfg, sc)
        methods, skipped = ib.build_methods(
            cfg, stations, sc, device=device, seed=args.seed, only=only,
            include_slow=args.slow and not args.smoke,
            include_deep=args.deep and not args.smoke)
        print(f"[{name}] {len(methods)} methods, {len(skipped)} skipped", flush=True)
        board, long = ib.run_benchmark(
            cfg, sc, methods, patterns=patterns, holdout_rate=args.holdout,
            seed=args.seed,
            max_stations=2 if args.smoke else None,
            max_slice_len=2000 if args.smoke else None)
        outdir = os.path.join("outputs", "imputation_benchmark", name)
        os.makedirs(outdir, exist_ok=True)
        board.to_csv(os.path.join(outdir, "leaderboard.csv"), index=False)
        long.to_csv(os.path.join(outdir, "per_variable.csv"), index=False)
        pd.DataFrame(skipped).to_csv(os.path.join(outdir, "skipped_methods.csv"), index=False)
        all_board.append(board)
        all_long.append(long)
        print(f"[{name}] done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    if not all_board:
        print("no datasets ran")
        return

    base = "outputs/imputation_benchmark"
    os.makedirs(base, exist_ok=True)
    board_all = pd.concat(all_board, ignore_index=True)
    pd.concat(all_long, ignore_index=True).to_csv(f"{base}/per_variable_all.csv", index=False)
    board_all.to_csv(f"{base}/leaderboard_all.csv", index=False)
    cov.to_csv(f"{base}/coverage_map.csv", index=False)
    print("coverage:", cov["status"].value_counts().to_dict())

    _figures(board_all, base)

    import shutil
    z = shutil.make_archive("imputation_benchmark_artifacts", "zip",
                            root_dir="outputs", base_dir="imputation_benchmark")
    print("wrote", z)


def _figures(board_all: pd.DataFrame, base: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except Exception:
        sns = None
    figdir = f"{base}/figures"
    os.makedirs(figdir, exist_ok=True)

    for name in board_all["dataset"].str.lower().unique():
        sub = board_all[(board_all.dataset.str.lower() == name)
                        & (board_all.pattern == "mcar")].sort_values("overall_std_rmse")
        if sub.empty:
            continue
        fams = sub["family"].astype("category")
        colors = plt.cm.tab20(fams.cat.codes / max(1, fams.cat.categories.size))
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(sub) + 1))
        ax.barh(sub["method"], sub["overall_std_rmse"], color=colors)
        ax.axvline(1.0, ls="--", c="k", lw=1, label="forward-fill")
        ax.set_xlabel("overall standardized RMSE (lower = better)")
        ax.set_title(f"{name.capitalize()} — reconstruction RMSE by method (MCAR)")
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{figdir}/rmse_bar_{name}.png", dpi=150)
        plt.close(fig)

    piv = (board_all[board_all.pattern == "mcar"]
           .pivot_table(index="method", columns="dataset", values="imputability"))
    if piv.shape[1] >= 1 and len(piv):
        fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 3, 0.33 * len(piv) + 1))
        if sns is not None:
            sns.heatmap(piv.sort_values(piv.columns[0]), annot=True, fmt=".2f",
                        center=0, cmap="RdYlGn", ax=ax,
                        cbar_kws={"label": "imputability"})
        else:
            im = ax.imshow(piv.values, cmap="RdYlGn")
            fig.colorbar(im)
        ax.set_title("Imputability vs forward-fill (>0 = beats ffill)")
        fig.tight_layout()
        fig.savefig(f"{figdir}/imputability_heatmap.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
