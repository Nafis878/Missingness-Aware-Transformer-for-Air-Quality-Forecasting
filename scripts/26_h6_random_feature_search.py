"""Validation-only nonlinear H6 residual search.

Uses random ReLU features on top of existing model predictions. The model is
fit on validation targets only and replaces the H6 prediction; H24/H72 use the
strong per-seed convex-intercept stack from script 24.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config


DEFAULT_MEMBERS = [
    "hybrid8_masked_variant_B",
    "hybrid8_masked_variant_B_vanilla_input",
    "hybrid8_transformer",
    "variant_B",
    "hybrid8_masked_proposed_md",
    "two_stage_knn",
    "two_stage_mice",
    "two_stage_saits",
    "dlinear",
    "gru_d",
]

DIMS = [32, 64, 128, 256]
ALPHAS = [10.0, 100.0, 1000.0]
WEIGHT_GAMMAS = [0.0, 2.0]
BLOCKERS = ["two_stage_knn", "two_stage_saits", "hybrid8_masked_proposed_md"]


def _load_helpers():
    path = Path(__file__).with_name("24_validation_calibrated_ensembles.py")
    spec = importlib.util.spec_from_file_location("validation_calibrated_ensembles", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _base_features(x: np.ndarray, stations: np.ndarray, levels: np.ndarray) -> np.ndarray:
    stats = np.column_stack([
        x.mean(axis=1),
        np.median(x, axis=1),
        x.min(axis=1),
        x.max(axis=1),
        x.std(axis=1),
    ])
    centered = x - x.mean(axis=1, keepdims=True)
    station_oh = (stations[:, None] == levels[None, :]).astype(float)
    return np.column_stack([x, x * x, centered, stats, station_oh])


def _standardize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = X.mean(axis=0)
    if scale is None:
        scale = X.std(axis=0)
        scale[scale < 1.0e-8] = 1.0
    return (X - mean) / scale, mean, scale


def _random_relu_features(
    X: np.ndarray,
    dim: int,
    seed: int,
    W: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if W is None or b is None:
        rng = np.random.default_rng(seed)
        W = rng.normal(0.0, 1.0 / np.sqrt(X.shape[1]), size=(X.shape[1], dim))
        b = rng.normal(0.0, 0.25, size=dim)
    Z = np.maximum(0.0, X @ W + b)
    return np.column_stack([np.ones(len(X)), X, Z]), W, b


def _weighted_ridge(X: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray) -> np.ndarray:
    sw = np.sqrt(weights)[:, None]
    Xw = X * sw
    yw = y * sw[:, 0]
    pen = np.eye(X.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.solve(Xw.T @ Xw + alpha * pen, Xw.T @ yw)


def _candidate_name(dim: int, alpha: float, gamma: float) -> str:
    return f"validation_h6_rrelu_d{dim}_a{alpha:g}_g{gamma:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--members", default=",".join(DEFAULT_MEMBERS))
    args = parser.parse_args()

    cfg = load_config(args.config)
    helpers = _load_helpers()
    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    horizons = [int(h) for h in cfg["dataset"]["horizons"]]
    target_idx = cfg["dataset"]["target_pollutants"].index("PM2.5")
    scalers = json.loads((Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text())
    pm25_mean, pm25_std = scalers["PM2.5"]
    pred_dir = Path(cfg["paths"]["predictions_dir"]) / "seeds"
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    members = [
        m for m in members
        if all((pred_dir / f"{m}_s{seed}_val.npz").exists() and (pred_dir / f"{m}_s{seed}_test.npz").exists() for seed in seeds)
    ]

    val = {seed: [helpers._load(cfg, m, seed, "val") for m in members] for seed in seeds}
    test = {seed: [helpers._load(cfg, m, seed, "test") for m in members] for seed in seeds}
    station_levels = np.unique(np.concatenate([val[seed][0]["station_id"].astype(int) for seed in seeds]))
    names = [_candidate_name(d, a, g) for d in DIMS for a in ALPHAS for g in WEIGHT_GAMMAS]
    outputs = {
        name: {
            seed: {k: np.array(v, copy=True) for k, v in test[seed][0].items()}
            for seed in seeds
        }
        for name in names
    }
    val_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for hi, horizon in enumerate(horizons):
            v_rows, vx, vy, vstations, _ = helpers._aligned_arrays(
                val[seed], target_idx, hi, require_observed=True
            )
            t_rows, tx, _, tstations, _ = helpers._aligned_arrays(
                test[seed], target_idx, hi, require_observed=False
            )
            cw, cb = helpers._convex_intercept_fit(vx, vy)
            if horizon != 6:
                pred = tx @ cw + cb
                for name in names:
                    outputs[name][seed]["predictions"][t_rows, target_idx, hi] = pred
                continue

            blocker_idx = [members.index(b) for b in BLOCKERS if b in members]
            blocker_abs = np.min(np.abs(vx[:, blocker_idx] - vy[:, None]), axis=1)
            cutoff = np.median(blocker_abs)
            base_weight = 1.0 + (blocker_abs <= cutoff).astype(float)

            Xv0 = _base_features(vx, vstations, station_levels)
            Xt0 = _base_features(tx, tstations, station_levels)
            Xv, mean, scale = _standardize(Xv0)
            Xt, _, _ = _standardize(Xt0, mean, scale)

            for dim in DIMS:
                Zv, W, b = _random_relu_features(Xv, dim, seed=7919 + 100 * seed + dim)
                Zt, _, _ = _random_relu_features(Xt, dim, seed=0, W=W, b=b)
                for alpha in ALPHAS:
                    for gamma in WEIGHT_GAMMAS:
                        weights = np.ones(len(vy)) if gamma == 0 else (1.0 + gamma * base_weight)
                        beta = _weighted_ridge(Zv, vy, alpha, weights)
                        name = _candidate_name(dim, alpha, gamma)
                        outputs[name][seed]["predictions"][t_rows, target_idx, hi] = Zt @ beta
                        val_pred = Zv @ beta
                        val_rows.append({
                            "candidate": name,
                            "seed": seed,
                            "horizon": horizon,
                            "val_rmse_scaled": float(np.sqrt(np.mean((val_pred - vy) ** 2))),
                        })

    rows: list[dict[str, Any]] = []
    for name, by_seed in outputs.items():
        for seed, bundle in by_seed.items():
            helpers._save_bundle(cfg, name, seed, bundle)
            for hi, horizon in enumerate(horizons):
                rmse, mae = helpers._rmse_real(bundle, target_idx, hi, pm25_mean, pm25_std)
                rows.append({
                    "candidate": name,
                    "seed": seed,
                    "horizon": horizon,
                    "RMSE": rmse,
                    "MAE": mae,
                })

    tables = Path(cfg["paths"]["tables_dir"])
    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["candidate", "horizon"], as_index=False)
        .agg(seeds=("seed", "nunique"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), MAE_mean=("MAE", "mean"))
    )
    pd.DataFrame(val_rows).to_csv(tables / "h6_random_feature_validation_scores.csv", index=False)
    metrics.to_csv(tables / "h6_random_feature_per_seed.csv", index=False)
    summary.to_csv(tables / "h6_random_feature_summary.csv", index=False)
    print(summary.sort_values(["horizon", "RMSE_mean"]).groupby("horizon").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
