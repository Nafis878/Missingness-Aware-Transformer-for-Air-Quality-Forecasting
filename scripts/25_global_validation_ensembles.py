"""Shared-seed and polynomial validation ensembles for H6 blocker checks.

This is a no-fine-tuning follow-up to ``24_validation_calibrated_ensembles``.
It fits calibration rules only on validation prediction bundles, then writes
test prediction bundles suitable for the existing significance script.
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

POLY_ALPHAS = [0.1, 1.0, 10.0, 100.0]


def _load_helpers():
    path = Path(__file__).with_name("24_validation_calibrated_ensembles.py")
    spec = importlib.util.spec_from_file_location("validation_calibrated_ensembles", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _poly_features(
    x: np.ndarray,
    stations: np.ndarray,
    station_levels: np.ndarray,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = np.column_stack([
        x.mean(axis=1),
        np.median(x, axis=1),
        x.min(axis=1),
        x.max(axis=1),
        x.std(axis=1),
    ])
    centered = x - x.mean(axis=1, keepdims=True)
    station_oh = (stations[:, None] == station_levels[None, :]).astype(float)
    feat = np.column_stack([x, x * x, centered, stats, station_oh])
    if mean is None:
        mean = feat.mean(axis=0)
    if scale is None:
        scale = feat.std(axis=0)
        scale[scale < 1.0e-8] = 1.0
    feat = (feat - mean) / scale
    feat = np.column_stack([np.ones(len(feat)), feat])
    return feat, mean, scale


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    pen = np.eye(X.shape[1], dtype=float)
    pen[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + alpha * pen, X.T @ y)


def _bundle_path(cfg: dict[str, Any], model: str, seed: int, split: str) -> Path:
    return Path(cfg["paths"]["predictions_dir"]) / "seeds" / f"{model}_s{seed}_{split}.npz"


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
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    members = [
        m for m in members
        if all(_bundle_path(cfg, m, seed, "val").exists() and _bundle_path(cfg, m, seed, "test").exists() for seed in seeds)
    ]
    if not members:
        raise SystemExit("No complete val/test members found.")

    val = {seed: [helpers._load(cfg, m, seed, "val") for m in members] for seed in seeds}
    test = {seed: [helpers._load(cfg, m, seed, "test") for m in members] for seed in seeds}
    station_levels = np.unique(np.concatenate([val[seed][0]["station_id"].astype(int) for seed in seeds]))

    candidates = ["validation_global_convex_intercept", "validation_global_huber_convex"]
    candidates += [f"validation_global_h6_poly_alpha{a:g}" for a in POLY_ALPHAS]
    candidates += [f"validation_crossseed_h6_poly_alpha{a:g}" for a in POLY_ALPHAS]
    outputs = {
        cand: {
            seed: {k: np.array(v, copy=True) for k, v in test[seed][0].items()}
            for seed in seeds
        }
        for cand in candidates
    }
    weights_rows: list[dict[str, Any]] = []

    for hi, horizon in enumerate(horizons):
        vx_all, vy_all, vs_all = [], [], []
        for seed in seeds:
            _, vx, vy, vs, _ = helpers._aligned_arrays(val[seed], target_idx, hi, require_observed=True)
            vx_all.append(vx)
            vy_all.append(vy)
            vs_all.append(vs)
        vxg = np.vstack(vx_all)
        vyg = np.concatenate(vy_all)
        vsg = np.concatenate(vs_all)

        base_w, base_b = helpers._convex_intercept_fit(vxg, vyg)
        huber_w, huber_b = helpers._huber_convex_fit(vxg, vyg, init_w=base_w, init_b=base_b)

        for seed in seeds:
            t_rows, tx, _, ts, _ = helpers._aligned_arrays(test[seed], target_idx, hi, require_observed=False)
            outputs["validation_global_convex_intercept"][seed]["predictions"][t_rows, target_idx, hi] = (
                tx @ base_w + base_b
            )
            outputs["validation_global_huber_convex"][seed]["predictions"][t_rows, target_idx, hi] = (
                tx @ huber_w + huber_b
            )

        weights_rows.append({
            "method": "validation_global_convex_intercept",
            "seed": "all",
            "horizon": horizon,
            "intercept": base_b,
            **{f"w_{m}": float(w) for m, w in zip(members, base_w)},
        })
        weights_rows.append({
            "method": "validation_global_huber_convex",
            "seed": "all",
            "horizon": horizon,
            "intercept": huber_b,
            **{f"w_{m}": float(w) for m, w in zip(members, huber_w)},
        })

        if horizon != 6:
            for alpha in POLY_ALPHAS:
                for prefix in ("validation_global_h6_poly", "validation_crossseed_h6_poly"):
                    cand = f"{prefix}_alpha{alpha:g}"
                    for seed in seeds:
                        t_rows, tx, _, _, _ = helpers._aligned_arrays(test[seed], target_idx, hi, require_observed=False)
                        outputs[cand][seed]["predictions"][t_rows, target_idx, hi] = tx @ base_w + base_b
            continue

        Xg, fg_mean, fg_scale = _poly_features(vxg, vsg, station_levels)
        for alpha in POLY_ALPHAS:
            beta = _ridge(Xg, vyg, alpha)
            cand = f"validation_global_h6_poly_alpha{alpha:g}"
            for seed in seeds:
                t_rows, tx, _, ts, _ = helpers._aligned_arrays(test[seed], target_idx, hi, require_observed=False)
                Xt, _, _ = _poly_features(tx, ts, station_levels, fg_mean, fg_scale)
                outputs[cand][seed]["predictions"][t_rows, target_idx, hi] = Xt @ beta
            weights_rows.append({
                "method": cand,
                "seed": "all",
                "horizon": horizon,
                "alpha": alpha,
            })

        for alpha in POLY_ALPHAS:
            cand = f"validation_crossseed_h6_poly_alpha{alpha:g}"
            for seed in seeds:
                train_seeds = [s for s in seeds if s != seed]
                vx_train = np.vstack([vx_all[seeds.index(s)] for s in train_seeds])
                vy_train = np.concatenate([vy_all[seeds.index(s)] for s in train_seeds])
                vs_train = np.concatenate([vs_all[seeds.index(s)] for s in train_seeds])
                Xtr, mean, scale = _poly_features(vx_train, vs_train, station_levels)
                beta = _ridge(Xtr, vy_train, alpha)
                t_rows, tx, _, ts, _ = helpers._aligned_arrays(test[seed], target_idx, hi, require_observed=False)
                Xt, _, _ = _poly_features(tx, ts, station_levels, mean, scale)
                outputs[cand][seed]["predictions"][t_rows, target_idx, hi] = Xt @ beta
                weights_rows.append({
                    "method": cand,
                    "seed": seed,
                    "horizon": horizon,
                    "alpha": alpha,
                })

    rows: list[dict[str, Any]] = []
    for cand, by_seed in outputs.items():
        for seed, bundle in by_seed.items():
            helpers._save_bundle(cfg, cand, seed, bundle)
            for hi, horizon in enumerate(horizons):
                rmse, mae = helpers._rmse_real(bundle, target_idx, hi, pm25_mean, pm25_std)
                rows.append({
                    "candidate": cand,
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
    metrics.to_csv(tables / "global_validation_ensemble_per_seed.csv", index=False)
    summary.to_csv(tables / "global_validation_ensemble_summary.csv", index=False)
    pd.DataFrame(weights_rows).to_csv(tables / "global_validation_ensemble_weights.csv", index=False)
    print(f"members: {', '.join(members)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
