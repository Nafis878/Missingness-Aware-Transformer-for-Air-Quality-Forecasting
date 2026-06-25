"""Refit tabular experts on train+validation after validation selection.

This is a final-model step: hyperparameters must already be selected on the
validation split.  The script trains one ExtraTrees regressor per PM2.5 horizon
on train+validation observed windows and writes a final test prediction bundle.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import AirQualityWindowDataset, feature_columns, make_datasets
from src.train import make_loader, save_predictions
from src.utils import load_config, seed_everything, setup_logging

logger = logging.getLogger("15_refit_tabular_trainval")


def collect_bundle(ds: AirQualityWindowDataset, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    targets, masks, sids, times = [], [], [], []
    for batch in make_loader(ds, cfg, shuffle=False):
        targets.append(batch["targets"].numpy())
        masks.append(batch["target_mask"].numpy())
        sids.append(batch["station_id"].numpy())
        times.append(batch["anchor_time"].numpy())
    targets_arr = np.concatenate(targets)
    return {
        "predictions": np.zeros_like(targets_arr),
        "targets": targets_arr,
        "target_mask": np.concatenate(masks),
        "station_id": np.concatenate(sids),
        "anchor_time": np.concatenate(times),
        "latency_ms_per_window": np.float64(0.0),
    }


def tabular_features(sample: dict[str, Any], cfg: dict[str, Any], feature_mode: str) -> np.ndarray:
    feats = feature_columns(cfg)
    targets = cfg["dataset"]["target_pollutants"]
    primary = cfg["dataset"]["primary_target"]
    pm_idx = feats.index(primary)
    values = sample["values"].numpy()
    mask = sample["mask"].numpy()
    time_feats = sample["time_feats"].numpy()

    lags = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]
    lag_bundle = [1, 6, 24, 72, 168]
    windows = [6, 12, 24, 48, 72, 168]
    if feature_mode == "lean":
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
        lag_bundle = [1, 24, 168]
        windows = [6, 24, 72, 168]

    x: list[float] = []
    for lag in lags:
        x.extend([float(values[-lag, pm_idx]), float(mask[-lag, pm_idx])])

    target_feats = [name for name in targets if name in feats]
    for name in target_feats:
        j = feats.index(name)
        for lag in lag_bundle:
            x.extend([float(values[-lag, j]), float(mask[-lag, j])])

    if feature_mode == "full":
        summary_idx = list(range(values.shape[1]))
    else:
        summary_idx = []
        for name in target_feats:
            summary_idx.append(feats.index(name))
        for j in range(values.shape[1]):
            if j not in summary_idx and len(summary_idx) < 12:
                summary_idx.append(j)

    for j in summary_idx:
        observed = np.where(mask[:, j] > 0, values[:, j], np.nan)
        for width in windows:
            segment = observed[-width:]
            finite = np.isfinite(segment)
            if finite.any():
                finite_vals = segment[finite]
                x.extend([
                    float(np.nanmean(segment)),
                    float(np.nanstd(segment)),
                    float(np.nanmin(segment)),
                    float(np.nanmax(segment)),
                    float(finite_vals[-1]),
                ])
            else:
                x.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            x.append(float(finite.mean()))

    for lag in [6, 24, 72, 168]:
        x.append(float(values[-1, pm_idx] - values[-lag, pm_idx]))
    x.extend(float(v) for v in time_feats[-1])

    station_id = int(sample["station_id"].item())
    for sid in range(32):
        x.append(1.0 if station_id == sid else 0.0)
    return np.asarray(x, dtype=np.float32)


def split_matrix(ds: AirQualityWindowDataset, cfg: dict[str, Any], horizon_idx: int,
                 feature_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_idx = cfg["dataset"]["target_pollutants"].index(cfg["dataset"]["primary_target"])
    rows, y, observed = [], [], []
    for i in range(len(ds)):
        sample = ds[i]
        rows.append(tabular_features(sample, cfg, feature_mode))
        is_obs = bool(sample["target_mask"][target_idx, horizon_idx].item() > 0)
        observed.append(is_obs)
        y.append(float(sample["targets"][target_idx, horizon_idx].item()) if is_obs else np.nan)
    return np.vstack(rows).astype(np.float32), np.asarray(y, dtype=np.float32), np.asarray(observed, dtype=bool)


def rmse_scaled(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, std: float) -> float:
    ok = mask & np.isfinite(pred) & np.isfinite(y)
    err = pred[ok] - y[ok]
    return float(np.sqrt(np.mean(err * err)) * std)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-val-model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--feature-mode", choices=["lean", "full"], default="full")
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("15_refit_tabular_trainval", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))
    datasets, _, scalers = make_datasets(cfg)

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    source_val = pred_dir / f"{args.source_val_model}_val.npz"
    if not source_val.exists():
        raise FileNotFoundError(source_val)

    val_bundle = dict(np.load(source_val))
    test_bundle = collect_bundle(datasets["test"], cfg)
    target_idx = cfg["dataset"]["target_pollutants"].index(cfg["dataset"]["primary_target"])
    std = float(scalers[cfg["dataset"]["primary_target"]][1])
    metrics = []

    for hi, horizon in enumerate(cfg["dataset"]["horizons"]):
        log.info("building train+val matrices for h=%s", horizon)
        x_train, y_train, obs_train = split_matrix(datasets["train"], cfg, hi, args.feature_mode)
        x_val, y_val, obs_val = split_matrix(datasets["val"], cfg, hi, args.feature_mode)
        x_test, y_test, obs_test = split_matrix(datasets["test"], cfg, hi, args.feature_mode)
        x_fit = np.vstack([x_train[obs_train], x_val[obs_val]])
        y_fit = np.concatenate([y_train[obs_train], y_val[obs_val]])

        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            min_samples_leaf=args.min_samples_leaf,
            random_state=cfg["seed"],
            n_jobs=-1,
        )
        t0 = time.perf_counter()
        model.fit(x_fit, y_fit)
        pred = model.predict(x_test).astype(np.float32)
        fit_s = time.perf_counter() - t0
        test_bundle["predictions"][:, target_idx, hi] = pred
        metrics.append({
            "model": args.name,
            "horizon": int(horizon),
            "test_RMSE": rmse_scaled(pred, y_test, obs_test, std),
            "n_fit": int(len(y_fit)),
            "fit_s": round(fit_s, 2),
        })
        log.info("h=%s test_RMSE=%.3f fit=%.1fs", horizon, metrics[-1]["test_RMSE"], fit_s)

    save_predictions(val_bundle, cfg, args.name, split="val")
    save_predictions(test_bundle, cfg, args.name, split="test")
    table_dir = Path(cfg["paths"]["tables_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    out_path = table_dir / f"{args.name}_pm25_metrics.csv"
    pd.DataFrame(metrics).to_csv(out_path, index=False)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
