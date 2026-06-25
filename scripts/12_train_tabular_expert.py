"""Train a lag-summary tabular forecasting expert.

This expert is meant as a conservative long-horizon complement to the neural
forecast ensemble.  It trains one ExtraTrees regressor per PM2.5 horizon on
train windows only, using lag, rolling-summary, missingness, calendar, and
station indicators.  Validation and test predictions are then saved in the
same aligned bundle format as the neural models.

Example:

    python scripts/12_train_tabular_expert.py --config config_delhi.yaml
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

logger = logging.getLogger("12_train_tabular_expert")


def collect_target_bundle(ds: AirQualityWindowDataset, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
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
            if name in feats:
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
    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(observed, dtype=bool),
    )


def rmse_unscaled(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, std: float) -> float:
    err = pred[mask] - y[mask]
    return float(np.sqrt(np.mean(err * err)) * std)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default="tabular_extra_full")
    parser.add_argument("--feature-mode", choices=["lean", "full"], default="full")
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger_obj = setup_logging("12_train_tabular_expert", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))

    datasets, _, scalers = make_datasets(cfg)
    target_idx = cfg["dataset"]["target_pollutants"].index(cfg["dataset"]["primary_target"])
    std = float(scalers[cfg["dataset"]["primary_target"]][1])

    bundles = {
        "val": collect_target_bundle(datasets["val"], cfg),
        "test": collect_target_bundle(datasets["test"], cfg),
    }
    metrics = []
    total_fit_s = 0.0

    for hi, horizon in enumerate(cfg["dataset"]["horizons"]):
        logger_obj.info("building tabular matrices for h=%s", horizon)
        x_train, y_train, obs_train = split_matrix(datasets["train"], cfg, hi, args.feature_mode)
        x_val, y_val, obs_val = split_matrix(datasets["val"], cfg, hi, args.feature_mode)
        x_test, y_test, obs_test = split_matrix(datasets["test"], cfg, hi, args.feature_mode)

        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            min_samples_leaf=args.min_samples_leaf,
            random_state=cfg["seed"],
            n_jobs=-1,
        )
        t0 = time.perf_counter()
        model.fit(x_train[obs_train], y_train[obs_train])
        fit_s = time.perf_counter() - t0
        total_fit_s += fit_s

        val_pred = model.predict(x_val).astype(np.float32)
        test_pred = model.predict(x_test).astype(np.float32)
        bundles["val"]["predictions"][:, target_idx, hi] = val_pred
        bundles["test"]["predictions"][:, target_idx, hi] = test_pred

        row = {
            "model": args.name,
            "feature_mode": args.feature_mode,
            "horizon": int(horizon),
            "val_RMSE": rmse_unscaled(val_pred, y_val, obs_val, std),
            "test_RMSE": rmse_unscaled(test_pred, y_test, obs_test, std),
            "n_train": int(obs_train.sum()),
            "n_val": int(obs_val.sum()),
            "n_test": int(obs_test.sum()),
            "fit_s": round(fit_s, 2),
        }
        metrics.append(row)
        logger_obj.info("h=%s val_RMSE=%.3f test_RMSE=%.3f fit=%.1fs",
                        horizon, row["val_RMSE"], row["test_RMSE"], fit_s)

    for split, bundle in bundles.items():
        bundle["latency_ms_per_window"] = np.float64(0.0)
        save_predictions(bundle, cfg, args.name, split=split)

    table_dir = Path(cfg["paths"]["tables_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    out_path = table_dir / f"{args.name}_pm25_metrics.csv"
    pd.DataFrame(metrics).to_csv(out_path, index=False)
    stats_path = Path(cfg["paths"]["checkpoints_dir"]) / f"{args.name}_stats.json"
    stats_path.write_text(
        pd.Series({
            "name": args.name,
            "feature_mode": args.feature_mode,
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "train_time_s": round(total_fit_s, 1),
            "n_parameters": 0,
        }).to_json(indent=2),
        encoding="utf-8",
    )
    logger_obj.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
