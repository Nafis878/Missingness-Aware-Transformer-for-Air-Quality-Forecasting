"""Phase 3: train and evaluate every baseline model.

Usage::

    python scripts/03_train_baselines.py --config config.yaml [--models m1,m2]

Models (all predictions saved to ``outputs/predictions/<name>_test.npz``):

* ``persistence``      last observed value of each target in the window
* ``seasonal_naive``   most recent observed same-hour-of-day value
* ``sarima``           per-station SARIMA on imputed univariate PM2.5
* ``lstm`` / ``gru``   2x128 RNNs on forward-fill+mean imputed inputs
* ``two_stage_knn``    KNNImputer (train-fit) -> vanilla Transformer
* ``two_stage_mice``   IterativeImputer (train-fit) -> vanilla Transformer

A quick PM2.5 RMSE preview (per horizon, unscaled ug/m3) is printed at the
end; the full evaluation (all metrics, significance tests) is Phase 5.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, seed_everything, setup_logging

logger = logging.getLogger("03_train_baselines")

ALL_MODELS = ["persistence", "seasonal_naive", "sarima", "lstm", "gru",
              "two_stage_knn", "two_stage_mice"]


def quick_pm25_rmse(npz_path: Path, cfg: dict, scalers: dict) -> dict[str, float]:
    """Unscaled PM2.5 RMSE per horizon over observed targets (preview only)."""
    data = np.load(npz_path)
    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    ti = targets.index(cfg["dataset"]["primary_target"])
    _, std = scalers[cfg["dataset"]["primary_target"]]
    out = {}
    for hi, h in enumerate(horizons):
        m = data["target_mask"][:, ti, hi] > 0
        p, y = data["predictions"][m, ti, hi], data["targets"][m, ti, hi]
        ok = np.isfinite(p)
        out[f"h{h}"] = float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)) * std)
    return out


def run_statistical(name: str, datasets, stations, cfg) -> None:
    from src.models.statistical import predict_sarima, predict_statistical
    from src.train import save_predictions, save_stats

    ds = datasets["test"]
    t0 = time.perf_counter()
    if name == "sarima":
        preds = predict_sarima(ds, stations, cfg)
    else:
        preds = predict_statistical(ds, cfg, name)
    elapsed = time.perf_counter() - t0

    # align with the standard prediction-bundle format
    import torch
    from src.train import make_loader

    targets, masks, sids, times = [], [], [], []
    for batch in make_loader(ds, cfg, shuffle=False):
        targets.append(batch["targets"].numpy())
        masks.append(batch["target_mask"].numpy())
        sids.append(batch["station_id"].numpy())
        times.append(batch["anchor_time"].numpy())
    out = {
        "predictions": preds,
        "targets": np.concatenate(targets),
        "target_mask": np.concatenate(masks),
        "station_id": np.concatenate(sids),
        "anchor_time": np.concatenate(times),
        "latency_ms_per_window": np.float64(elapsed / max(len(ds), 1) * 1000),
    }
    save_predictions(out, cfg, name)
    save_stats(
        {"name": name, "seed": cfg["seed"], "n_parameters": 0,
         "train_time_s": 0.0 if name != "sarima" else round(elapsed, 1),
         "latency_ms_per_window": out["latency_ms_per_window"]},
        cfg, name,
    )


def run_neural(name: str, datasets, stations, cfg, scalers) -> None:
    import torch

    from src.data.dataset import AirQualityWindowDataset, feature_columns
    from src.data.impute import FfillImputedDataset, impute_full_series, replace_inputs
    from src.models.lstm import RNNForecaster
    from src.models.vanilla_transformer import VanillaTransformer
    from src.train import predict, save_predictions, save_stats, train_model

    n_feat = len(feature_columns(cfg))
    n_stations = len(stations)
    n_targets = len(cfg["dataset"]["target_pollutants"])
    n_horizons = len(cfg["dataset"]["horizons"])
    seed = cfg["seed"]
    torch.manual_seed(seed)

    if name in ("lstm", "gru"):
        model = RNNForecaster(n_feat, n_stations, n_targets, n_horizons, name, cfg)
        wrapped = {k: FfillImputedDataset(v) for k, v in datasets.items()}
    elif name in ("two_stage_knn", "two_stage_mice"):
        method = name.split("_")[-1]
        t0 = time.perf_counter()
        imputed = impute_full_series(stations, cfg, method, seed)
        impute_time = time.perf_counter() - t0
        logger.info("%s imputation took %.1fs", method, impute_time)
        stations_imp = replace_inputs(stations, imputed)
        wrapped = {
            split: AirQualityWindowDataset(stations_imp, split, cfg)
            for split in ("train", "val", "test")
        }
        model = VanillaTransformer(n_feat, n_stations, n_targets, n_horizons, cfg)
    else:
        raise ValueError(name)

    stats = train_model(model, wrapped["train"], wrapped["val"], cfg, name, seed)
    if name.startswith("two_stage"):
        stats["impute_time_s"] = round(impute_time, 1)
    out = predict(model, wrapped["test"], cfg)
    stats["latency_ms_per_window"] = float(out["latency_ms_per_window"])
    save_predictions(out, cfg, name)
    save_stats(stats, cfg, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", default=",".join(ALL_MODELS),
                        help="comma-separated subset of: " + ", ".join(ALL_MODELS))
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("03_train_baselines", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))

    from src.data.dataset import make_datasets

    datasets, stations, scalers = make_datasets(cfg)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")

    for name in models:
        logger.info("=== %s ===", name)
        t0 = time.perf_counter()
        if name in ("persistence", "seasonal_naive", "sarima"):
            run_statistical(name, datasets, stations, cfg)
        else:
            run_neural(name, datasets, stations, cfg, scalers)
        logger.info("=== %s done in %.1fs ===", name, time.perf_counter() - t0)

    # preview table
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    preview = {}
    for name in models:
        path = pred_dir / f"{name}_test.npz"
        if path.exists():
            preview[name] = quick_pm25_rmse(path, cfg, scalers)
    logger.info("PM2.5 test RMSE preview (ug/m3):\n%s", json.dumps(preview, indent=2))
    (Path(cfg["paths"]["outputs_dir"]) / "baseline_preview.json").write_text(
        json.dumps(preview, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
