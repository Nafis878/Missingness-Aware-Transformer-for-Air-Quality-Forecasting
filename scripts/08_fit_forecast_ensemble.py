"""Fit a validation-weighted forecast ensemble from existing model checkpoints.

This script is intentionally post-hoc: it does not retrain base forecasters.
It reconstructs validation prediction bundles from saved checkpoints, learns
non-negative weights on validation PM2.5 for each forecast horizon, then applies
those weights to the saved/test base predictions.

Example:

    python scripts/08_fit_forecast_ensemble.py --config config.yaml

Outputs:

* ``outputs/predictions/ensemble_weighted_val.npz``
* ``outputs/predictions/ensemble_weighted_test.npz``
* ``outputs/tables/ensemble_weights_pm25.csv``
* ``outputs/tables/ensemble_pm25_metrics.csv``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, seed_everything, setup_logging

logger = logging.getLogger("08_fit_forecast_ensemble")


DEFAULT_MODELS = [
    "persistence",
    "seasonal_naive",
    "dlinear",
    "gru_d",
    "proposed",
    "variant_B",
    "proposed_md",
    "hybrid8_masked_transformer",
    "hybrid8_masked_variant_B",
    "hybrid8_masked_proposed_md",
]


def _model_seed_path(cfg: dict[str, Any], model: str, seed: int, split: str) -> Path:
    return (Path(cfg["paths"]["predictions_dir"]) / "seeds"
            / f"{model}_s{seed}_{split}.npz")


def _model_top_path(cfg: dict[str, Any], model: str, split: str) -> Path:
    return Path(cfg["paths"]["predictions_dir"]) / f"{model}_{split}.npz"


def _checkpoint_path(cfg: dict[str, Any], model: str, seed: int) -> Path:
    return Path(cfg["paths"]["checkpoints_dir"]) / f"{model}_seed{seed}.pt"


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))


def _bundle_exists(cfg: dict[str, Any], model: str, split: str) -> bool:
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    if (pred_dir / f"{model}_{split}.npz").exists():
        return True
    seeds = cfg.get("ablation", {}).get("seeds", [cfg["seed"]])
    return any((pred_dir / "seeds" / f"{model}_s{seed}_{split}.npz").exists()
               for seed in seeds)


def _save_bundle(out: dict[str, np.ndarray], cfg: dict[str, Any], name: str,
                 split: str, subdir: str | None = None) -> Path:
    from src.train import save_predictions

    return save_predictions(out, cfg, name, split=split, subdir=subdir)


def _collect_targets(ds, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    from src.train import make_loader

    targets, masks, sids, times = [], [], [], []
    for batch in make_loader(ds, cfg, shuffle=False):
        targets.append(batch["targets"].numpy())
        masks.append(batch["target_mask"].numpy())
        sids.append(batch["station_id"].numpy())
        times.append(batch["anchor_time"].numpy())
    return {
        "targets": np.concatenate(targets),
        "target_mask": np.concatenate(masks),
        "station_id": np.concatenate(sids),
        "anchor_time": np.concatenate(times),
    }


def _base_datasets_for_model(model: str, cfg: dict[str, Any], datasets, stations,
                             seed: int):
    """Return split datasets for a model name, matching training-time wrapping."""
    from src.data.dataset import AirQualityWindowDataset, RandomMissingnessAugment
    from src.data.impute import FfillImputedDataset, impute_full_series, replace_inputs

    if model in ("lstm", "gru"):
        return {k: FfillImputedDataset(v) for k, v in datasets.items()}
    if model in ("dlinear", "gru_d", "patchtst", "proposed", "variant_B"):
        return datasets
    if model == "proposed_md":
        wrapped = dict(datasets)
        wrapped["train"] = RandomMissingnessAugment(
            wrapped["train"], max_level=0.5, seed=seed
        )
        return wrapped
    if model.startswith("two_stage_"):
        method = model.split("_")[-1]
        imputed = impute_full_series(stations, cfg, method, seed)
        stations_imp = replace_inputs(stations, imputed)
        return {
            split: AirQualityWindowDataset(stations_imp, split, cfg)
            for split in ("train", "val", "test")
        }
    if model.startswith("hybrid8_"):
        preserve_mask = model.startswith("hybrid8_masked_")
        imputed = impute_full_series(stations, cfg, "hybrid8", seed)
        stations_imp = replace_inputs(stations, imputed, preserve_mask=preserve_mask)
        wrapped = {
            split: AirQualityWindowDataset(stations_imp, split, cfg)
            for split in ("train", "val", "test")
        }
        if model.endswith("_proposed_md"):
            wrapped["train"] = RandomMissingnessAugment(
                wrapped["train"], max_level=0.5, seed=seed
            )
        return wrapped
    raise ValueError(f"unsupported model for dataset wrapping: {model}")


def _build_model(model: str, cfg: dict[str, Any], n_stations: int):
    from src.data.dataset import feature_columns
    from src.models.factory import build_model
    from src.models.lstm import RNNForecaster

    feats = feature_columns(cfg)
    n_feat = len(feats)
    n_targets = len(cfg["dataset"]["target_pollutants"])
    n_horizons = len(cfg["dataset"]["horizons"])
    target_indices = [feats.index(p) for p in cfg["dataset"]["target_pollutants"]]
    target_feature_idx = feats.index(cfg["dataset"]["primary_target"])

    name = model
    if model.startswith("hybrid8_masked_"):
        name = model[len("hybrid8_masked_"):]
    elif model.startswith("hybrid8_"):
        name = model[len("hybrid8_"):]
    if name == "transformer":
        name = "vanilla_transformer"

    if name in ("lstm", "gru"):
        return RNNForecaster(n_feat, n_stations, n_targets, n_horizons, name, cfg)
    return build_model(
        name, cfg, n_feat, n_stations, n_targets, n_horizons,
        target_feature_idx=target_feature_idx,
        target_indices=target_indices,
    )


def generate_statistical_bundle(model: str, split: str, cfg: dict[str, Any],
                                datasets, stations) -> dict[str, np.ndarray]:
    from src.data.dataset import AirQualityWindowDataset
    from src.data.impute import impute_full_series, replace_inputs
    from src.models.statistical import predict_sarima, predict_statistical

    method = model
    stat_stations = stations
    ds = datasets[split]
    if model.startswith("hybrid8_"):
        method = model[len("hybrid8_"):]
        imputed = impute_full_series(stations, cfg, "hybrid8", cfg["seed"])
        stat_stations = replace_inputs(stations, imputed)
        ds = AirQualityWindowDataset(stat_stations, split, cfg)

    if method == "sarima":
        preds = predict_sarima(ds, stat_stations, cfg)
    else:
        preds = predict_statistical(ds, cfg, method)
    base = _collect_targets(ds, cfg)
    base["predictions"] = preds
    base["latency_ms_per_window"] = np.float64(0.0)
    return base


def ensure_prediction_bundle(model: str, split: str, seed: int | None,
                             cfg: dict[str, Any], datasets, stations) -> Path | None:
    """Ensure a prediction bundle exists; return its path when available."""
    stat_models = {
        "persistence", "seasonal_naive", "sarima",
        "hybrid8_persistence", "hybrid8_seasonal_naive", "hybrid8_sarima",
    }
    if seed is None:
        top = _model_top_path(cfg, model, split)
        if top.exists():
            return top
        if model not in stat_models:
            return None
        logger.info("generating %s %s bundle", model, split)
        out = generate_statistical_bundle(model, split, cfg, datasets, stations)
        return _save_bundle(out, cfg, model, split=split)

    seed_path = _model_seed_path(cfg, model, seed, split)
    if seed_path.exists():
        return seed_path
    if split == "test":
        top = _model_top_path(cfg, model, split)
        if seed == cfg["seed"] and top.exists():
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            seed_path.write_bytes(top.read_bytes())
            return seed_path
        return None

    ckpt = _checkpoint_path(cfg, model, seed)
    if not ckpt.exists():
        logger.warning("%s seed %d: checkpoint missing (%s)", model, seed, ckpt)
        return None

    import torch
    from src.train import predict

    wrapped = _base_datasets_for_model(model, cfg, datasets, stations, seed)
    model_obj = _build_model(model, cfg, n_stations=len(stations))
    try:
        model_obj.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"])
    except RuntimeError as exc:
        logger.warning(
            "%s seed %d: checkpoint incompatible with current model registry (%s)",
            model, seed, exc,
        )
        return None
    out = predict(model_obj, wrapped[split], cfg)
    return _save_bundle(out, cfg, f"{model}_s{seed}", split=split, subdir="seeds")


def average_model_bundle(model: str, split: str, cfg: dict[str, Any],
                         datasets, stations) -> dict[str, np.ndarray] | None:
    """Return seed-mean predictions for a learned model or single statistical bundle."""
    stat_models = {
        "persistence", "seasonal_naive", "sarima",
        "hybrid8_persistence", "hybrid8_seasonal_naive", "hybrid8_sarima",
    }
    if model in stat_models:
        path = ensure_prediction_bundle(model, split, None, cfg, datasets, stations)
        return _load_bundle(path) if path else None

    bundles = []
    for seed in cfg.get("ablation", {}).get("seeds", [cfg["seed"]]):
        path = ensure_prediction_bundle(model, split, int(seed), cfg, datasets, stations)
        if path is not None:
            bundles.append(_load_bundle(path))
    if not bundles:
        logger.warning("%s: no %s bundles available", model, split)
        return None

    ref = bundles[0]
    out = {
        "predictions": np.mean([b["predictions"] for b in bundles], axis=0),
        "targets": ref["targets"],
        "target_mask": ref["target_mask"],
        "station_id": ref["station_id"],
        "anchor_time": ref["anchor_time"],
        "latency_ms_per_window": np.float64(
            np.mean([float(b.get("latency_ms_per_window", 0.0)) for b in bundles])
        ),
    }
    return out


def _assert_aligned(ref: dict[str, np.ndarray], other: dict[str, np.ndarray],
                    name: str) -> None:
    for key in ("targets", "target_mask", "station_id", "anchor_time"):
        if not np.array_equal(ref[key], other[key]):
            raise ValueError(f"{name}: bundle alignment mismatch on {key}")


def _fit_simplex_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nonnegative weights summing to one, minimizing validation MSE."""
    from scipy.optimize import minimize

    k = x.shape[1]
    if k == 1:
        return np.ones(1, dtype=np.float64)

    def obj(w):
        err = x @ w - y
        return float(np.mean(err * err))

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * k
    starts = [np.full(k, 1.0 / k)]
    for j in range(k):
        s = np.zeros(k)
        s[j] = 1.0
        starts.append(s)

    best = None
    for start in starts:
        res = minimize(obj, start, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
        if not res.success:
            logger.warning("ensemble optimizer warning: %s", res.message)
        cand = np.clip(res.x, 0.0, 1.0)
        cand = cand / cand.sum() if cand.sum() > 0 else np.full(k, 1.0 / k)
        score = obj(cand)
        if best is None or score < best[0]:
            best = (score, cand)
    return best[1]


def fit_and_apply_ensemble(
    cfg: dict[str, Any], models: list[str], name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.data.dataset import make_datasets

    datasets, stations, scalers = make_datasets(cfg)
    val_bundles = {}
    test_bundles = {}
    for model in models:
        val = average_model_bundle(model, "val", cfg, datasets, stations)
        test = average_model_bundle(model, "test", cfg, datasets, stations)
        if val is None or test is None:
            logger.warning("%s: skipping (missing val/test)", model)
            continue
        val_bundles[model] = val
        test_bundles[model] = test

    if not val_bundles:
        raise SystemExit("no usable base bundles")

    model_names = list(val_bundles)
    ref_val = val_bundles[model_names[0]]
    ref_test = test_bundles[model_names[0]]
    for model in model_names[1:]:
        _assert_aligned(ref_val, val_bundles[model], f"{model} val")
        _assert_aligned(ref_test, test_bundles[model], f"{model} test")

    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    pol = cfg["dataset"]["primary_target"]
    ti = targets.index(pol)
    weights_by_horizon = {}
    weight_rows = []
    for hi, h in enumerate(horizons):
        m = ref_val["target_mask"][:, ti, hi] > 0
        x = np.column_stack([
            val_bundles[model]["predictions"][m, ti, hi]
            for model in model_names
        ])
        y = ref_val["targets"][m, ti, hi]
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        w = _fit_simplex_weights(x[ok], y[ok])
        weights_by_horizon[hi] = w
        for model, weight in zip(model_names, w):
            weight_rows.append({"horizon": h, "model": model, "weight": weight})

    def apply(split_bundles, ref):
        pred = np.zeros_like(ref["predictions"])
        stack = np.stack([split_bundles[model]["predictions"] for model in model_names])
        for hi in range(len(horizons)):
            w = weights_by_horizon[hi].reshape(-1, 1, 1)
            pred[:, :, hi] = (stack[:, :, :, hi] * w).sum(axis=0)
        return {
            "predictions": pred,
            "targets": ref["targets"],
            "target_mask": ref["target_mask"],
            "station_id": ref["station_id"],
            "anchor_time": ref["anchor_time"],
            "latency_ms_per_window": np.float64(
                sum(float(test_bundles[m].get("latency_ms_per_window", 0.0))
                    for m in model_names)
            ),
        }

    val_out = apply(val_bundles, ref_val)
    test_out = apply(test_bundles, ref_test)
    _save_bundle(val_out, cfg, name, split="val")
    _save_bundle(test_out, cfg, name, split="test")

    mean, std = scalers[pol]
    metric_rows = []
    for split, bundle in (("val", val_out), ("test", test_out)):
        for hi, h in enumerate(horizons):
            m = bundle["target_mask"][:, ti, hi] > 0
            p = bundle["predictions"][m, ti, hi] * std + mean
            y = bundle["targets"][m, ti, hi] * std + mean
            ok = np.isfinite(p) & np.isfinite(y)
            err = p[ok] - y[ok]
            metric_rows.append({
                "split": split,
                "model": name,
                "pollutant": pol,
                "horizon": h,
                "RMSE": float(np.sqrt(np.mean(err * err))),
                "MAE": float(np.mean(np.abs(err))),
                "n": int(ok.sum()),
            })

    weights_df = pd.DataFrame(weight_rows)
    metrics_df = pd.DataFrame(metric_rows)
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(tables_dir / f"{name}_weights_pm25.csv", index=False)
    metrics_df.to_csv(tables_dir / f"{name}_pm25_metrics.csv", index=False)
    return weights_df, metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated base model names")
    parser.add_argument("--name", default="ensemble_weighted")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("08_fit_forecast_ensemble", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    weights, metrics = fit_and_apply_ensemble(cfg, models, args.name)
    logger.info("ensemble weights:\n%s", weights.to_string(index=False))
    logger.info("ensemble metrics:\n%s", metrics.to_string(index=False))
    print(metrics[metrics["split"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
