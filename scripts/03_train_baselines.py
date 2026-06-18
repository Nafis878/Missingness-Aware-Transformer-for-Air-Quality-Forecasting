"""Phase 3: train and evaluate every baseline model.

Usage::

    python scripts/03_train_baselines.py --config config.yaml [--models m1,m2]
        [--seeds 42,43,44] [--force]

Models (canonical-seed predictions saved to
``outputs/predictions/<name>_test.npz``; every seed additionally to
``outputs/predictions/seeds/<name>_s<seed>_test.npz``):

* ``persistence``      last observed value of each target in the window
* ``seasonal_naive``   most recent observed same-hour-of-day value
* ``sarima``           per-station SARIMA on imputed univariate PM2.5
* ``lstm`` / ``gru``   2x128 RNNs on forward-fill+mean imputed inputs
* ``two_stage_knn``    KNNImputer (train-fit) -> vanilla Transformer
* ``two_stage_mice``   IterativeImputer (train-fit) -> vanilla Transformer

Statistical baselines are deterministic and run once; learned models loop
over ``--seeds`` (default ``ablation.seeds``). A (model, seed) combination is
skipped when its prediction bundle already exists (incremental resume),
unless ``--force``.

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

STATISTICAL_MODELS = ["persistence", "seasonal_naive", "sarima"]
HYBRID8_STATISTICAL_MODELS = [
    "hybrid8_persistence", "hybrid8_seasonal_naive", "hybrid8_sarima",
]
LEARNED_MODELS = ["lstm", "gru", "gru_d", "dlinear", "patchtst",
                  "two_stage_knn", "two_stage_mice", "two_stage_saits",
                  "hybrid8_lstm", "hybrid8_gru", "hybrid8_gru_d",
                  "hybrid8_dlinear", "hybrid8_patchtst",
                  "hybrid8_transformer", "hybrid8_proposed",
                  "hybrid8_variant_B", "hybrid8_proposed_md",
                  "hybrid8_masked_transformer", "hybrid8_masked_proposed",
                  "hybrid8_masked_variant_B", "hybrid8_masked_proposed_md"]
ALL_MODELS = STATISTICAL_MODELS + HYBRID8_STATISTICAL_MODELS + LEARNED_MODELS


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
    from src.data.dataset import AirQualityWindowDataset
    from src.data.impute import impute_full_series, replace_inputs
    from src.models.statistical import predict_sarima, predict_statistical
    from src.train import save_predictions, save_stats

    t0 = time.perf_counter()
    method = name
    impute_time = None
    stat_stations = stations
    if name.startswith("hybrid8_"):
        method = name[8:]
        t_imp = time.perf_counter()
        imputed = impute_full_series(stations, cfg, "hybrid8", cfg["seed"])
        impute_time = time.perf_counter() - t_imp
        logger.info("hybrid8 imputation took %.1fs", impute_time)
        stat_stations = replace_inputs(stations, imputed)
        ds = AirQualityWindowDataset(stat_stations, "test", cfg)
    else:
        ds = datasets["test"]

    if method == "sarima":
        preds = predict_sarima(ds, stat_stations, cfg)
    else:
        preds = predict_statistical(ds, cfg, method)
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
    stats = {
        "name": name, "seed": cfg["seed"], "n_parameters": 0,
        "train_time_s": 0.0 if method != "sarima" else round(elapsed, 1),
        "latency_ms_per_window": out["latency_ms_per_window"],
    }
    if impute_time is not None:
        stats["impute_time_s"] = round(impute_time, 1)
        stats["base_method"] = method
    save_stats(stats, cfg, name)


def run_neural(name: str, datasets, stations, cfg, scalers, seed: int) -> None:
    import torch

    from src.data.dataset import AirQualityWindowDataset, feature_columns
    from src.data.impute import FfillImputedDataset, impute_full_series, replace_inputs
    from src.models.lstm import RNNForecaster
    from src.models.vanilla_transformer import VanillaTransformer
    from src.train import predict, save_predictions, save_stats, train_model
    from src.utils import seed_everything

    n_feat = len(feature_columns(cfg))
    n_stations = len(stations)
    n_targets = len(cfg["dataset"]["target_pollutants"])
    n_horizons = len(cfg["dataset"]["horizons"])
    seed_everything(seed, cfg.get("num_threads"))
    torch.manual_seed(seed)

    if name in ("lstm", "gru"):
        model = RNNForecaster(n_feat, n_stations, n_targets, n_horizons, name, cfg)
        wrapped = {k: FfillImputedDataset(v) for k, v in datasets.items()}
    elif name in ("gru_d", "dlinear", "patchtst"):
        from src.models.factory import build_model

        feats = feature_columns(cfg)
        model = build_model(
            name, cfg, n_feat, n_stations, n_targets, n_horizons,
            target_indices=[feats.index(p)
                            for p in cfg["dataset"]["target_pollutants"]],
        )
        wrapped = datasets  # mask-aware: consume the raw window datasets
    elif name in ("two_stage_knn", "two_stage_mice", "two_stage_saits"):
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
    elif name.startswith("hybrid8_"):
        # merged top-8 imputer -> same forecasting architecture on imputed inputs
        from src.models.factory import build_model

        preserve_mask = name.startswith("hybrid8_masked_")
        prefix_len = len("hybrid8_masked_") if preserve_mask else len("hybrid8_")
        backbone_name = name[prefix_len:]
        backbone = {"transformer": "vanilla_transformer"}.get(
            backbone_name, backbone_name
        )
        feats = feature_columns(cfg)
        t0 = time.perf_counter()
        imputed = impute_full_series(stations, cfg, "hybrid8", seed)  # cached -> ~0s
        impute_time = time.perf_counter() - t0
        logger.info("hybrid8 imputation took %.1fs", impute_time)
        stations_imp = replace_inputs(stations, imputed, preserve_mask=preserve_mask)
        wrapped = {
            split: AirQualityWindowDataset(stations_imp, split, cfg)
            for split in ("train", "val", "test")
        }
        if backbone in ("lstm", "gru"):
            model = RNNForecaster(
                n_feat, n_stations, n_targets, n_horizons, backbone, cfg
            )
        else:
            model = build_model(
                backbone, cfg, n_feat, n_stations, n_targets, n_horizons,
                target_indices=[feats.index(p)
                                for p in cfg["dataset"]["target_pollutants"]],
                target_feature_idx=feats.index(cfg["dataset"]["primary_target"]),
            )
        if backbone == "proposed_md":
            from src.data.dataset import RandomMissingnessAugment

            wrapped = dict(wrapped)
            wrapped["train"] = RandomMissingnessAugment(
                wrapped["train"], max_level=0.5, seed=seed
            )
    else:
        raise ValueError(name)

    stats = train_model(model, wrapped["train"], wrapped["val"], cfg, name, seed)
    if name.startswith("two_stage") or name.startswith("hybrid8_"):
        stats["impute_time_s"] = round(impute_time, 1)
    if name.startswith("hybrid8_"):
        stats["base_model"] = backbone_name
        stats["imputer"] = "hybrid_top8"
        stats["preserve_original_mask"] = preserve_mask
    out = predict(model, wrapped["test"], cfg)
    stats["latency_ms_per_window"] = float(out["latency_ms_per_window"])
    save_predictions(out, cfg, f"{name}_s{seed}", subdir="seeds")
    canonical = seed == cfg["seed"]
    if canonical:
        save_predictions(out, cfg, name)
    save_stats(stats, cfg, name if canonical else f"{name}_s{seed}")


def _seed_bundle_done(pred_dir: Path, name: str, seed: int, canonical: bool) -> bool:
    """True when this (model, seed) run's prediction bundles already exist.

    Pre-existing single-seed runs left only the top-level bundle; in that case
    the ``seeds/`` copy is backfilled here (cheap file copy, no inference).
    """
    seed_path = pred_dir / "seeds" / f"{name}_s{seed}_test.npz"
    top_path = pred_dir / f"{name}_test.npz"
    if seed_path.exists():
        return not canonical or top_path.exists()
    if canonical and top_path.exists():
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_bytes(top_path.read_bytes())
        logger.info("backfilled %s from %s", seed_path.name, top_path.name)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", default=",".join(ALL_MODELS),
                        help="comma-separated subset of: " + ", ".join(ALL_MODELS))
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds for learned models "
                             "(default: ablation.seeds)")
    parser.add_argument("--force", action="store_true",
                        help="retrain even when prediction bundles exist")
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
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else list(cfg["ablation"]["seeds"]))
    pred_dir = Path(cfg["paths"]["predictions_dir"])

    statistical_names = set(STATISTICAL_MODELS) | set(HYBRID8_STATISTICAL_MODELS)
    for name in models:
        if name in statistical_names:
            if not args.force and (pred_dir / f"{name}_test.npz").exists():
                logger.info("=== %s: bundle exists, skipping ===", name)
                continue
            logger.info("=== %s ===", name)
            t0 = time.perf_counter()
            run_statistical(name, datasets, stations, cfg)
            logger.info("=== %s done in %.1fs ===", name, time.perf_counter() - t0)
            continue
        for seed in seeds:
            canonical = seed == cfg["seed"]
            if not args.force and _seed_bundle_done(pred_dir, name, seed, canonical):
                logger.info("=== %s seed %d: bundle exists, skipping ===", name, seed)
                continue
            logger.info("=== %s seed %d ===", name, seed)
            t0 = time.perf_counter()
            run_neural(name, datasets, stations, cfg, scalers, seed)
            logger.info("=== %s seed %d done in %.1fs ===",
                        name, seed, time.perf_counter() - t0)

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
