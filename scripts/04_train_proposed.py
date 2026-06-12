"""Phase 4: train the proposed Missingness-Aware Transformer.

Usage::

    python scripts/04_train_proposed.py --config config.yaml [--seed 42]
        [--name proposed] [--variant A|B]

Trains on the standard (non-imputed) window datasets — the model consumes the
observation mask natively. Saves checkpoint, stats, test predictions, and
prints a PM2.5 RMSE preview against the saved baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, seed_everything, setup_logging

logger = logging.getLogger("04_train_proposed")


def build_proposed(cfg: dict, n_stations: int, **model_kwargs):
    """Construct the proposed model from config (shared with script 05)."""
    from src.data.dataset import feature_columns
    from src.models.missingness_transformer import MissingnessTransformer

    feats = feature_columns(cfg)
    return MissingnessTransformer(
        n_features=len(feats),
        n_stations=n_stations,
        n_targets=len(cfg["dataset"]["target_pollutants"]),
        n_horizons=len(cfg["dataset"]["horizons"]),
        cfg=cfg,
        target_feature_idx=feats.index(cfg["dataset"]["primary_target"]),
        **model_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default="proposed")
    parser.add_argument("--variant", default=None, choices=["A", "B"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.variant:
        cfg["model"]["attention_variant"] = args.variant
    seed = args.seed if args.seed is not None else cfg["seed"]
    setup_logging(f"04_train_proposed_{args.name}_seed{seed}", cfg["paths"]["logs_dir"])
    seed_everything(seed, cfg.get("num_threads"))

    import torch

    from src.data.dataset import make_datasets
    from src.train import predict, save_predictions, save_stats, train_model

    datasets, stations, scalers = make_datasets(cfg)
    torch.manual_seed(seed)
    model = build_proposed(cfg, n_stations=len(stations))

    stats = train_model(model, datasets["train"], datasets["val"], cfg, args.name, seed)
    out = predict(model, datasets["test"], cfg)
    stats["latency_ms_per_window"] = float(out["latency_ms_per_window"])
    stats["attention_variant"] = cfg["model"]["attention_variant"]
    save_predictions(out, cfg, args.name)
    save_stats(stats, cfg, args.name)

    # preview against baselines
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = __import__("03_train_baselines")
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    preview = {}
    for npz in sorted(pred_dir.glob("*_test.npz")):
        name = npz.stem.replace("_test", "")
        preview[name] = mod.quick_pm25_rmse(npz, cfg, scalers)
    logger.info("PM2.5 test RMSE (ug/m3) including proposed:\n%s",
                json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
