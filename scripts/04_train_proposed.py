"""Phase 4: train the proposed Missingness-Aware Transformer.

Usage::

    python scripts/04_train_proposed.py --config config.yaml [--seed 42]
        [--seeds 42,43,44] [--name proposed] [--variant A|B] [--miss-dropout]

Trains on the standard (non-imputed) window datasets — the model consumes the
observation mask natively. ``--seeds`` loops over multiple seeds with
file-existence resume (per-seed bundles in ``predictions/seeds/``, canonical
seed also top-level); ``--miss-dropout`` wraps the train set in
:class:`RandomMissingnessAugment` (max_level 0.5) and defaults the name to
``proposed_md``. Saves checkpoint, stats, test predictions, and prints a
PM2.5 RMSE preview against the saved baselines.
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


def run_one_seed(cfg: dict, name: str, seed: int, miss_dropout: bool) -> None:
    """Train one (name, seed) run with file-existence resume."""
    import torch

    from src.data.dataset import RandomMissingnessAugment, make_datasets
    from src.train import predict, save_predictions, save_stats, train_model

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    canonical = seed == cfg["seed"]
    seed_path = pred_dir / "seeds" / f"{name}_s{seed}_test.npz"
    top_path = pred_dir / f"{name}_test.npz"
    if seed_path.exists() and (not canonical or top_path.exists()):
        logger.info("%s seed %d: bundles exist, skipping", name, seed)
        return

    seed_everything(seed, cfg.get("num_threads"))
    datasets, stations, scalers = make_datasets(cfg)
    torch.manual_seed(seed)
    model = build_proposed(cfg, n_stations=len(stations))

    train_ds = datasets["train"]
    if miss_dropout:
        train_ds = RandomMissingnessAugment(train_ds, max_level=0.5, seed=seed)

    stats = train_model(model, train_ds, datasets["val"], cfg, name, seed)
    out = predict(model, datasets["test"], cfg)
    stats["latency_ms_per_window"] = float(out["latency_ms_per_window"])
    stats["attention_variant"] = cfg["model"]["attention_variant"]
    stats["miss_dropout"] = miss_dropout
    save_predictions(out, cfg, f"{name}_s{seed}", subdir="seeds")
    if canonical:
        save_predictions(out, cfg, name)
    save_stats(stats, cfg, name if canonical else f"{name}_s{seed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds; overrides --seed and "
                             "loops with resume")
    parser.add_argument("--name", default=None)
    parser.add_argument("--variant", default=None, choices=["A", "B"])
    parser.add_argument("--miss-dropout", action="store_true",
                        help="train with RandomMissingnessAugment(max 0.5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.variant:
        cfg["model"]["attention_variant"] = args.variant
    name = args.name or ("proposed_md" if args.miss_dropout else "proposed")
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds
             else [args.seed if args.seed is not None else cfg["seed"]])
    setup_logging(f"04_train_proposed_{name}", cfg["paths"]["logs_dir"])

    for seed in seeds:
        run_one_seed(cfg, name, seed, args.miss_dropout)

    # preview against baselines
    from src.data.dataset import load_scalers

    scalers = load_scalers(Path(cfg["paths"]["processed_dir"]) / "scalers.json")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = __import__("03_train_baselines")
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    preview = {}
    for npz in sorted(pred_dir.glob("*_test.npz")):
        preview[npz.stem.replace("_test", "")] = mod.quick_pm25_rmse(npz, cfg, scalers)
    logger.info("PM2.5 test RMSE (ug/m3) including proposed:\n%s",
                json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
