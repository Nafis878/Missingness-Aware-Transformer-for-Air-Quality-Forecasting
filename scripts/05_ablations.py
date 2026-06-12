"""Phase 6: ablation study + robustness experiment.

Usage::

    python scripts/05_ablations.py --config config.yaml              # everything
    python scripts/05_ablations.py --config config.yaml --robustness # robustness only
    python scripts/05_ablations.py --config config.yaml --variants full,no_met

Ablation variants (each trained with seeds from ``ablation.seeds``; the
``full`` seed-42 run reuses the checkpoint from script 04 when present):

* ``full``           proposed model as configured (variant A)
* ``no_miss_embed``  missingness embedding removed (missing = zero-fill only)
* ``variant_B``      attention additionally masked to PM2.5-missing timesteps
* ``no_met``         pollutant inputs only (WS/WD/Temp/RH/BP/SR excluded)
* ``no_time``        calendar time features removed
* ``seq72``/``seq336``  input window 72 h / 336 h (vs 168 h)
* ``single_h24``     single-horizon training (loss weights [0,1,0]);
                     compared on h24 — a full per-horizon grid would triple
                     the training budget for a secondary question.

Robustness: corrupts observed test-period input cells MCAR at the configured
levels (10/30/50%), then re-evaluates the trained seed-42 proposed model and
the two-stage pipelines (re-imputing the corrupted series with imputers fit
on uncorrupted train rows, exactly as in Phase 3). Saves
``<model>_test_miss<level>.npz`` bundles for the robustness figure.

Results: ``outputs/ablation_results.json`` + ablation table via evaluate.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, seed_everything, setup_logging

logger = logging.getLogger("05_ablations")

MET_VARS = ["WS", "WD", "Temp", "RH", "BP", "SR"]
ALL_VARIANTS = ["full", "no_miss_embed", "variant_B", "no_met", "no_time",
                "seq72", "seq336", "single_h24", "miss_dropout"]


def variant_setup(variant: str, base_cfg: dict) -> tuple[dict, dict, int | None]:
    """Return (cfg, model_kwargs, input_length) for one ablation variant."""
    cfg = copy.deepcopy(base_cfg)
    kwargs: dict = {}
    input_length = None
    if variant == "no_miss_embed":
        kwargs["use_missingness_embedding"] = False
    elif variant == "variant_B":
        cfg["model"]["attention_variant"] = "B"
    elif variant == "no_met":
        cfg["data"]["exclude_features"] = list(
            set(cfg["data"]["exclude_features"]) | set(MET_VARS)
        )
    elif variant == "no_time":
        kwargs["use_time_features"] = False
    elif variant == "seq72":
        input_length = 72
    elif variant == "seq336":
        input_length = 336
        # 336-step attention has ~4x the activation memory of 168; halve the
        # batch to keep peak RAM bounded (the 168-step run with batch 64 was
        # killed by the OS at this length). Documented in the paper.
        cfg["train"]["batch_size"] = 32
    elif variant == "single_h24":
        cfg["train"]["horizon_loss_weights"] = [0.0, 1.0, 0.0]
    elif variant == "miss_dropout":
        pass  # handled via train-set wrapper in run_ablations
    elif variant != "full":
        raise ValueError(variant)
    return cfg, kwargs, input_length


def pm25_rmse_per_horizon(out: dict, cfg: dict, scalers: dict) -> dict[str, float]:
    """Unscaled PM2.5 RMSE per horizon from a prediction dict."""
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    _, std = scalers["PM2.5"]
    res = {}
    for hi, h in enumerate(cfg["dataset"]["horizons"]):
        m = out["target_mask"][:, ti, hi] > 0
        err = out["predictions"][m, ti, hi] - out["targets"][m, ti, hi]
        res[f"h{h}"] = float(np.sqrt((err ** 2).mean()) * std)
    return res


def run_ablations(base_cfg: dict, variants: list[str]) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    builder = __import__("04_train_proposed")

    from src.data.dataset import load_scalers, make_datasets
    from src.train import predict, save_stats, train_model

    scalers = load_scalers(
        Path(base_cfg["paths"]["processed_dir"]) / "scalers.json"
    )
    results_path = Path(base_cfg["paths"]["outputs_dir"]) / "ablation_results.json"
    results: dict = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists() else {}
    )

    for variant in variants:
        cfg, kwargs, input_length = variant_setup(variant, base_cfg)
        datasets, stations, _ = make_datasets(cfg, input_length=input_length)
        results.setdefault(variant, {})
        for seed in base_cfg["ablation"]["seeds"]:
            key = f"seed{seed}"
            if key in results[variant]:
                logger.info("%s %s already done, skipping", variant, key)
                continue
            name = f"abl_{variant}"
            seed_everything(seed, base_cfg.get("num_threads"))
            torch.manual_seed(seed)
            model = builder.build_proposed(cfg, n_stations=len(stations), **kwargs)

            train_ds = datasets["train"]
            if variant == "miss_dropout":
                from src.data.dataset import RandomMissingnessAugment

                train_ds = RandomMissingnessAugment(train_ds, max_level=0.5, seed=seed)

            ckpt = Path(cfg["paths"]["checkpoints_dir"]) / f"proposed_seed{seed}.pt"
            if variant == "full" and ckpt.exists():
                logger.info("full seed %d: reusing %s", seed, ckpt)
                model.load_state_dict(
                    torch.load(ckpt, weights_only=False)["model_state"]
                )
                stats = {"reused_checkpoint": str(ckpt)}
            else:
                stats = train_model(model, train_ds, datasets["val"],
                                    cfg, f"{name}_s{seed}", seed)
                save_stats(stats, cfg, f"{name}_s{seed}")
            out = predict(model, datasets["test"], cfg)
            rmse = pm25_rmse_per_horizon(out, cfg, scalers)
            results[variant][key] = {"pm25_rmse": rmse,
                                     "train_time_s": stats.get("train_time_s")}
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            logger.info("ablation %s %s: %s", variant, key, rmse)


def run_robustness(base_cfg: dict) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    builder = __import__("04_train_proposed")

    from src.data.dataset import AirQualityWindowDataset, load_scalers, make_datasets
    from src.data.impute import (
        corrupt_test_inputs,
        corrupt_test_outages,
        impute_full_series,
        replace_inputs,
    )
    from src.models.vanilla_transformer import VanillaTransformer
    from src.data.dataset import feature_columns
    from src.train import predict, save_predictions

    cfg = base_cfg
    seed = cfg["seed"]
    datasets, stations, _ = make_datasets(cfg)
    n_feat = len(feature_columns(cfg))
    n_t = len(cfg["dataset"]["target_pollutants"])
    n_h = len(cfg["dataset"]["horizons"])
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])

    proposed = builder.build_proposed(cfg, n_stations=len(stations))
    proposed.load_state_dict(torch.load(
        ckpt_dir / f"proposed_seed{seed}.pt", weights_only=False)["model_state"])
    models_direct = {"proposed": proposed}

    # proposed + training-time missingness dropout (ablation variant), if trained
    md_ckpt = ckpt_dir / f"abl_miss_dropout_s{seed}_seed{seed}.pt"
    if md_ckpt.exists():
        md = builder.build_proposed(cfg, n_stations=len(stations))
        md.load_state_dict(torch.load(md_ckpt, weights_only=False)["model_state"])
        models_direct["proposed_md"] = md
        clean_path = Path(cfg["paths"]["predictions_dir"]) / "proposed_md_test.npz"
        if not clean_path.exists():
            save_predictions(predict(md, datasets["test"], cfg), cfg, "proposed_md")

    two_stage = {}
    for ts_name in ("two_stage_knn", "two_stage_mice"):
        m = VanillaTransformer(n_feat, len(stations), n_t, n_h, cfg)
        m.load_state_dict(torch.load(
            ckpt_dir / f"{ts_name}_seed{seed}.pt", weights_only=False)["model_state"])
        two_stage[ts_name] = m

    # Two corruption mechanisms (both reported in the paper):
    #   miss = cell-wise MCAR (per spec; easy case for row-wise imputers)
    #   out  = station-outage blocks (mechanism dominating real missingness
    #          per the Phase 1 analysis; hard case for row-wise imputers)
    corruptors = {"miss": corrupt_test_inputs, "out": corrupt_test_outages}
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    for mode, corrupt_fn in corruptors.items():
        for level in cfg["dataset"]["synthetic_missingness"]:
            suffix = f"test_{mode}{int(level * 100)}"
            expected = [pred_dir / f"{m}_{suffix}.npz"
                        for m in (*models_direct, *two_stage)]
            if all(p.exists() for p in expected):
                logger.info("robustness %s: bundles exist, skipping", suffix)
                continue
            corrupted = corrupt_fn(stations, cfg, level, seed)

            ds = AirQualityWindowDataset(corrupted, "test", cfg)
            assert np.array_equal(ds.index, datasets["test"].index), \
                "corruption must not change window enumeration"
            for d_name, d_model in models_direct.items():
                if (pred_dir / f"{d_name}_{suffix}.npz").exists():
                    continue
                save_predictions(predict(d_model, ds, cfg), cfg, d_name, suffix)

            for ts_name, model in two_stage.items():
                if (pred_dir / f"{ts_name}_{suffix}.npz").exists():
                    continue
                method = ts_name.split("_")[-1]
                t0 = time.perf_counter()
                imputed = impute_full_series(corrupted, cfg, method, seed)
                logger.info("%s %s level %.0f%%: re-imputation took %.1fs",
                            method, mode, level * 100, time.perf_counter() - t0)
                ds_imp = AirQualityWindowDataset(replace_inputs(corrupted, imputed),
                                                 "test", cfg)
                save_predictions(predict(model, ds_imp, cfg), cfg, ts_name, suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--variants", default=",".join(ALL_VARIANTS))
    parser.add_argument("--robustness", action="store_true",
                        help="run only the robustness experiment")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    setup_logging("05_ablations", base_cfg["paths"]["logs_dir"])
    seed_everything(base_cfg["seed"], base_cfg.get("num_threads"))

    if args.robustness:
        run_robustness(base_cfg)
        return
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(ALL_VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    run_ablations(base_cfg, variants)
    run_robustness(base_cfg)


if __name__ == "__main__":
    main()
