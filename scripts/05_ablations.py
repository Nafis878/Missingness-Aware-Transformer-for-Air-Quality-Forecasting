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
* ``no_station_embed`` station identity embedding removed
* ``no_pos_enc``     sequence positional encoding removed
* ``no_met``         pollutant inputs only (WS/WD/Temp/RH/BP/SR excluded)
* ``no_time``        calendar time features removed
* ``seq72``/``seq336``  input window 72 h / 336 h (vs 168 h)
* ``heads4``/``heads16`` attention-head count 4 / 16 (vs 8)
* ``layers2``/``layers4`` Transformer depth 2 / 4 (vs 3)
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
ALL_VARIANTS = [
    "full",
    "variant_B",
    "no_attention_mask",
    "no_miss_embed",
    "no_station_embed",
    "no_pos_enc",
    "no_met",
    "no_time",
    "seq72",
    "seq336",
    "heads4",
    "heads16",
    "layers2",
    "layers4",
    "single_h24",
    "miss_dropout",
]


def _test_input_missingness(stations, cfg: dict) -> float:
    """Mean fraction of MISSING input cells over the test period (times > val_end).

    The realized severity of a (possibly corrupted) set of station arrays;
    used as the crossover study's x-axis.
    """
    import pandas as pd

    val_end = pd.Timestamp(cfg["splits"]["val_end"]).to_datetime64()
    miss = tot = 0.0
    for st in stations:
        m = st.mask[st.times > val_end]
        miss += float((m == 0).sum())
        tot += float(m.size)
    return miss / max(tot, 1.0)


def variant_setup(variant: str, base_cfg: dict) -> tuple[dict, dict, int | None]:
    """Return (cfg, model_kwargs, input_length) for one ablation variant."""
    cfg = copy.deepcopy(base_cfg)
    kwargs: dict = {}
    input_length = None
    if variant == "no_miss_embed":
        kwargs["use_missingness_embedding"] = False
    elif variant == "variant_B":
        cfg["model"]["attention_variant"] = "B"
    elif variant == "no_attention_mask":
        cfg["model"]["attention_variant"] = "A"
    elif variant == "no_station_embed":
        kwargs["use_station_embedding"] = False
    elif variant == "no_pos_enc":
        kwargs["use_positional_encoding"] = False
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
    elif variant == "heads4":
        cfg["model"]["n_heads"] = 4
    elif variant == "heads16":
        cfg["model"]["n_heads"] = 16
    elif variant == "layers2":
        cfg["model"]["n_layers"] = 2
    elif variant == "layers4":
        cfg["model"]["n_layers"] = 4
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


def export_seed_predictions(base_cfg: dict) -> None:
    """Per-seed test prediction bundles for the proposed-model family.

    The ablation runner already trained ``full``, ``variant_B`` and
    ``miss_dropout`` with all seeds but stored only their RMSE numbers; the
    multi-seed main tables and per-seed DM tests need the full bundles. This
    re-runs inference from the existing checkpoints (no training) and writes
    ``predictions/seeds/{alias}_s{seed}_test.npz``. The seed-42 ``variant_B``
    bundle is also written top-level (it was never exported by scripts 03/04)
    together with a ``variant_B_stats.json`` for the efficiency table.
    """
    import shutil

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    builder = __import__("04_train_proposed")

    from src.data.dataset import make_datasets
    from src.train import predict, save_predictions

    # ablation variant -> (bundle alias, build kwargs)
    aliases = {"full": "proposed", "variant_B": "variant_B",
               "miss_dropout": "proposed_md"}
    pred_dir = Path(base_cfg["paths"]["predictions_dir"])
    ckpt_dir = Path(base_cfg["paths"]["checkpoints_dir"])
    datasets, stations, _ = make_datasets(base_cfg)

    for variant, alias in aliases.items():
        cfg, kwargs, _ = variant_setup(variant, base_cfg)
        for seed in base_cfg["ablation"]["seeds"]:
            canonical = seed == base_cfg["seed"]
            seed_path = pred_dir / "seeds" / f"{alias}_s{seed}_test.npz"
            top_path = pred_dir / f"{alias}_test.npz"
            if seed_path.exists() and (not canonical or top_path.exists()):
                logger.info("%s seed %d: bundles exist, skipping", alias, seed)
                continue
            if canonical and top_path.exists() and not seed_path.exists():
                seed_path.parent.mkdir(parents=True, exist_ok=True)
                seed_path.write_bytes(top_path.read_bytes())
                logger.info("backfilled %s from %s", seed_path.name, top_path.name)
                continue
            ckpt = ckpt_dir / f"abl_{variant}_s{seed}_seed{seed}.pt"
            if variant == "full" and (ckpt_dir / f"proposed_seed{seed}.pt").exists():
                ckpt = ckpt_dir / f"proposed_seed{seed}.pt"
            if not ckpt.exists():
                logger.warning("%s seed %d: no checkpoint (%s), skipping",
                               alias, seed, ckpt.name)
                continue
            model = builder.build_proposed(cfg, n_stations=len(stations), **kwargs)
            model.load_state_dict(
                torch.load(ckpt, weights_only=False)["model_state"]
            )
            out = predict(model, datasets["test"], cfg)
            save_predictions(out, cfg, f"{alias}_s{seed}", subdir="seeds")
            if canonical and not top_path.exists():
                save_predictions(out, cfg, alias)

    # stats json so variant_B shows up in the efficiency table
    vb_stats_src = ckpt_dir / "abl_variant_B_s42_stats.json"
    vb_stats_dst = ckpt_dir / "variant_B_stats.json"
    if vb_stats_src.exists() and not vb_stats_dst.exists():
        stats = json.loads(vb_stats_src.read_text(encoding="utf-8"))
        stats["name"] = "variant_B"
        vb_stats_dst.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info("wrote %s (copied from %s)", vb_stats_dst.name, vb_stats_src.name)


def run_robustness(base_cfg: dict) -> None:
    import torch

    from src.data.dataset import (
        AirQualityWindowDataset,
        feature_columns,
        load_scalers,
        make_datasets,
    )
    from src.data.impute import (
        corrupt_test_inputs,
        corrupt_test_outages,
        impute_full_series,
        replace_inputs,
    )
    from src.models.factory import build_model
    from src.train import predict, save_predictions

    cfg = base_cfg
    seed = cfg["seed"]
    datasets, stations, _ = make_datasets(cfg)
    feats = feature_columns(cfg)
    n_feat = len(feats)
    n_t = len(cfg["dataset"]["target_pollutants"])
    n_h = len(cfg["dataset"]["horizons"])
    t_feat_idx = feats.index(cfg["dataset"]["primary_target"])
    t_indices = [feats.index(p) for p in cfg["dataset"]["target_pollutants"]]
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    rcfg = cfg.get("robustness", {
        "direct": ["proposed", "proposed_md"],
        "two_stage": ["two_stage_knn", "two_stage_mice"],
    })

    def build(name: str):
        return build_model(name, cfg, n_feat, len(stations), n_t, n_h,
                           target_feature_idx=t_feat_idx, target_indices=t_indices)

    models_direct = {}
    for name in rcfg["direct"]:
        # proposed_md may come from script 04 (--miss-dropout, e.g. Beijing)
        # or from the miss_dropout ablation run (Dhaka)
        candidates = [ckpt_dir / f"{name}_seed{seed}.pt"]
        if name == "proposed_md":
            candidates.append(ckpt_dir / f"abl_miss_dropout_s{seed}_seed{seed}.pt")
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            logger.warning("robustness: no checkpoint for %s, skipping", name)
            continue
        m = build(name)
        m.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"])
        models_direct[name] = m
        clean_path = Path(cfg["paths"]["predictions_dir"]) / f"{name}_test.npz"
        if not clean_path.exists():
            save_predictions(predict(m, datasets["test"], cfg), cfg, name)

    two_stage = {}
    for ts_name in rcfg["two_stage"]:
        ckpt = ckpt_dir / f"{ts_name}_seed{seed}.pt"
        if not ckpt.exists():
            logger.warning("robustness: no checkpoint for %s, skipping", ts_name)
            continue
        m = build(ts_name)
        m.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"])
        two_stage[ts_name] = m

    # Two corruption mechanisms (both reported in the paper):
    #   miss = cell-wise MCAR (per spec; easy case for row-wise imputers)
    #   out  = station-outage blocks (mechanism dominating real missingness
    #          per the Phase 1 analysis; hard case for row-wise imputers)
    corruptors = {"miss": corrupt_test_inputs, "out": corrupt_test_outages}
    pred_dir = Path(cfg["paths"]["predictions_dir"])

    # Effective (realized) mean test-input missingness per (mode, level), the
    # x-axis of the missingness-severity crossover study. Stored alongside the
    # bundles so src.evaluate can plot the gap against true severity rather than
    # the nominal corruption level.
    levels_path = Path(cfg["paths"]["outputs_dir"]) / "robustness_levels.json"
    levels_map = (json.loads(levels_path.read_text(encoding="utf-8"))
                  if levels_path.exists() else {})
    levels_map.setdefault("clean", _test_input_missingness(stations, cfg))

    for mode, corrupt_fn in corruptors.items():
        for level in cfg["dataset"]["synthetic_missingness"]:
            suffix = f"test_{mode}{int(level * 100)}"
            key = f"{mode}{int(level * 100)}"
            expected = [pred_dir / f"{m}_{suffix}.npz"
                        for m in (*models_direct, *two_stage)]
            if all(p.exists() for p in expected) and key in levels_map:
                logger.info("robustness %s: bundles exist, skipping", suffix)
                continue
            corrupted = corrupt_fn(stations, cfg, level, seed)
            levels_map[key] = _test_input_missingness(corrupted, cfg)
            levels_path.write_text(json.dumps(levels_map, indent=2), encoding="utf-8")

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
    parser.add_argument("--seeds", default=None,
                        help="comma-separated ablation seeds; defaults to "
                             "ablation.seeds from the config")
    parser.add_argument("--skip-robustness", action="store_true",
                        help="do not run the robustness experiment after "
                             "training ablations")
    parser.add_argument("--robustness", action="store_true",
                        help="run only the robustness experiment")
    parser.add_argument("--export-seed-predictions", action="store_true",
                        help="only export per-seed test bundles for the "
                             "proposed family from existing checkpoints")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    if args.seeds:
        base_cfg["ablation"]["seeds"] = [
            int(s) for s in args.seeds.split(",") if s.strip()
        ]
    setup_logging("05_ablations", base_cfg["paths"]["logs_dir"])
    seed_everything(base_cfg["seed"], base_cfg.get("num_threads"))

    if args.export_seed_predictions:
        export_seed_predictions(base_cfg)
        return
    if args.robustness:
        run_robustness(base_cfg)
        return
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(ALL_VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    run_ablations(base_cfg, variants)
    if not args.skip_robustness:
        run_robustness(base_cfg)


if __name__ == "__main__":
    main()
