"""Export ablation prediction bundles and significance tables.

This fills the review gap left by the main evaluation tables:

* every clean MAT ablation variant vs the full MAT;
* after-imputation MAT variants vs the imputed vanilla Transformer;
* after-imputation variants vs imputed full MAT.

The statistical test mirrors ``src.evaluate``: per-seed Diebold-Mariano tests
on squared errors with a Newey-West lag, plus paired bootstrap confidence
intervals for RMSE(model A) - RMSE(model B). Negative RMSE differences mean
model A is better.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_scalers, make_datasets
from src.train import predict, save_predictions
from src.utils import load_config, setup_logging

logger = logging.getLogger("22_ablation_significance")


CLEAN_VARIANTS = [
    "full",
    "no_miss_embed",
    "variant_B",
    "no_met",
    "no_time",
    "seq72",
    "seq336",
    "single_h24",
    "miss_dropout",
]

IMPUTED_MODELS = {
    "vanilla_transformer": "hybrid8_transformer",
    "vanilla_transformer_mask_preserved": "hybrid8_masked_transformer",
    "mat_full": "hybrid8_masked_proposed",
    "mat_variant_B": "hybrid8_masked_variant_B",
    "mat_variant_B_vanilla_input": "hybrid8_masked_variant_B_vanilla_input",
    "mat_variant_B_dual_input_ridge": "variant_B_dual_input_ridge",
    "mat_miss_dropout": "hybrid8_masked_proposed_md",
}


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundle_path(cfg: dict[str, Any], name: str, seed: int, *, clean: bool) -> Path:
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    if clean:
        return pred_dir / "ablation_significance" / f"{name}_s{seed}_test.npz"
    return pred_dir / "seeds" / f"{name}_s{seed}_test.npz"


def _checkpoint_path(cfg: dict[str, Any], variant: str, seed: int) -> Path:
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    if variant == "full" and (ckpt_dir / f"proposed_seed{seed}.pt").exists():
        return ckpt_dir / f"proposed_seed{seed}.pt"
    return ckpt_dir / f"abl_{variant}_s{seed}_seed{seed}.pt"


def export_clean_ablation_predictions(cfg: dict[str, Any], *, force: bool = False) -> None:
    """Export per-sample predictions for all clean ablation checkpoints."""
    scripts_dir = Path(__file__).resolve().parent
    train_proposed = _load_script_module(scripts_dir / "04_train_proposed.py", "train_proposed_04")
    ablations = _load_script_module(scripts_dir / "05_ablations.py", "ablations_05")

    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    dataset_cache: dict[str, tuple[dict[str, Any], dict[str, Any], list[Any]]] = {}

    for variant in CLEAN_VARIANTS:
        variant_cfg, model_kwargs, input_length = ablations.variant_setup(variant, cfg)
        cache_key = f"{variant}:{input_length}"
        if cache_key not in dataset_cache:
            datasets, stations, _ = make_datasets(variant_cfg, input_length=input_length)
            dataset_cache[cache_key] = (datasets, variant_cfg, stations)
        datasets, variant_cfg, stations = dataset_cache[cache_key]

        for seed in seeds:
            ckpt = _checkpoint_path(cfg, variant, seed)
            out_path = _bundle_path(cfg, variant, seed, clean=True)
            if out_path.exists() and not force:
                logger.info("%s seed %d: prediction bundle exists", variant, seed)
                continue
            if not ckpt.exists():
                logger.warning("%s seed %d: checkpoint missing (%s)", variant, seed, ckpt.name)
                continue
            model = train_proposed.build_proposed(
                variant_cfg, n_stations=len(stations), **model_kwargs
            )
            model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"])
            out = predict(model, datasets["test"], variant_cfg)
            save_predictions(out, cfg, f"{variant}_s{seed}", subdir="ablation_significance")


def export_vanilla_input_variant_b_predictions(cfg: dict[str, Any], *, force: bool = False) -> None:
    """Evaluate Variant B with the vanilla Transformer's input pathway/weights.

    This isolates the attention-mask idea from MAT's learned missingness
    embedding after imputation: the model receives imputed values, time
    features, and station embeddings exactly like ``hybrid8_transformer``, but
    uses the preserved original mask to block attention to PM2.5-missing keys.
    """
    import copy

    from src.data.dataset import AirQualityWindowDataset, feature_columns
    from src.data.impute import impute_full_series, replace_inputs
    from src.models.missingness_transformer import MissingnessTransformer

    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    _, stations, _ = make_datasets(cfg)
    feats = feature_columns(cfg)
    cfg_b = copy.deepcopy(cfg)
    cfg_b["model"]["attention_variant"] = "B"

    for seed in seeds:
        out_path = _bundle_path(
            cfg, "hybrid8_masked_variant_B_vanilla_input", seed, clean=False
        )
        if out_path.exists() and not force:
            logger.info("variant_B_vanilla_input seed %d: prediction bundle exists", seed)
            continue
        ckpt = Path(cfg["paths"]["checkpoints_dir"]) / f"hybrid8_transformer_seed{seed}.pt"
        if not ckpt.exists():
            logger.warning("variant_B_vanilla_input seed %d: checkpoint missing", seed)
            continue
        imputed = impute_full_series(stations, cfg, "hybrid8", seed)
        stations_imp = replace_inputs(stations, imputed, preserve_mask=True)
        test_ds = AirQualityWindowDataset(stations_imp, "test", cfg)
        model = MissingnessTransformer(
            n_features=len(feats),
            n_stations=len(stations),
            n_targets=len(cfg["dataset"]["target_pollutants"]),
            n_horizons=len(cfg["dataset"]["horizons"]),
            cfg=cfg_b,
            target_feature_idx=feats.index(cfg["dataset"]["primary_target"]),
            use_missingness_embedding=False,
        )
        model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"], strict=True)
        out = predict(model, test_ds, cfg_b)
        save_predictions(
            out,
            cfg,
            f"hybrid8_masked_variant_B_vanilla_input_s{seed}",
            subdir="seeds",
        )


def export_native_variant_b_val_predictions(cfg: dict[str, Any], *, force: bool = False) -> None:
    """Export validation bundles for native Variant B, needed for stacking."""
    import copy

    train_proposed = _load_script_module(
        Path(__file__).resolve().parent / "04_train_proposed.py", "train_proposed_04_val"
    )
    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    cfg_b = copy.deepcopy(cfg)
    cfg_b["model"]["attention_variant"] = "B"
    datasets, stations, _ = make_datasets(cfg_b)
    for seed in seeds:
        out_path = Path(cfg["paths"]["predictions_dir"]) / "seeds" / f"variant_B_s{seed}_val.npz"
        if out_path.exists() and not force:
            logger.info("native variant_B seed %d: validation bundle exists", seed)
            continue
        ckpt = Path(cfg["paths"]["checkpoints_dir"]) / f"abl_variant_B_s{seed}_seed{seed}.pt"
        if not ckpt.exists():
            logger.warning("native variant_B seed %d: checkpoint missing", seed)
            continue
        model = train_proposed.build_proposed(cfg_b, n_stations=len(stations))
        model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"])
        out = predict(model, datasets["val"], cfg_b)
        save_predictions(out, cfg, f"variant_B_s{seed}", split="val", subdir="seeds")


def _aligned_pm25_matrix(
    cfg: dict[str, Any],
    scalers: dict[str, Any],
    bundles: list[dict[str, np.ndarray]],
    horizon_idx: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[np.ndarray]]:
    """Return aligned PM2.5 predictions, target, keys, and bundle row indexes."""
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    mean, std = scalers["PM2.5"]
    key_lists = [
        [(int(sid), int(anchor)) for sid, anchor in zip(b["station_id"], b["anchor_time"])]
        for b in bundles
    ]
    maps = [{key: i for i, key in enumerate(keys)} for keys in key_lists]
    common = [key for key in key_lists[0] if all(key in mp for mp in maps[1:])]
    idxs = [np.asarray([mp[key] for key in common], dtype=int) for mp in maps]
    observed = np.ones(len(common), dtype=bool)
    for b, idx in zip(bundles, idxs):
        observed &= b["target_mask"][idx, ti, horizon_idx] > 0
        observed &= np.isfinite(b["predictions"][idx, ti, horizon_idx])
    idxs = [idx[observed] for idx in idxs]
    keys = [key for key, ok in zip(common, observed) if ok]
    preds = [
        b["predictions"][idx, ti, horizon_idx] * std + mean
        for b, idx in zip(bundles, idxs)
    ]
    y = bundles[0]["targets"][idxs[0], ti, horizon_idx] * std + mean
    return np.vstack(preds).T, y, keys, idxs


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    X2 = np.c_[X, np.ones(len(X))]
    reg = np.eye(X2.shape[1]) * lam
    reg[-1, -1] = 0.0
    return np.linalg.solve(X2.T @ X2 + reg, X2.T @ y)


def _predict_ridge(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.c_[X, np.ones(len(X))] @ coef


def export_dual_input_variant_b_ridge(
    cfg: dict[str, Any], scalers: dict[str, Any], *, force: bool = False
) -> None:
    """Validation-fitted ridge stack of the two Variant-B input paths.

    Inputs are native Variant B and vanilla-input Variant B. Coefficients are
    fit on validation PM2.5 predictions separately for each seed/horizon, then
    applied once to the test bundles.
    """
    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    pred_dir = Path(cfg["paths"]["predictions_dir"]) / "seeds"
    sources = ["variant_B", "hybrid8_masked_variant_B_vanilla_input"]
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    mean, std = scalers["PM2.5"]
    horizons = cfg["dataset"]["horizons"]

    for seed in seeds:
        out_path = pred_dir / f"variant_B_dual_input_ridge_s{seed}_test.npz"
        if out_path.exists() and not force:
            logger.info("dual-input Variant B ridge seed %d: bundle exists", seed)
            continue
        val_bundles = [
            dict(np.load(pred_dir / f"{src}_s{seed}_val.npz", allow_pickle=True))
            for src in sources
        ]
        test_bundles = [
            dict(np.load(pred_dir / f"{src}_s{seed}_test.npz", allow_pickle=True))
            for src in sources
        ]
        out = {k: np.array(v, copy=True) for k, v in test_bundles[0].items()}
        key_to_row = {
            (int(sid), int(anchor)): i
            for i, (sid, anchor) in enumerate(zip(out["station_id"], out["anchor_time"]))
        }
        for hi, _h in enumerate(horizons):
            X_val, y_val, _, _ = _aligned_pm25_matrix(cfg, scalers, val_bundles, hi)
            coef = _fit_ridge(X_val, y_val, lam=10.0)
            X_test, _, keys, _ = _aligned_pm25_matrix(cfg, scalers, test_bundles, hi)
            pred = _predict_ridge(X_test, coef)
            rows = np.asarray([key_to_row[key] for key in keys], dtype=int)
            out["predictions"][rows, ti, hi] = (pred - mean) / std
        save_predictions(out, cfg, f"variant_B_dual_input_ridge_s{seed}", subdir="seeds")


def _load_bundle(cfg: dict[str, Any], model: str, seed: int, *, clean: bool) -> dict[str, np.ndarray]:
    return dict(np.load(_bundle_path(cfg, model, seed, clean=clean), allow_pickle=True))


def _aligned_errors(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    cfg: dict[str, Any],
    scalers: dict[str, Any],
    horizon_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    mean, std = scalers["PM2.5"]

    rows_a = {}
    for i, (sid, anchor) in enumerate(zip(a["station_id"], a["anchor_time"])):
        rows_a[(int(sid), int(anchor))] = i

    idx_a, idx_b = [], []
    for j, (sid, anchor) in enumerate(zip(b["station_id"], b["anchor_time"])):
        i = rows_a.get((int(sid), int(anchor)))
        if i is not None:
            idx_a.append(i)
            idx_b.append(j)
    ia = np.asarray(idx_a, dtype=int)
    ib = np.asarray(idx_b, dtype=int)

    mask = (a["target_mask"][ia, ti, horizon_idx] > 0) & (b["target_mask"][ib, ti, horizon_idx] > 0)
    mask &= np.isfinite(a["predictions"][ia, ti, horizon_idx])
    mask &= np.isfinite(b["predictions"][ib, ti, horizon_idx])
    ia, ib = ia[mask], ib[mask]

    y = a["targets"][ia, ti, horizon_idx] * std + mean
    pa = a["predictions"][ia, ti, horizon_idx] * std + mean
    pb = b["predictions"][ib, ti, horizon_idx] * std + mean
    return (pa - y) ** 2, (pb - y) ** 2, a["anchor_time"][ia]


def diebold_mariano(
    e1_sq: np.ndarray, e2_sq: np.ndarray, order: np.ndarray, nw_lag: int
) -> tuple[float, float]:
    d = (e1_sq - e2_sq)[np.argsort(order, kind="stable")]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float((dc @ dc) / n)
    lrv = gamma0
    for k in range(1, nw_lag + 1):
        gk = float((dc[k:] @ dc[:-k]) / n)
        lrv += 2 * (1 - k / (nw_lag + 1)) * gk
    if lrv <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(lrv / n)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def paired_bootstrap_rmse_diff(
    e1_sq: np.ndarray, e2_sq: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(e1_sq)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(e1_sq[idx].mean()) - np.sqrt(e2_sq[idx].mean())
    point = float(np.sqrt(e1_sq.mean()) - np.sqrt(e2_sq.mean()))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def metrics_table(
    cfg: dict[str, Any], scalers: dict[str, Any], models: dict[str, str], *, clean: bool
) -> pd.DataFrame:
    rows = []
    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    horizons = cfg["dataset"]["horizons"]
    for label, model in models.items():
        for hi, h in enumerate(horizons):
            rmses, maes, r2s, ns = [], [], [], []
            for seed in seeds:
                path = _bundle_path(cfg, model, seed, clean=clean)
                if not path.exists():
                    continue
                b = _load_bundle(cfg, model, seed, clean=clean)
                e, _, _ = _aligned_errors(b, b, cfg, scalers, hi)
                rmse = float(np.sqrt(e.mean()))
                ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
                mean, std = scalers["PM2.5"]
                mask = b["target_mask"][:, ti, hi] > 0
                y = b["targets"][mask, ti, hi] * std + mean
                p = b["predictions"][mask, ti, hi] * std + mean
                mae = float(np.mean(np.abs(p - y)))
                ss_res = float(np.sum((p - y) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                rmses.append(rmse)
                maes.append(mae)
                r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
                ns.append(int(mask.sum()))
            if rmses:
                rows.append({
                    "model": label,
                    "horizon": h,
                    "seeds": len(rmses),
                    "RMSE_mean": float(np.mean(rmses)),
                    "RMSE_std": float(np.std(rmses, ddof=0)),
                    "MAE_mean": float(np.mean(maes)),
                    "R2_mean": float(np.mean(r2s)),
                    "n": int(ns[0]),
                })
    return pd.DataFrame(rows)


def significance_table(
    cfg: dict[str, Any],
    scalers: dict[str, Any],
    comparisons: list[tuple[str, str, str, str]],
    *,
    clean: bool,
) -> pd.DataFrame:
    rows = []
    horizons = cfg["dataset"]["horizons"]
    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    for comparison, model_a, model_b, note in comparisons:
        for hi, h in enumerate(horizons):
            if comparison == "single_h24_vs_full" and h != 24:
                continue
            seed_rows = []
            for seed in seeds:
                pa = _bundle_path(cfg, model_a, seed, clean=clean)
                pb = _bundle_path(cfg, model_b, seed, clean=clean)
                if not pa.exists() or not pb.exists():
                    continue
                a = _load_bundle(cfg, model_a, seed, clean=clean)
                b = _load_bundle(cfg, model_b, seed, clean=clean)
                e_a, e_b, order = _aligned_errors(a, b, cfg, scalers, hi)
                dm, pval = diebold_mariano(e_a, e_b, order, nw_lag=max(1, -(-h // 24)))
                diff, lo, hi_ci = paired_bootstrap_rmse_diff(e_a, e_b, 1000, seed)
                seed_rows.append({
                    "seed": seed,
                    "n": int(len(e_a)),
                    "DM_stat": dm,
                    "DM_p": pval,
                    "RMSE_diff": diff,
                    "CI_lo": lo,
                    "CI_hi": hi_ci,
                })
            if not seed_rows:
                continue
            pvals = [r["DM_p"] for r in seed_rows]
            rows.append({
                "comparison": comparison,
                "model_A": model_a,
                "model_B": model_b,
                "horizon": h,
                "seeds": len(seed_rows),
                "n": seed_rows[0]["n"],
                "DM_p_median": float(np.nanmedian(pvals)),
                "DM_p_min": float(np.nanmin(pvals)),
                "DM_p_max": float(np.nanmax(pvals)),
                "sig_all_seeds": bool(all(p < 0.05 for p in pvals)),
                "RMSE_diff_mean_A_minus_B": float(np.mean([r["RMSE_diff"] for r in seed_rows])),
                "CI_lo_min": float(np.min([r["CI_lo"] for r in seed_rows])),
                "CI_hi_max": float(np.max([r["CI_hi"] for r in seed_rows])),
                "directional_A_better": bool(np.mean([r["RMSE_diff"] for r in seed_rows]) < 0),
                "note": note,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="re-export prediction bundles")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("22_ablation_significance", cfg["paths"]["logs_dir"])
    scalers = load_scalers(Path(cfg["paths"]["processed_dir"]) / "scalers.json")
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    export_clean_ablation_predictions(cfg, force=args.force)
    export_vanilla_input_variant_b_predictions(cfg, force=args.force)
    export_native_variant_b_val_predictions(cfg, force=args.force)
    export_dual_input_variant_b_ridge(cfg, scalers, force=args.force)

    clean_models = {variant: variant for variant in CLEAN_VARIANTS}
    clean_metrics = metrics_table(cfg, scalers, clean_models, clean=True)
    clean_metrics.to_csv(tables_dir / "ablation_metrics_clean.csv", index=False)

    clean_comparisons = [
        (f"{variant}_vs_full", variant, "full", "clean ablation vs full MAT")
        for variant in CLEAN_VARIANTS
        if variant != "full"
    ]
    clean_sig = significance_table(cfg, scalers, clean_comparisons, clean=True)
    clean_sig.to_csv(tables_dir / "ablation_significance_clean_vs_full.csv", index=False)

    imputed_metrics = metrics_table(cfg, scalers, IMPUTED_MODELS, clean=False)
    imputed_metrics.to_csv(tables_dir / "ablation_metrics_after_imputation.csv", index=False)

    imputed_vs_vanilla = [
        (f"{label}_vs_vanilla_transformer", model, IMPUTED_MODELS["vanilla_transformer"],
         "after-imputation variant vs vanilla Transformer")
        for label, model in IMPUTED_MODELS.items()
        if label != "vanilla_transformer"
    ]
    sig_vs_vanilla = significance_table(cfg, scalers, imputed_vs_vanilla, clean=False)
    sig_vs_vanilla.to_csv(
        tables_dir / "ablation_significance_after_imputation_vs_vanilla.csv",
        index=False,
    )

    imputed_vs_full = [
        (f"{label}_vs_mat_full", model, IMPUTED_MODELS["mat_full"],
         "after-imputation variant vs imputed full MAT")
        for label, model in IMPUTED_MODELS.items()
        if label != "mat_full"
    ]
    sig_vs_full = significance_table(cfg, scalers, imputed_vs_full, clean=False)
    sig_vs_full.to_csv(
        tables_dir / "ablation_significance_after_imputation_vs_full_mat.csv",
        index=False,
    )

    logger.info("wrote %s", tables_dir / "ablation_metrics_clean.csv")
    logger.info("wrote %s", tables_dir / "ablation_significance_clean_vs_full.csv")
    logger.info("wrote %s", tables_dir / "ablation_metrics_after_imputation.csv")
    logger.info("wrote %s", tables_dir / "ablation_significance_after_imputation_vs_vanilla.csv")
    logger.info("wrote %s", tables_dir / "ablation_significance_after_imputation_vs_full_mat.csv")


if __name__ == "__main__":
    main()
