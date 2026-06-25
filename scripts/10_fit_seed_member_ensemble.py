"""Fit a validation-weighted ensemble over individual model seeds.

Compared with ``08_fit_forecast_ensemble.py``, this script treats each
``model_s{seed}`` prediction bundle as a separate base learner instead of
averaging seeds before fitting weights. Statistical baselines remain single
members. This is useful when the final comparator is itself a deployable
seed-averaged prediction rather than a mean-of-RMSE table row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import save_predictions
from src.utils import load_config


DATASET_DEFAULTS = {
    "config.yaml": [
        "persistence", "seasonal_naive", "dlinear", "gru_d", "proposed",
        "hybrid8_transformer", "hybrid8_masked_transformer",
        "hybrid8_masked_variant_B", "hybrid8_masked_proposed_md",
    ],
    "config_delhi.yaml": [
        "persistence", "seasonal_naive", "dlinear", "gru_d", "proposed",
        "variant_B", "proposed_md",
    ],
    "config_beijing.yaml": [
        "persistence", "seasonal_naive", "dlinear", "gru_d", "proposed",
        "variant_B", "proposed_md",
    ],
}

STATISTICAL = {"persistence", "seasonal_naive", "sarima"}


def _load(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))


def _assert_aligned(ref: dict[str, np.ndarray], other: dict[str, np.ndarray],
                    label: str) -> None:
    for key in ("targets", "target_mask", "station_id", "anchor_time"):
        if not np.array_equal(ref[key], other[key]):
            raise ValueError(f"{label}: not aligned on {key}")


def _members(cfg: dict[str, Any], models: list[str], split: str
             ) -> list[tuple[str, dict[str, np.ndarray]]]:
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    out = []
    for model in models:
        if model in STATISTICAL:
            path = pred_dir / f"{model}_{split}.npz"
            if path.exists():
                out.append((model, _load(path)))
            continue
        for seed in cfg.get("ablation", {}).get("seeds", [cfg["seed"]]):
            path = pred_dir / "seeds" / f"{model}_s{seed}_{split}.npz"
            if path.exists():
                out.append((f"{model}_s{seed}", _load(path)))
    return out


def _fit_simplex(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    k = x.shape[1]
    if k == 1:
        return np.ones(1)

    def obj(w):
        err = x @ w - y
        return float(np.mean(err * err))

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * k
    starts = [np.full(k, 1.0 / k)]
    # Add strong single-member starts; this helps SLSQP avoid dull local points.
    for j in range(k):
        s = np.zeros(k)
        s[j] = 1.0
        starts.append(s)

    best = None
    for start in starts:
        res = minimize(obj, start, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
        w = np.clip(res.x, 0.0, 1.0)
        w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
        score = obj(w)
        if best is None or score < best[0]:
            best = (score, w)
    return best[1]


def _metrics(bundle: dict[str, np.ndarray], cfg: dict[str, Any],
             scalers: dict[str, list[float]], name: str, split: str
             ) -> list[dict[str, Any]]:
    pol = cfg["dataset"]["primary_target"]
    ti = cfg["dataset"]["target_pollutants"].index(pol)
    mean, std = scalers[pol]
    rows = []
    for hi, h in enumerate(cfg["dataset"]["horizons"]):
        m = bundle["target_mask"][:, ti, hi] > 0
        p = bundle["predictions"][m, ti, hi] * std + mean
        y = bundle["targets"][m, ti, hi] * std + mean
        ok = np.isfinite(p) & np.isfinite(y)
        err = p[ok] - y[ok]
        rows.append({
            "split": split,
            "model": name,
            "pollutant": pol,
            "horizon": h,
            "RMSE": float(np.sqrt(np.mean(err * err))),
            "MAE": float(np.mean(np.abs(err))),
            "n": int(ok.sum()),
        })
    return rows


def fit_seed_member_ensemble(cfg: dict[str, Any], models: list[str], name: str):
    val_members = _members(cfg, models, "val")
    test_members = _members(cfg, models, "test")
    test_by_name = dict(test_members)
    paired = [(n, v, test_by_name[n]) for n, v in val_members if n in test_by_name]
    if not paired:
        raise SystemExit("no paired val/test members found")

    names = [p[0] for p in paired]
    val = [p[1] for p in paired]
    test = [p[2] for p in paired]
    ref_val, ref_test = val[0], test[0]
    for n, v, t in paired[1:]:
        _assert_aligned(ref_val, v, f"{n} val")
        _assert_aligned(ref_test, t, f"{n} test")

    pol = cfg["dataset"]["primary_target"]
    ti = cfg["dataset"]["target_pollutants"].index(pol)
    horizons = cfg["dataset"]["horizons"]
    weight_rows = []
    weights = {}
    for hi, h in enumerate(horizons):
        m = ref_val["target_mask"][:, ti, hi] > 0
        x = np.column_stack([b["predictions"][m, ti, hi] for b in val])
        y = ref_val["targets"][m, ti, hi]
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        w = _fit_simplex(x[ok], y[ok])
        weights[hi] = w
        for member, weight in zip(names, w):
            weight_rows.append({"horizon": h, "member": member, "weight": weight})

    def apply(bundles, ref):
        stack = np.stack([b["predictions"] for b in bundles])
        pred = np.zeros_like(ref["predictions"])
        for hi in range(len(horizons)):
            pred[:, :, hi] = (stack[:, :, :, hi]
                              * weights[hi].reshape(-1, 1, 1)).sum(axis=0)
        return {
            "predictions": pred,
            "targets": ref["targets"],
            "target_mask": ref["target_mask"],
            "station_id": ref["station_id"],
            "anchor_time": ref["anchor_time"],
            "latency_ms_per_window": np.float64(0.0),
        }

    val_out = apply(val, ref_val)
    test_out = apply(test, ref_test)
    save_predictions(val_out, cfg, name, split="val")
    save_predictions(test_out, cfg, name, split="test")

    scalers = json.loads((Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text())
    tables = Path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weight_rows).to_csv(tables / f"{name}_weights_pm25.csv", index=False)
    metrics = pd.DataFrame(
        _metrics(val_out, cfg, scalers, name, "val")
        + _metrics(test_out, cfg, scalers, name, "test")
    )
    metrics.to_csv(tables / f"{name}_pm25_metrics.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--name", default="ensemble_seed_member")
    parser.add_argument("--models", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = DATASET_DEFAULTS.get(Path(args.config).name, DATASET_DEFAULTS["config.yaml"])
    metrics = fit_seed_member_ensemble(cfg, models, args.name)
    print(metrics[metrics["split"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
