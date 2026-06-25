"""Fit a validation-selected DLinear + tabular safeguard blend.

The blend is designed for cases where long-horizon linear structure survives
distribution shift better than a pure neural ensemble, while tabular lag
summaries correct residual nonlinearities.  For each PM2.5 horizon, the script
chooses the DLinear seed, tabular expert, and convex weight that minimize
validation RMSE, then freezes that choice for test.
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

from src.utils import load_config


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))


def assert_aligned(a: dict[str, np.ndarray], b: dict[str, np.ndarray], label: str) -> None:
    for key in ("targets", "target_mask", "station_id", "anchor_time"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"{label}: not aligned on {key}")


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, std: float) -> float:
    ok = mask & np.isfinite(pred) & np.isfinite(target)
    err = pred[ok] - target[ok]
    return float(np.sqrt(np.mean(err * err)) * std)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default="linear_tabular_blend")
    parser.add_argument(
        "--tabular-models",
        default="tabular_extra_leaf8,tabular_extra_leaf12,tabular_extra_leaf20,tabular_extra_full",
    )
    parser.add_argument("--weight-step", type=float, default=0.01)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    table_dir = Path(cfg["paths"]["tables_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)

    scalers = json.loads((Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text())
    primary = cfg["dataset"]["primary_target"]
    target_idx = cfg["dataset"]["target_pollutants"].index(primary)
    std = float(scalers[primary][1])

    dlinear_models = [f"seeds/dlinear_s{seed}" for seed in cfg["ablation"]["seeds"]]
    tabular_models = [m.strip() for m in args.tabular_models.split(",") if m.strip()]
    pairs = [(d, t) for d in dlinear_models for t in tabular_models]
    weights = np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step)

    bundles: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model in dlinear_models + tabular_models:
        val_path = pred_dir / f"{model}_val.npz"
        test_path = pred_dir / f"{model}_test.npz"
        if val_path.exists() and test_path.exists():
            bundles[model] = {
                "val": load_bundle(val_path),
                "test": load_bundle(test_path),
            }

    missing = sorted(set(dlinear_models + tabular_models) - set(bundles))
    if missing:
        print(f"skipping missing candidates: {missing}")
    if not any(d in bundles for d in dlinear_models):
        raise RuntimeError("no DLinear seed bundles available")
    if not any(t in bundles for t in tabular_models):
        raise RuntimeError("no tabular bundles available")

    ref = next(iter(bundles.values()))["val"]
    for model, split_bundles in bundles.items():
        assert_aligned(ref, split_bundles["val"], f"{model} val")

    out = {
        split: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in next(iter(bundles.values()))[split].items()
        }
        for split in ("val", "test")
    }

    choices = []
    metrics = []
    for hi, horizon in enumerate(cfg["dataset"]["horizons"]):
        mask = ref["target_mask"][:, target_idx, hi] > 0
        target = ref["targets"][:, target_idx, hi]
        best: dict[str, Any] | None = None
        for dlinear, tabular in pairs:
            if dlinear not in bundles or tabular not in bundles:
                continue
            d_pred = bundles[dlinear]["val"]["predictions"][:, target_idx, hi]
            t_pred = bundles[tabular]["val"]["predictions"][:, target_idx, hi]
            for w in weights:
                pred = w * d_pred + (1.0 - w) * t_pred
                score = rmse(pred, target, mask, std)
                if best is None or score < best["val_RMSE"]:
                    best = {
                        "horizon": int(horizon),
                        "dlinear": dlinear,
                        "tabular": tabular,
                        "dlinear_weight": float(w),
                        "tabular_weight": float(1.0 - w),
                        "val_RMSE": score,
                    }
        if best is None:
            raise RuntimeError(f"no valid blend for horizon {horizon}")

        for split in ("val", "test"):
            d_pred = bundles[best["dlinear"]][split]["predictions"][:, target_idx, hi]
            t_pred = bundles[best["tabular"]][split]["predictions"][:, target_idx, hi]
            out[split]["predictions"][:, target_idx, hi] = (
                best["dlinear_weight"] * d_pred + best["tabular_weight"] * t_pred
            )

        for split in ("val", "test"):
            b = out[split]
            score = rmse(
                b["predictions"][:, target_idx, hi],
                b["targets"][:, target_idx, hi],
                b["target_mask"][:, target_idx, hi] > 0,
                std,
            )
            metrics.append({
                "model": args.name,
                "split": split,
                "horizon": int(horizon),
                "RMSE": score,
            })
        choices.append(best)

    for split in ("val", "test"):
        out[split]["latency_ms_per_window"] = np.float64(0.0)
        np.savez_compressed(pred_dir / f"{args.name}_{split}.npz", **out[split])

    pd.DataFrame(choices).to_csv(table_dir / f"{args.name}_choices.csv", index=False)
    pd.DataFrame(metrics).to_csv(table_dir / f"{args.name}_pm25_metrics.csv", index=False)
    print(pd.DataFrame(choices).to_string(index=False))
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
