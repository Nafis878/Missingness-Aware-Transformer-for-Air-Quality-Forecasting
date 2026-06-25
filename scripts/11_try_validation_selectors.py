"""Try validation-only selector/calibration approaches for a stronger claim.

This is an exploratory script, but it obeys the main rule for a defensible
paper result: all choices are made on validation predictions only, then frozen
for test.

Approaches:

* global candidate selector per horizon;
* station-aware candidate selector per horizon;
* affine calibration of each candidate per horizon;
* station-bias calibration of each candidate per horizon.

The script writes PM2.5 metrics and chosen candidates to each dataset's tables
directory, plus an overall comparison to the previous best table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DATASETS = {
    "Dhaka": ("config.yaml", Path("outputs")),
    "Delhi": ("config_delhi.yaml", Path("outputs/delhi")),
    "Beijing": ("config_beijing.yaml", Path("outputs/beijing")),
}

STAT = {"persistence", "seasonal_naive", "sarima"}

DEFAULT_MODELS = {
    "Dhaka": [
        "persistence", "seasonal_naive", "dlinear", "gru_d", "proposed",
        "hybrid8_transformer", "hybrid8_masked_transformer",
        "hybrid8_masked_variant_B", "hybrid8_masked_proposed_md",
        "ensemble_weighted_hybrid8", "ensemble_seed_member",
        "ensemble_ridge_seed_member",
    ],
    "Delhi": [
        "persistence", "seasonal_naive", "dlinear", "gru_d", "proposed",
        "variant_B", "proposed_md", "ensemble_weighted",
        "ensemble_seed_member", "ensemble_ridge_seed_member",
    ],
    "Beijing": [
        "persistence", "seasonal_naive", "dlinear", "gru_d", "proposed",
        "variant_B", "proposed_md", "ensemble_weighted",
        "ensemble_seed_member", "ensemble_ridge_seed_member",
    ],
}


def load(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))


def first_float(x: Any) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else float("nan")


def model_bundle(cfg: dict[str, Any], model: str, split: str):
    pred = Path(cfg["paths"]["predictions_dir"])
    top = pred / f"{model}_{split}.npz"
    if top.exists():
        return load(top)
    bundles = []
    for seed in cfg["ablation"]["seeds"]:
        path = pred / "seeds" / f"{model}_s{seed}_{split}.npz"
        if path.exists():
            bundles.append(load(path))
    if not bundles:
        return None
    ref = bundles[0]
    return {
        "predictions": np.mean([b["predictions"] for b in bundles], axis=0),
        "targets": ref["targets"],
        "target_mask": ref["target_mask"],
        "station_id": ref["station_id"],
        "anchor_time": ref["anchor_time"],
        "latency_ms_per_window": np.float64(0.0),
    }


def aligned(ref: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return all(np.array_equal(ref[k], b[k])
               for k in ("targets", "target_mask", "station_id", "anchor_time"))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(y)
    if ok.sum() == 0:
        return float("inf")
    err = pred[ok] - y[ok]
    return float(np.sqrt(np.mean(err * err)))


def metric_rows(bundle, cfg, scalers, name, split):
    pol = cfg["dataset"]["primary_target"]
    ti = cfg["dataset"]["target_pollutants"].index(pol)
    mean, std = scalers[pol]
    rows = []
    for hi, h in enumerate(cfg["dataset"]["horizons"]):
        m = bundle["target_mask"][:, ti, hi] > 0
        p = bundle["predictions"][m, ti, hi] * std + mean
        y = bundle["targets"][m, ti, hi] * std + mean
        ok = np.isfinite(p) & np.isfinite(y)
        e = p[ok] - y[ok]
        rows.append({
            "split": split, "model": name, "horizon": h,
            "RMSE": float(np.sqrt(np.mean(e * e))),
            "MAE": float(np.mean(np.abs(e))), "n": int(ok.sum()),
        })
    return rows


def affine_fit(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 10 or np.nanstd(p[ok]) < 1e-8:
        return 1.0, 0.0
    x = np.column_stack([p[ok], np.ones(ok.sum())])
    a, b = np.linalg.lstsq(x, y[ok], rcond=None)[0]
    # Conservative guardrail against wild validation overfit.
    a = float(np.clip(a, 0.5, 1.5))
    return a, float(b)


def station_bias_fit(pred: np.ndarray, y: np.ndarray, sid: np.ndarray,
                     global_shrink: float = 20.0) -> dict[int, float]:
    ok = np.isfinite(pred) & np.isfinite(y)
    out = {}
    for s in np.unique(sid[ok]):
        m = ok & (sid == s)
        n = int(m.sum())
        if n == 0:
            continue
        raw = float(np.mean(y[m] - pred[m]))
        out[int(s)] = raw * n / (n + global_shrink)
    return out


def apply_station_bias(base: np.ndarray, sid: np.ndarray, bias: dict[int, float]):
    out = base.copy()
    for s, b in bias.items():
        out[sid == s] += b
    return out


def build_outputs(dataset: str, cfg: dict[str, Any], out_dir: Path):
    scalers = json.loads((Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text())
    pol = cfg["dataset"]["primary_target"]
    ti = cfg["dataset"]["target_pollutants"].index(pol)
    mean, std = scalers[pol]

    candidates = {}
    for model in DEFAULT_MODELS[dataset]:
        val = model_bundle(cfg, model, "val")
        test = model_bundle(cfg, model, "test")
        if val is None or test is None:
            continue
        if candidates and (not aligned(next(iter(candidates.values()))["val"], val)
                           or not aligned(next(iter(candidates.values()))["test"], test)):
            continue
        candidates[model] = {"val": val, "test": test}
    if not candidates:
        raise RuntimeError(f"{dataset}: no candidates")

    ref_val = next(iter(candidates.values()))["val"]
    ref_test = next(iter(candidates.values()))["test"]
    methods = {
        "selector_global": {
            "val": np.zeros_like(ref_val["predictions"]),
            "test": np.zeros_like(ref_test["predictions"]),
            "choices": [],
        },
        "selector_station": {
            "val": np.zeros_like(ref_val["predictions"]),
            "test": np.zeros_like(ref_test["predictions"]),
            "choices": [],
        },
        "affine_selector": {
            "val": np.zeros_like(ref_val["predictions"]),
            "test": np.zeros_like(ref_test["predictions"]),
            "choices": [],
        },
        "station_bias_selector": {
            "val": np.zeros_like(ref_val["predictions"]),
            "test": np.zeros_like(ref_test["predictions"]),
            "choices": [],
        },
    }

    # Copy non-primary-target predictions from the first candidate; PM2.5 is
    # the claim target and is overwritten below.
    for m in methods.values():
        m["val"][:] = ref_val["predictions"]
        m["test"][:] = ref_test["predictions"]

    for hi, h in enumerate(cfg["dataset"]["horizons"]):
        mv = ref_val["target_mask"][:, ti, hi] > 0
        mt = ref_test["target_mask"][:, ti, hi] > 0
        yv = ref_val["targets"][:, ti, hi] * std + mean

        # Global selector.
        scores = {}
        for name, b in candidates.items():
            pv = b["val"]["predictions"][:, ti, hi] * std + mean
            scores[name] = rmse(pv[mv], yv[mv])
        best = min(scores, key=scores.get)
        methods["selector_global"]["val"][:, ti, hi] = candidates[best]["val"]["predictions"][:, ti, hi]
        methods["selector_global"]["test"][:, ti, hi] = candidates[best]["test"]["predictions"][:, ti, hi]
        methods["selector_global"]["choices"].append({
            "dataset": dataset, "horizon": h, "scope": "global",
            "selected": best, "val_RMSE": scores[best],
        })

        # Station selector.
        sidv = ref_val["station_id"]
        sidt = ref_test["station_id"]
        for s in np.unique(sidt):
            vals = {}
            sv = mv & (sidv == s)
            st = sidt == s
            for name, b in candidates.items():
                pv = b["val"]["predictions"][:, ti, hi] * std + mean
                vals[name] = rmse(pv[sv], yv[sv])
            chosen = min(vals, key=vals.get)
            methods["selector_station"]["val"][sidv == s, ti, hi] = candidates[chosen]["val"]["predictions"][sidv == s, ti, hi]
            methods["selector_station"]["test"][st, ti, hi] = candidates[chosen]["test"]["predictions"][st, ti, hi]
            methods["selector_station"]["choices"].append({
                "dataset": dataset, "horizon": h, "scope": f"station_{int(s)}",
                "selected": chosen, "val_RMSE": vals[chosen],
            })

        # Affine calibration selector.
        affine_scores = {}
        affine_params = {}
        for name, b in candidates.items():
            pv = b["val"]["predictions"][:, ti, hi] * std + mean
            a, c = affine_fit(pv[mv], yv[mv])
            affine_params[name] = (a, c)
            affine_scores[name] = rmse(a * pv[mv] + c, yv[mv])
        chosen = min(affine_scores, key=affine_scores.get)
        a, c = affine_params[chosen]
        for split, ref, key in (("val", ref_val, "val"), ("test", ref_test, "test")):
            raw = candidates[chosen][split]["predictions"][:, ti, hi] * std + mean
            calibrated = (a * raw + c - mean) / std
            methods["affine_selector"][key][:, ti, hi] = calibrated
        methods["affine_selector"]["choices"].append({
            "dataset": dataset, "horizon": h, "scope": "global",
            "selected": chosen, "val_RMSE": affine_scores[chosen],
            "a": a, "b": c,
        })

        # Station bias calibration selector.
        bias_scores = {}
        bias_params = {}
        for name, b in candidates.items():
            pv = b["val"]["predictions"][:, ti, hi] * std + mean
            bias = station_bias_fit(pv, yv, sidv)
            bias_params[name] = bias
            bias_scores[name] = rmse(apply_station_bias(pv, sidv, bias)[mv], yv[mv])
        chosen = min(bias_scores, key=bias_scores.get)
        bias = bias_params[chosen]
        for split, ref, key in (("val", ref_val, "val"), ("test", ref_test, "test")):
            raw = candidates[chosen][split]["predictions"][:, ti, hi] * std + mean
            calibrated = apply_station_bias(raw, ref["station_id"], bias)
            methods["station_bias_selector"][key][:, ti, hi] = (calibrated - mean) / std
        methods["station_bias_selector"]["choices"].append({
            "dataset": dataset, "horizon": h, "scope": "global",
            "selected": chosen, "val_RMSE": bias_scores[chosen],
        })

    metric_rows_all = []
    choice_rows = []
    for name, item in methods.items():
        for split, ref, arr in (
            ("val", ref_val, item["val"]), ("test", ref_test, item["test"])
        ):
            bundle = {
                "predictions": arr,
                "targets": ref["targets"],
                "target_mask": ref["target_mask"],
                "station_id": ref["station_id"],
                "anchor_time": ref["anchor_time"],
            }
            metric_rows_all.extend(metric_rows(bundle, cfg, scalers, name, split))
            np.savez_compressed(
                Path(cfg["paths"]["predictions_dir"]) / f"{name}_{split}.npz",
                **bundle,
                latency_ms_per_window=np.float64(0.0),
            )
        choice_rows.extend({"method": name, **r} for r in item["choices"])

    pd.DataFrame(metric_rows_all).to_csv(
        Path(cfg["paths"]["tables_dir"]) / "validation_selector_trials_pm25.csv",
        index=False,
    )
    pd.DataFrame(choice_rows).to_csv(
        Path(cfg["paths"]["tables_dir"]) / "validation_selector_trials_choices.csv",
        index=False,
    )
    return pd.DataFrame(metric_rows_all)


def main():
    all_rows = []
    comp_rows = []
    for dataset, (config_path, out_dir) in DATASETS.items():
        cfg = yaml.safe_load(Path(config_path).read_text())
        metrics = build_outputs(dataset, cfg, out_dir)
        metrics.insert(0, "dataset", dataset)
        all_rows.append(metrics)
        main = pd.read_csv(
            Path(cfg["paths"]["tables_dir"]) / "main_results_pm25.csv",
            header=[0, 1], index_col=0,
        )
        for h in cfg["dataset"]["horizons"]:
            prev = main[("RMSE", f"h{h}")].map(first_float).astype(float)
            best_prev = float(prev.min())
            for method in sorted(metrics["model"].unique()):
                sub = metrics[(metrics["split"] == "test")
                              & (metrics["model"] == method)
                              & (metrics["horizon"] == h)]
                comp_rows.append({
                    "dataset": dataset, "horizon": h, "method": method,
                    "previous_table_best": best_prev,
                    "test_RMSE": float(sub["RMSE"].iloc[0]),
                    "delta_vs_table_best": float(sub["RMSE"].iloc[0]) - best_prev,
                })
    all_metrics = pd.concat(all_rows, ignore_index=True)
    comp = pd.DataFrame(comp_rows)
    out = Path("outputs") / "tables"
    all_metrics.to_csv(out / "validation_selector_trials_all_pm25.csv", index=False)
    comp.to_csv(out / "validation_selector_trials_vs_table_best.csv", index=False)
    print(comp.pivot_table(index=["dataset", "method"], columns="horizon",
                           values="delta_vs_table_best").to_string())


if __name__ == "__main__":
    main()
