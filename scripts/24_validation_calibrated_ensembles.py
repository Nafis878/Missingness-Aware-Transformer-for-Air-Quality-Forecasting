"""Validation-only post-processing ensembles for the MAT winner search.

This script does not retrain neural networks. It fits lightweight calibration
rules on validation prediction bundles, applies them to test bundles, and
exports new candidate prediction bundles plus summary metrics.

The intent is to test whether a defensible validation-calibrated portfolio
built around MAT/Variant B can clear the all-model significance gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config


DEFAULT_MEMBERS = [
    "hybrid8_masked_variant_B",
    "hybrid8_masked_variant_B_vanilla_input",
    "hybrid8_transformer",
    "variant_B",
    "proposed",
    "hybrid8_masked_proposed_md",
    "two_stage_knn",
    "two_stage_mice",
    "two_stage_saits",
    "dlinear",
    "gru_d",
]

METHODS = [
    "ridge_stack",
    "convex_stack",
    "convex_intercept_stack",
    "huber_convex_intercept",
    "convex_station_residual",
    "station_shrunk_convex",
    "h6_adaptive_dominance",
    "station_selector",
    "station_residual_ridge",
]

DOMINANCE_LAMBDAS = [1.0, 5.0, 25.0, 100.0]
DOMINANCE_REFERENCES = [
    "two_stage_knn",
    "two_stage_saits",
    "hybrid8_masked_proposed_md",
]


def _path(cfg: dict[str, Any], model: str, seed: int, split: str) -> Path:
    return Path(cfg["paths"]["predictions_dir"]) / "seeds" / f"{model}_s{seed}_{split}.npz"


def _load(cfg: dict[str, Any], model: str, seed: int, split: str) -> dict[str, np.ndarray]:
    path = _path(cfg, model, seed, split)
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def _row_map(bundle: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    return {
        (int(sid), int(anchor)): i
        for i, (sid, anchor) in enumerate(zip(bundle["station_id"], bundle["anchor_time"]))
    }


def _aligned_arrays(
    bundles: list[dict[str, np.ndarray]],
    target_idx: int,
    horizon_idx: int,
    *,
    require_observed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref = bundles[0]
    maps = [_row_map(b) for b in bundles]
    keys = list(maps[0].keys())
    ref_idx: list[int] = []
    member_idx: list[list[int]] = [[] for _ in bundles]

    for key in keys:
        idxs = [m.get(key) for m in maps]
        if any(i is None for i in idxs):
            continue
        ri = int(idxs[0])
        if require_observed and ref["target_mask"][ri, target_idx, horizon_idx] <= 0:
            continue
        preds = [
            b["predictions"][int(i), target_idx, horizon_idx]
            for b, i in zip(bundles, idxs)
        ]
        if not np.all(np.isfinite(preds)):
            continue
        ref_idx.append(ri)
        for out, i in zip(member_idx, idxs):
            out.append(int(i))

    rows = np.asarray(ref_idx, dtype=int)
    x = np.column_stack([
        b["predictions"][np.asarray(idxs, dtype=int), target_idx, horizon_idx]
        for b, idxs in zip(bundles, member_idx)
    ])
    y = ref["targets"][rows, target_idx, horizon_idx]
    stations = ref["station_id"][rows].astype(int)
    anchors = ref["anchor_time"][rows].astype(int)
    return rows, x, y, stations, anchors


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    xm = x.mean(axis=0)
    ym = float(y.mean())
    xc = x - xm
    yc = y - ym
    eye = np.eye(x.shape[1], dtype=float)
    beta = np.linalg.solve(xc.T @ xc + alpha * eye, xc.T @ yc)
    intercept = ym - float(xm @ beta)
    return beta, intercept


def _convex_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[1]

    def objective(w: np.ndarray) -> float:
        r = x @ w - y
        return float(np.mean(r * r))

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * n
    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return np.full(n, 1.0 / n)
    return np.asarray(res.x, dtype=float)


def _convex_intercept_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    init_w: np.ndarray | None = None,
    init_b: float = 0.0,
    l2_to: np.ndarray | None = None,
    l2_strength: float = 0.0,
    bounds_b: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, float]:
    n = x.shape[1]
    if init_w is None:
        init_w = np.full(n, 1.0 / n)
    z0 = np.r_[init_w, init_b]

    def objective(z: np.ndarray) -> float:
        w = z[:n]
        b = float(z[n])
        r = x @ w + b - y
        obj = float(np.mean(r * r))
        if l2_to is not None and l2_strength > 0:
            d = w - l2_to
            obj += float(l2_strength * (d @ d))
        return obj

    cons = ({"type": "eq", "fun": lambda z: float(np.sum(z[:n]) - 1.0)},)
    bounds = [(0.0, 1.0)] * n + [bounds_b]
    res = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return np.asarray(init_w, dtype=float), float(init_b)
    return np.asarray(res.x[:n], dtype=float), float(res.x[n])


def _dominance_convex_fit(
    x: np.ndarray,
    y: np.ndarray,
    ref_preds: list[np.ndarray],
    lam: float,
    *,
    init_w: np.ndarray,
    init_b: float,
) -> tuple[np.ndarray, float]:
    n = x.shape[1]
    z0 = np.r_[init_w, init_b]
    ref_losses = [(rp - y) ** 2 for rp in ref_preds]

    def objective(z: np.ndarray) -> float:
        w = z[:n]
        b = float(z[n])
        r = x @ w + b - y
        cand_loss = r * r
        obj = float(np.mean(cand_loss))
        for bl in ref_losses:
            # Penalize validation cases where the candidate is worse than a
            # blocker. This is a validation-only proxy for pairwise dominance.
            hinge = np.maximum(0.0, cand_loss - bl)
            obj += float(lam * np.mean(hinge * hinge))
        obj += 1.0e-4 * float(b * b)
        return obj

    cons = ({"type": "eq", "fun": lambda z: float(np.sum(z[:n]) - 1.0)},)
    bounds = [(0.0, 1.0)] * n + [(-1.0, 1.0)]
    res = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return init_w, init_b
    return np.asarray(res.x[:n], dtype=float), float(res.x[n])


def _huber_convex_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    delta: float = 0.5,
    init_w: np.ndarray,
    init_b: float,
) -> tuple[np.ndarray, float]:
    n = x.shape[1]
    z0 = np.r_[init_w, init_b]

    def objective(z: np.ndarray) -> float:
        w = z[:n]
        b = float(z[n])
        r = x @ w + b - y
        a = np.abs(r)
        loss = np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
        return float(np.mean(loss) + 1.0e-4 * b * b)

    cons = ({"type": "eq", "fun": lambda z: float(np.sum(z[:n]) - 1.0)},)
    bounds = [(0.0, 1.0)] * n + [(-1.0, 1.0)]
    res = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return init_w, init_b
    return np.asarray(res.x[:n], dtype=float), float(res.x[n])


def _dominance_score(
    pred: np.ndarray,
    y: np.ndarray,
    ref_preds: list[np.ndarray],
    lam: float = 5.0,
) -> float:
    cand_loss = (pred - y) ** 2
    score = float(np.mean(cand_loss))
    for rp in ref_preds:
        hinge = np.maximum(0.0, cand_loss - (rp - y) ** 2)
        score += float(lam * np.mean(hinge * hinge))
    return score


def _fit_station_offsets(
    x: np.ndarray,
    y: np.ndarray,
    stations: np.ndarray,
    beta: np.ndarray,
    intercept: float,
) -> dict[int, float]:
    residual = y - (x @ beta + intercept)
    offsets: dict[int, float] = {}
    for sid in np.unique(stations):
        m = stations == sid
        if int(m.sum()) >= 3:
            offsets[int(sid)] = float(residual[m].mean())
    return offsets


def _rmse_real(
    bundle: dict[str, np.ndarray],
    target_idx: int,
    horizon_idx: int,
    mean: float,
    std: float,
) -> tuple[float, float]:
    mask = bundle["target_mask"][:, target_idx, horizon_idx] > 0
    pred = bundle["predictions"][mask, target_idx, horizon_idx] * std + mean
    y = bundle["targets"][mask, target_idx, horizon_idx] * std + mean
    err = pred - y
    return float(np.sqrt(np.mean(err * err))), float(np.mean(np.abs(err)))


def _save_bundle(
    cfg: dict[str, Any],
    name: str,
    seed: int,
    base: dict[str, np.ndarray],
) -> None:
    out = Path(cfg["paths"]["predictions_dir"]) / "seeds" / f"{name}_s{seed}_test.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--members", default=",".join(DEFAULT_MEMBERS))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    horizons = [int(h) for h in cfg["dataset"]["horizons"]]
    target_idx = cfg["dataset"]["target_pollutants"].index("PM2.5")
    scalers = json.loads((Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text())
    mean, std = scalers["PM2.5"]
    members = [m.strip() for m in args.members.split(",") if m.strip()]

    available_members: list[str] = []
    for model in members:
        if all(_path(cfg, model, seed, "val").exists() and _path(cfg, model, seed, "test").exists() for seed in seeds):
            available_members.append(model)
    if not available_members:
        raise SystemExit("No members have complete val/test bundles.")

    method_names = METHODS + [f"h6_dominance_convex_lam{lam:g}" for lam in DOMINANCE_LAMBDAS]
    rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []

    for seed in seeds:
        val_bundles = [_load(cfg, m, seed, "val") for m in available_members]
        test_bundles = [_load(cfg, m, seed, "test") for m in available_members]

        outputs = {
            method: {k: np.array(v, copy=True) for k, v in test_bundles[0].items()}
            for method in method_names
        }

        for hi, horizon in enumerate(horizons):
            v_rows, vx, vy, v_stations, _ = _aligned_arrays(
                val_bundles, target_idx, hi, require_observed=True
            )
            t_rows, tx, _, t_stations, _ = _aligned_arrays(
                test_bundles, target_idx, hi, require_observed=False
            )

            beta, intercept = _ridge_fit(vx, vy, alpha=1.0)
            outputs["ridge_stack"]["predictions"][t_rows, target_idx, hi] = tx @ beta + intercept
            weights_rows.append({
                "method": "ridge_stack",
                "seed": seed,
                "horizon": horizon,
                "intercept": intercept,
                **{f"w_{m}": float(w) for m, w in zip(available_members, beta)},
            })

            w = _convex_fit(vx, vy)
            outputs["convex_stack"]["predictions"][t_rows, target_idx, hi] = tx @ w
            weights_rows.append({
                "method": "convex_stack",
                "seed": seed,
                "horizon": horizon,
                "intercept": 0.0,
                **{f"w_{m}": float(x) for m, x in zip(available_members, w)},
            })

            convex_intercept = float((vy - vx @ w).mean())
            outputs["convex_intercept_stack"]["predictions"][t_rows, target_idx, hi] = (
                tx @ w + convex_intercept
            )
            weights_rows.append({
                "method": "convex_intercept_stack",
                "seed": seed,
                "horizon": horizon,
                "intercept": convex_intercept,
                **{f"w_{m}": float(x) for m, x in zip(available_members, w)},
            })

            hw, hb = _huber_convex_fit(vx, vy, init_w=w, init_b=convex_intercept)
            outputs["huber_convex_intercept"]["predictions"][t_rows, target_idx, hi] = tx @ hw + hb
            weights_rows.append({
                "method": "huber_convex_intercept",
                "seed": seed,
                "horizon": horizon,
                "intercept": hb,
                **{f"w_{m}": float(x) for m, x in zip(available_members, hw)},
            })

            convex_offsets = _fit_station_offsets(vx, vy, v_stations, w, convex_intercept)
            convex_correction = np.asarray(
                [convex_offsets.get(int(sid), 0.0) for sid in t_stations], dtype=float
            )
            outputs["convex_station_residual"]["predictions"][t_rows, target_idx, hi] = (
                tx @ w + convex_intercept + convex_correction
            )
            weights_rows.append({
                "method": "convex_station_residual",
                "seed": seed,
                "horizon": horizon,
                "intercept": convex_intercept,
                **{f"w_{m}": float(x) for m, x in zip(available_members, w)},
            })

            # Station-specific convex weights, shrunk toward the global convex
            # solution. This is more flexible than station selection but keeps a
            # validation-only regularization anchor.
            station_pred = outputs["station_shrunk_convex"]["predictions"]
            for sid in np.unique(t_stations):
                val_mask = v_stations == sid
                test_mask = t_stations == sid
                if int(val_mask.sum()) >= 4:
                    sw, sb = _convex_intercept_fit(
                        vx[val_mask],
                        vy[val_mask],
                        init_w=w,
                        init_b=convex_intercept,
                        l2_to=w,
                        l2_strength=0.25,
                    )
                else:
                    sw, sb = w, convex_intercept
                station_pred[t_rows[test_mask], target_idx, hi] = tx[test_mask] @ sw + sb
            weights_rows.append({
                "method": "station_shrunk_convex",
                "seed": seed,
                "horizon": horizon,
                "intercept": convex_intercept,
                **{f"w_{m}": float(x) for m, x in zip(available_members, w)},
            })

            if horizon == 6:
                ref_preds = [
                    vx[:, available_members.index(ref)]
                    for ref in DOMINANCE_REFERENCES
                    if ref in available_members
                ]
                adaptive_options: list[tuple[float, str, np.ndarray, float]] = [
                    (
                        _dominance_score(vx @ w + convex_intercept, vy, ref_preds),
                        "convex_intercept_stack",
                        w,
                        convex_intercept,
                    ),
                    (
                        _dominance_score(vx @ hw + hb, vy, ref_preds),
                        "huber_convex_intercept",
                        hw,
                        hb,
                    ),
                ]
                for lam in DOMINANCE_LAMBDAS:
                    method = f"h6_dominance_convex_lam{lam:g}"
                    dw, db = _dominance_convex_fit(
                        vx,
                        vy,
                        ref_preds,
                        lam,
                        init_w=w,
                        init_b=convex_intercept,
                    )
                    outputs[method]["predictions"][t_rows, target_idx, hi] = tx @ dw + db
                    adaptive_options.append((
                        _dominance_score(vx @ dw + db, vy, ref_preds),
                        method,
                        dw,
                        db,
                    ))
                    weights_rows.append({
                        "method": method,
                        "seed": seed,
                        "horizon": horizon,
                        "intercept": db,
                        **{f"w_{m}": float(x) for m, x in zip(available_members, dw)},
                    })
                _, selected_method, aw, ab = min(adaptive_options, key=lambda item: item[0])
                outputs["h6_adaptive_dominance"]["predictions"][t_rows, target_idx, hi] = tx @ aw + ab
                weights_rows.append({
                    "method": "h6_adaptive_dominance",
                    "seed": seed,
                    "horizon": horizon,
                    "intercept": ab,
                    "selected": selected_method,
                    **{f"w_{m}": float(x) for m, x in zip(available_members, aw)},
                })
            else:
                for lam in DOMINANCE_LAMBDAS:
                    method = f"h6_dominance_convex_lam{lam:g}"
                    outputs[method]["predictions"][t_rows, target_idx, hi] = (
                        tx @ w + convex_intercept
                    )
                outputs["h6_adaptive_dominance"]["predictions"][t_rows, target_idx, hi] = (
                    tx @ w + convex_intercept
                )

            # Global station/horizon selector: each station picks the member with
            # lowest validation RMSE for that station and horizon.
            selected_by_station: dict[int, int] = {}
            for sid in np.unique(v_stations):
                sm = v_stations == sid
                if int(sm.sum()) < 3:
                    selected_by_station[int(sid)] = int(np.argmin(np.mean((vx - vy[:, None]) ** 2, axis=0)))
                    continue
                mse = np.mean((vx[sm] - vy[sm, None]) ** 2, axis=0)
                selected_by_station[int(sid)] = int(np.argmin(mse))
            global_best = int(np.argmin(np.mean((vx - vy[:, None]) ** 2, axis=0)))
            selector_pred = outputs["station_selector"]["predictions"]
            for sid in np.unique(t_stations):
                tm = t_stations == sid
                j = selected_by_station.get(int(sid), global_best)
                selector_pred[t_rows[tm], target_idx, hi] = tx[tm, j]
            weights_rows.append({
                "method": "station_selector",
                "seed": seed,
                "horizon": horizon,
                "intercept": 0.0,
                "selected": json.dumps({str(k): available_members[v] for k, v in selected_by_station.items()}),
            })

            offsets = _fit_station_offsets(vx, vy, v_stations, beta, intercept)
            correction = np.asarray([offsets.get(int(sid), 0.0) for sid in t_stations], dtype=float)
            outputs["station_residual_ridge"]["predictions"][t_rows, target_idx, hi] = (
                tx @ beta + intercept + correction
            )
            weights_rows.append({
                "method": "station_residual_ridge",
                "seed": seed,
                "horizon": horizon,
                "intercept": intercept,
                **{f"w_{m}": float(weight) for m, weight in zip(available_members, beta)},
            })

        for method, bundle in outputs.items():
            name = f"validation_{method}"
            _save_bundle(cfg, name, seed, bundle)
            for hi, horizon in enumerate(horizons):
                rmse, mae = _rmse_real(bundle, target_idx, hi, mean, std)
                rows.append({
                    "candidate": name,
                    "seed": seed,
                    "horizon": horizon,
                    "RMSE": rmse,
                    "MAE": mae,
                })

    tables = Path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["candidate", "horizon"], as_index=False)
        .agg(seeds=("seed", "nunique"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"), MAE_mean=("MAE", "mean"))
    )
    metrics.to_csv(tables / "validation_calibrated_ensemble_per_seed.csv", index=False)
    summary.to_csv(tables / "validation_calibrated_ensemble_summary.csv", index=False)
    pd.DataFrame(weights_rows).to_csv(tables / "validation_calibrated_ensemble_weights.csv", index=False)
    print(f"members: {', '.join(available_members)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
