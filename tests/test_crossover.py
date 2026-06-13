"""Tests for the missingness-severity crossover study and window stratification.

Pure-table logic tested with synthetic bundles + a synthetic robustness_levels
map, mirroring tests/test_multiseed.py. The crossover interpolation is checked
against a constructed monotone gap with a known zero-crossing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (
    crossover_long,
    crossover_points,
    robustness_long,
    stratified_gap_table,
)

SCALERS = {"PM2.5": [50.0, 30.0], "PM10": [80.0, 40.0]}


def cfg(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "paths": {"predictions_dir": str(tmp_path / "pred"),
                  "outputs_dir": str(tmp_path)},
        "dataset": {"target_pollutants": ["PM2.5", "PM10"],
                    "horizons": [6, 24],
                    "synthetic_missingness": [0.1, 0.3, 0.5],
                    "primary_target": "PM2.5"},
        "robustness": {"direct": ["proposed", "proposed_md"],
                       "two_stage": ["two_stage_saits"]},
    }


def bundle(n: int, err_scaled: float) -> dict:
    """(n,2,2) bundle whose every prediction is off by err_scaled (scaled)."""
    y = np.zeros((n, 2, 2), np.float32)
    p = y + err_scaled
    return {"predictions": p, "targets": y,
            "target_mask": np.ones((n, 2, 2), np.float32),
            "station_id": np.zeros(n, np.int64),
            "anchor_time": (np.arange(n) * 3600).astype(np.int64)}


def write(path: Path, b: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **b)


def test_robustness_long_and_crossover_zero_crossing(tmp_path) -> None:
    c = cfg(tmp_path)
    pred = Path(c["paths"]["predictions_dir"])
    levels = [0, 10, 30, 50]
    # end-to-end error grows slowly with severity; two-stage grows fast and
    # starts lower -> the gap (ts - e2e) crosses zero somewhere in the middle.
    # scaled error -> RMSE in ug/m3 = err * 30 (std).
    e2e_err = {0: 0.5, 10: 0.55, 30: 0.6, 50: 0.7}      # 15.0 .. 21.0 ug/m3
    ts_err = {0: 0.40, 10: 0.50, 30: 0.65, 50: 0.85}    # 12.0 .. 25.5 ug/m3
    for lv in levels:
        for mode in ("miss", "out"):
            for name, errs in (("proposed", e2e_err), ("proposed_md", e2e_err),
                               ("two_stage_saits", ts_err)):
                suffix = "test" if lv == 0 else f"test_{mode}{lv}"
                write(pred / f"{name}_{suffix}.npz", bundle(200, errs[lv]))
    # effective-missingness map (monotone with level)
    (tmp_path / "robustness_levels.json").write_text(json.dumps({
        "clean": 0.10,
        **{f"{m}{lv}": round(0.10 + lv / 100.0, 3)
           for m in ("miss", "out") for lv in (10, 30, 50)},
    }))

    rob = robustness_long(c, SCALERS)
    assert rob is not None and set(rob["level"]) == {0, 10, 30, 50}

    cross = crossover_long(c, rob)
    assert cross is not None
    # gap = best_two_stage - best_end_to_end, in ug/m3
    g0 = cross[(cross["mode"] == "out") & (cross["horizon"] == 6)
               & (cross["level"] == 0)]["gap"].iloc[0]
    g50 = cross[(cross["mode"] == "out") & (cross["horizon"] == 6)
                & (cross["level"] == 50)]["gap"].iloc[0]
    assert g0 < 0 < g50, "two-stage better when clean, end-to-end better when severe"

    pts = crossover_points(cross)
    cp = pts[(pts["mode"] == "out") & (pts["horizon"] == 6)
             ]["crossover_missing_pct"].iloc[0]
    # crossing must be a number strictly inside the tested effective range
    assert isinstance(cp, (int, float))
    assert 10.0 < cp < 60.0
    assert "end-to-end above" in pts[(pts["mode"] == "out")
                                     & (pts["horizon"] == 6)]["recommendation"].iloc[0]


def test_crossover_points_handles_no_crossing(tmp_path) -> None:
    # gap positive everywhere -> end-to-end always wins -> "<min"
    cross = pd.DataFrame({
        "mode": ["out"] * 3, "horizon": [6] * 3, "level": [0, 30, 50],
        "eff_missing_pct": [10.0, 40.0, 60.0],
        "best_two_stage": [20.0, 25.0, 30.0],
        "best_end_to_end": [18.0, 20.0, 22.0],
        "gap": [2.0, 5.0, 8.0],
    })
    pts = crossover_points(cross)
    assert pts["crossover_missing_pct"].iloc[0] == "<min"

    cross["gap"] = [-2.0, -5.0, -8.0]  # two-stage always wins
    assert crossover_points(cross)["crossover_missing_pct"].iloc[0] == ">max"


def test_stratified_gap_widens_with_window_missingness(tmp_path) -> None:
    """Inject a known per-window missingness and a bundle where the end-to-end
    advantage is larger on high-missingness windows; the gap must increase."""
    c = cfg(tmp_path)
    pred = Path(c["paths"]["predictions_dir"])
    n = 400
    sid = np.zeros(n, np.int64)
    at = (np.arange(n) * 3600).astype(np.int64)
    # first half low missingness, second half high missingness
    wim = {(0, int(a)): (0.1 if i < n // 2 else 0.6) for i, a in enumerate(at)}
    high = np.arange(n) >= n // 2

    pred.mkdir(parents=True, exist_ok=True)
    y = np.zeros((n, 2, 2), np.float32)
    # end-to-end: small error everywhere
    pe = y + 0.2
    # two-stage: matches on low-missingness windows, much worse on high ones
    pt = y + np.where(high, 0.9, 0.2)[:, None, None]
    for name, p in (("proposed_md", pe), ("two_stage_saits", pt)):
        np.savez_compressed(
            pred / f"{name}_test.npz", predictions=p.astype(np.float32),
            targets=y, target_mask=np.ones((n, 2, 2), np.float32),
            station_id=sid, anchor_time=at)

    tbl = stratified_gap_table(c, SCALERS, n_bins=2, horizon=6, wim=wim)
    assert tbl is not None and len(tbl) == 2
    gaps = tbl["gap (two-stage − end-to-end)"].to_numpy()
    assert gaps[0] < gaps[1], "gap must grow on the higher-missingness bin"
    assert gaps[1] > 0


def test_crossover_returns_none_without_levels_map(tmp_path) -> None:
    c = cfg(tmp_path)
    pred = Path(c["paths"]["predictions_dir"])
    for lv in (0, 10, 30, 50):
        for mode in ("miss", "out"):
            for name in ("proposed", "proposed_md", "two_stage_saits"):
                suffix = "test" if lv == 0 else f"test_{mode}{lv}"
                write(pred / f"{name}_{suffix}.npz", bundle(200, 0.5))
    rob = robustness_long(c, SCALERS)
    # no robustness_levels.json written -> crossover cannot place the x-axis
    assert crossover_long(c, rob) is None
