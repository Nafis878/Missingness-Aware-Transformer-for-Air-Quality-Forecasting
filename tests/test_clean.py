"""Unit tests for src/data/clean.py: bounds, dedup, stuck runs, hourly reindex."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import (
    CleanReport,
    apply_bounds,
    apply_sentinels,
    clean,
    flag_stuck_runs,
    reindex_hourly,
)

MEAS = ["PM2.5", "Temp"]
CFG = {
    "data": {"measurement_cols": MEAS},
    "clean": {
        "bounds": {"PM2.5": [0, 1000], "Temp": [-10, 55]},
        "stuck_run_length": 4,
    },
}


def _frame(**overrides) -> pd.DataFrame:
    n = 10
    base = {
        "station": ["A"] * n,
        "datetime": pd.date_range("2022-01-01", periods=n, freq="h"),
        "year": [2022] * n,
        "PM2.5": np.linspace(10, 100, n),
        "Temp": np.linspace(15, 25, n),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_bounds_flag_and_nan_counts() -> None:
    df = _frame(**{"PM2.5": [-5, 50, 1200, 30, 40, 50, 60, 70, 80, 90]})
    report = CleanReport()
    out = apply_bounds(df, CFG["clean"]["bounds"], report)
    assert report.bounds_flagged["PM2.5"] == 2  # -5 and 1200
    assert out["PM2.5"].isna().sum() == 2
    assert len(out) == len(df)  # rows never dropped


def test_sentinel_values_flagged() -> None:
    df = _frame(**{"PM2.5": [985.0, 50.0, 999.99, 30.0, 985.0, 50.0, 60.0, 70.0, 80.0, 90.0]})
    report = CleanReport()
    out = apply_sentinels(df, {"PM2.5": [985.0, 999.99]}, report)
    assert report.sentinel_flagged["PM2.5"] == 3
    assert out["PM2.5"].isna().sum() == 3
    assert out.loc[1, "PM2.5"] == 50.0  # nearby legitimate values untouched


def test_stuck_run_detection() -> None:
    s = pd.Series([1.0, 7.0, 7.0, 7.0, 7.0, 7.0, 2.0, 3.0])
    mask = flag_stuck_runs(s, min_run=4)
    assert mask.tolist() == [False, True, True, True, True, True, False, False]


def test_stuck_run_zero_exempt() -> None:
    """Long runs of exact zeros (Rain, nighttime SR) must NOT be flagged."""
    s = pd.Series([0.0] * 50 + [1.0, 2.0])
    assert not flag_stuck_runs(s, min_run=4).any()


def test_stuck_run_nan_breaks_run() -> None:
    s = pd.Series([5.0, 5.0, np.nan, 5.0, 5.0])
    assert not flag_stuck_runs(s, min_run=4).any()


def test_reindex_inserts_explicit_gaps() -> None:
    df = _frame()
    df = df.drop(index=[3, 4]).reset_index(drop=True)  # implicit 2-hour gap
    out = reindex_hourly(df, CleanReport())
    assert len(out) == 10  # full hourly range restored
    assert out["PM2.5"].isna().sum() == 2
    assert out["datetime"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert (out["year"] == 2022).all()  # year recomputed for inserted rows


def test_dedup_keeps_first() -> None:
    df = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    out, report = clean(df, CFG)
    assert report.duplicates_dropped == 1
    assert not out.duplicated(subset=["station", "datetime"]).any()


def test_clean_full_pipeline_row_count() -> None:
    """clean() must never lose observed in-bounds values."""
    df = _frame()
    out, report = clean(df, CFG)
    assert len(out) == len(df)
    assert out["PM2.5"].notna().sum() == df["PM2.5"].notna().sum()
