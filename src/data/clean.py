"""Cleaning: plausibility bounds, dedup, stuck-sensor detection, hourly reindex.

Philosophy: **flag-and-NaN, never silently drop rows.** Every rule logs the
exact number of values it removed, per column, into a :class:`CleanReport`
that feeds the markdown data-cleaning report.

Steps (in order):

1. Deduplicate exact (station, timestamp) pairs (keep first).
2. Plausibility bounds from config -> out-of-range values set to NaN.
3. Stuck-sensor detection: runs of >= N identical consecutive *non-zero*
   values per (station, variable) set to NaN (zero runs are legitimate:
   Rain = 0 for weeks, SR = 0 at night).
4. Reindex each station to its full hourly range so implicit gaps become
   explicit NaN rows (required for windowing and missingness analysis).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleanReport:
    """Accounting of values flagged by each cleaning rule."""

    duplicates_dropped: int = 0
    sentinel_flagged: dict[str, int] = field(default_factory=dict)
    bounds_flagged: dict[str, int] = field(default_factory=dict)
    stuck_flagged: dict[str, int] = field(default_factory=dict)
    rows_before_reindex: int = 0
    rows_after_reindex: int = 0
    station_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)


def apply_sentinels(
    df: pd.DataFrame, sentinels: dict[str, list[float]], report: CleanReport
) -> pd.DataFrame:
    """Set instrument error/saturation code values (exact match) to NaN.

    These are values like 985.0 that repeat verbatim hundreds of times at the
    instrument ceiling, including in seasons where such concentrations are
    physically impossible -- error codes, not measurements.
    """
    df = df.copy()
    for col, codes in (sentinels or {}).items():
        if col not in df.columns:
            continue
        bad = df[col].isin(codes)
        n_bad = int(bad.sum())
        report.sentinel_flagged[col] = n_bad
        if n_bad:
            logger.info("sentinel: %s -> %d exact matches of %s set to NaN",
                        col, n_bad, codes)
            df.loc[bad, col] = np.nan
    return df


def apply_bounds(
    df: pd.DataFrame, bounds: dict[str, list[float]], report: CleanReport
) -> pd.DataFrame:
    """Set values outside configured [lo, hi] plausibility bounds to NaN."""
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        bad = (df[col] < lo) | (df[col] > hi)
        n_bad = int(bad.sum())
        report.bounds_flagged[col] = n_bad
        if n_bad:
            logger.info(
                "bounds: %s -> %d values outside [%g, %g] set to NaN", col, n_bad, lo, hi
            )
            df.loc[bad, col] = np.nan
    return df


def flag_stuck_runs(series: pd.Series, min_run: int) -> pd.Series:
    """Return a boolean mask of values belonging to stuck-sensor runs.

    A stuck run is ``min_run`` or more consecutive identical **non-zero,
    non-NaN** values. NaNs break runs; zeros are exempt.
    """
    vals = series.to_numpy()
    same_as_prev = np.zeros(len(vals), dtype=bool)
    if len(vals) > 1:
        with np.errstate(invalid="ignore"):
            same_as_prev[1:] = (vals[1:] == vals[:-1]) & ~pd.isna(vals[1:])
    # run_id increments whenever the value changes (or is NaN).
    run_id = np.cumsum(~same_as_prev)
    run_lengths = pd.Series(run_id).groupby(run_id).transform("size").to_numpy()
    return (run_lengths >= min_run) & ~pd.isna(vals) & (vals != 0)


def apply_stuck_detection(
    df: pd.DataFrame,
    measurement_cols: list[str],
    min_run: int,
    report: CleanReport,
) -> pd.DataFrame:
    """Flag-and-NaN stuck-sensor runs per (station, variable)."""
    df = df.copy()
    for col in measurement_cols:
        if col not in df.columns:
            continue
        total = 0
        for _, grp in df.groupby("station", sort=False):
            mask = flag_stuck_runs(grp[col], min_run)
            if mask.any():
                df.loc[grp.index[mask], col] = np.nan
                total += int(mask.sum())
        report.stuck_flagged[col] = total
        if total:
            logger.info(
                "stuck-sensor: %s -> %d values in runs >= %d set to NaN", col, total, min_run
            )
    return df


def reindex_hourly(df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """Reindex each station to its full hourly range (implicit gaps -> NaN rows).

    Timestamps are floored to the hour first (raw data is on the hour already;
    this is a guard). The ``year`` column is recomputed from the index so the
    inserted rows carry it too.
    """
    report.rows_before_reindex = len(df)
    out = []
    for station, grp in df.groupby("station", sort=True):
        grp = grp.copy()
        grp["datetime"] = grp["datetime"].dt.floor("h")
        grp = grp.drop_duplicates(subset="datetime").set_index("datetime").sort_index()
        full = pd.date_range(grp.index.min(), grp.index.max(), freq="h")
        grp = grp.reindex(full)
        grp.index.name = "datetime"
        grp["station"] = station
        grp["year"] = grp.index.year
        report.station_ranges[station] = (str(full.min()), str(full.max()))
        out.append(grp.reset_index())
    res = pd.concat(out, ignore_index=True)
    report.rows_after_reindex = len(res)
    logger.info(
        "hourly reindex: %d -> %d rows (%d explicit gap rows inserted)",
        report.rows_before_reindex, len(res), len(res) - report.rows_before_reindex,
    )
    return res


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, CleanReport]:
    """Run the full cleaning pipeline. Returns (cleaned frame, report)."""
    report = CleanReport()
    meas = cfg["data"]["measurement_cols"]

    n_dup = int(df.duplicated(subset=["station", "datetime"]).sum())
    report.duplicates_dropped = n_dup
    if n_dup:
        logger.info("dropping %d duplicated (station, datetime) rows (keep first)", n_dup)
        df = df.drop_duplicates(subset=["station", "datetime"], keep="first")

    df = apply_sentinels(df, cfg["clean"].get("sentinel_values", {}), report)
    df = apply_bounds(df, cfg["clean"]["bounds"], report)
    df = apply_stuck_detection(df, meas, cfg["clean"]["stuck_run_length"], report)
    df = reindex_hourly(df, report)
    return df, report
