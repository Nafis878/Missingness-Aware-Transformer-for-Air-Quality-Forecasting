"""Raw data loading: read the yearly Excel files into one tidy DataFrame.

Handles every known quirk of the raw files:

* Each file is a concatenation of per-station blocks and **each block starts
  with its own unit row** (strings like ``ppb``, ``ug/m3``, ``degre-----``)
  -> detected by regex over all measurement columns and stripped.
* Columns are aligned strictly **by name** (the 2024 file has PM2.5/PM10 in
  swapped positional order; 2023 has an extra near-empty ``RS`` column).
* All measurement columns arrive as ``object`` dtype -> coerced with
  ``pd.to_numeric(errors="coerce")`` and the number of values lost to
  coercion is logged per column.
* Station name variants (``TV center``/``TV Sation``/``Narayangonj``/...)
  are normalized via the config rename map.
* Rows with NaT timestamps (unit rows plus a small number of genuine
  corrupt rows) are dropped with logging.

The per-file/per-column accounting is collected into a :class:`LoadReport`
so the cleaning report can cite exact numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    """Accounting of everything removed or altered during loading."""

    rows_in: dict[int, int] = field(default_factory=dict)
    rows_out: dict[int, int] = field(default_factory=dict)
    unit_rows: dict[int, int] = field(default_factory=dict)
    nat_rows_non_unit: dict[int, int] = field(default_factory=dict)
    coercion_failures: dict[int, dict[str, int]] = field(default_factory=dict)
    dropped_cols: dict[int, dict[str, int]] = field(default_factory=dict)
    station_renames: dict[str, str] = field(default_factory=dict)


def detect_unit_rows(
    df: pd.DataFrame, measurement_cols: list[str], unit_regex: str
) -> pd.Series:
    """Return a boolean mask of rows that are embedded unit/header rows.

    A row counts as a unit row if **any** measurement column contains a unit
    string (e.g. ``ppb``, ``ug/m3``, ``W/m2``). This catches the unit row at
    the top of every per-station block, not just row 0.
    """
    present = [c for c in measurement_cols if c in df.columns]
    hits = pd.DataFrame(
        {
            c: df[c].astype(str).str.contains(unit_regex, na=False, regex=True)
            for c in present
        }
    )
    return hits.any(axis=1)


def load_year(
    path: str | Path,
    year: int,
    cfg: dict[str, Any],
    report: LoadReport,
) -> pd.DataFrame:
    """Load one yearly Excel file into a tidy numeric DataFrame.

    Returns a frame with columns ``[station, datetime, year, *measurement_cols]``
    where every measurement column is float64 and unit/NaT rows are removed.
    """
    dcfg = cfg["data"]
    dt_col: str = dcfg["datetime_col"]
    st_col: str = dcfg["station_col"]
    meas: list[str] = dcfg["measurement_cols"]

    df = pd.read_excel(path, sheet_name=0)
    report.rows_in[year] = len(df)
    logger.info("%s: read %d rows, columns=%s", path, len(df), list(df.columns))

    # Drop known junk columns (e.g. 2023's RS) after logging their content.
    report.dropped_cols[year] = {}
    for col in dcfg.get("drop_cols", []):
        if col in df.columns:
            non_null = int(df[col].notna().sum())
            report.dropped_cols[year][col] = non_null
            logger.info(
                "%d: dropping column %r (%d non-null of %d rows)",
                year, col, non_null, len(df),
            )
            df = df.drop(columns=[col])

    missing = [c for c in [st_col, dt_col, *meas] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: expected columns missing: {missing}")

    # Strip embedded unit rows (one per station block).
    unit_mask = detect_unit_rows(df, meas, dcfg["unit_row_regex"])
    report.unit_rows[year] = int(unit_mask.sum())
    logger.info(
        "%d: stripping %d unit rows at positions %s",
        year, unit_mask.sum(), df.index[unit_mask].tolist(),
    )
    df = df.loc[~unit_mask]

    # Drop remaining NaT timestamps (genuine corrupt rows, distinct from unit rows).
    nat_mask = pd.to_datetime(df[dt_col], errors="coerce").isna()
    report.nat_rows_non_unit[year] = int(nat_mask.sum())
    if nat_mask.any():
        logger.info("%d: dropping %d non-unit rows with NaT timestamps", year, nat_mask.sum())
        df = df.loc[~nat_mask]
    df[dt_col] = pd.to_datetime(df[dt_col])

    # Normalize station names.
    df[st_col] = df[st_col].astype(str).str.strip()
    rename = dcfg.get("station_rename", {})
    applied = sorted(set(df[st_col]) & set(rename))
    for old in applied:
        report.station_renames[old] = rename[old]
        logger.info("%d: renaming station %r -> %r", year, old, rename[old])
    df[st_col] = df[st_col].replace(rename)

    # Coerce measurement columns by NAME and log coercion losses.
    report.coercion_failures[year] = {}
    for col in meas:
        before = int(df[col].notna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lost = before - int(df[col].notna().sum())
        report.coercion_failures[year][col] = lost
        if lost:
            logger.info("%d: column %s lost %d values to numeric coercion", year, col, lost)

    df = df.rename(columns={st_col: "station", dt_col: "datetime"})
    df["year"] = year
    # Select by NAME -> consistent column order regardless of raw file layout.
    df = df[["station", "datetime", "year", *meas]].reset_index(drop=True)
    report.rows_out[year] = len(df)
    logger.info("%d: %d rows after loading", year, len(df))
    return df


def load_all(cfg: dict[str, Any]) -> tuple[pd.DataFrame, LoadReport]:
    """Load all configured yearly files and concatenate them.

    Returns the combined frame (sorted by station, datetime) and the
    :class:`LoadReport` with full accounting.
    """
    report = LoadReport()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    frames = [
        load_year(raw_dir / fname, int(year), cfg, report)
        for year, fname in sorted(cfg["data"]["raw_files"].items())
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    n_dup = int(df.duplicated(subset=["station", "datetime"]).sum())
    if n_dup:
        logger.warning("combined frame has %d duplicated (station, datetime) rows", n_dup)
    logger.info(
        "combined: %d rows, %d stations: %s",
        len(df), df["station"].nunique(), sorted(df["station"].unique()),
    )
    return df, report
