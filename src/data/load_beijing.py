"""Loader for the Beijing Multi-Site Air Quality dataset (UCI id 501).

12 monitoring stations, hourly, 2013-03-01 .. 2017-02-28, with natural
missingness: PM2.5/PM10/SO2/NO2/CO/O3 (all ug/m3; note CO reaches ~10,000)
plus meteorology (TEMP, PRES, DEWP, RAIN, wd, WSPM). One CSV per station,
``PRSA_Data_<station>_20130301-20170228.csv``.

Output matches the Dhaka tidy-frame contract exactly
(``[station, datetime, year, *measurement_cols]``, float64 measurements), so
:func:`src.data.clean.clean` and the entire downstream pipeline run unchanged
via ``--config config_beijing.yaml``.

Dataset-specific handling:

* ``year/month/day/hour`` integer columns -> one ``datetime`` column.
* ``wd`` (16-point compass strings) -> ``wd_sin``/``wd_cos`` via the compass
  degree map; unknown tokens and missing wd become NaN in both (logged).
* Download is cached: the UCI zip is fetched only when the per-station CSVs
  are not already in ``data/raw/beijing/``.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load import LoadReport

logger = logging.getLogger(__name__)

#: 16-point compass -> degrees (N = 0, clockwise).
WD_DEGREES: dict[str, float] = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

CSV_GLOB = "PRSA_Data_*.csv"
#: Raw numeric columns taken from the CSVs as-is.
RAW_NUMERIC = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
               "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def download_beijing(raw_dir: str | Path, url: str, force: bool = False) -> list[Path]:
    """Download + extract the UCI zip unless the station CSVs already exist.

    Returns the sorted list of per-station CSV paths. The zip nests the CSVs
    inside a ``PRSA_Data_20130301-20170228/`` folder; they are flattened into
    ``raw_dir``.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_dir.glob(CSV_GLOB))
    if existing and not force:
        logger.info("beijing: %d station CSVs already in %s, skipping download",
                    len(existing), raw_dir)
        return existing

    logger.info("beijing: downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    logger.info("beijing: downloaded %.1f MB", len(payload) / 2**20)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = [m for m in zf.namelist()
                   if Path(m).name.startswith("PRSA_Data_")
                   and m.endswith(".csv")]
        # the outer zip may nest an inner zip with the CSVs
        if not members:
            inner = [m for m in zf.namelist() if m.endswith(".zip")]
            if not inner:
                raise FileNotFoundError("no PRSA_Data_*.csv files in the UCI zip")
            with zipfile.ZipFile(io.BytesIO(zf.read(inner[0]))) as zf2:
                members = [m for m in zf2.namelist()
                           if Path(m).name.startswith("PRSA_Data_")
                           and m.endswith(".csv")]
                for m in members:
                    (raw_dir / Path(m).name).write_bytes(zf2.read(m))
        else:
            for m in members:
                (raw_dir / Path(m).name).write_bytes(zf.read(m))
    out = sorted(raw_dir.glob(CSV_GLOB))
    logger.info("beijing: extracted %d station CSVs to %s", len(out), raw_dir)
    return out


def wd_to_sin_cos(wd: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compass strings -> (sin, cos) of the angle; unknown/missing -> NaN."""
    deg = wd.map(WD_DEGREES)
    unknown = wd.notna() & deg.isna()
    if unknown.any():
        logger.info("beijing: %d unknown wd tokens -> NaN (%s)",
                    int(unknown.sum()),
                    sorted(wd[unknown].astype(str).unique())[:8])
    rad = np.deg2rad(deg.astype(float))
    return np.sin(rad), np.cos(rad)


def load_all_beijing(cfg: dict[str, Any]) -> tuple[pd.DataFrame, LoadReport]:
    """Load all station CSVs into one tidy frame + accounting report.

    Mirrors :func:`src.data.load.load_all`: returns
    ``[station, datetime, year, *measurement_cols]`` sorted by
    (station, datetime), measurements float64.
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    meas: list[str] = cfg["data"]["measurement_cols"]
    report = LoadReport()
    paths = sorted(raw_dir.glob(CSV_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"no {CSV_GLOB} in {raw_dir} — run download_beijing() / "
            "scripts/01b_prepare_beijing.py, or place the UCI CSVs there"
        )

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        report.rows_in[path.name] = len(df)
        report.unit_rows[path.name] = 0  # CSVs have no embedded unit rows
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]], errors="coerce"
        )
        nat = df["datetime"].isna()
        report.nat_rows_non_unit[path.name] = int(nat.sum())
        df = df[~nat].copy()

        df["wd_sin"], df["wd_cos"] = wd_to_sin_cos(df["wd"])
        coerce: dict[str, int] = {}
        for col in RAW_NUMERIC:
            before = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            lost = int(before - df[col].notna().sum())
            if lost:
                coerce[col] = lost
        report.coercion_failures[path.name] = coerce

        keep = [c for c in meas if c in df.columns]
        missing_cols = [c for c in meas if c not in df.columns]
        if missing_cols:
            raise KeyError(f"{path.name}: missing expected columns {missing_cols}")
        tidy = df[["station", "datetime"]].copy()
        tidy["year"] = df["datetime"].dt.year
        for c in keep:
            tidy[c] = df[c].astype(np.float64)
        report.rows_out[path.name] = len(tidy)
        frames.append(tidy)

    out = (pd.concat(frames, ignore_index=True)
           .sort_values(["station", "datetime"]).reset_index(drop=True))
    logger.info("beijing: loaded %d rows, %d stations, %s .. %s",
                len(out), out["station"].nunique(),
                out["datetime"].min(), out["datetime"].max())
    return out, report
