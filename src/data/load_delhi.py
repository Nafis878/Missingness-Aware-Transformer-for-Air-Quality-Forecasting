"""Loader for the Delhi Multi-Site Air-Quality dataset (CPCB; Mendeley
``bzhzr9b64v``).

Six CPCB monitoring stations in Delhi, hourly, 2018-06-01 .. 2019-10-01
(~11,704 rows/site), with natural missingness. Variables: PM2.5, PM10, NO,
NO2, NOx, NH3, SO2, CO, Ozone, Benzene plus meteorology (Temp, RH, WS, WD,
Solar Radiation, Barometric Pressure). One CSV per station.

This is the **intermediate-imputability** third network in the crossover study:
more complete and more structured than Dhaka, less complete than Beijing.

Output matches the Dhaka/Beijing tidy-frame contract exactly
(``[station, datetime, year, *measurement_cols]``, float64 measurements), so
:func:`src.data.clean.clean` and the whole downstream pipeline run unchanged
via ``--config config_delhi.yaml``.

Robustness to packaging variation (the public Mendeley archive does not
publish a fixed header spec): raw column headers are **canonicalized** —
the unit parenthetical is stripped (``"PM2.5 (ug/m3)" -> "PM2.5"``) and a small
alias map folds CPCB names onto the project's canonical names
(``Ozone -> O3``, ``NOx -> NOX``, ``Temperature -> Temp`` ...). Anything still
unmatched can be remapped from the config via ``data.column_rename``. The
station name is taken from the per-file ``station`` column if present, else the
file stem (with an optional ``data.station_rename`` map). Wind direction is
numeric **degrees** here (not compass strings), so it is converted to
``wd_sin``/``wd_cos`` with :func:`wd_deg_to_sin_cos`.

NOTE: the canonical column set, units and ``clean.bounds`` in
``config_delhi.yaml`` are sensible CPCB defaults; verify them against the
cleaning report produced on the first real run and adjust the config if a
column is missing or a bound clips legitimate values.
"""

from __future__ import annotations

import io
import logging
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load import LoadReport

logger = logging.getLogger(__name__)

CSV_GLOB = "*.csv"

#: Canonical names after stripping unit parentheticals. Folds the common CPCB
#: header spellings onto the project's canonical measurement names.
HEADER_ALIASES: dict[str, str] = {
    "PM2.5": "PM2.5", "PM 2.5": "PM2.5", "PM25": "PM2.5",
    "PM10": "PM10", "PM 10": "PM10",
    "NO": "NO", "NO2": "NO2",
    "NOx": "NOX", "NOX": "NOX",
    "NH3": "NH3", "SO2": "SO2", "CO": "CO",
    "Ozone": "O3", "O3": "O3", "OZONE": "O3",
    "Benzene": "Benzene",
    "Temp": "Temp", "Temperature": "Temp", "AT": "Temp",
    "RH": "RH", "Relative Humidity": "RH",
    "WS": "WS", "Wind Speed": "WS",
    "WD": "WD", "Wind Direction": "WD",
    "SR": "SR", "Solar Radiation": "SR", "Solar Radiaton": "SR",
    "BP": "BP", "Barometric Pressure": "BP", "Pressure": "BP",
    "RF": "Rain", "Rain": "Rain", "Rainfall": "Rain",
}

#: Common timestamp column spellings in CPCB exports (first match wins).
DATETIME_CANDIDATES = ["From Date", "datetime", "Datetime", "Date Time",
                       "Timestamp", "date", "Date"]


def download_delhi(raw_dir: str | Path, url: str | None = None,
                   force: bool = False) -> list[Path]:
    """Return per-station CSV paths, downloading a zip archive if configured.

    Cached: if CSVs already exist in ``raw_dir`` they are returned untouched.
    Otherwise, when ``url`` is given (a direct ``.zip`` link to the Mendeley
    archive), it is downloaded and every CSV inside is flattened into
    ``raw_dir``. If no URL is available (the Mendeley "Download All" link is
    session-scoped), download the archive manually from the dataset page and
    unzip the CSVs into ``raw_dir`` — this loader and the prep script then run
    offline, mirroring the Beijing fallback.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_dir.glob(CSV_GLOB))
    if existing and not force:
        logger.info("delhi: %d station CSVs already in %s, skipping download",
                    len(existing), raw_dir)
        return existing
    if not url:
        raise FileNotFoundError(
            f"no CSVs in {raw_dir} and no data.archive_url configured — "
            "download the Delhi Multi-Site archive from "
            "https://data.mendeley.com/datasets/bzhzr9b64v/1 and unzip the "
            "station CSVs into this directory, then re-run."
        )
    logger.info("delhi: downloading %s", url)
    with urllib.request.urlopen(url, timeout=180) as resp:
        payload = resp.read()
    logger.info("delhi: downloaded %.1f MB", len(payload) / 2**20)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:  # archive may nest a second zip
            inner = [m for m in zf.namelist() if m.lower().endswith(".zip")]
            if not inner:
                raise FileNotFoundError("no .csv files in the Delhi archive")
            with zipfile.ZipFile(io.BytesIO(zf.read(inner[0]))) as zf2:
                members = [m for m in zf2.namelist() if m.lower().endswith(".csv")]
                for m in members:
                    (raw_dir / Path(m).name).write_bytes(zf2.read(m))
        else:
            for m in members:
                (raw_dir / Path(m).name).write_bytes(zf.read(m))
    out = sorted(raw_dir.glob(CSV_GLOB))
    logger.info("delhi: extracted %d station CSVs to %s", len(out), raw_dir)
    return out


def canon_header(h: str) -> str:
    """Strip the unit parenthetical and fold onto a canonical name."""
    base = re.sub(r"\(.*?\)", "", str(h)).strip()
    return HEADER_ALIASES.get(base, base)


def wd_deg_to_sin_cos(wd: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Numeric wind-direction degrees -> (sin, cos); non-numeric/NaN -> NaN."""
    deg = pd.to_numeric(wd, errors="coerce")
    rad = np.deg2rad(deg.astype(float))
    return np.sin(rad), np.cos(rad)


def _station_name(df: pd.DataFrame, path: Path, cfg: dict[str, Any]) -> str:
    dcfg = cfg["data"]
    col = dcfg.get("station_col")
    if col and col in df.columns and df[col].notna().any():
        name = str(df[col].dropna().iloc[0]).strip()
    else:
        name = path.stem.strip()
    return dcfg.get("station_rename", {}).get(name, name)


def load_all_delhi(cfg: dict[str, Any]) -> tuple[pd.DataFrame, LoadReport]:
    """Load all Delhi station CSVs into one tidy frame + accounting report.

    Mirrors :func:`src.data.load.load_all`: returns
    ``[station, datetime, year, *measurement_cols]`` sorted by
    (station, datetime), measurements float64.
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    dcfg = cfg["data"]
    meas: list[str] = dcfg["measurement_cols"]
    extra_rename: dict[str, str] = dcfg.get("column_rename", {})
    report = LoadReport()
    paths = sorted(raw_dir.glob(CSV_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"no {CSV_GLOB} in {raw_dir} — run download_delhi() / "
            "scripts/01c_prepare_delhi.py, or place the Delhi CSVs there"
        )

    frames = []
    for path in paths:
        raw = pd.read_csv(path)
        report.rows_in[path.name] = len(raw)
        report.unit_rows[path.name] = 0
        # canonicalize headers (unit-strip + alias), then any config overrides
        raw = raw.rename(columns={c: canon_header(c) for c in raw.columns})
        if extra_rename:
            raw = raw.rename(columns=extra_rename)

        # timestamp: configured column or first known candidate
        dt_col = dcfg.get("datetime_col")
        if not dt_col or dt_col not in raw.columns:
            dt_col = next((c for c in DATETIME_CANDIDATES if c in raw.columns), None)
        if dt_col is None:
            raise KeyError(f"{path.name}: no timestamp column found "
                           f"(looked for {DATETIME_CANDIDATES}); set "
                           "data.datetime_col")
        dt = pd.to_datetime(raw[dt_col], errors="coerce", dayfirst=True)
        nat = dt.isna()
        report.nat_rows_non_unit[path.name] = int(nat.sum())
        raw = raw[~nat].copy()
        dt = dt[~nat]

        station = _station_name(raw, path, cfg)
        # numeric wind direction -> wd_sin/wd_cos (when WD present and requested)
        if "WD" in raw.columns and ("wd_sin" in meas or "wd_cos" in meas):
            raw["wd_sin"], raw["wd_cos"] = wd_deg_to_sin_cos(raw["WD"])

        coerce: dict[str, int] = {}
        for col in meas:
            if col not in raw.columns:
                continue
            before = int(raw[col].notna().sum())
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
            lost = before - int(raw[col].notna().sum())
            if lost:
                coerce[col] = lost
        report.coercion_failures[path.name] = coerce

        missing_cols = [c for c in meas if c not in raw.columns]
        if missing_cols:
            raise KeyError(
                f"{path.name}: missing expected columns {missing_cols}; "
                f"available after canonicalization: {sorted(raw.columns)}. "
                "Fix data.measurement_cols or add data.column_rename entries."
            )
        tidy = pd.DataFrame({"station": station, "datetime": dt.to_numpy()})
        tidy["year"] = pd.DatetimeIndex(dt).year
        for c in meas:
            tidy[c] = raw[c].astype(np.float64).to_numpy()
        report.rows_out[path.name] = len(tidy)
        frames.append(tidy)

    out = (pd.concat(frames, ignore_index=True)
           .sort_values(["station", "datetime"]).reset_index(drop=True))
    n_dup = int(out.duplicated(subset=["station", "datetime"]).sum())
    if n_dup:
        logger.warning("delhi: %d duplicated (station, datetime) rows", n_dup)
    logger.info("delhi: loaded %d rows, %d stations, %s .. %s",
                len(out), out["station"].nunique(),
                out["datetime"].min(), out["datetime"].max())
    return out, report
