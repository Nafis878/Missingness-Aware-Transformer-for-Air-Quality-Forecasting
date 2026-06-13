"""Loader for the Delhi Multi-Site Air-Quality dataset (CPCB; Mendeley
``bzhzr9b64v``, DOI 10.17632/bzhzr9b64v.1, CC BY 4.0).

Six CPCB monitoring stations in Delhi (AshokVihar, DCStadium, DwarkaSec8,
Najafgarh, NehruNagar, Okhla), hourly, 2018-06-01 .. 2019-10-01 (11,704
rows/site). One ``<Station>_Hourly.csv`` per station. Real header (verified
against the published files):

    ``,PM2.5,year,month,day,hour,PM10,AT,BP,SR,RH,WS,WD,NO,NO2,SO2,Ozone,CO,
      Benzene,NH3,NOx``

i.e. an unnamed index column, the timestamp as integer ``year/month/day/hour``
columns (as in Beijing — NOT a "From Date" string), ambient temperature as
``AT``, ``Ozone``/``NOx`` (folded to ``O3``/``NOX``), and numeric ``WD``
degrees. The published series is **already cleaned to completeness** (no missing
values, gap-free hourly), so Delhi is the *complete-network* anchor of the
crossover study: where it lands is governed by series **imputability**
(structure vs noise), not by natural missingness. The robustness suite injects
synthetic missingness and the imputability metric reconstructs held-out cells,
both of which work regardless of the (near-zero) natural gaps.

Output matches the Dhaka/Beijing tidy-frame contract exactly
(``[station, datetime, year, *measurement_cols]``, float64 measurements), so
:func:`src.data.clean.clean` and the whole downstream pipeline run unchanged
via ``--config config_delhi.yaml``.

Robustness to packaging variation: raw headers are **canonicalized** — the unit
parenthetical is stripped and a small alias map folds CPCB spellings onto the
project's canonical names (``Ozone -> O3``, ``NOx -> NOX``, ``AT -> Temp`` ...).
Anything still unmatched can be remapped from the config via
``data.column_rename``. The station name is the file stem with a trailing
``_Hourly`` removed (override via ``data.station_rename``). The timestamp is
built from ``year/month/day/hour`` when present, else parsed from a configured /
auto-detected datetime column. Numeric ``WD`` degrees -> ``wd_sin``/``wd_cos``.
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

#: Published file manifest (filename -> Mendeley file id) for auto-download.
#: The per-file URLs are content-addressed (stable), unlike the session-scoped
#: "Download All" link.
DELHI_FILES: dict[str, str] = {
    "AshokVihar_Hourly.csv": "22eebee2-c4aa-4d3f-b276-f72fbfdd0d55",
    "DCStadium_Hourly.csv": "a5db639a-4c2a-43a5-94e1-9cd99a2536f1",
    "DwarkaSec8_Hourly.csv": "d09aa1b1-2858-43bd-9e39-d1521bd7d989",
    "Najafgarh_Hourly.csv": "296426d0-1d00-4bc3-98bc-7c95aee8dc7c",
    "NehruNagar_Hourly.csv": "a6ac1c86-6e54-414b-825c-9a7cc259d309",
    "Okhla_Hourly.csv": "35e9438b-baa8-4e11-b97f-28da31498289",
}
MENDELEY_FILE_URL = ("https://data.mendeley.com/public-files/datasets/"
                     "bzhzr9b64v/files/{fid}/file_downloaded")

#: Canonical names after stripping unit parentheticals.
HEADER_ALIASES: dict[str, str] = {
    "PM2.5": "PM2.5", "PM 2.5": "PM2.5", "PM25": "PM2.5",
    "PM10": "PM10", "PM 10": "PM10",
    "NO": "NO", "NO2": "NO2",
    "NOx": "NOX", "NOX": "NOX",
    "NH3": "NH3", "SO2": "SO2", "CO": "CO",
    "Ozone": "O3", "O3": "O3", "OZONE": "O3",
    "Benzene": "Benzene",
    "AT": "Temp", "Temp": "Temp", "Temperature": "Temp",
    "RH": "RH", "Relative Humidity": "RH",
    "WS": "WS", "Wind Speed": "WS",
    "WD": "WD", "Wind Direction": "WD",
    "SR": "SR", "Solar Radiation": "SR",
    "BP": "BP", "Barometric Pressure": "BP", "Pressure": "BP",
    "RF": "Rain", "Rain": "Rain", "Rainfall": "Rain",
}

#: Used only if the integer year/month/day/hour columns are absent.
DATETIME_CANDIDATES = ["From Date", "datetime", "Datetime", "Date Time",
                       "Timestamp", "date", "Date"]
_YMDH = ["year", "month", "day", "hour"]


def download_delhi(raw_dir: str | Path, url: str | None = None,
                   force: bool = False) -> list[Path]:
    """Return per-station CSV paths, downloading them if needed.

    Cached: existing CSVs in ``raw_dir`` are returned untouched. Otherwise,
    if ``url`` (a direct ``.zip``) is given it is downloaded and its CSVs
    flattened into ``raw_dir``; if not, the six published station files are
    fetched individually from their content-addressed Mendeley URLs.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_dir.glob(CSV_GLOB))
    if existing and not force:
        logger.info("delhi: %d station CSVs already in %s, skipping download",
                    len(existing), raw_dir)
        return existing

    if url:
        logger.info("delhi: downloading archive %s", url)
        with urllib.request.urlopen(url, timeout=180) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                inner = [m for m in zf.namelist() if m.lower().endswith(".zip")]
                if not inner:
                    raise FileNotFoundError("no .csv files in the Delhi archive")
                with zipfile.ZipFile(io.BytesIO(zf.read(inner[0]))) as zf2:
                    for m in (x for x in zf2.namelist() if x.lower().endswith(".csv")):
                        (raw_dir / Path(m).name).write_bytes(zf2.read(m))
            else:
                for m in members:
                    (raw_dir / Path(m).name).write_bytes(zf.read(m))
    else:
        logger.info("delhi: downloading %d station files from Mendeley",
                    len(DELHI_FILES))
        for name, fid in DELHI_FILES.items():
            dest = raw_dir / name
            if dest.exists() and not force:
                continue
            file_url = MENDELEY_FILE_URL.format(fid=fid)
            with urllib.request.urlopen(file_url, timeout=180) as resp:
                dest.write_bytes(resp.read())
            logger.info("delhi: fetched %s (%.1f MB)", name,
                        dest.stat().st_size / 2**20)

    out = sorted(raw_dir.glob(CSV_GLOB))
    if not out:
        raise FileNotFoundError(
            f"no CSVs in {raw_dir} after download — set data.archive_url or "
            "place the Delhi CSVs there manually "
            "(https://data.mendeley.com/datasets/bzhzr9b64v/1)."
        )
    logger.info("delhi: %d station CSVs ready in %s", len(out), raw_dir)
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
        name = re.sub(r"_Hourly$", "", path.stem, flags=re.IGNORECASE).strip()
    return dcfg.get("station_rename", {}).get(name, name)


def _build_datetime(df: pd.DataFrame, dcfg: dict[str, Any],
                    path: Path) -> pd.Series:
    if set(_YMDH).issubset(df.columns):
        return pd.to_datetime(df[_YMDH].astype("Int64").astype(float),
                              errors="coerce")
    dt_col = dcfg.get("datetime_col")
    if not dt_col or dt_col not in df.columns:
        dt_col = next((c for c in DATETIME_CANDIDATES if c in df.columns), None)
    if dt_col is None:
        raise KeyError(f"{path.name}: no year/month/day/hour columns and no "
                       f"timestamp column found (looked for {DATETIME_CANDIDATES})")
    return pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True)


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
        raw = raw.rename(columns={c: canon_header(c) for c in raw.columns})
        if extra_rename:
            raw = raw.rename(columns=extra_rename)

        dt = _build_datetime(raw, dcfg, path)
        nat = dt.isna()
        report.nat_rows_non_unit[path.name] = int(nat.sum())
        raw = raw[~nat].copy()
        dt = dt[~nat]

        station = _station_name(raw, path, cfg)
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
