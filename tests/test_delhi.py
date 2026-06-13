"""Tests for the Delhi Multi-Site (CPCB) loader against synthetic fixtures.

No network and no real data needed: small CPCB-format CSVs (unit-suffixed
headers, dd-mm-yyyy "From Date", numeric wind direction, "None" missing tokens)
are written to tmp_path. Covers header canonicalization, datetime parsing,
station naming (column vs filename), wd-degree -> sin/cos, tidy-frame schema
parity with the Dhaka/Beijing contract, clean() + windowing integration, the
helpful missing-column error, and the download cache / manual fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import clean
from src.data.dataset import (
    AirQualityWindowDataset,
    build_station_arrays,
    compute_scalers,
)
from src.data.load_delhi import (
    DELHI_FILES,
    canon_header,
    download_delhi,
    load_all_delhi,
    wd_deg_to_sin_cos,
)

DELHI_COLS = ["PM2.5", "PM10", "NO2", "O3", "CO", "Temp", "RH", "WS",
              "wd_sin", "wd_cos"]


def delhi_cfg(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "paths": {"raw_dir": str(tmp_path / "raw")},
        "data": {
            "datetime_col": "From Date",
            "station_col": "station",
            "column_rename": {},
            "station_rename": {},
            "measurement_cols": DELHI_COLS,
            "exclude_features": [],
        },
        "clean": {
            "bounds": {
                "PM2.5": [0, 1000], "PM10": [0, 2000], "NO2": [0, 500],
                "O3": [0, 1000], "CO": [0, 50], "Temp": [-5, 52],
                "RH": [0, 100], "WS": [0, 30],
                "wd_sin": [-1, 1], "wd_cos": [-1, 1],
            },
            "stuck_run_length": 24,
            "sentinel_values": {},
        },
        "splits": {"train_end": "2018-06-25 23:00:00",
                   "val_end": "2018-07-01 23:00:00"},
        "dataset": {
            "input_length": 24,
            "horizons": [3, 6],
            "stride_train": 6,
            "stride_eval": 6,
            "target_pollutants": ["PM2.5", "PM10"],
            "primary_target": "PM2.5",
            "synthetic_missingness": [0.3],
        },
    }


def write_cpcb_csv(raw_dir: Path, station: str, fname: str | None = None,
                   n_hours: int = 24 * 40, seed: int = 0,
                   with_station_col: bool = True) -> Path:
    """Synthetic CPCB-format CSV: unit-suffixed headers, From/To Date,
    numeric WD, 'None' missing tokens."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2018-06-01", periods=n_hours, freq="h")
    to_times = times + pd.Timedelta(hours=1)
    fmt = "%d-%m-%Y %H:%M"
    data = {
        "From Date": times.strftime(fmt),
        "To Date": to_times.strftime(fmt),
        "PM2.5 (ug/m3)": rng.uniform(10, 400, n_hours),
        "PM10 (ug/m3)": rng.uniform(20, 600, n_hours),
        "NO2 (ug/m3)": rng.uniform(5, 150, n_hours),
        "Ozone (ug/m3)": rng.uniform(1, 200, n_hours),   # -> O3 via alias
        "CO (mg/m3)": rng.uniform(0.2, 6.0, n_hours),
        "Temp (degree C)": rng.uniform(15, 45, n_hours),
        "RH (%)": rng.uniform(10, 95, n_hours),
        "WS (m/s)": rng.uniform(0, 8, n_hours),
        "WD (deg)": rng.uniform(0, 360, n_hours),        # numeric degrees
    }
    df = pd.DataFrame(data)
    df["PM2.5 (ug/m3)"] = df["PM2.5 (ug/m3)"].astype(object)
    df.loc[20:50, "PM2.5 (ug/m3)"] = "None"  # CPCB missing token (default NA)
    if with_station_col:
        df["station"] = station
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / (fname or f"{station}.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Header canonicalization + wind direction
# ---------------------------------------------------------------------------

def test_canon_header_strips_units_and_aliases() -> None:
    assert canon_header("PM2.5 (ug/m3)") == "PM2.5"
    assert canon_header("Ozone (ug/m3)") == "O3"
    assert canon_header("NOx (ppb)") == "NOX"
    assert canon_header("Temperature (degree C)") == "Temp"
    assert canon_header("Solar Radiation (W/mt2)") == "SR"
    assert canon_header("WD (deg)") == "WD"


def test_wd_deg_to_sin_cos_unit_circle_and_invalid_to_nan() -> None:
    wd = pd.Series([0.0, 90.0, 180.0, 270.0, "junk", None])
    s, c = wd_deg_to_sin_cos(wd)
    valid = s[:4].to_numpy() ** 2 + c[:4].to_numpy() ** 2
    assert np.allclose(valid, 1.0)
    assert s.iloc[0] == pytest.approx(0.0, abs=1e-12)  # 0 deg
    assert c.iloc[0] == pytest.approx(1.0)
    assert s.iloc[1] == pytest.approx(1.0)             # 90 deg -> sin 1
    assert np.isnan(s.iloc[4]) and np.isnan(c.iloc[4])  # non-numeric
    assert np.isnan(s.iloc[5]) and np.isnan(c.iloc[5])  # missing


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_all_delhi_schema_matches_contract(tmp_path) -> None:
    cfg = delhi_cfg(tmp_path)
    raw = Path(cfg["paths"]["raw_dir"])
    write_cpcb_csv(raw, "AnandVihar", seed=0)
    write_cpcb_csv(raw, "RKPuram", seed=1)

    df, report = load_all_delhi(cfg)
    assert list(df.columns) == ["station", "datetime", "year"] + DELHI_COLS
    assert df["station"].nunique() == 2
    assert set(df["station"]) == {"AnandVihar", "RKPuram"}
    assert (df.groupby("station")["datetime"]
            .apply(lambda s: s.is_monotonic_increasing).all())
    for c in DELHI_COLS:
        assert df[c].dtype == np.float64, c
    # dd-mm-yyyy parsed day-first
    first = df[df["station"] == "AnandVihar"].iloc[0]
    assert first["datetime"] == pd.Timestamp("2018-06-01 00:00:00")
    # 'None' tokens became NaN
    assert df[df["station"] == "AnandVihar"]["PM2.5"].isna().sum() >= 31
    # wd present and on the unit circle where observed
    wsum = df["wd_sin"] ** 2 + df["wd_cos"] ** 2
    assert np.allclose(wsum.dropna(), 1.0)
    assert report.rows_in and report.rows_out


def test_load_all_delhi_station_from_filename(tmp_path) -> None:
    cfg = delhi_cfg(tmp_path)
    raw = Path(cfg["paths"]["raw_dir"])
    write_cpcb_csv(raw, "Ignored", fname="Punjabi_Bagh.csv",
                   with_station_col=False)
    df, _ = load_all_delhi(cfg)
    assert set(df["station"]) == {"Punjabi_Bagh"}  # from the file stem


def test_load_all_delhi_missing_csvs_raise(tmp_path) -> None:
    cfg = delhi_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_all_delhi(cfg)


def test_load_all_delhi_missing_column_raises_helpful(tmp_path) -> None:
    cfg = delhi_cfg(tmp_path)
    cfg["data"]["measurement_cols"] = DELHI_COLS + ["SO2"]  # not in fixture
    raw = Path(cfg["paths"]["raw_dir"])
    write_cpcb_csv(raw, "AnandVihar")
    with pytest.raises(KeyError, match="missing expected columns"):
        load_all_delhi(cfg)


# ---------------------------------------------------------------------------
# Clean + windowing integration
# ---------------------------------------------------------------------------

def test_delhi_clean_and_window_pipeline(tmp_path) -> None:
    cfg = delhi_cfg(tmp_path)
    raw = Path(cfg["paths"]["raw_dir"])
    write_cpcb_csv(raw, "AnandVihar", seed=0)
    write_cpcb_csv(raw, "RKPuram", seed=1)

    df, _ = load_all_delhi(cfg)
    df, clean_rep = clean(df, cfg)
    assert clean_rep.rows_after_reindex >= clean_rep.rows_before_reindex

    scalers = compute_scalers(df, cfg)
    stations = build_station_arrays(df, cfg, scalers)
    assert len(stations) == 2
    assert stations[0].values.shape[1] == len(DELHI_COLS)

    for split in ("train", "val", "test"):
        ds = AirQualityWindowDataset(stations, split, cfg)
        assert len(ds) > 0, f"no {split} windows"
        sample = ds[0]
        assert sample["values"].shape == (24, len(DELHI_COLS))
        assert (sample["values"][sample["mask"] == 0] == 0).all()


# ---------------------------------------------------------------------------
# Download cache / manual fallback
# ---------------------------------------------------------------------------

def test_download_delhi_short_circuits_when_csvs_exist(tmp_path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    write_cpcb_csv(raw, "AnandVihar")

    import urllib.request

    def boom(*a, **k):
        raise AssertionError("network must not be touched when CSVs exist")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    out = download_delhi(raw, "https://example.invalid/data.zip")
    assert len(out) == 1


class _FakeResp:
    def __init__(self, data: bytes) -> None:
        self._d = data

    def read(self) -> bytes:
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_delhi_auto_downloads_manifest(tmp_path, monkeypatch) -> None:
    """No URL + no CSVs -> fetch the 6 published station files (mocked)."""
    import urllib.request

    calls: list[str] = []

    def fake_urlopen(url, timeout=0):
        calls.append(url)
        return _FakeResp(b"PM2.5,year,month,day,hour\n1.0,2018,6,1,0\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = download_delhi(tmp_path / "raw", None)
    assert len(out) == len(DELHI_FILES) == 6
    assert all(u.endswith("file_downloaded") for u in calls)
