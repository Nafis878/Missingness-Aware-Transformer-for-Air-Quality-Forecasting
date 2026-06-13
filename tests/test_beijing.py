"""Tests for the Beijing Multi-Site loader against synthetic PRSA fixtures.

No network and no real data needed: small PRSA-format CSVs are written to
tmp_path. Covers the compass mapping, datetime assembly, tidy-frame schema
parity with the Dhaka contract, clean() integration, windowing on the loaded
frame, and the download cache short-circuit.
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
from src.data.load_beijing import (
    WD_DEGREES,
    download_beijing,
    load_all_beijing,
    wd_to_sin_cos,
)

BEIJING_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                "TEMP", "PRES", "DEWP", "RAIN", "wd_sin", "wd_cos", "WSPM"]


def beijing_cfg(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "paths": {"raw_dir": str(tmp_path / "raw")},
        "data": {
            "measurement_cols": BEIJING_COLS,
            "exclude_features": [],
        },
        "clean": {
            "bounds": {
                "PM2.5": [0, 1000], "PM10": [0, 2000], "SO2": [0, 1000],
                "NO2": [0, 500], "CO": [0, 20000], "O3": [0, 1200],
                "TEMP": [-30, 45], "PRES": [950, 1060], "DEWP": [-45, 35],
                "RAIN": [0, 200], "wd_sin": [-1, 1], "wd_cos": [-1, 1],
                "WSPM": [0, 60],
            },
            "stuck_run_length": 24,
            "sentinel_values": {},
        },
        "splits": {"train_end": "2013-03-25 23:00:00",
                   "val_end": "2013-04-01 23:00:00"},
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


def write_prsa_csv(raw_dir: Path, station: str, n_hours: int = 24 * 40,
                   seed: int = 0) -> Path:
    """Synthetic PRSA-format CSV: real column layout, injected NaNs + wd."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2013-03-01", periods=n_hours, freq="h")
    wd = rng.choice(list(WD_DEGREES), size=n_hours).astype(object)
    wd[10] = np.nan          # missing compass reading
    wd[11] = "XX"            # invalid token
    df = pd.DataFrame({
        "No": np.arange(1, n_hours + 1),
        "year": times.year, "month": times.month,
        "day": times.day, "hour": times.hour,
        "PM2.5": rng.uniform(5, 300, n_hours),
        "PM10": rng.uniform(10, 400, n_hours),
        "SO2": rng.uniform(1, 80, n_hours),
        "NO2": rng.uniform(5, 150, n_hours),
        "CO": rng.uniform(100, 8000, n_hours),
        "O3": rng.uniform(1, 250, n_hours),
        "TEMP": rng.uniform(-10, 35, n_hours),
        "PRES": rng.uniform(990, 1040, n_hours),
        "DEWP": rng.uniform(-20, 25, n_hours),
        # mostly dry with occasional rain (all-zero would trip the
        # zero-variance scaler guard, which real Beijing RAIN does not)
        "RAIN": np.where(rng.random(n_hours) < 0.05,
                         rng.uniform(0.1, 20, n_hours), 0.0),
        "wd": wd,
        "WSPM": rng.uniform(0, 10, n_hours),
        "station": station,
    })
    df.loc[20:50, "PM2.5"] = np.nan  # natural-missingness hole
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"PRSA_Data_{station}_20130301-20170228.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Compass mapping
# ---------------------------------------------------------------------------

def test_wd_degrees_covers_all_16_compass_points() -> None:
    assert len(WD_DEGREES) == 16
    degs = sorted(WD_DEGREES.values())
    assert degs == [i * 22.5 for i in range(16)]


def test_wd_to_sin_cos_unit_circle_and_invalid_to_nan() -> None:
    wd = pd.Series(list(WD_DEGREES) + ["XX", None])
    s, c = wd_to_sin_cos(wd)
    valid = s[:16].to_numpy() ** 2 + c[:16].to_numpy() ** 2
    assert np.allclose(valid, 1.0)
    assert np.isnan(s.iloc[16]) and np.isnan(c.iloc[16])  # invalid token
    assert np.isnan(s.iloc[17]) and np.isnan(c.iloc[17])  # missing
    # N = 0 deg -> sin 0, cos 1
    assert s.iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert c.iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_all_beijing_schema_matches_dhaka_contract(tmp_path) -> None:
    cfg = beijing_cfg(tmp_path)
    raw = Path(cfg["paths"]["raw_dir"])
    write_prsa_csv(raw, "Aoti", seed=0)
    write_prsa_csv(raw, "Dingling", seed=1)

    df, report = load_all_beijing(cfg)
    assert list(df.columns) == ["station", "datetime", "year"] + BEIJING_COLS
    assert df["station"].nunique() == 2
    assert df["datetime"].is_monotonic_increasing or True  # sorted per station
    assert (df.groupby("station")["datetime"]
            .apply(lambda s: s.is_monotonic_increasing).all())
    for c in BEIJING_COLS:
        assert df[c].dtype == np.float64, c
    # datetime assembled from year/month/day/hour
    first = df[df["station"] == "Aoti"].iloc[0]
    assert first["datetime"] == pd.Timestamp("2013-03-01 00:00:00")
    # injected NaN hole survives
    assert df[df["station"] == "Aoti"]["PM2.5"].isna().sum() >= 31
    assert report.rows_in and report.rows_out


def test_load_all_beijing_missing_csvs_raise(tmp_path) -> None:
    cfg = beijing_cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="PRSA_Data"):
        load_all_beijing(cfg)


# ---------------------------------------------------------------------------
# Clean + windowing integration
# ---------------------------------------------------------------------------

def test_beijing_clean_and_window_pipeline(tmp_path) -> None:
    cfg = beijing_cfg(tmp_path)
    raw = Path(cfg["paths"]["raw_dir"])
    write_prsa_csv(raw, "Aoti", seed=0)
    write_prsa_csv(raw, "Dingling", seed=1)

    df, _ = load_all_beijing(cfg)
    df, clean_rep = clean(df, cfg)
    assert clean_rep.rows_after_reindex >= clean_rep.rows_before_reindex

    scalers = compute_scalers(df, cfg)
    stations = build_station_arrays(df, cfg, scalers)
    assert len(stations) == 2
    assert stations[0].values.shape[1] == len(BEIJING_COLS)

    sizes = {}
    for split in ("train", "val", "test"):
        ds = AirQualityWindowDataset(stations, split, cfg)
        sizes[split] = len(ds)
        assert len(ds) > 0, f"no {split} windows"
        sample = ds[0]
        assert sample["values"].shape == (24, len(BEIJING_COLS))
        # contract: missing cells are exactly 0 with mask 0
        assert (sample["values"][sample["mask"] == 0] == 0).all()
    assert sizes["train"] > sizes["val"]


# ---------------------------------------------------------------------------
# Download cache
# ---------------------------------------------------------------------------

def test_download_beijing_short_circuits_when_csvs_exist(tmp_path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    write_prsa_csv(raw, "Aoti")

    def boom(*args, **kwargs):  # any network use fails the test
        raise AssertionError("network must not be touched when CSVs exist")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    out = download_beijing(raw, "https://example.invalid/data.zip")
    assert len(out) == 1 and out[0].name.startswith("PRSA_Data_Aoti")
