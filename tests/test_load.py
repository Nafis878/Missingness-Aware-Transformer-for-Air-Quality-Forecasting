"""Unit tests for src/data/load.py: unit-row stripping, name-based column
alignment, station renaming, coercion accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import LoadReport, detect_unit_rows, load_year

MEAS = ["SO2", "NO", "NO2", "NOX", "CO", "O3", "PM10", "PM2.5",
        "WS", "WD", "Temp", "RH", "BP", "SR", "Rain", "VWS"]

CFG = {
    "data": {
        "datetime_col": "Date & Time",
        "station_col": "Location",
        "measurement_cols": MEAS,
        "drop_cols": ["RS"],
        "unit_row_regex": r"ppb|ppm|ug/m3|W/m2|HPa|degre|m\s?m|m/s",
        "station_rename": {"TV Sation": "TV Center", "Narayangonj": "Narayanganj"},
    }
}


def _make_raw_frame(swap_pm: bool = False, with_rs: bool = False) -> pd.DataFrame:
    """Two station blocks, each starting with a unit row (as in the real files)."""
    unit = {"Location": "X", "Date & Time": pd.NaT,
            **{c: "ppb" for c in MEAS}}
    unit["PM2.5"] = "ug/m3"
    unit["WD"] = "degre-----"

    def data_row(station: str, hour: int, pm25: float) -> dict:
        row = {"Location": station,
               "Date & Time": pd.Timestamp(f"2022-01-01 {hour:02d}:00:00"),
               **{c: 1.0 for c in MEAS}}
        row["PM2.5"] = pm25
        return row

    rows = [
        {**unit, "Location": "TV Sation"},
        data_row("TV Sation", 1, 50.0),
        data_row("TV Sation", 2, 60.0),
        {**unit, "Location": "Narayangonj"},
        data_row("Narayangonj", 1, 70.0),
        data_row("Narayangonj", 2, "bad-value"),  # coercion failure
    ]
    df = pd.DataFrame(rows)
    if with_rs:
        df["RS"] = None
    if swap_pm:
        cols = list(df.columns)
        i, j = cols.index("PM10"), cols.index("PM2.5")
        cols[i], cols[j] = cols[j], cols[i]
        df = df[cols]  # swapped positional order, same names
    return df


def _load(df: pd.DataFrame, tmp_path: Path) -> tuple[pd.DataFrame, LoadReport]:
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)
    report = LoadReport()
    return load_year(path, 2022, CFG, report), report


def test_detect_unit_rows_mid_file() -> None:
    df = _make_raw_frame()
    mask = detect_unit_rows(df, MEAS, CFG["data"]["unit_row_regex"])
    assert mask.sum() == 2
    assert mask.iloc[0] and mask.iloc[3]  # one unit row per station block


def test_unit_rows_stripped_and_counted(tmp_path: Path) -> None:
    out, report = _load(_make_raw_frame(), tmp_path)
    assert report.unit_rows[2022] == 2
    assert len(out) == 4
    # no unit strings survive anywhere
    assert not out[MEAS].astype(str).apply(
        lambda c: c.str.contains("ppb|ug/m3", na=False)).any().any()


def test_column_alignment_by_name_not_position(tmp_path: Path) -> None:
    """The 2024-style PM2.5/PM10 positional swap must not corrupt values."""
    normal, _ = _load(_make_raw_frame(swap_pm=False), tmp_path)
    swapped, _ = _load(_make_raw_frame(swap_pm=True), tmp_path)
    pd.testing.assert_series_equal(normal["PM2.5"], swapped["PM2.5"])
    assert normal.loc[0, "PM2.5"] == 50.0
    assert swapped.loc[0, "PM2.5"] == 50.0
    assert (swapped["PM10"].dropna() == 1.0).all()


def test_station_rename(tmp_path: Path) -> None:
    out, report = _load(_make_raw_frame(), tmp_path)
    assert set(out["station"]) == {"TV Center", "Narayanganj"}
    assert report.station_renames == {
        "TV Sation": "TV Center", "Narayangonj": "Narayanganj"}


def test_coercion_failures_logged(tmp_path: Path) -> None:
    out, report = _load(_make_raw_frame(), tmp_path)
    assert report.coercion_failures[2022]["PM2.5"] == 1  # the "bad-value" cell
    assert out["PM2.5"].dtype == float
    assert out["PM2.5"].isna().sum() == 1


def test_rs_column_dropped(tmp_path: Path) -> None:
    out, report = _load(_make_raw_frame(with_rs=True), tmp_path)
    assert "RS" not in out.columns
    assert report.dropped_cols[2022]["RS"] == 0  # all-null in synthetic frame


def test_missing_expected_column_raises(tmp_path: Path) -> None:
    df = _make_raw_frame().drop(columns=["O3"])
    path = tmp_path / "bad.xlsx"
    df.to_excel(path, index=False)
    with pytest.raises(ValueError, match="O3"):
        load_year(path, 2022, CFG, LoadReport())
