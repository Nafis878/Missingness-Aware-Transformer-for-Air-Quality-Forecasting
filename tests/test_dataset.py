"""Unit tests for src/data/dataset.py.

Covers the four reproducibility-critical guarantees:
1. scalers are fit on the training period ONLY;
2. no temporal leakage between splits;
3. a masked value can never leak into the loss (values==0 where mask==0,
   masked-MSE invariant to perturbations of masked cells);
4. synthetic extra-missingness is deterministic and only flips observed cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (
    AirQualityWindowDataset,
    ExtraMissingnessDataset,
    build_station_arrays,
    build_window_index,
    compute_scalers,
    feature_columns,
    split_ranges,
)

VARS = ["PM2.5", "PM10", "Temp"]


def make_cfg(**dataset_overrides) -> dict:
    ds = {
        "input_length": 24,
        "horizons": [3, 6],
        "stride_train": 6,
        "stride_eval": 6,
        "target_pollutants": ["PM2.5", "PM10"],
        "primary_target": "PM2.5",
        "synthetic_missingness": [0.3],
    }
    ds.update(dataset_overrides)
    return {
        "data": {"measurement_cols": VARS, "exclude_features": []},
        "splits": {"train_end": "2022-03-31 23:00:00", "val_end": "2022-04-30 23:00:00"},
        "dataset": ds,
        "seed": 42,
    }


def make_frame(n_hours: int = 24 * 31 * 5, stations: tuple[str, ...] = ("A",),
               start: str = "2022-01-01") -> pd.DataFrame:
    """Gap-free hourly frame Jan-May 2022 with deterministic values."""
    rng = np.random.default_rng(0)
    frames = []
    for st in stations:
        times = pd.date_range(start, periods=n_hours, freq="h")
        df = pd.DataFrame({
            "station": st,
            "datetime": times,
            "year": times.year,
            "PM2.5": 50 + 10 * rng.standard_normal(n_hours),
            "PM10": 90 + 20 * rng.standard_normal(n_hours),
            "Temp": 25 + 5 * rng.standard_normal(n_hours),
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Scalers
# ---------------------------------------------------------------------------

def test_scaler_fit_on_train_only() -> None:
    """Val/test rows get a shifted distribution; scaler must ignore them."""
    cfg = make_cfg()
    df = make_frame()
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    df.loc[df["datetime"] > train_end, "PM2.5"] += 1000.0  # poison non-train rows

    scalers = compute_scalers(df, cfg)
    train_vals = df.loc[df["datetime"] <= train_end, "PM2.5"]
    assert scalers["PM2.5"][0] == pytest.approx(train_vals.mean(), rel=1e-9)
    assert scalers["PM2.5"][1] == pytest.approx(train_vals.std(ddof=0), rel=1e-9)
    assert scalers["PM2.5"][0] < 100  # poisoned values did not contaminate


def test_scaler_raises_on_all_missing_train() -> None:
    cfg = make_cfg()
    df = make_frame()
    df.loc[df["datetime"] <= pd.Timestamp(cfg["splits"]["train_end"]), "Temp"] = np.nan
    with pytest.raises(ValueError, match="Temp"):
        compute_scalers(df, cfg)


# ---------------------------------------------------------------------------
# 2. No temporal leakage
# ---------------------------------------------------------------------------

def _build(cfg, df):
    scalers = compute_scalers(df, cfg)
    stations = build_station_arrays(df, cfg, scalers)
    return stations


def test_no_temporal_leakage_between_splits() -> None:
    cfg = make_cfg()
    df = make_frame(stations=("A", "B"))
    stations = _build(cfg, df)
    ranges = split_ranges(cfg)
    horizons = cfg["dataset"]["horizons"]

    seen: dict[str, set[tuple[int, int]]] = {}
    for split in ("train", "val", "test"):
        idx, _ = build_window_index(stations, split, cfg)
        assert len(idx) > 0
        lo, hi = ranges[split]
        for s_i, anchor in idx:
            times = pd.DatetimeIndex(stations[s_i].times)
            for h in horizons:
                t = times[anchor + h]
                assert lo <= t <= hi, f"{split} target {t} outside [{lo}, {hi}]"
        seen[split] = {tuple(r) for r in idx}

    assert not (seen["train"] & seen["val"])
    assert not (seen["val"] & seen["test"])
    assert not (seen["train"] & seen["test"])


# ---------------------------------------------------------------------------
# 3. Mask correctness / no leak through values
# ---------------------------------------------------------------------------

def test_mask_and_values_consistent() -> None:
    cfg = make_cfg()
    df = make_frame()
    df.loc[5:200, "PM10"] = np.nan  # carve a hole into the inputs
    stations = _build(cfg, df)
    ds = AirQualityWindowDataset(stations, "train", cfg)
    sample = ds[0]

    values, mask = sample["values"].numpy(), sample["mask"].numpy()
    assert values.shape == (24, 3) and mask.shape == (24, 3)
    assert ((mask == 0) | (mask == 1)).all()
    assert (values[mask == 0] == 0).all(), "missing cells must be exactly 0"

    # observed cell equals hand-computed scaled value
    feats = feature_columns(cfg)
    scalers = compute_scalers(df, cfg)
    s_i, anchor = ds.index[0]
    st = stations[s_i]
    t0 = pd.Timestamp(st.times[anchor - ds.input_length + 1])
    raw = df.set_index("datetime").loc[t0, "PM2.5"]
    expected = (raw - scalers["PM2.5"][0]) / scalers["PM2.5"][1]
    j = feats.index("PM2.5")
    assert values[0, j] == pytest.approx(expected, rel=1e-5)
    assert mask[0, j] == 1


def test_target_mask_marks_missing_targets() -> None:
    cfg = make_cfg()
    df = make_frame()
    # PM10 observed only in the first 100 train hours (varying values so the
    # scaler stays fittable); unobserved everywhere else, including all of val.
    keep = df["PM10"].iloc[:100].to_numpy()
    df["PM10"] = np.nan
    df.iloc[:100, df.columns.get_loc("PM10")] = keep
    stations = _build(cfg, df)
    ds = AirQualityWindowDataset(stations, "val", cfg)
    sample = ds[0]
    tm = sample["target_mask"].numpy()  # (T, H); PM10 row must be all zero in val
    assert tm.shape == (2, 2)
    assert tm[1].sum() == 0          # PM10 unobserved in val
    assert tm[0].sum() > 0           # PM2.5 observed
    assert (sample["targets"].numpy()[tm == 0] == 0).all()


def test_masked_mse_invariant_to_masked_values() -> None:
    """Reference masked-MSE must not change when masked cells are perturbed.

    This is the contract Phase 4's loss must satisfy; the dataset guarantees
    targets==0 where target_mask==0, and this test pins the loss formula.
    """
    cfg = make_cfg()
    stations = _build(cfg, make_frame())
    ds = AirQualityWindowDataset(stations, "train", cfg)
    sample = ds[0]
    targets, t_mask = sample["targets"], sample["target_mask"]
    pred = torch.randn_like(targets)

    def masked_mse(p: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return ((p - y) ** 2 * m).sum() / m.sum().clamp(min=1)

    base = masked_mse(pred, targets, t_mask)
    poisoned = targets + (1 - t_mask) * 9999.0
    assert torch.allclose(base, masked_mse(pred, poisoned, t_mask))


# ---------------------------------------------------------------------------
# 4. Validity rule
# ---------------------------------------------------------------------------

def test_window_excluded_when_all_targets_missing() -> None:
    cfg = make_cfg()
    df = make_frame()
    stations = _build(cfg, df)
    n_before = len(build_window_index(stations, "train", cfg)[0])

    # destroy both targets everywhere except the first 300 hours (varying
    # values keep the scalers fittable) -> zero VAL windows survive
    df2 = df.copy()
    keep = df2[["PM2.5", "PM10"]].iloc[:300].to_numpy()
    df2[["PM2.5", "PM10"]] = np.nan
    df2.iloc[:300, [df2.columns.get_loc("PM2.5"), df2.columns.get_loc("PM10")]] = keep
    stations2 = _build(cfg, df2)
    idx2, _ = build_window_index(stations2, "val", cfg)
    assert n_before > 0 and len(idx2) == 0


def test_window_kept_with_single_observed_pair() -> None:
    cfg = make_cfg()
    df = make_frame()
    val_lo = pd.Timestamp(cfg["splits"]["train_end"]) + pd.Timedelta(hours=1)
    # blank both targets in val, then restore exactly one timestamp's PM2.5
    in_val = df["datetime"] >= val_lo
    df.loc[in_val, ["PM2.5", "PM10"]] = np.nan
    keep_time = val_lo + pd.Timedelta(hours=26)  # reachable as anchor+h
    df.loc[df["datetime"] == keep_time, "PM2.5"] = 55.0
    stations = _build(cfg, df)
    idx, _ = build_window_index(stations, "val", cfg)
    assert len(idx) >= 1
    ds = AirQualityWindowDataset(stations, "val", cfg)
    total_obs = sum(ds[i]["target_mask"].sum().item() for i in range(len(ds)))
    assert total_obs >= 1


# ---------------------------------------------------------------------------
# 5. Synthetic extra missingness
# ---------------------------------------------------------------------------

def test_extra_missingness_only_flips_observed_and_is_deterministic() -> None:
    cfg = make_cfg()
    df = make_frame()
    df.loc[10:400, "Temp"] = np.nan
    stations = _build(cfg, df)
    base = AirQualityWindowDataset(stations, "test", cfg)
    wrapped_a = ExtraMissingnessDataset(base, level=0.3, seed=42)
    wrapped_b = ExtraMissingnessDataset(base, level=0.3, seed=42)

    s0 = base[0]
    a0, b0 = wrapped_a[0], wrapped_b[0]

    # deterministic
    assert torch.equal(a0["mask"], b0["mask"])
    assert torch.equal(a0["values"], b0["values"])
    # only observed cells flipped, none un-flipped
    flipped = (s0["mask"] == 1) & (a0["mask"] == 0)
    assert ((s0["mask"] == 0) <= (a0["mask"] == 0)).all()  # missing stays missing
    n_obs = int(s0["mask"].sum())
    assert int(flipped.sum()) == round(n_obs * 0.3)
    assert (a0["values"][a0["mask"] == 0] == 0).all()
    # targets untouched
    assert torch.equal(s0["targets"], a0["targets"])
    assert torch.equal(s0["target_mask"], a0["target_mask"])
    # base dataset unchanged by wrapping
    assert torch.equal(base[0]["mask"], s0["mask"])


def test_extra_missingness_levels_differ() -> None:
    cfg = make_cfg()
    stations = _build(cfg, make_frame())
    base = AirQualityWindowDataset(stations, "test", cfg)
    m10 = ExtraMissingnessDataset(base, 0.1, 42)[0]["mask"].sum()
    m50 = ExtraMissingnessDataset(base, 0.5, 42)[0]["mask"].sum()
    assert m50 < m10


# ---------------------------------------------------------------------------
# 6. Sequence-length flexibility
# ---------------------------------------------------------------------------

def test_input_length_override() -> None:
    cfg = make_cfg()
    stations = _build(cfg, make_frame())
    short = AirQualityWindowDataset(stations, "train", cfg, input_length=12)
    long = AirQualityWindowDataset(stations, "train", cfg, input_length=48)
    assert short[0]["values"].shape == (12, 3)
    assert long[0]["values"].shape == (48, 3)
    assert len(short) > len(long)
