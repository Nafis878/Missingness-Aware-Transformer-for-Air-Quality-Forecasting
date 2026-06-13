"""Tests for the series-imputability metric (the measured x-axis of the
crossover study).

The skill core ``_impute_skill`` is tested torch-free with injected imputers
(perfect oracle -> score ~ 1; an imputer identical to forward-fill -> score
exactly 0; held-out cells are always a subset of observed cells). The
dataset-level ``imputability_score`` is tested end-to-end on a small synthetic
gap-free network with the imputers injected, plus the graceful skip when no
SAITS checkpoint exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import compute_scalers
from src.data.impute import ffill_mean_impute
from src.evaluate import _impute_skill, imputability_score


# ---------------------------------------------------------------------------
# Skill core (torch-free, injected imputers)
# ---------------------------------------------------------------------------

def test_impute_skill_perfect_model_scores_near_one() -> None:
    rng = np.random.default_rng(0)
    vals = rng.standard_normal((200, 3)).astype(np.float32)
    mask = np.ones((200, 3), np.float32)
    oracle = lambda x_in, m_in: vals  # reconstructs the truth exactly  # noqa: E731
    out = _impute_skill([(vals, mask)], oracle, ffill_mean_impute, 0.3, seed=1)
    assert out["n_cells"] > 0
    assert out["rmse_model"] < 1e-6
    assert out["rmse_ffill"] > 0
    assert out["imputability"] > 0.99


def test_impute_skill_equal_to_ffill_scores_zero() -> None:
    rng = np.random.default_rng(2)
    vals = rng.standard_normal((150, 2)).astype(np.float32)
    mask = np.ones_like(vals)
    out = _impute_skill([(vals, mask)], ffill_mean_impute, ffill_mean_impute,
                        0.25, seed=3)
    # identical imputers -> identical RMSE -> imputability exactly 0
    assert out["imputability"] == 0.0


def test_impute_skill_holdout_subset_of_observed() -> None:
    vals = np.ones((50, 2), np.float32)
    mask = np.zeros((50, 2), np.float32)
    mask[:25] = 1.0  # only the first half is observed
    seen: dict[str, float] = {}

    def rec(x_in, m_in):
        seen["m_in_sum"] = float(m_in.sum())
        return x_in

    out = _impute_skill([(vals, mask)], rec, rec, 0.5, seed=7)
    assert 0 < out["n_cells"] <= 25 * 2          # never hide a missing cell
    assert seen["m_in_sum"] <= 25 * 2            # reduced input <= observed


def test_impute_skill_empty_when_nothing_observed() -> None:
    vals = np.zeros((10, 2), np.float32)
    mask = np.zeros((10, 2), np.float32)
    out = _impute_skill([(vals, mask)], lambda x, m: x, ffill_mean_impute,
                        0.5, seed=0)
    assert out["n_cells"] == 0 and np.isnan(out["imputability"])


# ---------------------------------------------------------------------------
# Dataset-level metric
# ---------------------------------------------------------------------------

def make_dataset(tmp_path: Path, name: str = "probe", n_days: int = 30) -> tuple[dict, dict]:
    proc = tmp_path / "proc"
    ckpt = tmp_path / "ckpt"
    proc.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    cols = ["PM2.5", "PM10"]
    times = pd.date_range("2020-01-01", periods=24 * n_days, freq="h")
    rng = np.random.default_rng(0)
    frames = []
    for st in ("S1", "S2"):
        base = 50 + 30 * np.sin(2 * np.pi * times.hour / 24)
        pm25 = base + rng.normal(0, 5, len(times))
        pm10 = 1.5 * pm25 + rng.normal(0, 8, len(times))
        d = pd.DataFrame({"station": st, "datetime": times,
                          "year": times.year, "PM2.5": pm25, "PM10": pm10})
        d.loc[10:40, "PM2.5"] = np.nan  # natural hole
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(proc / "all_stations.parquet", index=False)
    cfg = {
        "seed": 42,
        "dataset_name": name,
        "paths": {"processed_dir": str(proc), "checkpoints_dir": str(ckpt)},
        "data": {"measurement_cols": cols, "exclude_features": []},
        "splits": {"train_end": "2020-01-20 23:00:00",
                   "val_end": "2020-01-25 23:00:00"},
        "dataset": {"input_length": 24, "horizons": [6, 24], "stride_train": 6,
                    "stride_eval": 6, "target_pollutants": cols,
                    "primary_target": "PM2.5"},
        "baselines": {"saits": {"segment_len": 24}},
    }
    scalers = compute_scalers(df, cfg)
    (proc / "scalers.json").write_text(json.dumps(scalers))
    return cfg, scalers


def test_imputability_score_injected_returns_one_row(tmp_path) -> None:
    cfg, scalers = make_dataset(tmp_path, name="probe")
    out = imputability_score(cfg, scalers, model_impute=ffill_mean_impute,
                             ffill_impute=ffill_mean_impute)
    assert out is not None and len(out) == 1
    assert {"dataset", "natural_PM25_missing_pct", "imputability",
            "rmse_model", "rmse_ffill", "n_cells"}.issubset(out.columns)
    assert out["dataset"].iloc[0] == "Probe"
    assert out["imputability"].iloc[0] == 0.0          # identical imputers
    assert out["natural_PM25_missing_pct"].iloc[0] > 0  # injected holes counted


def test_imputability_score_no_checkpoint_returns_none(tmp_path) -> None:
    cfg, scalers = make_dataset(tmp_path)
    # default model path needs a SAITS checkpoint; none exists -> graceful skip
    assert imputability_score(cfg, scalers) is None
