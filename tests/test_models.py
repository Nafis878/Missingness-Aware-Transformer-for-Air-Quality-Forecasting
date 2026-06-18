"""Unit tests for baseline models, imputation, and the masked training loop."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.impute import FfillImputedDataset, ffill_mean_impute, replace_inputs
from src.models.lstm import RNNForecaster
from src.models.statistical import (
    _predict_window_persistence,
    _predict_window_seasonal_naive,
)
from src.models.vanilla_transformer import VanillaTransformer
from src.train import masked_mse

from tests.test_dataset import make_cfg, make_frame, _build
from src.data.dataset import AirQualityWindowDataset


def model_cfg() -> dict:
    cfg = make_cfg()
    cfg["model"] = {
        "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64,
        "dropout": 0.1, "norm_first": True, "pooling": "last",
        "attention_variant": "A", "positional_encoding": "sinusoidal",
    }
    cfg["baselines"] = {
        "rnn": {"hidden_size": 32, "num_layers": 2, "dropout": 0.1},
    }
    return cfg


def fake_batch(B=4, L=24, V=3, T=2, H=2, n_time=6):
    g = torch.Generator().manual_seed(0)
    return {
        "values": torch.randn(B, L, V, generator=g),
        "mask": torch.randint(0, 2, (B, L, V), generator=g).float(),
        "time_feats": torch.randn(B, L, n_time, generator=g),
        "station_id": torch.randint(0, 3, (B,), generator=g),
        "targets": torch.randn(B, T, H, generator=g),
        "target_mask": torch.randint(0, 2, (B, T, H), generator=g).float(),
    }


# ---------------------------------------------------------------------------
# Model output shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_rnn_output_shape(cell: str) -> None:
    model = RNNForecaster(3, 3, 2, 2, cell, model_cfg())
    out = model(fake_batch())
    assert out.shape == (4, 2, 2)


def test_vanilla_transformer_output_shape() -> None:
    model = VanillaTransformer(3, 3, 2, 2, model_cfg())
    out = model(fake_batch())
    assert out.shape == (4, 2, 2)


# ---------------------------------------------------------------------------
# Masked loss: a masked value must never leak into the loss
# ---------------------------------------------------------------------------

def test_masked_mse_ignores_masked_cells() -> None:
    b = fake_batch()
    pred = torch.randn_like(b["targets"])
    base = masked_mse(pred, b["targets"], b["target_mask"])
    poisoned = b["targets"] + (1 - b["target_mask"]) * 1e6
    assert torch.allclose(base, masked_mse(pred, poisoned, b["target_mask"]))


def test_masked_mse_gradient_zero_for_masked_predictions() -> None:
    """Gradients w.r.t. predictions at masked targets must be exactly zero."""
    b = fake_batch()
    pred = torch.randn_like(b["targets"], requires_grad=True)
    loss = masked_mse(pred, b["targets"], b["target_mask"])
    loss.backward()
    assert (pred.grad[b["target_mask"] == 0] == 0).all()
    assert (pred.grad[b["target_mask"] == 1] != 0).any()


def test_masked_mse_horizon_weights() -> None:
    b = fake_batch()
    pred = torch.zeros_like(b["targets"])
    hw0 = torch.tensor([1.0, 0.0])  # zero out second horizon
    only_h0 = masked_mse(pred, b["targets"], b["target_mask"], hw0)
    manual = ((b["targets"][:, :, 0] ** 2) * b["target_mask"][:, :, 0]).sum() / \
        b["target_mask"][:, :, 0].sum()
    assert torch.allclose(only_h0, manual)


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def test_ffill_mean_impute() -> None:
    values = np.array([[0.0, 5.0], [2.0, 0.0], [0.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    mask = np.array([[0, 1], [1, 0], [0, 0], [1, 0]], dtype=np.float32)
    out = ffill_mean_impute(values, mask)
    # col0: leading missing -> 0 (mean), then 2 carried forward, then 3
    assert out[:, 0].tolist() == [0.0, 2.0, 2.0, 3.0]
    # col1: 5 observed then carried forward forever
    assert out[:, 1].tolist() == [5.0, 5.0, 5.0, 5.0]


def test_ffill_dataset_targets_untouched() -> None:
    cfg = make_cfg()
    df = make_frame()
    df.loc[50:120, "PM2.5"] = np.nan
    stations = _build(cfg, df)
    base = AirQualityWindowDataset(stations, "train", cfg)
    wrapped = FfillImputedDataset(base)
    s, w = base[1], wrapped[1]
    assert torch.equal(s["targets"], w["targets"])
    assert torch.equal(s["target_mask"], w["target_mask"])
    assert (w["mask"] == 1).all()
    # observed cells unchanged by imputation
    obs = s["mask"] == 1
    assert torch.equal(s["values"][obs], w["values"][obs])


def test_replace_inputs_keeps_targets_and_windows() -> None:
    cfg = make_cfg()
    stations = _build(cfg, make_frame())
    fake_imputed = [np.full_like(st.values, 7.0) for st in stations]
    swapped = replace_inputs(stations, fake_imputed)
    ds_orig = AirQualityWindowDataset(stations, "test", cfg)
    ds_swap = AirQualityWindowDataset(swapped, "test", cfg)
    # identical window enumeration and targets, different inputs
    assert np.array_equal(ds_orig.index, ds_swap.index)
    a, b = ds_orig[0], ds_swap[0]
    assert torch.equal(a["targets"], b["targets"])
    assert torch.equal(a["target_mask"], b["target_mask"])
    assert (b["values"] == 7.0).all() and (b["mask"] == 1).all()


def test_replace_inputs_can_preserve_original_mask() -> None:
    cfg = make_cfg()
    df = make_frame()
    df.loc[30:70, "PM2.5"] = np.nan
    stations = _build(cfg, df)
    fake_imputed = [np.full_like(st.values, 7.0) for st in stations]
    swapped = replace_inputs(stations, fake_imputed, preserve_mask=True)
    ds_orig = AirQualityWindowDataset(stations, "test", cfg)
    ds_swap = AirQualityWindowDataset(swapped, "test", cfg)

    assert np.array_equal(ds_orig.index, ds_swap.index)
    a, b = ds_orig[0], ds_swap[0]
    assert torch.equal(a["targets"], b["targets"])
    assert torch.equal(a["target_mask"], b["target_mask"])
    assert torch.equal(a["mask"], b["mask"])
    assert (b["values"] == 7.0).all()


def test_corrupt_test_outages_blocks_and_targets() -> None:
    from src.data.impute import corrupt_test_outages

    cfg = make_cfg()
    df = make_frame()
    stations = _build(cfg, df)
    corrupted = corrupt_test_outages(stations, cfg, level=0.3, seed=42)
    st, st_c = stations[0], corrupted[0]
    val_end = pd.Timestamp(cfg["splits"]["val_end"]).to_datetime64()
    in_test = st.times > val_end
    # ~30% of observed test cells dropped, in all-variable blocks
    obs_before = st.mask[in_test].sum()
    obs_after = st_c.mask[in_test].sum()
    frac = 1 - obs_after / obs_before
    assert 0.25 < frac < 0.35
    # non-test rows untouched; targets untouched
    assert np.array_equal(st.mask[~in_test], st_c.mask[~in_test])
    assert np.array_equal(st.raw_targets, st_c.raw_targets, equal_nan=True)
    # dropped rows are all-variable (outage) blocks: any dropped cell in a row
    # implies the whole observed row was dropped
    dropped_rows = (st.mask > st_c.mask).any(axis=1)
    fully_dropped = ~(st_c.mask[dropped_rows] > 0).any(axis=1)
    assert fully_dropped.all()
    # deterministic
    corrupted2 = corrupt_test_outages(stations, cfg, level=0.3, seed=42)
    assert np.array_equal(st_c.mask, corrupted2[0].mask)


# ---------------------------------------------------------------------------
# Statistical baselines
# ---------------------------------------------------------------------------

def test_persistence_picks_last_observed() -> None:
    L, V = 6, 2
    values = np.zeros((L, V), dtype=np.float32)
    mask = np.zeros((L, V), dtype=np.float32)
    values[2, 0], mask[2, 0] = 3.3, 1   # last observed for col0 at t=2
    values[5, 1], mask[5, 1] = 1.1, 1   # last observed for col1 at t=5
    out = _predict_window_persistence(values, mask, t_cols=[0, 1], n_horizons=2)
    assert np.allclose(out[0], 3.3) and np.allclose(out[1], 1.1)  # broadcast across horizons


def test_persistence_all_missing_falls_back_to_mean() -> None:
    values = np.zeros((4, 1), dtype=np.float32)
    mask = np.zeros((4, 1), dtype=np.float32)
    out = _predict_window_persistence(values, mask, t_cols=[0], n_horizons=3)
    assert (out == 0).all()  # 0 == train mean in scaled space


def test_seasonal_naive_picks_same_hour_lag() -> None:
    """For h=6 with L=48 the same-hour candidate is index L-1+6-24 = L-19."""
    L = 48
    values = np.arange(L, dtype=np.float32).reshape(L, 1)
    mask = np.ones((L, 1), dtype=np.float32)
    out = _predict_window_seasonal_naive(values, mask, t_cols=[0], horizons=[6, 24])
    assert out[0, 0] == L - 1 + 6 - 24   # h=6  -> index 29
    assert out[0, 1] == L - 1            # h=24 -> the anchor itself


def test_seasonal_naive_steps_back_when_missing() -> None:
    L = 48
    values = np.arange(L, dtype=np.float32).reshape(L, 1)
    mask = np.ones((L, 1), dtype=np.float32)
    mask[29] = 0  # first candidate for h=6 missing -> step back 24h to index 5
    out = _predict_window_seasonal_naive(values, mask, t_cols=[0], horizons=[6])
    assert out[0, 0] == 5
