"""Unit tests for the minimal SAITS imputer and its two-stage integration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import StationArrays
from src.data.impute import impute_full_series
from src.models.saits import (
    SAITS,
    _train_segments,
    diagonal_attn_mask,
    train_saits,
)


def saits_cfg(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "paths": {"checkpoints_dir": str(tmp_path / "checkpoints")},
        "splits": {"train_end": "2022-01-14 23:00:00",
                   "val_end": "2022-01-21 23:00:00"},
        "baselines": {
            "saits": {
                "d_model": 16, "n_heads": 2, "d_ff": 32,
                "n_layers_per_group": 1, "dropout": 0.0,
                "epochs": 2, "patience": 5, "batch_size": 16, "lr": 1.0e-3,
                "mit_rate": 0.2, "segment_len": 48, "segment_stride": 24,
            },
        },
    }


def make_stations(n_hours: int = 24 * 28, n_vars: int = 3,
                  miss_rate: float = 0.3) -> list[StationArrays]:
    rng = np.random.default_rng(0)
    out = []
    for sid, name in enumerate(("A", "B")):
        times = pd.date_range("2022-01-01", periods=n_hours, freq="h")
        values = rng.standard_normal((n_hours, n_vars)).astype(np.float32)
        mask = (rng.random((n_hours, n_vars)) > miss_rate).astype(np.float32)
        values = values * mask  # contract: missing cells are exactly 0
        out.append(StationArrays(
            station=name, station_id=sid,
            times=times.to_numpy().astype("datetime64[s]"),
            values=values, mask=mask,
            raw_targets=values[:, :2].copy(),
            time_feats=np.zeros((n_hours, 6), dtype=np.float32),
        ))
    return out


def test_diagonal_attn_mask_blocks_self_attention_only() -> None:
    m = diagonal_attn_mask(5)
    assert (torch.diag(m) == float("-inf")).all()
    off = m[~torch.eye(5, dtype=torch.bool)]
    assert (off == 0).all()


def test_saits_impute_preserves_observed_cells_exactly(tmp_path) -> None:
    cfg = saits_cfg(tmp_path)
    torch.manual_seed(0)
    model = SAITS(n_features=3, cfg=cfg)
    model.eval()
    g = torch.Generator().manual_seed(1)
    mask = torch.randint(0, 2, (2, 48, 3), generator=g).float()
    x = torch.randn(2, 48, 3, generator=g) * mask
    out = model.impute(x, mask)
    assert torch.equal(out[mask > 0], x[mask > 0])
    assert torch.isfinite(out).all()


def test_saits_deterministic_construction_and_forward(tmp_path) -> None:
    cfg = saits_cfg(tmp_path)
    g = torch.Generator().manual_seed(2)
    mask = torch.randint(0, 2, (2, 48, 3), generator=g).float()
    x = torch.randn(2, 48, 3, generator=g) * mask
    torch.manual_seed(0)
    m1 = SAITS(3, cfg)
    m1.eval()
    torch.manual_seed(0)
    m2 = SAITS(3, cfg)
    m2.eval()
    assert torch.equal(m1.impute(x, mask), m2.impute(x, mask))
    assert torch.equal(m1.impute(x, mask), m1.impute(x, mask))


def test_train_segments_use_train_period_only(tmp_path) -> None:
    cfg = saits_cfg(tmp_path)
    stations = make_stations()
    xs, ms = _train_segments(stations, cfg)
    # 14 train days = 336 rows; L=48, stride=24 -> 13 segments per station
    n_train = 24 * 14
    scfg = cfg["baselines"]["saits"]
    expected = len(range(0, n_train - scfg["segment_len"] + 1,
                         scfg["segment_stride"])) * len(stations)
    assert len(xs) == expected
    assert xs.shape[1:] == (48, 3) and ms.shape == xs.shape


def test_train_saits_writes_checkpoint_and_resumes(tmp_path) -> None:
    cfg = saits_cfg(tmp_path)
    stations = make_stations()
    torch.manual_seed(42)
    model, stats = train_saits(stations, cfg, seed=42)
    ckpt = Path(cfg["paths"]["checkpoints_dir"]) / "saits_imputer_seed42.pt"
    assert ckpt.exists()
    assert stats["fit_time_s"] >= 0 and "best_val_mit_mae" in stats
    assert "ffill_val_mit_mae" in stats  # quality gate recorded
    # second call resumes from the checkpoint instead of retraining
    model2, stats2 = train_saits(stations, cfg, seed=42)
    assert stats2.get("reused_checkpoint") == str(ckpt)
    p1 = torch.cat([p.flatten() for p in model.parameters()])
    p2 = torch.cat([p.flatten() for p in model2.parameters()])
    assert torch.equal(p1, p2)


def test_impute_full_series_saits_contract(tmp_path) -> None:
    """method="saits" matches the KNN/MICE contract via the same entry point."""
    cfg = saits_cfg(tmp_path)
    stations = make_stations()
    torch.manual_seed(42)
    imputed = impute_full_series(stations, cfg, method="saits", seed=42)
    assert len(imputed) == len(stations)
    for st, arr in zip(stations, imputed):
        assert arr.shape == st.values.shape and arr.dtype == np.float32
        obs = st.mask > 0
        assert np.array_equal(arr[obs], st.values[obs]), \
            "observed cells must be preserved exactly"
        assert np.isfinite(arr).all()
