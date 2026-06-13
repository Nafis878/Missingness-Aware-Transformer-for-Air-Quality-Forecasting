"""Unit tests for the Phase-2 baselines: DLinear, PatchTST, GRU-D + factory.

Every new model gets shape, determinism, and mask-handling (poisoning) tests;
GRU-D's vectorized delta/x_last computation is checked against hand-computed
values; production-size configs are checked against the ~400k parameter
budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.common import count_parameters
from src.models.dlinear import DLinearForecaster
from src.models.factory import build_model
from src.models.gru_d import GRUDForecaster
from src.models.patchtst import PatchTSTForecaster

from tests.test_models import fake_batch, model_cfg


def new_model_cfg() -> dict:
    cfg = model_cfg()
    cfg["baselines"].update({
        "dlinear": {"kernel_size": 25, "station_bias": True},
        "patchtst": {"patch_len": 16, "stride": 8, "d_model": 32,
                     "n_layers": 2, "n_heads": 4, "d_ff": 64, "dropout": 0.1},
        "gru_d": {"hidden_size": 32, "dropout": 0.1},
    })
    return cfg


def build(name: str, cfg: dict | None = None):
    cfg = cfg or new_model_cfg()
    return build_model(name, cfg, n_features=3, n_stations=3, n_targets=2,
                       n_horizons=2, target_feature_idx=0, target_indices=[0, 1])


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dlinear", "patchtst", "gru_d"])
def test_new_model_output_shape(name: str) -> None:
    out = build(name)(fake_batch())
    assert out.shape == (4, 2, 2)


def test_patchtst_handles_length_not_divisible_by_stride() -> None:
    # L=21: (21-16) % 8 != 0 -> end is replicate-padded so patches tile
    out = build("patchtst")(fake_batch(L=21))
    assert out.shape == (4, 2, 2)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dlinear", "patchtst", "gru_d"])
def test_new_model_deterministic(name: str) -> None:
    """Same seed -> identical weights; eval mode -> identical repeat outputs."""
    b = fake_batch()
    torch.manual_seed(0)
    m1 = build(name)
    m1.eval()
    torch.manual_seed(0)
    m2 = build(name)
    m2.eval()
    out1, out1b, out2 = m1(b), m1(b), m2(b)
    assert torch.equal(out1, out1b)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Mask handling: values at masked cells must never influence the output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dlinear", "patchtst", "gru_d"])
def test_new_model_ignores_poisoned_masked_values(name: str) -> None:
    model = build(name)
    model.eval()
    b = fake_batch()
    poisoned = dict(b)
    poisoned["values"] = b["values"] + (1 - b["mask"]) * 1e6
    assert torch.equal(model(b), model(poisoned))


# ---------------------------------------------------------------------------
# GRU-D specifics
# ---------------------------------------------------------------------------

def test_gru_d_delta_and_last_hand_computed() -> None:
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # (1, 4, 1)
    mask = torch.tensor([[[1.0], [0.0], [0.0], [1.0]]])
    delta, x_last = GRUDForecaster.delta_and_last(values * mask, mask)
    # delta_t = steps since last observation strictly before t (Che et al.)
    assert delta.squeeze().tolist() == [0.0, 1.0, 2.0, 3.0]
    # x_last_t = most recent observed value at or before t
    assert x_last.squeeze().tolist() == [1.0, 1.0, 1.0, 4.0]


def test_gru_d_delta_with_leading_missing() -> None:
    values = torch.tensor([[[0.0], [2.0], [0.0], [0.0]]])
    mask = torch.tensor([[[0.0], [1.0], [0.0], [0.0]]])
    delta, x_last = GRUDForecaster.delta_and_last(values, mask)
    # never-observed prefix accumulates from sequence start
    assert delta.squeeze().tolist() == [0.0, 1.0, 1.0, 2.0]
    # 0 (= scaled train mean) before the first observation
    assert x_last.squeeze().tolist() == [0.0, 2.0, 2.0, 2.0]


# ---------------------------------------------------------------------------
# Parameter budget at production scale
# ---------------------------------------------------------------------------

def production_cfg() -> dict:
    return {
        "dataset": {"input_length": 168,
                    "target_pollutants": ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2"]},
        "model": {"d_model": 128, "n_layers": 3, "n_heads": 8, "d_ff": 256,
                  "dropout": 0.1, "norm_first": True, "pooling": "last",
                  "attention_variant": "A", "positional_encoding": "sinusoidal"},
        "baselines": {
            "rnn": {"hidden_size": 128, "num_layers": 2, "dropout": 0.1},
            "dlinear": {"kernel_size": 25, "station_bias": True},
            "patchtst": {"patch_len": 16, "stride": 8, "d_model": 128,
                         "n_layers": 3, "n_heads": 8, "d_ff": 256, "dropout": 0.1},
            "gru_d": {"hidden_size": 128, "dropout": 0.1},
        },
    }


@pytest.mark.parametrize("name", ["dlinear", "patchtst", "gru_d"])
def test_new_model_parameter_budget(name: str) -> None:
    cfg = production_cfg()
    model = build_model(name, cfg, n_features=14, n_stations=16, n_targets=6,
                        n_horizons=3, target_feature_idx=7,
                        target_indices=[7, 6, 2, 5, 4, 0])
    n = count_parameters(model)
    assert n < 450_000, f"{name}: {n} params exceeds the ~400k budget"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_dispatch_returns_expected_classes() -> None:
    assert isinstance(build("dlinear"), DLinearForecaster)
    assert isinstance(build("patchtst"), PatchTSTForecaster)
    assert isinstance(build("gru_d"), GRUDForecaster)
    from src.models.lstm import RNNForecaster
    from src.models.missingness_transformer import MissingnessTransformer
    from src.models.vanilla_transformer import VanillaTransformer

    assert isinstance(build("gru"), RNNForecaster)
    assert isinstance(build("two_stage_saits"), VanillaTransformer)
    assert isinstance(build("proposed"), MissingnessTransformer)
    vb = build("variant_B")
    assert isinstance(vb, MissingnessTransformer)
    assert vb.attention_variant == "B"


def test_factory_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build("transformer_xl")


def test_factory_dlinear_requires_target_indices() -> None:
    with pytest.raises(ValueError, match="target_indices"):
        build_model("dlinear", new_model_cfg(), 3, 3, 2, 2)
