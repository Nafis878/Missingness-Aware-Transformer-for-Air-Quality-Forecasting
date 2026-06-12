"""Unit tests for the proposed MissingnessTransformer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.missingness_transformer import MissingnessTransformer

from tests.test_models import fake_batch, model_cfg


def make_model(**kwargs) -> MissingnessTransformer:
    defaults = dict(
        n_features=3, n_stations=3, n_targets=2, n_horizons=2,
        cfg=model_cfg(), target_feature_idx=0,
    )
    defaults.update(kwargs)
    return MissingnessTransformer(**defaults)


def test_output_shape() -> None:
    model = make_model()
    assert model(fake_batch()).shape == (4, 2, 2)


def test_missingness_embedding_is_live() -> None:
    """Same values, different mask => different output (absent != zero)."""
    torch.manual_seed(0)
    model = make_model().eval()
    b = fake_batch()
    b["values"] = b["values"] * b["mask"]  # enforce dataset invariant
    out1 = model(b)
    b2 = dict(b)
    b2["mask"] = torch.ones_like(b["mask"])  # claim everything observed
    out2 = model(b2)
    assert not torch.allclose(out1, out2)


def test_no_missingness_embedding_ablation_ignores_mask() -> None:
    """With the embedding ablated (variant A), the mask has no pathway."""
    torch.manual_seed(0)
    model = make_model(use_missingness_embedding=False).eval()
    b = fake_batch()
    out1 = model(b)
    b2 = dict(b)
    b2["mask"] = torch.zeros_like(b["mask"])
    out2 = model(b2)
    assert torch.allclose(out1, out2)


def test_variant_b_masks_attention_and_guards_all_missing() -> None:
    cfg = model_cfg()
    cfg["model"]["attention_variant"] = "B"
    torch.manual_seed(0)
    model = make_model(cfg=cfg).eval()
    b = fake_batch()
    # window 0: PM2.5 (feature 0) missing at EVERY timestep -> guard must kick in
    b["mask"][0, :, 0] = 0
    out = model(b)
    assert torch.isfinite(out).all()
    # variant B must actually change outputs vs variant A weights-equal model
    torch.manual_seed(0)
    model_a = make_model().eval()
    model_a.load_state_dict(model.state_dict())
    assert not torch.allclose(out, model_a(b))


def test_use_time_features_flag() -> None:
    model = make_model(use_time_features=False).eval()
    b = fake_batch()
    out1 = model(b)
    b2 = dict(b)
    b2["time_feats"] = torch.randn_like(b["time_feats"])
    assert torch.allclose(out1, model(b2))  # time features have no pathway


def test_attention_pooling_switch() -> None:
    cfg = model_cfg()
    cfg["model"]["pooling"] = "attention"
    model = make_model(cfg=cfg)
    assert model(fake_batch()).shape == (4, 2, 2)


def test_learned_positional_encoding_switch() -> None:
    cfg = model_cfg()
    cfg["model"]["positional_encoding"] = "learned"
    model = make_model(cfg=cfg)
    assert model(fake_batch()).shape == (4, 2, 2)
