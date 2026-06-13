"""Single construction point for every trainable forecaster.

Scripts and tests build models by name through :func:`build_model` so that
registering a new baseline means: add a module, add a branch here, append the
name to the script registries and ``MODEL_LABELS``.
"""

from __future__ import annotations

import copy
from typing import Any

import torch.nn as nn


def build_model(
    name: str,
    cfg: dict[str, Any],
    n_features: int,
    n_stations: int,
    n_targets: int,
    n_horizons: int,
    target_feature_idx: int | None = None,
    target_indices: list[int] | None = None,
) -> nn.Module:
    """Construct a forecaster by registry name.

    Parameters
    ----------
    name:
        One of: ``lstm``, ``gru``, ``gru_d``, ``dlinear``, ``patchtst``,
        ``vanilla_transformer`` (also any ``two_stage_*`` alias), ``proposed``,
        ``proposed_md`` (same architecture; the dropout wrapper is a dataset
        concern), ``variant_B``.
    target_feature_idx:
        Primary-target column index (needed by ``variant_B``).
    target_indices:
        Target-pollutant column indices (needed by ``dlinear``).
    """
    if name in ("lstm", "gru"):
        from src.models.lstm import RNNForecaster

        return RNNForecaster(n_features, n_stations, n_targets, n_horizons, name, cfg)
    if name == "vanilla_transformer" or name.startswith("two_stage"):
        from src.models.vanilla_transformer import VanillaTransformer

        return VanillaTransformer(n_features, n_stations, n_targets, n_horizons, cfg)
    if name == "gru_d":
        from src.models.gru_d import GRUDForecaster

        return GRUDForecaster(n_features, n_stations, n_targets, n_horizons, cfg)
    if name == "dlinear":
        from src.models.dlinear import DLinearForecaster

        if target_indices is None:
            raise ValueError("dlinear requires target_indices")
        return DLinearForecaster(
            n_features, n_stations, n_targets, n_horizons, cfg, target_indices
        )
    if name == "patchtst":
        from src.models.patchtst import PatchTSTForecaster

        return PatchTSTForecaster(n_features, n_stations, n_targets, n_horizons, cfg)
    if name in ("proposed", "proposed_md", "variant_B"):
        from src.models.missingness_transformer import MissingnessTransformer

        if name == "variant_B":
            cfg = copy.deepcopy(cfg)
            cfg["model"]["attention_variant"] = "B"
        return MissingnessTransformer(
            n_features, n_stations, n_targets, n_horizons, cfg,
            target_feature_idx=target_feature_idx,
        )
    raise ValueError(f"unknown model name {name!r}")
