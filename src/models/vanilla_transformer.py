"""Vanilla Transformer encoder for the two-stage baseline.

Identical in size and structure to the proposed missingness-aware model
(d_model, layers, heads, FFN, pre-norm, positional encoding, station
embedding, pooling, multi-horizon heads) but it consumes **imputed** inputs
and has no observation-mask pathway. The only difference between this and the
proposed model is the native missingness handling - which is exactly the
contribution the two-stage comparison isolates.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.common import (
    AttentionPooling,
    MultiHorizonHead,
    SinusoidalPositionalEncoding,
)


class VanillaTransformer(nn.Module):
    """Standard pre-norm Transformer encoder over imputed inputs."""

    def __init__(
        self,
        n_features: int,
        n_stations: int,
        n_targets: int,
        n_horizons: int,
        cfg: dict[str, Any],
        n_time_feats: int = 6,
    ) -> None:
        super().__init__()
        mcfg = cfg["model"]
        d_model = int(mcfg["d_model"])

        self.value_proj = nn.Linear(n_features, d_model)
        self.time_proj = nn.Linear(n_time_feats, d_model)
        self.station_embed = nn.Embedding(n_stations, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(mcfg["n_heads"]),
            dim_feedforward=int(mcfg["d_ff"]),
            dropout=float(mcfg["dropout"]),
            batch_first=True,
            norm_first=bool(mcfg["norm_first"]),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(mcfg["n_layers"]))
        self.pooling = (
            AttentionPooling(d_model) if mcfg["pooling"] == "attention" else None
        )
        self.head = MultiHorizonHead(d_model, n_targets, n_horizons)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, L, V) imputed values -> (B, T, H) predictions."""
        x = (
            self.value_proj(batch["values"])
            + self.time_proj(batch["time_feats"])
            + self.station_embed(batch["station_id"]).unsqueeze(1)
        )
        x = self.pos_enc(x)
        x = self.encoder(x)
        pooled = self.pooling(x) if self.pooling is not None else x[:, -1]
        return self.head(pooled)
