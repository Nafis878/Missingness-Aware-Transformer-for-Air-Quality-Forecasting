"""THE proposed model: Missingness-Aware Transformer.

Identical encoder backbone to the vanilla Transformer baseline; the entire
contribution is the native missingness handling:

* **Learned missingness embedding**: ``miss_proj = Linear(V, d_model, bias
  =False)`` applied to ``(1 - mask)`` — equivalently, a trainable per-variable
  embedding vector added wherever that variable is missing, so the model can
  distinguish "absent" from "measured zero" (values are zero-filled where
  missing, hence ``value_proj(values) == value_proj(values * mask)``).
* **Attention masking variant B** (config switch): additionally prevents
  attention *to* timesteps where the primary target (PM2.5) is unobserved,
  via ``src_key_padding_mask``. Variant A relies on missingness embeddings
  only. Windows whose every timestep would be masked fall back to an
  unmasked window (guard against NaN attention).

Config switches (``model.*``): ``attention_variant`` A/B, ``pooling``
last/attention, ``positional_encoding`` sinusoidal/learned,
plus constructor flags ``use_missingness_embedding`` and
``use_time_features`` for the ablations.
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


class LearnedPositionalEncoding(nn.Module):
    """Trainable positional embedding (config alternative to sinusoidal)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class MissingnessTransformer(nn.Module):
    """Missingness-aware Transformer encoder forecaster.

    Parameters
    ----------
    n_features, n_stations, n_targets, n_horizons:
        Data dimensions (V, S, T, H).
    cfg:
        Full config dict; reads ``model.*`` and ``dataset.*``.
    target_feature_idx:
        Column index of the primary target (PM2.5) within the feature axis;
        used by attention variant B. Required when variant B is active.
    use_missingness_embedding / use_time_features:
        Ablation switches (default on).
    """

    def __init__(
        self,
        n_features: int,
        n_stations: int,
        n_targets: int,
        n_horizons: int,
        cfg: dict[str, Any],
        target_feature_idx: int | None = None,
        use_missingness_embedding: bool = True,
        use_time_features: bool = True,
        n_time_feats: int = 6,
    ) -> None:
        super().__init__()
        mcfg = cfg["model"]
        d_model = int(mcfg["d_model"])
        self.attention_variant = str(mcfg.get("attention_variant", "A")).upper()
        if self.attention_variant == "B" and target_feature_idx is None:
            raise ValueError("variant B requires target_feature_idx")
        self.target_feature_idx = target_feature_idx
        self.use_missingness_embedding = use_missingness_embedding
        self.use_time_features = use_time_features

        self.value_proj = nn.Linear(n_features, d_model)
        self.miss_proj = (
            nn.Linear(n_features, d_model, bias=False)
            if use_missingness_embedding else None
        )
        self.time_proj = nn.Linear(n_time_feats, d_model) if use_time_features else None
        self.station_embed = nn.Embedding(n_stations, d_model)
        self.pos_enc = (
            LearnedPositionalEncoding(d_model)
            if mcfg.get("positional_encoding") == "learned"
            else SinusoidalPositionalEncoding(d_model)
        )

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

    def _variant_b_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Key-padding mask (B, L): True = do not attend to this timestep.

        Masks timesteps where the primary target is unobserved. Guard: if a
        window would have every key masked, unmask it entirely (otherwise
        softmax over an empty key set yields NaN).
        """
        kpm = mask[:, :, self.target_feature_idx] == 0  # (B, L) True = missing
        all_masked = kpm.all(dim=1)
        if all_masked.any():
            kpm = kpm.clone()
            kpm[all_masked] = False
        return kpm

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample dict (values/mask/time_feats/station_id) -> (B, T, H)."""
        values, mask = batch["values"], batch["mask"]
        x = self.value_proj(values)
        if self.miss_proj is not None:
            x = x + self.miss_proj(1.0 - mask)
        if self.time_proj is not None:
            x = x + self.time_proj(batch["time_feats"])
        x = x + self.station_embed(batch["station_id"]).unsqueeze(1)
        x = self.pos_enc(x)

        kpm = self._variant_b_mask(mask) if self.attention_variant == "B" else None
        x = self.encoder(x, src_key_padding_mask=kpm)
        pooled = self.pooling(x) if self.pooling is not None else x[:, -1]
        return self.head(pooled)
