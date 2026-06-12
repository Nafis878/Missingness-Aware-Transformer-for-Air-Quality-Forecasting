"""Shared building blocks for all neural models.

Every model in this project implements ``forward(batch: dict) -> Tensor`` with
output shape ``(B, T, H)`` (T target pollutants x H horizons) so the training
loop in :mod:`src.train` is model-agnostic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sin/cos positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape (B, L, d_model)."""
        return x + self.pe[: x.size(1)]


class MultiHorizonHead(nn.Module):
    """Separate linear head per horizon, each predicting all targets.

    Output shape (B, T, H) to align with the dataset's ``targets`` tensor.
    """

    def __init__(self, d_model: int, n_targets: int, n_horizons: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Linear(d_model, n_targets) for _ in range(n_horizons)
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(pooled) for head in self.heads], dim=-1)


class AttentionPooling(nn.Module):
    """Learned single-query attention pooling over the sequence dimension."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool (B, L, d) -> (B, d)."""
        scores = (x @ self.query) / math.sqrt(x.size(-1))   # (B, L)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
