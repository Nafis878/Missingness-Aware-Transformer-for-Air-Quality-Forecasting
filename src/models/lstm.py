"""LSTM and GRU baselines.

Standard-practice recurrent forecasters: inputs are the forward-fill+mean
imputed values (imputation happens upstream, see
:class:`src.data.impute.FfillImputedDataset`), concatenated with calendar
time features; a station embedding is added to the input projection. The last
hidden state feeds the same multi-horizon head used by all other models.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.common import MultiHorizonHead


class RNNForecaster(nn.Module):
    """2-layer LSTM/GRU forecaster with multi-horizon, multi-target heads.

    Parameters
    ----------
    n_features:
        Number of input variables V.
    n_stations:
        Number of stations (embedding table size).
    n_targets, n_horizons:
        Output dimensions (T, H).
    cell:
        ``"lstm"`` or ``"gru"``.
    cfg:
        Full config; reads ``baselines.rnn``.
    """

    def __init__(
        self,
        n_features: int,
        n_stations: int,
        n_targets: int,
        n_horizons: int,
        cell: str,
        cfg: dict[str, Any],
        n_time_feats: int = 6,
    ) -> None:
        super().__init__()
        rcfg = cfg["baselines"]["rnn"]
        hidden = int(rcfg["hidden_size"])
        layers = int(rcfg["num_layers"])
        dropout = float(rcfg["dropout"])

        self.input_proj = nn.Linear(n_features + n_time_feats, hidden)
        self.station_embed = nn.Embedding(n_stations, hidden)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = MultiHorizonHead(hidden, n_targets, n_horizons)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, L, V) imputed values -> (B, T, H) predictions."""
        x = torch.cat([batch["values"], batch["time_feats"]], dim=-1)
        x = self.input_proj(x) + self.station_embed(batch["station_id"]).unsqueeze(1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1])
