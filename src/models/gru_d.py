"""GRU-D baseline (Che et al., 2018, "Recurrent Neural Networks for
Multivariate Time Series with Missing Values").

The canonical missingness-native recurrent model and the natural head-to-head
competitor for this paper's claim. Faithful to the original:

* **Input decay**: missing values decay from the last observation toward the
  empirical mean with a learned per-variable rate,
  ``gamma_x = exp(-relu(w_x * delta + b_x))`` (diagonal weights) and
  ``x_hat = m*x + (1-m) * (gamma_x * x_last + (1-gamma_x) * x_mean)``.
  Inputs here are standardized with train-period scalers, so the empirical
  train mean is exactly 0 in scaled space and the mean term vanishes.
* **Hidden-state decay**: ``gamma_h = exp(-relu(W_gh @ delta + b_gh))``
  (full matrix) multiplies the hidden state before every step.
* **Mask as input**: the GRU consumes ``[x_hat, m, time_feats]``.

``delta`` (time since last observation, per variable) and ``x_last`` are
computed inside ``forward`` from the mask alone, vectorized with a
cumulative-max index trick (no scatter ops, deterministic-algorithms safe);
only the hidden recurrence loops over time. ``delta`` is expressed in days
(hours / 24) purely for conditioning — the learned rates absorb the scale.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.common import MultiHorizonHead


class GRUDForecaster(nn.Module):
    """Single-layer GRU-D with multi-horizon, multi-target heads.

    Parameters
    ----------
    n_features, n_stations, n_targets, n_horizons:
        Data dimensions (V, S, T, H).
    cfg:
        Full config; reads ``baselines.gru_d``.
    """

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
        gcfg = cfg["baselines"]["gru_d"]
        hidden = int(gcfg["hidden_size"])
        self.hidden = hidden

        # per-variable (diagonal) input-decay rates; small positive init so
        # gamma starts near 1 (trust the last observation) with live gradients
        self.decay_x_w = nn.Parameter(torch.full((n_features,), 0.1))
        self.decay_x_b = nn.Parameter(torch.zeros(n_features))
        self.decay_h = nn.Linear(n_features, hidden)
        nn.init.uniform_(self.decay_h.weight, 0.0, 0.05)
        nn.init.zeros_(self.decay_h.bias)

        in_dim = 2 * n_features + n_time_feats  # [x_hat, mask, time_feats]
        self.in_proj = nn.Linear(in_dim, 3 * hidden)
        self.h_proj = nn.Linear(hidden, 3 * hidden)
        self.station_embed = nn.Embedding(n_stations, hidden)
        self.dropout = nn.Dropout(float(gcfg.get("dropout", 0.0)))
        self.head = MultiHorizonHead(hidden, n_targets, n_horizons)

    @staticmethod
    def delta_and_last(
        values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-cell time-since-last-observation and last observed value.

        ``delta[t, v]`` = steps since variable v was last observed strictly
        before t (Che et al. eq. 2 with unit step; t itself when never
        observed yet). ``x_last[t, v]`` = most recent observed value at or
        before t, 0 (= scaled train mean) when none exists.
        """
        B, L, V = mask.shape
        t_idx = torch.arange(L, device=mask.device).view(1, L, 1)
        obs_idx = torch.where(mask > 0, t_idx.expand(B, L, V),
                              t_idx.new_tensor(-1).expand(B, L, V))
        last_idx = torch.cummax(obs_idx, dim=1).values          # (B, L, V)
        prev_idx = torch.cat(
            [last_idx.new_full((B, 1, V), -1), last_idx[:, :-1]], dim=1
        )
        delta = (t_idx - prev_idx.clamp(min=0)).float()
        x_last = torch.gather(values, 1, last_idx.clamp(min=0))
        x_last = torch.where(last_idx >= 0, x_last, torch.zeros_like(x_last))
        return delta, x_last

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample dict (values/mask/time_feats/station_id) -> (B, T, H)."""
        values = batch["values"] * batch["mask"]  # defensive zero-fill
        mask = batch["mask"]
        delta, x_last = self.delta_and_last(values, mask)
        delta = delta / 24.0  # days, for conditioning only

        gamma_x = torch.exp(-torch.relu(self.decay_x_w * delta + self.decay_x_b))
        # empirical mean is 0 in scaled space, so the mean term vanishes
        x_hat = mask * values + (1.0 - mask) * gamma_x * x_last
        gamma_h = torch.exp(-torch.relu(self.decay_h(delta)))   # (B, L, hidden)

        u = torch.cat([x_hat, mask, batch["time_feats"]], dim=-1)
        gates_in = self.in_proj(u)                              # (B, L, 3*hidden)
        h = self.station_embed(batch["station_id"])             # (B, hidden)
        for t in range(u.size(1)):
            h = gamma_h[:, t] * h
            i_r, i_z, i_n = gates_in[:, t].chunk(3, dim=-1)
            h_r, h_z, h_n = self.h_proj(h).chunk(3, dim=-1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1.0 - z) * n + z * h
        return self.head(self.dropout(h))
