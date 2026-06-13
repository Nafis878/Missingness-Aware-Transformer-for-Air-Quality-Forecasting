"""DLinear baseline (Zeng et al., 2023, "Are Transformers Effective for Time
Series Forecasting?").

The embarrassingly simple linear baseline: decompose each target series into
trend (moving average) + seasonal (residual) and map each component to the
forecast horizons with one linear layer per channel ("individual" DLinear).
Adapted to this project's masked contract:

* Channel-independent over the T **target** pollutants (the model is
  univariate per channel by design; the remaining covariates are not used,
  exactly as in the original).
* Missingness handling: inputs are zero-filled where missing (``values *
  mask``), and each channel's binary mask vector enters through a third
  linear term, so the model sees the same information as the proposed model
  (zero-fill + mask channel) without changing the linear model class.
* A per-station bias completes the parity with the station embeddings used by
  the neural models.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinearForecaster(nn.Module):
    """Per-target trend/seasonal linear forecaster on zero-filled + mask inputs.

    Parameters
    ----------
    n_features, n_stations, n_targets, n_horizons:
        Data dimensions (V, S, T, H).
    cfg:
        Full config; reads ``baselines.dlinear`` and ``dataset.input_length``.
    target_indices:
        Column indices of the T target pollutants within the feature axis.
    """

    def __init__(
        self,
        n_features: int,
        n_stations: int,
        n_targets: int,
        n_horizons: int,
        cfg: dict[str, Any],
        target_indices: list[int],
    ) -> None:
        super().__init__()
        if len(target_indices) != n_targets:
            raise ValueError("target_indices must have length n_targets")
        dcfg = cfg["baselines"]["dlinear"]
        kernel = int(dcfg["kernel_size"])
        if kernel % 2 == 0:
            raise ValueError("dlinear.kernel_size must be odd")
        self.kernel = kernel
        L = int(cfg["dataset"]["input_length"])
        self.register_buffer(
            "target_idx", torch.tensor(target_indices, dtype=torch.long)
        )

        def linear_weight() -> nn.Parameter:
            w = torch.empty(n_targets, n_horizons, L)
            nn.init.uniform_(w, -L ** -0.5, L ** -0.5)
            return nn.Parameter(w)

        self.w_seasonal = linear_weight()
        self.w_trend = linear_weight()
        self.w_mask = linear_weight()
        self.bias = nn.Parameter(torch.zeros(n_targets, n_horizons))
        self.station_bias = (
            nn.Embedding(n_stations, n_targets * n_horizons)
            if bool(dcfg.get("station_bias", True)) else None
        )
        if self.station_bias is not None:
            nn.init.zeros_(self.station_bias.weight)
        self.n_targets, self.n_horizons = n_targets, n_horizons

    def _decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, L) -> (seasonal, trend) via replicate-padded moving average."""
        pad_l = (self.kernel - 1) // 2
        pad_r = self.kernel - 1 - pad_l
        trend = F.avg_pool1d(
            F.pad(x, (pad_l, pad_r), mode="replicate"), self.kernel, stride=1
        )
        return x - trend, trend

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample dict (values/mask/station_id) -> (B, T, H)."""
        values = batch["values"] * batch["mask"]  # defensive zero-fill
        x = values.index_select(-1, self.target_idx).permute(0, 2, 1)  # (B, T, L)
        m = batch["mask"].index_select(-1, self.target_idx).permute(0, 2, 1)
        seasonal, trend = self._decompose(x)
        out = (
            torch.einsum("btl,thl->bth", seasonal, self.w_seasonal)
            + torch.einsum("btl,thl->bth", trend, self.w_trend)
            + torch.einsum("btl,thl->bth", m, self.w_mask)
            + self.bias
        )
        if self.station_bias is not None:
            out = out + self.station_bias(batch["station_id"]).view(
                -1, self.n_targets, self.n_horizons
            )
        return out
