"""PatchTST baseline (Nie et al., 2023, "A Time Series is Worth 64 Words").

Channel-independent patching Transformer at a CPU-friendly scale matched to
the proposed model (d_model 128, 3 layers, d_ff 256). Adapted to the masked
contract: the channel set is the 2V channels ``(values * mask) || mask`` —
zero-filled values plus the binary mask as extra channels — so the model sees
the same information as the proposed model.

Per channel the series is unfolded into overlapping patches (replicate-padded
at the end so the last patch covers the final timestep), linearly embedded,
given a learned patch-position embedding, and run through a shared
Transformer encoder with the channel folded into the batch dimension
(channel independence). The supervised multi-horizon head needs one vector
per window, so per-channel representations (last patch token) are combined
with a learned channel-identity embedding and attention pooling over the
channel axis — the standard adaptation of PatchTST's per-channel design to a
multi-target forecast head.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import AttentionPooling, MultiHorizonHead


class PatchTSTForecaster(nn.Module):
    """Channel-independent patching Transformer on zero-fill + mask channels.

    Parameters
    ----------
    n_features, n_stations, n_targets, n_horizons:
        Data dimensions (V, S, T, H). Channels = 2V.
    cfg:
        Full config; reads ``baselines.patchtst`` and ``dataset.input_length``
        (the latter only to size the learned position embedding; shorter
        sequences at inference are fine).
    """

    def __init__(
        self,
        n_features: int,
        n_stations: int,
        n_targets: int,
        n_horizons: int,
        cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        pcfg = cfg["baselines"]["patchtst"]
        self.patch_len = int(pcfg["patch_len"])
        self.stride = int(pcfg["stride"])
        d_model = int(pcfg["d_model"])
        L = int(cfg["dataset"]["input_length"])
        max_patches = self._n_patches(max(L, self.patch_len))

        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.channel_embed = nn.Parameter(torch.zeros(2 * n_features, d_model))
        nn.init.normal_(self.channel_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(pcfg["n_heads"]),
            dim_feedforward=int(pcfg["d_ff"]),
            dropout=float(pcfg["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(pcfg["n_layers"]))
        self.channel_pool = AttentionPooling(d_model)
        self.station_embed = nn.Embedding(n_stations, d_model)
        self.dropout = nn.Dropout(float(pcfg["dropout"]))
        self.head = MultiHorizonHead(d_model, n_targets, n_horizons)

    def _n_patches(self, length: int) -> int:
        """Patch count after end-padding so the last patch reaches the end."""
        if length <= self.patch_len:
            return 1
        return -(-(length - self.patch_len) // self.stride) + 1

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample dict (values/mask/station_id) -> (B, T, H)."""
        values = batch["values"] * batch["mask"]  # defensive zero-fill
        x = torch.cat([values, batch["mask"]], dim=-1)   # (B, L, 2V)
        x = x.permute(0, 2, 1)                           # (B, C, L)
        B, C, L = x.shape
        n_patches = self._n_patches(L)
        # replicate-pad the end so patches tile the full window
        padded_len = self.patch_len + (n_patches - 1) * self.stride
        if padded_len > L:
            x = F.pad(x, (0, padded_len - L), mode="replicate")
        patches = x.unfold(-1, self.patch_len, self.stride)  # (B, C, P, patch_len)

        z = self.patch_embed(patches) + self.pos_embed[:n_patches]
        z = z.reshape(B * C, n_patches, -1)
        z = self.encoder(z)                              # channel-independent
        rep = z[:, -1].reshape(B, C, -1)                 # last patch token
        pooled = self.channel_pool(rep + self.channel_embed)
        pooled = pooled + self.station_embed(batch["station_id"])
        return self.head(self.dropout(pooled))
