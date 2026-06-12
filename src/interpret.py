"""Phase 7: attention extraction and analysis for the proposed model.

Attention weights are recovered **without modifying training code**:
``nn.TransformerEncoderLayer`` does not expose weights, so a forward
pre-hook captures each layer's input ``x`` during a normal forward pass and
the per-head attention is recomputed offline as
``layer.self_attn(norm1(x), norm1(x), norm1(x), need_weights=True,
average_attn_weights=False)`` (pre-norm aware; identical computation in
eval mode).

Analyses (aggregated on the fly to bound memory):

* attention-by-lag profile from the last-token query row (forecast token);
* mean attention maps for high- vs low-missingness windows;
* attention mass on meteorology-observed vs PM2.5-observed timesteps when
  PM2.5 history is largely missing;
* per-head specialization (peak lag, entropy);
* monsoon vs winter lag profiles;
* permutation feature importance vs gradient saliency (variable-level
  complement to the timestep-level attention).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def layer_attention(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> list[torch.Tensor]:
    """Per-layer per-head attention maps for one batch.

    Returns a list (n_layers) of tensors (B, n_heads, L, L); rows are
    queries, columns keys.
    """
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, args: tuple) -> None:
        captured.append(args[0].detach())

    handles = [layer.register_forward_pre_hook(hook) for layer in model.encoder.layers]
    try:
        model.eval()
        model(batch)
    finally:
        for h in handles:
            h.remove()

    maps = []
    for layer, x in zip(model.encoder.layers, captured):
        attn_in = layer.norm1(x) if layer.norm_first else x
        _, weights = layer.self_attn(
            attn_in, attn_in, attn_in,
            need_weights=True, average_attn_weights=False,
        )
        maps.append(weights.detach())  # (B, n_heads, L, L)
    return maps


class AttentionAggregator:
    """Streaming aggregation of attention statistics over many batches."""

    def __init__(self, n_layers: int, n_heads: int, seq_len: int) -> None:
        self.L = seq_len
        # last-token query row per (layer, head): sum over windows
        self.lag_sum = np.zeros((n_layers, n_heads, seq_len))
        self.lag_n = 0
        # full maps (head-averaged) for missingness groups
        self.group_maps: dict[str, np.ndarray] = {
            "high": np.zeros((seq_len, seq_len)), "low": np.zeros((seq_len, seq_len))
        }
        self.group_n = {"high": 0, "low": 0}
        # seasonal lag profiles (layer/head-averaged)
        self.season_sum: dict[str, np.ndarray] = {
            "Monsoon": np.zeros(seq_len), "Winter": np.zeros(seq_len)
        }
        self.season_n = {"Monsoon": 0, "Winter": 0}
        # attention mass split for PM2.5-sparse windows
        self.mass_met = []
        self.mass_pm = []

    def update(
        self,
        maps: list[torch.Tensor],
        batch: dict[str, torch.Tensor],
        pm_idx: int,
        met_idx: list[int],
        seasons: np.ndarray,
    ) -> None:
        stack = torch.stack(maps)                      # (layers, B, heads, L, L)
        last_q = stack[:, :, :, -1, :].numpy()         # (layers, B, heads, L)
        self.lag_sum += last_q.sum(axis=1).transpose(0, 1, 2)
        self.lag_n += last_q.shape[1]

        head_avg = stack.mean(dim=(0, 2)).numpy()      # (B, L, L)
        mask = batch["mask"].numpy()
        miss_frac = 1.0 - mask.mean(axis=(1, 2))
        profile = last_q.mean(axis=(0, 2))             # (B, L)
        for b in range(head_avg.shape[0]):
            group = "high" if miss_frac[b] > 0.6 else ("low" if miss_frac[b] < 0.15 else None)
            if group:
                self.group_maps[group] += head_avg[b]
                self.group_n[group] += 1
            season = seasons[b]
            if season in self.season_sum:
                self.season_sum[season] += profile[b]
                self.season_n[season] += 1
            # PM2.5-sparse windows: where does the forecast token look?
            pm_obs = mask[b, :, pm_idx] > 0
            met_obs = (mask[b][:, met_idx] > 0).any(axis=1)
            if pm_obs.mean() < 0.3:
                self.mass_pm.append(float(profile[b][pm_obs].sum()))
                self.mass_met.append(float(profile[b][met_obs & ~pm_obs].sum()))

    # -- results ------------------------------------------------------------

    def lag_profile(self) -> np.ndarray:
        """(n_layers, n_heads, L) mean last-token attention by key position."""
        return self.lag_sum / max(self.lag_n, 1)

    def head_specialization(self) -> list[dict[str, Any]]:
        """Peak lag (hours before forecast token) and entropy per layer/head."""
        prof = self.lag_profile()
        rows = []
        for li in range(prof.shape[0]):
            for hi in range(prof.shape[1]):
                p = prof[li, hi]
                p = p / p.sum()
                lag = (self.L - 1) - int(np.argmax(p))
                entropy = float(-(p * np.log(p + 1e-12)).sum())
                rows.append({"layer": li, "head": hi, "peak_lag_h": lag,
                             "entropy_nats": round(entropy, 3)})
        return rows


@torch.no_grad()
def permutation_importance(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    feats: list[str],
    pm_idx: int,
    h_idx: int,
    seed: int,
) -> dict[str, float]:
    """Delta masked-RMSE (scaled) when variable v is permuted across windows.

    Values and mask of variable v are permuted JOINTLY across the batch so
    the (value, mask) pairing stays valid.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)

    def rmse(b: dict[str, torch.Tensor]) -> float:
        pred = model(b)[:, pm_idx, h_idx]
        m = b["target_mask"][:, pm_idx, h_idx] > 0
        return float(torch.sqrt(((pred[m] - b["targets"][:, pm_idx, h_idx][m]) ** 2).mean()))

    base = rmse(batch)
    out = {}
    for vi, name in enumerate(feats):
        perm = torch.randperm(batch["values"].shape[0], generator=g)
        b2 = dict(batch)
        b2["values"] = batch["values"].clone()
        b2["mask"] = batch["mask"].clone()
        b2["values"][:, :, vi] = batch["values"][perm, :, vi]
        b2["mask"][:, :, vi] = batch["mask"][perm, :, vi]
        out[name] = rmse(b2) - base
    logger.info("permutation importance (base scaled RMSE %.4f): %s", base,
                {k: round(v, 4) for k, v in out.items()})
    return out


def gradient_saliency(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    feats: list[str],
    pm_idx: int,
    h_idx: int,
) -> dict[str, float]:
    """Mean |d yhat_PM2.5,h / d values_v| over observed input cells."""
    model.eval()
    values = batch["values"].clone().requires_grad_(True)
    b2 = dict(batch)
    b2["values"] = values
    pred = model(b2)[:, pm_idx, h_idx].sum()
    pred.backward()
    grad = values.grad.abs()  # (B, L, V)
    mask = batch["mask"]
    out = {}
    for vi, name in enumerate(feats):
        m = mask[:, :, vi] > 0
        out[name] = float(grad[:, :, vi][m].mean()) if m.any() else 0.0
    return out
