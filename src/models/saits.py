"""Minimal SAITS imputer (Du et al., 2023, "SAITS: Self-Attention-based
Imputation for Time Series") for the deep-imputer two-stage baseline.

Implemented in-repo rather than via the ``pypots`` package so that the
project's contracts hold and are testable: config-driven hyperparameters,
``seed_everything`` determinism, fitting on train-period rows only, and CPU
checkpoints with the repo's resume convention.

Faithful at small scale to the original architecture:

* Two groups of **diagonally-masked self-attention** (DMSA) blocks — the
  attention mask forbids position t from attending to itself, so each
  timestep's reconstruction is driven by the rest of the series.
* First group reconstructs from ``[x, mask]``; its output fills the missing
  cells of the second group's input; learned per-cell combining weights blend
  the two estimates (computed from the mask — a simplification of the
  paper's attention-map-based weights, noted here for transparency).
* Joint **ORT + MIT** objective: MAE reconstruction of observed cells (ORT)
  plus MAE on a fraction ``mit_rate`` of observed cells artificially hidden
  from the input each batch (MIT), which directly supervises imputation.

:func:`train_saits` fits on overlapping train-period segments with early
stopping on held-back segments; :func:`impute_full_series_saits` transforms
whole station series segment-wise and matches the
:func:`src.data.impute.impute_full_series` contract, so the downstream
two-stage protocol (vanilla Transformer on imputed inputs) is identical to
KNN/MICE. Training also logs a quality gate: SAITS imputation MAE vs
window-level ffill on the same artificially-masked validation cells.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.common import SinusoidalPositionalEncoding, count_parameters

if TYPE_CHECKING:  # avoid circular import (impute.py imports this module)
    from src.data.dataset import StationArrays

logger = logging.getLogger(__name__)


class DMSABlock(nn.Module):
    """Pre-norm diagonally-masked self-attention + feed-forward block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.norm2(x)))


def diagonal_attn_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """(L, L) float mask, -inf on the diagonal: no position attends to itself."""
    m = torch.zeros(length, length, device=device)
    m.fill_diagonal_(float("-inf"))
    return m


class SAITS(nn.Module):
    """Two-group DMSA imputer.

    Parameters
    ----------
    n_features:
        Number of variables V.
    cfg:
        Full config; reads ``baselines.saits``.
    """

    def __init__(self, n_features: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        scfg = cfg["baselines"]["saits"]
        d_model = int(scfg["d_model"])
        n_heads = int(scfg["n_heads"])
        d_ff = int(scfg["d_ff"])
        n_layers = int(scfg["n_layers_per_group"])
        dropout = float(scfg["dropout"])

        self.embed1 = nn.Linear(2 * n_features, d_model)
        self.embed2 = nn.Linear(2 * n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.group1 = nn.ModuleList(
            DMSABlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.group2 = nn.ModuleList(
            DMSABlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.out1 = nn.Linear(d_model, n_features)
        self.out2 = nn.Linear(d_model, n_features)
        self.combine = nn.Linear(n_features, n_features)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, L, V) zero-filled values + mask -> three reconstructions."""
        am = diagonal_attn_mask(x.size(1), x.device)
        h = self.pos_enc(self.embed1(torch.cat([x, mask], dim=-1)))
        for blk in self.group1:
            h = blk(h, am)
        x1 = self.out1(h)

        xc = mask * x + (1.0 - mask) * x1
        h2 = self.pos_enc(self.embed2(torch.cat([xc, mask], dim=-1)))
        for blk in self.group2:
            h2 = blk(h2, am)
        x2 = self.out2(h2)

        eta = torch.sigmoid(self.combine(mask))
        x3 = (1.0 - eta) * x1 + eta * x2
        return x1, x2, x3

    @torch.no_grad()
    def impute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Imputed series: observed cells preserved exactly."""
        _, _, x3 = self(x, mask)
        return mask * x + (1.0 - mask) * x3


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1.0)


def _train_segments(
    stations: list["StationArrays"], cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlapping (values, mask) segments from TRAIN-period rows only."""
    scfg = cfg["baselines"]["saits"]
    seg_len = int(scfg["segment_len"])
    seg_stride = int(scfg["segment_stride"])
    train_end = pd.Timestamp(cfg["splits"]["train_end"]).to_datetime64()
    xs, ms = [], []
    for st in stations:
        n_train = int((st.times <= train_end).sum())
        vals, mask = st.values[:n_train], st.mask[:n_train]
        if n_train < seg_len:
            continue
        for start in range(0, n_train - seg_len + 1, seg_stride):
            m = mask[start: start + seg_len]
            if m.sum() == 0:
                continue
            xs.append(vals[start: start + seg_len])
            ms.append(m)
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms)))


def train_saits(
    stations: list["StationArrays"], cfg: dict[str, Any], seed: int
) -> tuple[SAITS, dict[str, Any]]:
    """Fit SAITS on train-period segments; resume from checkpoint if present.

    Returns the model and a stats dict (fit time, best val MIT-MAE, and the
    ffill quality-gate MAE on the same artificially-masked validation cells).
    """
    from src.train import resolve_device

    scfg = cfg["baselines"]["saits"]
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"saits_imputer_seed{seed}.pt"
    n_features = stations[0].values.shape[1]
    model = SAITS(n_features, cfg)

    if ckpt_path.exists():
        logger.info("saits seed %d: reusing %s", seed, ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
        stats_path = ckpt_dir / f"saits_imputer_seed{seed}_stats.json"
        stats = (json.loads(stats_path.read_text(encoding="utf-8"))
                 if stats_path.exists() else {})
        stats["reused_checkpoint"] = str(ckpt_path)
        return model, stats

    device = resolve_device(cfg)
    model.to(device)
    x_all, m_all = _train_segments(stations, cfg)
    val_sel = np.arange(len(x_all)) % 10 == 9  # held-back 10% for early stop
    x_tr, m_tr = x_all[~val_sel], m_all[~val_sel]
    x_va, m_va = x_all[val_sel].to(device), m_all[val_sel].to(device)
    logger.info("saits seed %d: %d train / %d val segments of length %d",
                seed, len(x_tr), len(x_va), x_all.shape[1])

    mit_rate = float(scfg["mit_rate"])
    batch_size = int(scfg["batch_size"])
    patience = int(scfg.get("patience", 5))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(scfg["lr"]))
    gen = torch.Generator().manual_seed(seed)
    # fixed artificial mask for validation (same cells every epoch)
    val_art = ((torch.rand(m_va.shape, generator=gen) < mit_rate).to(device)
               & (m_va > 0))

    def val_mae() -> float:
        model.eval()
        with torch.no_grad():
            m_in = m_va * (~val_art)
            _, _, x3 = model(x_va * m_in, m_in)
            return float(_masked_mae(x3, x_va, val_art.float()))

    best_val, best_epoch = float("inf"), -1
    t0 = time.perf_counter()
    for epoch in range(int(scfg["epochs"])):
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        ep_loss, n_batches = 0.0, 0
        for lo in range(0, len(perm), batch_size):
            idx = perm[lo: lo + batch_size]
            x = x_tr[idx].to(device)
            m = m_tr[idx].to(device)
            art = ((torch.rand(m.shape, generator=gen) < mit_rate).to(device)
                   & (m > 0))
            m_in = m * (~art)
            x_in = x * m_in
            x1, x2, x3 = model(x_in, m_in)
            ort = (_masked_mae(x1, x, m_in) + _masked_mae(x2, x, m_in)
                   + _masked_mae(x3, x, m_in)) / 3.0
            mit = _masked_mae(x3, x, art.float())
            loss = ort + mit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.detach())
            n_batches += 1
        v = val_mae()
        logger.info("saits epoch %02d: train=%.4f val_mit_mae=%.4f",
                    epoch, ep_loss / max(n_batches, 1), v)
        if v < best_val:
            best_val, best_epoch = v, epoch
            torch.save(
                {"model_state": {k: t.cpu() for k, t in model.state_dict().items()},
                 "epoch": epoch, "val_loss": v,
                 "name": "saits_imputer", "seed": seed},
                ckpt_path,
            )
        elif epoch - best_epoch >= patience:
            logger.info("saits: early stop at epoch %d (best %d)", epoch, best_epoch)
            break
    fit_time = time.perf_counter() - t0
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])

    # quality gate: window-level ffill on the SAME artificially-masked cells
    from src.data.impute import ffill_mean_impute

    ffill_err, ffill_n = 0.0, 0
    m_in_va = (m_va * (~val_art)).cpu().numpy()
    x_va_np, art_np = x_va.cpu().numpy(), val_art.cpu().numpy()
    for i in range(len(x_va_np)):
        imp = ffill_mean_impute(x_va_np[i] * m_in_va[i], m_in_va[i])
        ffill_err += float(np.abs(imp - x_va_np[i])[art_np[i]].sum())
        ffill_n += int(art_np[i].sum())
    ffill_mae = ffill_err / max(ffill_n, 1)

    stats = {
        "name": "saits_imputer", "seed": seed, "checkpoint": str(ckpt_path),
        "fit_time_s": round(fit_time, 1), "best_epoch": best_epoch,
        "best_val_mit_mae": round(best_val, 4),
        "ffill_val_mit_mae": round(ffill_mae, 4),
        "n_parameters": count_parameters(model),
    }
    (ckpt_dir / f"saits_imputer_seed{seed}_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    gate = "PASS" if best_val < ffill_mae else "FAIL (worse than ffill!)"
    logger.info("saits quality gate: val MIT-MAE %.4f vs ffill %.4f -> %s",
                best_val, ffill_mae, gate)
    return model, stats


def impute_full_series_saits(
    stations: list["StationArrays"],
    cfg: dict[str, Any],
    seed: int,
    model: SAITS | None = None,
) -> list[np.ndarray]:
    """Impute whole station series with SAITS, segment-wise.

    Matches the :func:`src.data.impute.impute_full_series` contract: returns
    one (N_station, V) float32 array per station, observed cells preserved
    exactly. Series are tiled with non-overlapping ``segment_len`` windows
    plus a final right-aligned window for the tail. When ``model`` is None it
    is trained (or resumed) on the train-period rows of ``stations`` — under
    test-time corruption those rows are uncorrupted, so reusing the trained
    imputer mirrors the KNN/MICE refit-on-train protocol.
    """
    from src.train import resolve_device

    if model is None:
        model, _ = train_saits(stations, cfg, seed)
    device = resolve_device(cfg)
    model.to(device)
    model.eval()
    seg_len = int(cfg["baselines"]["saits"]["segment_len"])

    out = []
    for st in stations:
        n = len(st.times)
        length = min(seg_len, n)
        starts = list(range(0, n - length + 1, length))
        if starts[-1] + length < n:
            starts.append(n - length)  # right-aligned tail (partially overlaps)
        imputed = np.empty_like(st.values)
        filled = np.zeros(n, dtype=bool)
        for lo in range(0, len(starts), 256):
            batch_starts = starts[lo: lo + 256]
            x = torch.from_numpy(
                np.stack([st.values[s: s + length] for s in batch_starts])
            ).to(device)
            m = torch.from_numpy(
                np.stack([st.mask[s: s + length] for s in batch_starts])
            ).to(device)
            seg = model.impute(x, m).cpu().numpy()
            for j, s in enumerate(batch_starts):
                rows = ~filled[s: s + length]
                imputed[s: s + length][rows] = seg[j][rows]
                filled[s: s + length] = True
        out.append(imputed.astype(np.float32))
    return out
