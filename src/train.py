"""Unified training loop for every neural model in the project.

* Masked MSE loss: only observed targets contribute, with configurable
  per-horizon weights.
* AdamW + cosine LR decay, gradient clipping, early stopping on validation
  masked MSE, best-checkpoint saving.
* :func:`predict` runs inference and returns aligned arrays (predictions,
  targets, masks, station ids, anchor times) that Phase 5 evaluation and the
  Diebold-Mariano tests consume.
* Efficiency statistics (parameter count, wall-clock train time, per-window
  inference latency) are recorded for the paper's CPU-deployability table.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.common import count_parameters

logger = logging.getLogger(__name__)


def masked_mse(
    pred: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE over observed targets only.

    All tensors are (B, T, H). ``horizon_weights`` (H,) rescales horizons.
    Cells with ``target_mask == 0`` contribute exactly nothing (the dataset
    guarantees ``targets == 0`` there, but the multiplication by the mask is
    what enforces the no-leakage contract).
    """
    se = (pred - targets) ** 2 * target_mask
    if horizon_weights is not None:
        se = se * horizon_weights.view(1, 1, -1)
        denom = (target_mask * horizon_weights.view(1, 1, -1)).sum()
    else:
        denom = target_mask.sum()
    return se.sum() / denom.clamp(min=1.0)


def make_loader(ds: Dataset, cfg: dict[str, Any], shuffle: bool, seed: int = 0) -> DataLoader:
    """DataLoader with a seeded generator for reproducible shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["train"]["num_workers"]),
        generator=gen if shuffle else None,
        drop_last=False,
    )


@torch.no_grad()
def _eval_loss(model: nn.Module, loader: DataLoader, hw: torch.Tensor) -> float:
    model.eval()
    total, weight = 0.0, 0.0
    for batch in loader:
        pred = model(batch)
        n = batch["target_mask"].sum().item()
        loss = masked_mse(pred, batch["targets"], batch["target_mask"], hw)
        total += loss.item() * n
        weight += n
    return total / max(weight, 1.0)


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: dict[str, Any],
    name: str,
    seed: int,
) -> dict[str, Any]:
    """Train with early stopping; save the best checkpoint.

    Returns a stats dict: checkpoint path, best val loss, epochs run,
    train time, parameter count, loss history.
    """
    tcfg = cfg["train"]
    hw = torch.tensor(tcfg["horizon_loss_weights"], dtype=torch.float32)
    train_loader = make_loader(train_ds, cfg, shuffle=True, seed=seed)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(tcfg["max_epochs"])
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_path = ckpt_dir / f"{name}_seed{seed}.pt"
    best_val, best_epoch = float("inf"), -1
    history: list[dict[str, float]] = []
    patience = int(tcfg["early_stopping_patience"])
    t0 = time.perf_counter()

    for epoch in range(int(tcfg["max_epochs"])):
        model.train()
        ep_loss, ep_n = 0.0, 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = masked_mse(pred, batch["targets"], batch["target_mask"], hw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            optimizer.step()
            n = batch["target_mask"].sum().item()
            ep_loss += loss.item() * n
            ep_n += n
        scheduler.step()

        train_loss = ep_loss / max(ep_n, 1.0)
        val_loss = _eval_loss(model, val_loader, hw)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        logger.info("%s epoch %02d: train=%.4f val=%.4f lr=%.2e",
                    name, epoch, train_loss, val_loss, scheduler.get_last_lr()[0])

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch,
                 "val_loss": val_loss, "name": name, "seed": seed},
                ckpt_path,
            )
        elif epoch - best_epoch >= patience:
            logger.info("%s: early stop at epoch %d (best %d)", name, epoch, best_epoch)
            break

    train_time = time.perf_counter() - t0
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
    stats = {
        "name": name, "seed": seed, "checkpoint": str(ckpt_path),
        "best_val_loss": best_val, "best_epoch": best_epoch,
        "epochs_run": len(history), "train_time_s": round(train_time, 1),
        "n_parameters": count_parameters(model), "history": history,
    }
    logger.info("%s: done in %.1fs, best val=%.4f (epoch %d), %d params",
                name, train_time, best_val, best_epoch, stats["n_parameters"])
    return stats


@torch.no_grad()
def predict(model: nn.Module, ds: Dataset, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run inference over ``ds`` and return aligned numpy arrays.

    Keys: predictions, targets, target_mask (n, T, H); station_id,
    anchor_time (n,); latency_ms_per_window (scalar).
    """
    model.eval()
    loader = make_loader(ds, cfg, shuffle=False)
    preds, targets, masks, sids, times = [], [], [], [], []
    t0 = time.perf_counter()
    for batch in loader:
        preds.append(model(batch).numpy())
        targets.append(batch["targets"].numpy())
        masks.append(batch["target_mask"].numpy())
        sids.append(batch["station_id"].numpy())
        times.append(batch["anchor_time"].numpy())
    latency_ms = (time.perf_counter() - t0) / max(len(ds), 1) * 1000
    return {
        "predictions": np.concatenate(preds),
        "targets": np.concatenate(targets),
        "target_mask": np.concatenate(masks),
        "station_id": np.concatenate(sids),
        "anchor_time": np.concatenate(times),
        "latency_ms_per_window": np.float64(latency_ms),
    }


def save_predictions(
    out: dict[str, np.ndarray], cfg: dict[str, Any], name: str, split: str = "test"
) -> Path:
    """Persist aligned prediction arrays for Phase 5 evaluation."""
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{name}_{split}.npz"
    np.savez_compressed(path, **out)
    logger.info("saved predictions to %s", path)
    return path


def save_stats(stats: dict[str, Any], cfg: dict[str, Any], name: str) -> None:
    """Persist training/efficiency stats next to the checkpoints."""
    path = Path(cfg["paths"]["checkpoints_dir"]) / f"{name}_stats.json"
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
