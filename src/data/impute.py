"""Imputation for the baselines (the two-stage pipeline and RNNs).

Two families:

* **Window-level forward-fill + mean** (:class:`FfillImputedDataset`) - the
  standard cheap practice for RNN baselines. Operates inside each input
  window only, so it cannot leak anything.
* **Full-series multivariate imputation** (:func:`impute_full_series`) with
  sklearn ``KNNImputer`` and ``IterativeImputer`` (MICE-style) for the
  two-stage baseline. Imputers are **fit on training-period rows only** and
  then applied to the whole series. Cyclic calendar features are appended as
  auxiliary predictors so the imputers see time-of-day/season structure.
  KNN transform cost is O(n_fit x n_query), so the fit set is a random
  subsample (size in config) of train rows - documented in the paper.

:func:`replace_inputs` clones the per-station arrays with imputed inputs
(mask = all ones) while keeping the original targets and window geometry, so
two-stage models are evaluated on *exactly* the same windows and targets as
every other model.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dataset import AirQualityWindowDataset, StationArrays

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window-level ffill + mean (for RNN baselines)
# ---------------------------------------------------------------------------

def ffill_mean_impute(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Forward-fill each column over time; remaining gaps -> 0 (= train mean).

    ``values`` (L, V) are scaled with missing cells already 0; ``mask`` (L, V)
    marks observations. Returns the imputed copy.
    """
    L, V = values.shape
    out = values.copy()
    observed = mask > 0
    # last observed row index per (row, col), -1 if none yet
    idx = np.where(observed, np.arange(L)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    has_prev = idx >= 0
    rows = np.clip(idx, 0, None)
    cols = np.broadcast_to(np.arange(V), (L, V))
    out = np.where(has_prev, values[rows, cols], 0.0)
    return out.astype(np.float32)


class FfillImputedDataset(Dataset):
    """Wraps a window dataset, replacing values with ffill+mean imputed ones.

    The mask is set to all-ones (models consuming this wrapper are mask-blind
    by design). Targets are untouched.
    """

    def __init__(self, base: AirQualityWindowDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sample = dict(self.base[i])
        values = sample["values"].numpy()
        mask = sample["mask"].numpy()
        sample["values"] = torch.from_numpy(ffill_mean_impute(values, mask))
        sample["mask"] = torch.ones_like(sample["mask"])
        return sample


# ---------------------------------------------------------------------------
# Full-series multivariate imputation (two-stage baseline)
# ---------------------------------------------------------------------------

def _calendar_features(times: np.ndarray) -> np.ndarray:
    """Cyclic hour/month features used as auxiliary imputation predictors."""
    dt = pd.DatetimeIndex(times)
    hour = dt.hour.to_numpy() / 24.0
    month = (dt.month.to_numpy() - 1) / 12.0
    return np.stack(
        [np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour),
         np.sin(2 * np.pi * month), np.cos(2 * np.pi * month)],
        axis=1,
    )


def _stacked_matrix(stations: list[StationArrays]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stack all stations into one (N, V+4) matrix with NaNs restored.

    Returns (matrix, is_train_placeholder_times, station row offsets).
    """
    blocks, times, offsets = [], [], [0]
    for st in stations:
        vals = np.where(st.mask > 0, st.values, np.nan).astype(np.float64)
        blocks.append(np.hstack([vals, _calendar_features(st.times)]))
        times.append(st.times)
        offsets.append(offsets[-1] + len(st.times))
    return np.vstack(blocks), np.concatenate(times), offsets


def impute_full_series(
    stations: list[StationArrays],
    cfg: dict[str, Any],
    method: str,
    seed: int,
) -> list[np.ndarray]:
    """Impute the full multivariate series with KNN or IterativeImputer (MICE).

    Fit on a random subsample of training-period rows ONLY; transform all rows.
    Returns one imputed (N_station, V) float32 array per station.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer

    icfg = cfg["baselines"]["impute"]
    train_end = pd.Timestamp(cfg["splits"]["train_end"]).to_datetime64()
    n_vars = stations[0].values.shape[1]

    X, times, offsets = _stacked_matrix(stations)
    train_rows = np.flatnonzero(times <= train_end)
    rng = np.random.default_rng(seed)

    if method == "knn":
        sub = int(icfg["knn_fit_subsample"])
        fit_rows = rng.choice(train_rows, size=min(sub, len(train_rows)), replace=False)
        imputer = KNNImputer(n_neighbors=int(icfg["knn_neighbors"]))
    elif method == "mice":
        sub = int(icfg["mice_fit_subsample"])
        fit_rows = rng.choice(train_rows, size=min(sub, len(train_rows)), replace=False)
        imputer = IterativeImputer(
            max_iter=int(icfg["mice_max_iter"]), random_state=seed,
            sample_posterior=False, keep_empty_features=True,
        )
    else:
        raise ValueError(f"unknown imputation method {method!r}")

    logger.info("%s imputer: fitting on %d train rows (of %d), transforming %d rows",
                method, len(fit_rows), len(train_rows), len(X))
    imputer.fit(X[fit_rows])
    # transform in chunks to bound memory (KNN distance matrices)
    imputed = np.empty_like(X)
    chunk = 20000
    for lo in range(0, len(X), chunk):
        imputed[lo: lo + chunk] = imputer.transform(X[lo: lo + chunk])
    # rows that remain NaN (e.g. KNN with no finite features) -> 0 = train mean
    n_residual = int(np.isnan(imputed[:, :n_vars]).sum())
    if n_residual:
        logger.info("%s: %d residual NaNs after imputation set to train mean (0)",
                    method, n_residual)
        imputed = np.nan_to_num(imputed, nan=0.0)

    return [
        imputed[offsets[i]: offsets[i + 1], :n_vars].astype(np.float32)
        for i in range(len(stations))
    ]


def corrupt_test_inputs(
    stations: list[StationArrays],
    cfg: dict[str, Any],
    level: float,
    seed: int,
) -> list[StationArrays]:
    """Series-level MCAR corruption of observed test-period input cells.

    Drops an additional fraction ``level`` of the OBSERVED cells in rows
    after ``splits.val_end`` (the test period): mask -> 0, value -> 0.
    Targets (``raw_targets``) are untouched, so evaluation targets are
    identical across corruption levels. The drop pattern is a pure function
    of (seed, level), identical for every model evaluated on it — including
    the two-stage pipeline, which re-imputes the corrupted series.
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    val_end = pd.Timestamp(cfg["splits"]["val_end"]).to_datetime64()
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(level * 1000)]))
    out = []
    total_dropped = 0
    for st in stations:
        values = st.values.copy()
        mask = st.mask.copy()
        in_test = st.times > val_end
        obs = np.argwhere((mask > 0) & in_test[:, None])
        n_drop = int(round(len(obs) * level))
        drop = obs[rng.choice(len(obs), size=n_drop, replace=False)]
        mask[drop[:, 0], drop[:, 1]] = 0.0
        values[drop[:, 0], drop[:, 1]] = 0.0
        total_dropped += n_drop
        out.append(replace(st, values=values, mask=mask))
    logger.info("corrupt_test_inputs(level=%.2f): dropped %d observed cells",
                level, total_dropped)
    return out


def corrupt_test_outages(
    stations: list[StationArrays],
    cfg: dict[str, Any],
    level: float,
    seed: int,
    block_hours: tuple[int, int] = (6, 48),
) -> list[StationArrays]:
    """Station-outage-style corruption: contiguous all-variable blocks.

    Mimics the dominant real-world missingness mechanism identified in the
    Phase 1 analysis (station-level outages: co-missingness correlation 0.38,
    ~39% of missing PM2.5 hours in gaps > 7 days). Repeatedly samples a
    block start in the test period and a length ~ U[6, 48] hours, then drops
    ALL observed input cells in that block, until a fraction ``level`` of the
    station's observed test cells is removed. Targets are untouched.

    This is the complement to :func:`corrupt_test_inputs` (cell-wise MCAR):
    cell-wise drops leave the same-timestep cross-section intact (easy for
    row-wise imputers), outage blocks do not.
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    val_end = pd.Timestamp(cfg["splits"]["val_end"]).to_datetime64()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919, int(level * 1000)]))
    out = []
    total_dropped = 0
    for st in stations:
        values = st.values.copy()
        mask = st.mask.copy()
        test_rows = np.flatnonzero(st.times > val_end)
        if len(test_rows) == 0:
            out.append(replace(st, values=values, mask=mask))
            continue
        lo_row, hi_row = test_rows[0], test_rows[-1]
        budget = int(round(mask[test_rows].sum() * level))
        dropped = 0
        attempts = 0
        while dropped < budget and attempts < 10000:
            attempts += 1
            start = int(rng.integers(lo_row, hi_row + 1))
            length = int(rng.integers(block_hours[0], block_hours[1] + 1))
            end = min(start + length, hi_row + 1)
            blk = mask[start:end] > 0
            n = int(blk.sum())
            if n == 0:
                continue
            mask[start:end][blk] = 0.0
            values[start:end][blk] = 0.0
            dropped += n
        total_dropped += dropped
        out.append(replace(st, values=values, mask=mask))
    logger.info("corrupt_test_outages(level=%.2f): dropped %d observed cells",
                level, total_dropped)
    return out


def replace_inputs(
    stations: list[StationArrays], imputed: list[np.ndarray]
) -> list[StationArrays]:
    """Clone station arrays with imputed inputs and all-ones masks.

    Targets (``raw_targets``) and times are untouched, so window enumeration
    and evaluation are identical to the non-imputed datasets.
    """
    out = []
    for st, vals in zip(stations, imputed):
        assert vals.shape == st.values.shape
        out.append(replace(st, values=vals, mask=np.ones_like(st.mask)))
    return out
