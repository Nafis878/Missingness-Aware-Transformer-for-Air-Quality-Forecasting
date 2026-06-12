"""PyTorch dataset layer: scalers, sliding windows, masks, splits.

Design (documented in README and the paper):

* **Splits** are chronological. A window consists of ``input_length`` hourly
  steps ending at anchor time ``t`` with targets at ``t + h`` for each horizon
  ``h``. A window belongs to a split iff **all** its horizon timestamps lie
  inside that split's range; inputs may reach back across the split boundary
  (deployment-realistic, leaks no future information).
* **Scalers** are per-variable mean/std computed on the training period only,
  ignoring NaNs, pooled across stations. Persisted to
  ``data/processed/scalers.json`` so every model sees identical scaling.
* **Window validity**: a window is kept iff at least one (target pollutant,
  horizon) pair is observed. Inputs may be arbitrarily incomplete - that is
  the point of the model.
* **Sample layout** (all float32 unless noted)::

      values      (L, V)  scaled, NaN -> 0 AFTER scaling
      mask        (L, V)  1 = observed
      time_feats  (L, 6)  sin/cos of hour-of-day, day-of-week, month
      station_id  ()      int64
      targets     (T, H)  scaled, NaN -> 0
      target_mask (T, H)  1 = observed
      anchor_time ()      int64, unix seconds of the last input step

* **Synthetic extra missingness** for the robustness ablation is a
  deterministic wrapper (:class:`ExtraMissingnessDataset`) that drops an
  additional MCAR fraction of *observed* input cells, identically
  reproducible for every model that evaluates on it.

CLI (build everything + export window-count tables)::

    python -m src.data.dataset --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

def compute_scalers(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, list[float]]:
    """Per-variable (mean, std) computed on the training period only, NaN-ignoring.

    Raises if a variable has no observed training values or zero variance.
    """
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    feats = feature_columns(cfg)
    train = df.loc[df["datetime"] <= train_end, feats]
    scalers: dict[str, list[float]] = {}
    for col in feats:
        vals = train[col].to_numpy(dtype=np.float64)
        n_obs = int(np.isfinite(vals).sum())
        if n_obs == 0:
            raise ValueError(f"scaler: no observed training values for {col!r}")
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        if std == 0:
            raise ValueError(f"scaler: zero variance for {col!r} on train")
        scalers[col] = [mean, std]
        logger.info("scaler %s: mean=%.3f std=%.3f (n_train_obs=%d)", col, mean, std, n_obs)
    return scalers


def save_scalers(scalers: dict[str, list[float]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(scalers, indent=2), encoding="utf-8")


def load_scalers(path: str | Path) -> dict[str, list[float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    """Model input variables: measurement columns minus excluded features."""
    excl = set(cfg["data"].get("exclude_features", []))
    return [c for c in cfg["data"]["measurement_cols"] if c not in excl]


# ---------------------------------------------------------------------------
# Station arrays
# ---------------------------------------------------------------------------

@dataclass
class StationArrays:
    """Contiguous per-station hourly arrays the Dataset slices into."""

    station: str
    station_id: int
    times: np.ndarray        # (N,) datetime64[s], hourly, gap-free
    values: np.ndarray       # (N, V) float32, scaled, NaN -> 0
    mask: np.ndarray         # (N, V) float32, 1 = observed
    raw_targets: np.ndarray  # (N, T) float32, scaled, NaN where missing
    time_feats: np.ndarray   # (N, 6) float32


def _time_features(times: pd.DatetimeIndex) -> np.ndarray:
    """Cyclic sin/cos encodings of hour-of-day, day-of-week, month."""
    hour = times.hour.to_numpy() / 24.0
    dow = times.dayofweek.to_numpy() / 7.0
    month = (times.month.to_numpy() - 1) / 12.0
    feats = []
    for frac in (hour, dow, month):
        feats.append(np.sin(2 * np.pi * frac))
        feats.append(np.cos(2 * np.pi * frac))
    return np.stack(feats, axis=1).astype(np.float32)


def build_station_arrays(
    df: pd.DataFrame, cfg: dict[str, Any], scalers: dict[str, list[float]]
) -> list[StationArrays]:
    """Build scaled value/mask/time arrays per station (sorted by name)."""
    feats = feature_columns(cfg)
    targets = cfg["dataset"]["target_pollutants"]
    out: list[StationArrays] = []
    for sid, (station, grp) in enumerate(df.groupby("station", sort=True)):
        grp = grp.sort_values("datetime")
        times = pd.DatetimeIndex(grp["datetime"])
        if len(times) > 1 and not (np.diff(times.asi8) == 3_600_000_000_000).all():
            raise ValueError(f"{station}: hourly index is not gap-free")
        raw = grp[feats].to_numpy(dtype=np.float64)
        mean = np.array([scalers[c][0] for c in feats])
        std = np.array([scalers[c][1] for c in feats])
        scaled = (raw - mean) / std
        mask = np.isfinite(scaled)
        values = np.where(mask, scaled, 0.0).astype(np.float32)

        tcols = [feats.index(t) for t in targets]
        raw_targets = np.where(mask, scaled, np.nan)[:, tcols].astype(np.float32)

        out.append(
            StationArrays(
                station=station,
                station_id=sid,
                times=times.to_numpy(),
                values=values,
                mask=mask.astype(np.float32),
                raw_targets=raw_targets,
                time_feats=_time_features(times),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Window index
# ---------------------------------------------------------------------------

def split_ranges(cfg: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Inclusive [start, end] target-time ranges for train/val/test."""
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    val_end = pd.Timestamp(cfg["splits"]["val_end"])
    return {
        "train": (pd.Timestamp.min, train_end),
        "val": (train_end + pd.Timedelta(hours=1), val_end),
        "test": (val_end + pd.Timedelta(hours=1), pd.Timestamp.max),
    }


def build_window_index(
    stations: list[StationArrays],
    split: str,
    cfg: dict[str, Any],
    input_length: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Enumerate valid (station_idx, anchor_pos) windows for one split.

    A window anchored at position ``p`` (last input step) uses inputs
    ``[p - L + 1, p]`` and targets at ``p + h`` for each horizon ``h``.
    Valid iff all horizon times fall in the split range and at least one
    (target, horizon) value is observed.

    Returns the index array (n, 2) and a per-station count DataFrame
    (total windows + windows with PM2.5 observed per horizon).
    """
    L = input_length or cfg["dataset"]["input_length"]
    horizons: list[int] = cfg["dataset"]["horizons"]
    stride = cfg["dataset"]["stride_train"] if split == "train" else cfg["dataset"]["stride_eval"]
    targets = cfg["dataset"]["target_pollutants"]
    pm_idx = targets.index(cfg["dataset"]["primary_target"])
    lo, hi = split_ranges(cfg)[split]

    rows = []
    index: list[tuple[int, int]] = []
    for s_i, st in enumerate(stations):
        n = len(st.times)
        times = pd.DatetimeIndex(st.times)
        max_h = max(horizons)
        anchors = np.arange(L - 1, n - max_h, stride)
        if len(anchors) == 0:
            rows.append({"station": st.station, "split": split, "windows": 0,
                         **{f"pm25_h{h}": 0 for h in horizons}})
            continue
        # all horizon target times must lie inside the split range
        h_times = {h: times[anchors + h] for h in horizons}
        in_range = np.ones(len(anchors), dtype=bool)
        for h in horizons:
            in_range &= (h_times[h] >= lo) & (h_times[h] <= hi)
        # at least one (target, horizon) observed
        any_obs = np.zeros(len(anchors), dtype=bool)
        pm_obs = {h: np.zeros(len(anchors), dtype=bool) for h in horizons}
        for h in horizons:
            t_vals = st.raw_targets[anchors + h]          # (n_anchors, T)
            obs = np.isfinite(t_vals)
            any_obs |= obs.any(axis=1)
            pm_obs[h] = obs[:, pm_idx]
        keep = in_range & any_obs
        index.extend((s_i, int(a)) for a in anchors[keep])
        rows.append({
            "station": st.station, "split": split, "windows": int(keep.sum()),
            **{f"pm25_h{h}": int((keep & pm_obs[h]).sum()) for h in horizons},
        })
    counts = pd.DataFrame(rows).set_index("station")
    idx_arr = np.array(index, dtype=np.int64).reshape(-1, 2)
    logger.info("%s: %d windows across %d stations (L=%d, stride=%d)",
                split, len(idx_arr), len(stations), L, stride)
    return idx_arr, counts


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class AirQualityWindowDataset(Dataset):
    """Sliding-window dataset over per-station hourly arrays.

    Parameters
    ----------
    stations:
        Output of :func:`build_station_arrays`.
    split:
        ``"train" | "val" | "test"`` -- selects target-time range and stride.
    cfg:
        Full config dict.
    input_length:
        Override of ``cfg['dataset']['input_length']`` (sequence-length ablation).
    """

    def __init__(
        self,
        stations: list[StationArrays],
        split: str,
        cfg: dict[str, Any],
        input_length: int | None = None,
    ) -> None:
        self.stations = stations
        self.split = split
        self.input_length = input_length or cfg["dataset"]["input_length"]
        self.horizons: list[int] = cfg["dataset"]["horizons"]
        self.index, self.window_counts = build_window_index(
            stations, split, cfg, self.input_length
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s_i, anchor = self.index[i]
        st = self.stations[s_i]
        L = self.input_length
        sl = slice(anchor - L + 1, anchor + 1)

        t_pos = anchor + np.asarray(self.horizons)
        t_vals = st.raw_targets[t_pos].T            # (T, H)
        t_mask = np.isfinite(t_vals)

        return {
            "values": torch.from_numpy(st.values[sl].copy()),
            "mask": torch.from_numpy(st.mask[sl].copy()),
            "time_feats": torch.from_numpy(st.time_feats[sl].copy()),
            "station_id": torch.tensor(st.station_id, dtype=torch.int64),
            "targets": torch.from_numpy(np.where(t_mask, t_vals, 0.0).astype(np.float32)),
            "target_mask": torch.from_numpy(t_mask.astype(np.float32)),
            "anchor_time": torch.tensor(
                int(pd.Timestamp(st.times[anchor]).timestamp()), dtype=torch.int64
            ),
        }


class ExtraMissingnessDataset(Dataset):
    """Deterministic MCAR extra-missingness wrapper for robustness ablations.

    For each window, an additional fraction ``level`` of **observed** input
    cells is flipped to missing (mask -> 0, value -> 0). The drop pattern is a
    pure function of (seed, level, window index), so every model evaluated on
    the same wrapper sees the same corrupted inputs. Targets are untouched.
    """

    def __init__(self, base: AirQualityWindowDataset, level: float, seed: int) -> None:
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1), got {level}")
        self.base = base
        self.level = level
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sample = self.base[i]
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(self.level * 1000), i])
        )
        mask = sample["mask"].numpy().copy()
        values = sample["values"].numpy().copy()
        obs = np.flatnonzero(mask)
        n_drop = int(round(len(obs) * self.level))
        drop = rng.choice(obs, size=n_drop, replace=False)
        mask.flat[drop] = 0.0
        values.flat[drop] = 0.0
        sample["mask"] = torch.from_numpy(mask)
        sample["values"] = torch.from_numpy(values)
        return sample


class RandomMissingnessAugment(Dataset):
    """Training-time missingness augmentation (ablation: ``miss_dropout``).

    Each access drops a random fraction U[0, max_level] of observed input
    cells (mask -> 0, value -> 0); targets untouched. The generator is seeded
    at construction and consumed in DataLoader access order, which is itself
    seeded, so full runs are reproducible while patterns still vary across
    epochs. Intended for the TRAIN split only.
    """

    def __init__(self, base: AirQualityWindowDataset, max_level: float, seed: int) -> None:
        if not 0 < max_level < 1:
            raise ValueError(f"max_level must be in (0, 1), got {max_level}")
        self.base = base
        self.max_level = max_level
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 104729]))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sample = dict(self.base[i])
        level = float(self.rng.uniform(0.0, self.max_level))
        mask = sample["mask"].numpy().copy()
        values = sample["values"].numpy().copy()
        obs = np.flatnonzero(mask)
        n_drop = int(round(len(obs) * level))
        if n_drop:
            drop = self.rng.choice(obs, size=n_drop, replace=False)
            mask.flat[drop] = 0.0
            values.flat[drop] = 0.0
        sample["mask"] = torch.from_numpy(mask)
        sample["values"] = torch.from_numpy(values)
        return sample


def make_datasets(
    cfg: dict[str, Any], input_length: int | None = None
) -> tuple[dict[str, AirQualityWindowDataset], list[StationArrays], dict[str, list[float]]]:
    """One-call constructor used by every training/evaluation script.

    Loads the processed parquet, computes (or loads) train-only scalers,
    builds station arrays and the three split datasets.
    """
    processed = Path(cfg["paths"]["processed_dir"])
    df = pd.read_parquet(processed / "all_stations.parquet")

    scaler_path = processed / "scalers.json"
    if scaler_path.exists():
        scalers = load_scalers(scaler_path)
        logger.info("loaded scalers from %s", scaler_path)
    else:
        scalers = compute_scalers(df, cfg)
        save_scalers(scalers, scaler_path)
        logger.info("computed and saved scalers to %s", scaler_path)

    stations = build_station_arrays(df, cfg, scalers)
    datasets = {
        split: AirQualityWindowDataset(stations, split, cfg, input_length)
        for split in ("train", "val", "test")
    }
    return datasets, stations, scalers


# ---------------------------------------------------------------------------
# CLI: build everything, export window-count tables
# ---------------------------------------------------------------------------

def _input_missingness_summary(ds: AirQualityWindowDataset) -> pd.Series:
    """Distribution of per-window input missingness (fraction of masked cells)."""
    fracs = []
    for s_i, anchor in ds.index:
        st = ds.stations[s_i]
        sl = slice(anchor - ds.input_length + 1, anchor + 1)
        fracs.append(1.0 - float(st.mask[sl].mean()))
    s = pd.Series(fracs)
    return s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils import export_table, load_config, seed_everything, setup_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("dataset_build", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))

    datasets, _, _ = make_datasets(cfg)

    counts = pd.concat([ds.window_counts for ds in datasets.values()])
    wide = counts.pivot_table(index="station", columns="split", values="windows",
                              fill_value=0)[["train", "val", "test"]]
    wide.loc["TOTAL"] = wide.sum()
    export_table(
        wide.astype(int), cfg["paths"]["tables_dir"], "window_counts",
        "Number of usable sliding windows per station and split "
        "(window kept iff at least one target pollutant is observed at one horizon).",
        "tab:window_counts", float_format="%d",
    )

    pm_cols = [c for c in counts.columns if c.startswith("pm25_h")]
    pm = counts.reset_index().pivot_table(index="station", columns="split",
                                          values=pm_cols, fill_value=0).astype(int)
    pm.columns = [f"{h.replace('pm25_', '')}_{s}" for h, s in pm.columns]
    export_table(
        pm, cfg["paths"]["tables_dir"], "window_counts_pm25",
        "Windows with PM2.5 observed, per horizon and split.",
        "tab:window_counts_pm25", float_format="%d",
    )

    miss = pd.DataFrame({s: _input_missingness_summary(ds) for s, ds in datasets.items()})
    export_table(
        miss.round(3), cfg["paths"]["tables_dir"], "window_input_missingness",
        "Distribution of per-window input missingness (fraction of input cells missing).",
        "tab:window_input_missingness", float_format="%.3f",
    )

    for split, ds in datasets.items():
        logger.info("%s: %d windows", split, len(ds))


if __name__ == "__main__":
    main()
