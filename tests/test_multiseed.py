"""Tests for the multi-seed aggregation and per-seed significance machinery.

Covers: per-seed bundle discovery (``seeds/`` subdir + top-level canonical
fallback), the mean +/- std headline-table math, and the per-seed
Diebold-Mariano table's all-seeds-significant flag on constructed errors with
known ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (
    build_main_results,
    episode_table,
    iter_seed_bundles,
    seed_metrics_long,
    significance_table_multiseed,
)

SCALERS = {"PM2.5": [50.0, 30.0], "PM10": [80.0, 40.0]}


def eval_cfg(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "paths": {"predictions_dir": str(tmp_path / "predictions")},
        "dataset": {
            "target_pollutants": ["PM2.5", "PM10"],
            "horizons": [6, 24],
            "primary_target": "PM2.5",
        },
        "ablation": {"seeds": [42, 43, 44]},
    }


def make_bundle(rng: np.random.Generator, n: int = 400, noise: float = 0.1,
                bias: float = 0.0) -> dict[str, np.ndarray]:
    """Synthetic (n, T=2, H=2) bundle; predictions = targets + noise + bias."""
    y = rng.normal(size=(n, 2, 2)).astype(np.float32)
    p = (y + rng.normal(scale=noise, size=(n, 2, 2)) + bias).astype(np.float32)
    return {
        "predictions": p,
        "targets": y,
        "target_mask": np.ones((n, 2, 2), dtype=np.float32),
        "station_id": np.zeros(n, dtype=np.int64),
        "anchor_time": (np.arange(n) * 3600).astype(np.int64),
        "latency_ms_per_window": np.float64(1.0),
    }


def write_bundle(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)


def test_iter_seed_bundles_discovery_and_canonical_fallback(tmp_path) -> None:
    cfg = eval_cfg(tmp_path)
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    rng = np.random.default_rng(0)
    # canonical seed 42 exists ONLY at the top level (pre-upgrade layout)
    write_bundle(pred_dir / "lstm_test.npz", make_bundle(rng))
    write_bundle(pred_dir / "seeds" / "lstm_s43_test.npz", make_bundle(rng))
    write_bundle(pred_dir / "seeds" / "lstm_s44_test.npz", make_bundle(rng))

    bundles = iter_seed_bundles(cfg, "lstm")
    assert sorted(bundles) == [42, 43, 44]

    # seeds/ takes precedence over the top-level file when both exist
    marked = make_bundle(rng)
    marked["predictions"] = marked["predictions"] + 123.0
    write_bundle(pred_dir / "seeds" / "lstm_s42_test.npz", marked)
    bundles = iter_seed_bundles(cfg, "lstm")
    assert float(bundles[42]["predictions"].mean()) > 100.0


def test_iter_seed_bundles_missing_seeds_skipped(tmp_path) -> None:
    cfg = eval_cfg(tmp_path)
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    write_bundle(pred_dir / "seeds" / "gru_s43_test.npz",
                 make_bundle(np.random.default_rng(1)))
    bundles = iter_seed_bundles(cfg, "gru")
    assert sorted(bundles) == [43]


def test_seed_metrics_long_and_main_results(tmp_path) -> None:
    cfg = eval_cfg(tmp_path)
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    rng = np.random.default_rng(2)
    write_bundle(pred_dir / "persistence_test.npz", make_bundle(rng, noise=0.5))
    for seed in (42, 43, 44):
        write_bundle(pred_dir / "seeds" / f"proposed_s{seed}_test.npz",
                     make_bundle(rng, noise=0.1))

    long_df = seed_metrics_long(cfg, SCALERS)
    pers = long_df[long_df["model"] == "persistence"]
    prop = long_df[long_df["model"] == "proposed"]
    # 2 pollutants x 2 horizons per (model, seed)
    assert len(pers) == 4 and pers["seed"].isna().all()
    assert len(prop) == 12 and sorted(prop["seed"].unique()) == [42, 43, 44]

    tbl = build_main_results(long_df, cfg)
    prop_cell = tbl.loc["Proposed (MAT)", ("RMSE", "h6")]
    pers_cell = tbl.loc["Persistence", ("RMSE", "h6")]
    assert "±" in prop_cell, "multi-seed model must report mean ± std"
    assert "±" not in pers_cell, "deterministic baseline must stay single-run"
    assert tbl.loc["Proposed (MAT)", ("", "seeds")] == 3
    assert tbl.loc["Persistence", ("", "seeds")] == 1

    # mean ± std math check against the long frame (population std)
    r = prop[(prop["pollutant"] == "PM2.5") & (prop["horizon"] == 6)]["RMSE"]
    assert prop_cell == f"{r.mean():.2f} ± {r.to_numpy().std():.2f}"


def test_episode_table_restricts_to_threshold_subset(tmp_path) -> None:
    cfg = eval_cfg(tmp_path)
    cfg["evaluation"] = {"episode_threshold": 150}
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    rng = np.random.default_rng(4)
    b = make_bundle(rng, n=400)
    # unscaled PM2.5 = scaled * 30 + 50 -> scaled 5.0 = 200 ug/m3 (episode),
    # scaled 0ish stays far below the 150 threshold
    b["targets"][:, 0, :] = 0.0
    b["targets"][:50, 0, 0] = 5.0
    b["predictions"] = b["targets"] + 0.1
    write_bundle(pred_dir / "persistence_test.npz", b)

    tbl = episode_table(cfg, SCALERS)
    assert tbl.loc["n (episode targets)", "h6"] == "50"
    assert tbl.loc["n (episode targets)", "h24"] == "0"
    # error is exactly 0.1 scaled = 3 ug/m3 on every episode target
    assert tbl.loc["Persistence", "h6"] == "3.00"


def test_significance_multiseed_flags_known_ordering(tmp_path) -> None:
    cfg = eval_cfg(tmp_path)
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    rng = np.random.default_rng(3)
    # baseline predictions biased by +1 scaled unit -> clearly worse
    write_bundle(pred_dir / "persistence_test.npz",
                 make_bundle(rng, noise=0.1, bias=1.0))
    for seed in (42, 43, 44):
        write_bundle(pred_dir / "seeds" / f"proposed_s{seed}_test.npz",
                     make_bundle(rng, noise=0.1))

    sig = significance_table_multiseed(cfg, SCALERS)
    pers = sig[sig["baseline"] == "Persistence"]
    assert len(pers) == 2  # one row per horizon
    assert pers["seeds"].eq(3).all()
    assert pers["sig_all_seeds"].all()
    assert (pers["DM_p_max"] < 0.05).all()
    assert (pers["RMSE_diff_mean"] < 0).all(), "negative = proposed better"
    assert (pers["DM_p_min"] <= pers["DM_p_median"]).all()
    assert (pers["DM_p_median"] <= pers["DM_p_max"]).all()
