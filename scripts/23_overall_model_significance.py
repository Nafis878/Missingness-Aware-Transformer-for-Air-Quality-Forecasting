"""Overall significance check for a candidate model against all saved baselines.

The default candidate is ``variant_B_dual_input_ridge``. The test mirrors the
project's main significance convention: per-seed Diebold-Mariano tests on
PM2.5 squared errors, plus paired bootstrap confidence intervals for
RMSE(candidate) - RMSE(baseline). Negative differences favor the candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config


DEFAULT_BASELINES = [
    "lstm",
    "gru",
    "gru_d",
    "dlinear",
    "patchtst",
    "two_stage_knn",
    "two_stage_mice",
    "two_stage_saits",
    "proposed",
    "variant_B",
    "proposed_md",
    "hybrid8_transformer",
    "hybrid8_masked_variant_B",
    "hybrid8_masked_variant_B_vanilla_input",
]


def _load(cfg: dict[str, Any], model: str, seed: int) -> dict[str, np.ndarray] | None:
    path = Path(cfg["paths"]["predictions_dir"]) / "seeds" / f"{model}_s{seed}_test.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def _aligned_errors(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    cfg: dict[str, Any],
    scalers: dict[str, Any],
    horizon_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ti = cfg["dataset"]["target_pollutants"].index("PM2.5")
    mean, std = scalers["PM2.5"]
    cand_rows = {
        (int(sid), int(anchor)): i
        for i, (sid, anchor) in enumerate(zip(candidate["station_id"], candidate["anchor_time"]))
    }
    ia, ib = [], []
    for j, (sid, anchor) in enumerate(zip(baseline["station_id"], baseline["anchor_time"])):
        i = cand_rows.get((int(sid), int(anchor)))
        if i is not None:
            ia.append(i)
            ib.append(j)
    ia = np.asarray(ia, dtype=int)
    ib = np.asarray(ib, dtype=int)
    mask = (candidate["target_mask"][ia, ti, horizon_idx] > 0) & (
        baseline["target_mask"][ib, ti, horizon_idx] > 0
    )
    mask &= np.isfinite(candidate["predictions"][ia, ti, horizon_idx])
    mask &= np.isfinite(baseline["predictions"][ib, ti, horizon_idx])
    ia, ib = ia[mask], ib[mask]
    y = candidate["targets"][ia, ti, horizon_idx] * std + mean
    pc = candidate["predictions"][ia, ti, horizon_idx] * std + mean
    pb = baseline["predictions"][ib, ti, horizon_idx] * std + mean
    return (pc - y) ** 2, (pb - y) ** 2, candidate["anchor_time"][ia]


def _dm(e1: np.ndarray, e2: np.ndarray, order: np.ndarray, h: int) -> tuple[float, float]:
    d = (e1 - e2)[np.argsort(order, kind="stable")]
    n = len(d)
    dc = d - d.mean()
    lag = max(1, -(-h // 24))
    lrv = float((dc @ dc) / n)
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float((dc[k:] @ dc[:-k]) / n)
    if lrv <= 0:
        return np.nan, np.nan
    stat = float(d.mean() / np.sqrt(lrv / n))
    p = float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
    return stat, p


def _bootstrap(e1: np.ndarray, e2: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(e1)
    diffs = np.empty(1000)
    for i in range(1000):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(e1[idx].mean()) - np.sqrt(e2[idx].mean())
    point = float(np.sqrt(e1.mean()) - np.sqrt(e2.mean()))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--candidate", default="variant_B_dual_input_ridge")
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    scalers = json.loads((Path(cfg["paths"]["processed_dir"]) / "scalers.json").read_text())
    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    horizons = cfg["dataset"]["horizons"]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    rows = []
    for baseline in baselines:
        if baseline == args.candidate:
            continue
        for hi, h in enumerate(horizons):
            per_seed = []
            for seed in seeds:
                cand = _load(cfg, args.candidate, seed)
                base = _load(cfg, baseline, seed)
                if cand is None or base is None:
                    continue
                e_c, e_b, order = _aligned_errors(cand, base, cfg, scalers, hi)
                dm_stat, dm_p = _dm(e_c, e_b, order, h)
                diff, lo, hi_ci = _bootstrap(e_c, e_b, seed)
                per_seed.append((seed, len(e_c), diff, lo, hi_ci, dm_stat, dm_p))
            if not per_seed:
                continue
            rows.append({
                "candidate": args.candidate,
                "baseline": baseline,
                "horizon": h,
                "seeds": len(per_seed),
                "n": per_seed[0][1],
                "RMSE_diff_mean_candidate_minus_baseline": float(np.mean([r[2] for r in per_seed])),
                "CI_lo_min": float(np.min([r[3] for r in per_seed])),
                "CI_hi_max": float(np.max([r[4] for r in per_seed])),
                "DM_p_median": float(np.nanmedian([r[6] for r in per_seed])),
                "DM_p_min": float(np.nanmin([r[6] for r in per_seed])),
                "DM_p_max": float(np.nanmax([r[6] for r in per_seed])),
                "directional_win": bool(np.mean([r[2] for r in per_seed]) < 0),
                "significant_all_seeds": bool(all(r[6] < 0.05 for r in per_seed)),
                "per_seed_RMSE_diff": ";".join(f"{r[0]}:{r[2]:+.4f}" for r in per_seed),
                "per_seed_DM_p": ";".join(f"{r[0]}:{r[6]:.6g}" for r in per_seed),
            })
    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else Path(cfg["paths"]["tables_dir"]) / f"overall_significance_{args.candidate}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    failures = df[~(df["directional_win"] & df["significant_all_seeds"])]
    print(f"comparisons: {len(df)}, failures: {len(failures)}")


if __name__ == "__main__":
    main()
