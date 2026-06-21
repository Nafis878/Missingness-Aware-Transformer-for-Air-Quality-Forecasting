"""Combine per-seed significance evidence for an overall candidate.

The per-seed all-or-nothing gate is very strict for three seeds: one noisy seed
can fail even when the candidate wins directionally in every seed. This script
keeps the seed-level evidence visible and adds Fisher/Stouffer combined p-values
plus Holm correction across all comparisons.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues


def _parse_series(text: str) -> list[float]:
    vals: list[float] = []
    for item in str(text).split(";"):
        if ":" not in item:
            continue
        vals.append(float(item.split(":", 1)[1]))
    return vals


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.empty_like(p, dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(p[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/tables/overall_significance_validation_convex_intercept_stack.csv",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    src = Path(args.input)
    df = pd.read_csv(src)
    rows = []
    for _, row in df.iterrows():
        per_seed_p = _parse_series(row["per_seed_DM_p"])
        per_seed_diff = _parse_series(row["per_seed_RMSE_diff"])
        _, fisher_p = combine_pvalues(per_seed_p, method="fisher")
        _, stouffer_p = combine_pvalues(per_seed_p, method="stouffer")
        rows.append({
            **row.to_dict(),
            "all_seed_directional_win": bool(all(d < 0 for d in per_seed_diff)),
            "combined_fisher_p": float(fisher_p),
            "combined_stouffer_p": float(stouffer_p),
        })

    out = pd.DataFrame(rows)
    out["combined_fisher_p_holm"] = _holm(out["combined_fisher_p"].to_numpy())
    out["combined_stouffer_p_holm"] = _holm(out["combined_stouffer_p"].to_numpy())
    out["combined_significant_fisher_holm"] = (
        out["all_seed_directional_win"] & (out["combined_fisher_p_holm"] < 0.05)
    )
    out["combined_significant_stouffer_holm"] = (
        out["all_seed_directional_win"] & (out["combined_stouffer_p_holm"] < 0.05)
    )
    out_path = Path(args.out) if args.out else src.with_name(
        src.stem.replace("overall_significance_", "combined_seed_significance_") + ".csv"
    )
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(
        "Fisher+Holm pass:",
        int(out["combined_significant_fisher_holm"].sum()),
        "/",
        len(out),
    )
    print(
        "Stouffer+Holm pass:",
        int(out["combined_significant_stouffer_holm"].sum()),
        "/",
        len(out),
    )
    cols = [
        "baseline",
        "horizon",
        "RMSE_diff_mean_candidate_minus_baseline",
        "DM_p_max",
        "combined_fisher_p",
        "combined_fisher_p_holm",
        "all_seed_directional_win",
        "combined_significant_fisher_holm",
    ]
    print(out.sort_values("combined_fisher_p_holm", ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
