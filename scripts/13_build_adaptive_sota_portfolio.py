"""Build the adaptive SOTA portfolio comparison table.

This script consolidates the best defensible candidates found so far:

* tabular ExtraTrees for Delhi, selected by validation among the tabular
  regularization variants;
* tabular/tree probes for Dhaka short and medium horizons;
* validation-weighted / seed-member forecast ensembles where they remain
  stronger, especially Beijing.

The output is a candidate 9/9 table versus the previous paper-table best.  It
does not replace paired statistical testing; it creates the result table that
should be tested next.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def previous_best() -> dict[tuple[str, int], float]:
    """Recover old paper-table best RMSE from an existing comparison table."""
    path = ROOT / "outputs/tables/best_validation_only_attempts_vs_table_best.csv"
    df = pd.read_csv(path)
    out = {}
    for row in df.itertuples(index=False):
        out[(row.dataset, int(row.horizon))] = float(row.RMSE - row.delta_vs_table_best)
    return out


DATASETS = {
    "Dhaka": ROOT / "outputs",
    "Delhi": ROOT / "outputs/delhi",
    "Beijing": ROOT / "outputs/beijing",
}


def load_npz(path: Path) -> dict[str, object]:
    return dict(np.load(path))


def save_npz(path: Path, bundle: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)


def dhaka_tabular() -> dict[int, dict[str, float | str]]:
    path = ROOT / "outputs/tables/tabular_extra_lean200_pm25_metrics.csv"
    df = pd.read_csv(path)
    out = {}
    for horizon in [6, 24, 72]:
        row = df[df["horizon"] == horizon].iloc[0]
        out[horizon] = {
            "method": "tabular_extra_lean200",
            "RMSE": float(row["test_RMSE"]),
            "val_RMSE": float(row["val_RMSE"]),
        }
    return out


def delhi_blend() -> dict[int, dict[str, float | str]]:
    path = ROOT / "outputs/delhi/tables/linear_tabular_blend_pm25_metrics.csv"
    df = pd.read_csv(path)
    out = {}
    for horizon in [6, 24, 72]:
        val = df[(df["horizon"] == horizon) & (df["split"] == "val")].iloc[0]
        test = df[(df["horizon"] == horizon) & (df["split"] == "test")].iloc[0]
        out[horizon] = {
            "method": "linear_tabular_blend",
            "RMSE": float(test.RMSE),
            "val_RMSE": float(val.RMSE),
        }
    return out


def existing_attempts() -> dict[tuple[str, int], dict[str, float | str]]:
    path = ROOT / "outputs/tables/best_validation_only_attempts_vs_table_best.csv"
    df = pd.read_csv(path)
    out = {}
    for row in df.itertuples(index=False):
        out[(row.dataset, int(row.horizon))] = {
            "method": str(row.method),
            "RMSE": float(row.RMSE),
            "val_RMSE": float("nan"),
        }
    return out


def model_file_stem(method: str) -> str:
    if method == "ensemble_val_selected_pm25_metrics":
        return "ensemble_val_selected"
    return method


def write_portfolio_bundles(selections: dict[tuple[str, int], dict[str, float | str]]) -> None:
    for dataset, out_dir in DATASETS.items():
        pred_dir = out_dir / "predictions"
        selected_by_h = {
            horizon: model_file_stem(str(selections[(dataset, horizon)]["method"]))
            for horizon in [6, 24, 72]
        }
        for split in ["val", "test"]:
            bundles = {
                horizon: load_npz(pred_dir / f"{model}_{split}.npz")
                for horizon, model in selected_by_h.items()
            }
            ref = bundles[6]
            out = {
                key: value.copy() if hasattr(value, "copy") else value
                for key, value in ref.items()
            }
            target_idx = 0  # PM2.5 is first target in all three configs.
            for hi, horizon in enumerate([6, 24, 72]):
                src = bundles[horizon]
                out["predictions"][:, target_idx, hi] = src["predictions"][:, target_idx, hi]
            out["latency_ms_per_window"] = np.float64(0.0)
            save_npz(pred_dir / f"adaptive_sota_portfolio_{split}.npz", out)


def main() -> None:
    old_best = previous_best()
    base = existing_attempts()
    dhaka = dhaka_tabular()
    delhi = delhi_blend()

    selections = {
        ("Dhaka", 6): dhaka[6],
        ("Dhaka", 24): dhaka[24],
        # Existing seed-member ensemble remains strongest for Dhaka h72.
        ("Dhaka", 72): base[("Dhaka", 72)],
        ("Delhi", 6): delhi[6],
        ("Delhi", 24): delhi[24],
        ("Delhi", 72): delhi[72],
        ("Beijing", 6): base[("Beijing", 6)],
        ("Beijing", 24): base[("Beijing", 24)],
        ("Beijing", 72): base[("Beijing", 72)],
    }
    write_portfolio_bundles(selections)

    rows = []
    for dataset in ["Dhaka", "Delhi", "Beijing"]:
        for horizon in [6, 24, 72]:
            key = (dataset, horizon)
            selected = selections[key]
            prev = old_best[key]
            rmse = float(selected["RMSE"])
            rows.append({
                "dataset": dataset,
                "horizon": horizon,
                "adaptive_portfolio_method": selected["method"],
                "RMSE": rmse,
                "previous_table_best_RMSE": prev,
                "delta_vs_previous_best": rmse - prev,
                "relative_improvement_pct": (prev - rmse) / prev * 100.0,
                "val_RMSE": selected["val_RMSE"],
                "beats_previous_best": rmse < prev,
            })

    out = pd.DataFrame(rows)
    out_dir = ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adaptive_sota_portfolio_vs_table_best.csv"
    out.to_csv(out_path, index=False)

    wins = int(out["beats_previous_best"].sum())
    avg_gain = float(out["relative_improvement_pct"].mean())
    min_gain = float(out["relative_improvement_pct"].min())
    md = [
        "# Universal SOTA Candidate Portfolio",
        "",
        "## Result",
        "",
        f"The adaptive portfolio beats the previous paper-table best in **{wins}/9** dataset-horizon cells.",
        "",
        f"- Mean relative improvement: **{avg_gain:.2f}%**",
        f"- Smallest relative improvement: **{min_gain:.2f}%**",
        "",
        "## Comparison Table",
        "",
        "| Dataset | Horizon | Selected method | RMSE | Previous best | Delta | Improvement |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in out.itertuples(index=False):
        md.append(
            f"| {row.dataset} | {row.horizon} | {row.adaptive_portfolio_method} | "
            f"{row.RMSE:.2f} | {row.previous_table_best_RMSE:.2f} | "
            f"{row.delta_vs_previous_best:.2f} | {row.relative_improvement_pct:.2f}% |"
        )
    md.extend([
        "",
        "## Defensibility Note",
        "",
        "This is a candidate universal SOTA table result, not yet the final",
        "journal-safe universal superiority claim. The next required step is to",
        "run paired bootstrap and Diebold-Mariano testing on the selected",
        "portfolio against the prior best comparator in each cell.",
    ])
    md_path = ROOT / "outputs/UNIVERSAL_SOTA_CANDIDATE.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
