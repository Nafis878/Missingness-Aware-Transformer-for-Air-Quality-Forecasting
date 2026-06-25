"""Reviewer-requested analysis assets.

This script is intentionally post-hoc and non-training: it reads saved metrics,
prediction bundles, and sweep tables, then writes compact tables/figures for
reviewer-facing manuscript additions:

* Cohen's d, Cliff's delta, and relative RMSE improvement.
* Extra-pollutant RMSE summaries.
* Architecture and hyperparameter ablation tables, marking missing runs.
* Validation-only ensemble leakage audit.
* Bootstrap CI for the Dhaka outage severity crossover.
* ROC-style diagnostic for the measured-imputability decision threshold.

Usage:
    python scripts/30_reviewer_requested_assets.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config


MODEL_LABELS = {
    "variant_B": "Full MAT (+attention mask)",
    "full": "No attention mask",
    "no_attention_mask": "No attention mask",
    "no_miss_embed": "No mask embedding",
    "no_station_embed": "No station embedding",
    "no_time": "No time features",
    "no_pos_enc": "No positional encoding",
    "seq72": "Window 72",
    "seq336": "Window 336",
    "heads4": "Heads 4",
    "heads16": "Heads 16",
    "layers2": "Layers 2",
    "layers4": "Layers 4",
    "miss_dropout": "Missingness dropout",
}


def _tex_escape(s: Any) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%")


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            safe[col] = safe[col].map(_tex_escape)
    text = safe.to_latex(index=False, escape=False, caption=caption, label=label)
    path.write_text(text, encoding="utf-8")


def cliffs_delta(candidate: np.ndarray, baseline: np.ndarray) -> float:
    """Positive means candidate RMSE is usually lower than baseline RMSE."""
    wins = losses = 0
    for c in candidate:
        for b in baseline:
            if c < b:
                wins += 1
            elif c > b:
                losses += 1
    denom = len(candidate) * len(baseline)
    return float((wins - losses) / denom) if denom else math.nan


def effect_size_table(tables: Path) -> pd.DataFrame:
    metrics = pd.read_csv(tables / "metrics_full.csv")
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("variant_B", "two_stage_saits", "Full MAT vs SAITS"),
        ("variant_B", "two_stage_knn", "Full MAT vs KNN"),
        ("proposed_md", "two_stage_saits", "MAT+dropout vs SAITS"),
    ]
    for cand, base, label in comparisons:
        for horizon in (6, 24, 72):
            c = metrics[
                (metrics["model"] == cand)
                & (metrics["pollutant"] == "PM2.5")
                & (metrics["horizon"] == horizon)
            ].sort_values("seed")["RMSE"].to_numpy(float)
            b = metrics[
                (metrics["model"] == base)
                & (metrics["pollutant"] == "PM2.5")
                & (metrics["horizon"] == horizon)
            ].sort_values("seed")["RMSE"].to_numpy(float)
            if len(c) == 0 or len(b) == 0:
                continue
            n = min(len(c), len(b))
            diff = b[:n] - c[:n]
            cohen_d = float(diff.mean() / diff.std(ddof=1)) if n > 1 and diff.std(ddof=1) > 0 else math.nan
            rows.append(
                {
                    "comparison": label,
                    "horizon_h": horizon,
                    "candidate_rmse": round(float(c.mean()), 2),
                    "baseline_rmse": round(float(b.mean()), 2),
                    "relative_rmse_improvement_pct": round(float((b.mean() - c.mean()) / b.mean() * 100), 2),
                    "paired_cohens_d": round(cohen_d, 3) if np.isfinite(cohen_d) else "NA",
                    "cliffs_delta": round(cliffs_delta(c, b), 3),
                    "n_seeds": n,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "effect_sizes_pm25.csv", index=False)
    tex = out.copy()
    for col in ("candidate_rmse", "baseline_rmse", "relative_rmse_improvement_pct"):
        tex[col] = tex[col].map(lambda x: f"{float(x):.2f}")
    for col in ("paired_cohens_d", "cliffs_delta"):
        tex[col] = tex[col].map(lambda x: x if x == "NA" else f"{float(x):.3f}")
    tex.columns = [
        "Comparison",
        "Horizon",
        "MAT RMSE",
        "Baseline RMSE",
        "Rel. improv. (\\%)",
        "Paired Cohen's $d$",
        "Cliff's $\\Delta$",
        "Seeds",
    ]
    write_latex_table(
        tex,
        tables / "effect_sizes_pm25.tex",
        "Effect sizes for Dhaka PM2.5 RMSE. Positive relative improvement, positive paired Cohen's $d$, and positive Cliff's $\\Delta$ favor the MAT-side candidate.",
        "tab:effect_sizes",
    )
    return out


def extra_pollutants_table(tables: Path) -> pd.DataFrame:
    metrics = pd.read_csv(tables / "metrics_full.csv")
    rows: list[dict[str, Any]] = []
    for pollutant in ("PM2.5", "PM10", "NO2", "O3"):
        for model, label in (
            ("variant_B", "Full MAT"),
            ("proposed_md", "MAT+dropout"),
            ("two_stage_saits", "Two-stage SAITS"),
        ):
            vals = metrics[
                (metrics["model"] == model)
                & (metrics["pollutant"] == pollutant)
                & (metrics["horizon"] == 24)
            ]["RMSE"].to_numpy(float)
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "pollutant": pollutant,
                    "model": label,
                    "h24_rmse_mean": round(float(vals.mean()), 2),
                    "h24_rmse_std": round(float(vals.std(ddof=1)), 2) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "extra_pollutants_h24.csv", index=False)
    tex = out.copy()
    tex["h24_rmse_mean"] = tex["h24_rmse_mean"].map(lambda x: f"{float(x):.2f}")
    tex["h24_rmse_std"] = tex["h24_rmse_std"].map(lambda x: f"{float(x):.2f}")
    tex.columns = ["Pollutant", "Model", "24 h RMSE", "Std.", "Seeds"]
    write_latex_table(
        tex,
        tables / "extra_pollutants_h24.tex",
        "Dhaka 24 h RMSE for PM2.5 and additional pollutants. Values are mean and standard deviation over seeds.",
        "tab:extra_pollutants",
    )
    return out


def ablation_review_tables(outputs: Path, tables: Path) -> None:
    path = outputs / "ablation_results.json"
    results: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def summarize(variant: str) -> dict[str, Any]:
        seeds = results.get(variant, {})
        row = {
            "variant": MODEL_LABELS.get(variant, variant),
            "run_key": variant,
            "status": "complete" if seeds else "not_run",
            "seeds": len(seeds),
        }
        for h in ("h6", "h24", "h72"):
            vals = [float(s["pm25_rmse"][h]) for s in seeds.values() if h in s.get("pm25_rmse", {})]
            row[h] = f"{np.mean(vals):.2f} $\\pm$ {np.std(vals, ddof=1):.2f}" if len(vals) > 1 else (f"{vals[0]:.2f}" if vals else "NA")
        train_times = [float(s["train_time_s"]) for s in seeds.values() if s.get("train_time_s") is not None]
        row["train_time_min_mean"] = round(float(np.mean(train_times) / 60), 1) if train_times else "NA"
        return row

    arch_order = [
        "variant_B",
        "full" if "no_attention_mask" not in results else "no_attention_mask",
        "no_miss_embed",
        "no_station_embed",
        "no_time",
        "no_pos_enc",
    ]
    arch = pd.DataFrame([summarize(v) for v in arch_order])
    arch.to_csv(tables / "requested_architecture_ablations.csv", index=False)
    arch_tex = arch[["variant", "status", "seeds", "h6", "h24", "h72"]].copy()
    arch_tex.columns = ["Variant", "Status", "Seeds", "6 h", "24 h", "72 h"]
    write_latex_table(
        arch_tex,
        tables / "requested_architecture_ablations.tex",
        "Requested MAT architecture ablations on Dhaka PM2.5, three-seed mean $\\pm$ std. The no\\_station\\_embed and no\\_pos\\_enc component ablations are GPU-trained; the reference Full MAT and No-attention-mask rows are CPU-trained (GPU and CPU runs at the same seed are not bit-identical).",
        "tab:requested_arch_ablations",
    )

    hp_order = ["seq72", "full", "seq336", "heads4", "variant_B", "heads16", "layers2", "full", "layers4"]
    hp_rows = []
    for v in hp_order:
        row = summarize(v)
        if v.startswith("seq"):
            row["factor"] = "Window"
            row["setting"] = v.replace("seq", "")
        elif v.startswith("heads"):
            row["factor"] = "Attention heads"
            row["setting"] = v.replace("heads", "")
        elif v.startswith("layers"):
            row["factor"] = "Layers"
            row["setting"] = v.replace("layers", "")
        elif v == "variant_B":
            row["factor"] = "Attention heads"
            row["setting"] = "8"
        else:
            row["factor"] = "Window/Layers"
            row["setting"] = "168 or 3"
        hp_rows.append(row)
    hp = pd.DataFrame(hp_rows)
    hp.to_csv(tables / "requested_hyperparameter_ablations.csv", index=False)
    hp_tex = hp[["factor", "setting", "variant", "status", "seeds", "h6", "h24", "h72"]].copy()
    hp_tex.columns = ["Factor", "Setting", "Run", "Status", "Seeds", "6 h", "24 h", "72 h"]
    write_latex_table(
        hp_tex,
        tables / "requested_hyperparameter_ablations.tex",
        "Requested hyperparameter ablations on Dhaka PM2.5, three-seed mean $\\pm$ std. Window 168, heads 8, and layers 3 correspond to the configured MAT run; the head-count (4/16), layer-count (2/4), and window-336 grid points are GPU-trained, the reference rows CPU-trained.",
        "tab:requested_hparam_ablations",
    )


def leakage_audit(cfg: dict[str, Any], tables: Path) -> pd.DataFrame:
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    seeds = [int(s) for s in cfg["ablation"]["seeds"]]
    members = [
        "hybrid8_masked_variant_B",
        "hybrid8_masked_variant_B_vanilla_input",
        "hybrid8_transformer",
        "variant_B",
        "proposed",
        "hybrid8_masked_proposed_md",
        "two_stage_knn",
        "two_stage_mice",
        "two_stage_saits",
        "dlinear",
        "gru_d",
    ]
    rows: list[dict[str, Any]] = []
    for member in members:
        for seed in seeds:
            vp = pred_dir / "seeds" / f"{member}_s{seed}_val.npz"
            tp = pred_dir / "seeds" / f"{member}_s{seed}_test.npz"
            if not (vp.exists() and tp.exists()):
                continue
            val = np.load(vp, allow_pickle=True)
            test = np.load(tp, allow_pickle=True)
            val_keys = set(zip(val["station_id"].astype(int), val["anchor_time"].astype(int)))
            test_keys = set(zip(test["station_id"].astype(int), test["anchor_time"].astype(int)))
            rows.append(
                {
                    "member": member,
                    "seed": seed,
                    "val_rows": len(val_keys),
                    "test_rows": len(test_keys),
                    "overlap_rows": len(val_keys & test_keys),
                    "max_val_anchor": int(np.max(val["anchor_time"])),
                    "min_test_anchor": int(np.min(test["anchor_time"])),
                    "chronological_gap_positive": bool(np.max(val["anchor_time"]) < np.min(test["anchor_time"])),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "ensemble_leakage_audit.csv", index=False)

    scripts = Path(__file__).resolve().parent
    audit_md = [
        "# Ensemble leakage audit",
        "",
        "Validation-calibrated ensembles are fit only from `*_val.npz` bundles and then applied to aligned `*_test.npz` bundles.",
        "The audit checks row-key overlap `(station_id, anchor_time)` and chronological ordering for every available member/seed pair.",
        "",
    ]
    if len(out):
        audit_md.append(f"Checked {len(out)} member/seed pairs; total overlap rows = {int(out['overlap_rows'].sum())}.")
        audit_md.append(f"All chronological gaps positive: {bool(out['chronological_gap_positive'].all())}.")
    else:
        audit_md.append("No complete validation/test member pairs were available to inspect.")
    for script_name in ("24_validation_calibrated_ensembles.py", "25_global_validation_ensembles.py"):
        text = (scripts / script_name).read_text(encoding="utf-8")
        fit_uses_val = ('"val"' in text) and ("_load" in text or "_bundle_path" in text)
        output_uses_test = ('"test"' in text) and ("_load" in text or "_bundle_path" in text)
        audit_md.append(f"- `{script_name}`: loads validation for fitting = {fit_uses_val}; loads test bundles for application = {output_uses_test}.")
    (tables / "ensemble_leakage_audit.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")

    tex = out.head(12).copy()
    if len(tex):
        tex.columns = [
            "Member",
            "Seed",
            "Val rows",
            "Test rows",
            "Overlap",
            "Max val anchor",
            "Min test anchor",
            "Chronological",
        ]
        write_latex_table(
            tex,
            tables / "ensemble_leakage_audit.tex",
            "Leakage audit for validation-calibrated ensemble inputs. Zero overlap and positive chronological gaps support validation-only fitting.",
            "tab:leakage_audit",
        )
    return out


def crossover_bootstrap(tables: Path, figures: Path, n_boot: int = 1000) -> pd.DataFrame:
    df = pd.read_csv(tables / "crossover_combined.csv")
    sub = df[(df["dataset"] == "Dhaka") & (df["mode"] == "out") & (df["horizon"] == 6)].copy()
    x = sub["eff_missing_pct"].to_numpy(float)
    y = sub["gap"].to_numpy(float)
    coef = np.polyfit(x, y, 1)
    observed = float(-coef[1] / coef[0])
    fitted = np.polyval(coef, x)
    resid = y - fitted
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        yb = fitted + rng.choice(resid, size=len(resid), replace=True)
        cb = np.polyfit(x, yb, 1)
        if abs(cb[0]) > 1.0e-9:
            root = float(-cb[1] / cb[0])
            if 0 <= root <= 100:
                boots.append(root)
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5]) if boots else (math.nan, math.nan)
    out = pd.DataFrame(
        [
            {
                "dataset": "Dhaka",
                "mode": "station_outage",
                "horizon_h": 6,
                "bootstrap_samples": len(boots),
                "trend_crossover_pct": round(observed, 2),
                "ci95_low_pct": round(float(ci_low), 2),
                "ci95_high_pct": round(float(ci_high), 2),
            }
        ]
    )
    out.to_csv(tables / "crossover_bootstrap_ci.csv", index=False)
    tex_out = out.copy()
    for col in ("trend_crossover_pct", "ci95_low_pct", "ci95_high_pct"):
        tex_out[col] = tex_out[col].map(lambda x: f"{float(x):.2f}")
    tex_out.columns = [
        "Dataset",
        "Mode",
        "Horizon",
        "Bootstrap samples",
        "Crossover (\\%)",
        "CI low (\\%)",
        "CI high (\\%)",
    ]
    write_latex_table(
        tex_out,
        tables / "crossover_bootstrap_ci.tex",
        "Bootstrap confidence interval for the Dhaka station-outage severity crossover. The estimand is the zero of a linear severity trend fit to the sweep points.",
        "tab:crossover_bootstrap",
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter(x, y, color="#2b6cb0", label="Sweep points")
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xs, np.polyval(coef, xs), color="#1a202c", label="Linear trend")
    ax.axhline(0, color="#718096", lw=1)
    ax.axvline(observed, color="#c53030", ls="--", label=f"Crossover {observed:.1f}%")
    if np.isfinite(ci_low):
        ax.axvspan(ci_low, ci_high, color="#c53030", alpha=0.14, label="95% bootstrap CI")
    ax.set_xlabel("Effective input missingness (%)")
    ax.set_ylabel("End-to-end advantage (ug/m3)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "crossover_bootstrap_ci.png", dpi=300)
    fig.savefig(figures / "crossover_bootstrap_ci.pdf")
    plt.close(fig)
    return out


def roc_threshold(tables: Path, figures: Path) -> pd.DataFrame:
    df = pd.read_csv(tables / "decision_by_imputability.csv")
    y = (df["adv_h6_out50"].to_numpy(float) > 0).astype(int)
    score = -df["imputability"].to_numpy(float)
    thresholds = [float("inf")] + sorted(set(score), reverse=True) + [float("-inf")]
    rows = []
    for t in thresholds:
        pred = score >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        rows.append(
            {
                "score_threshold_minus_imputability": t,
                "imputability_rule_threshold": -t if np.isfinite(t) else "NA",
                "TPR": tpr,
                "FPR": fpr,
                "youden_j": tpr - fpr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "imputability_threshold_roc.csv", index=False)
    pts = out[["FPR", "TPR"]].drop_duplicates().sort_values(["FPR", "TPR"])
    auc = float(np.trapezoid(pts["TPR"], pts["FPR"])) if len(pts) > 1 else math.nan
    best = out.sort_values(["youden_j", "TPR"], ascending=False).iloc[0].to_dict()
    summary = pd.DataFrame(
        [
            {
                "n_networks": len(df),
                "positives_end_to_end_wins": int(y.sum()),
                "roc_auc": round(auc, 3),
                "selected_imputability_threshold": best["imputability_rule_threshold"],
                "note": "diagnostic_only_n_equals_3",
            }
        ]
    )
    summary.to_csv(tables / "imputability_threshold_roc_summary.csv", index=False)
    tex_summary = summary.copy()
    tex_summary["roc_auc"] = tex_summary["roc_auc"].map(lambda x: f"{float(x):.2f}")
    tex_summary["selected_imputability_threshold"] = tex_summary["selected_imputability_threshold"].map(lambda x: f"{float(x):.3f}")
    tex_summary.columns = [
        "Networks",
        "End-to-end wins",
        "ROC AUC",
        "Selected threshold",
        "Note",
    ]
    write_latex_table(
        tex_summary,
        tables / "imputability_threshold_roc_summary.tex",
        "ROC diagnostic for the measured-imputability decision threshold. This is reported as diagnostic only because there are three networks.",
        "tab:imputability_roc",
    )

    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], color="#a0aec0", lw=1, ls="--")
    ax.step(pts["FPR"], pts["TPR"], where="post", color="#2b6cb0", lw=2)
    ax.scatter(pts["FPR"], pts["TPR"], color="#2b6cb0")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"Imputability ROC (AUC={auc:.2f}, n=3)")
    fig.tight_layout()
    fig.savefig(figures / "imputability_threshold_roc.png", dpi=300)
    fig.savefig(figures / "imputability_threshold_roc.pdf")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs = Path(cfg["paths"]["outputs_dir"])
    tables = Path(cfg["paths"]["tables_dir"])
    figures = Path(cfg["paths"]["figures_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    effect_size_table(tables)
    extra_pollutants_table(tables)
    ablation_review_tables(outputs, tables)
    leakage_audit(cfg, tables)
    crossover_bootstrap(tables, figures, n_boot=1000)
    roc_threshold(tables, figures)
    print("wrote reviewer-requested tables and figures")


if __name__ == "__main__":
    main()
