"""Backfill peak inference memory into the efficiency-table stats files.

The efficiency table (``src.evaluate.efficiency_table``) reports a
``Peak memory (MB)`` column read from ``peak_memory_mb`` in each
``outputs/checkpoints/<name>_stats.json``. That field was added to
``src.train`` after the headline models were trained, so the existing stats
carry no value. This script backfills it: for every learned efficiency model
with a stats file and a resolvable seed-42 checkpoint, it loads the checkpoint,
runs one CPU inference pass over the test set, and records the process peak
working-set size.

Peak working set is *process-cumulative*, so each model is measured in a fresh
subprocess (``--model NAME`` does exactly one model). The no-argument driver
dispatches one subprocess per model. Statistical baselines (persistence,
seasonal-naive, SARIMA) have no forward pass and are left untouched.

Usage::

    python scripts/measure_efficiency_memory.py --config config.yaml
    python scripts/measure_efficiency_memory.py --config config.yaml --model proposed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, peak_memory_mb, seed_everything

# Learned models that appear in the efficiency table (mask out hybrids/statistical).
EFFICIENCY_MODELS = [
    "lstm", "gru", "gru_d", "dlinear", "patchtst",
    "two_stage_knn", "two_stage_mice", "two_stage_saits",
    "proposed", "variant_B", "proposed_md",
]


def resolve_checkpoint(name: str, ckpt_dir: Path, seed: int) -> Path | None:
    """Local checkpoint path for an efficiency model (stored paths are stale)."""
    candidates: list[Path] = []
    if name == "variant_B":
        candidates.append(ckpt_dir / f"abl_variant_B_s{seed}_seed{seed}.pt")
    if name == "proposed_md":
        candidates.append(ckpt_dir / f"proposed_md_seed{seed}.pt")
        candidates.append(ckpt_dir / f"abl_miss_dropout_s{seed}_seed{seed}.pt")
    candidates.append(ckpt_dir / f"{name}_seed{seed}.pt")
    return next((c for c in candidates if c.exists()), None)


def measure_one(cfg: dict, name: str) -> float | None:
    """Build ``name``, load its checkpoint, run a CPU test pass, return peak MB."""
    import torch

    from src.data.dataset import feature_columns, make_datasets
    from src.models.factory import build_model
    from src.train import predict

    seed = cfg["seed"]
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    stats_path = ckpt_dir / f"{name}_stats.json"
    if not stats_path.exists():
        print(f"{name}: no stats file, skipping")
        return None
    ckpt = resolve_checkpoint(name, ckpt_dir, seed)
    if ckpt is None:
        print(f"{name}: no local checkpoint, skipping")
        return None

    datasets, stations, _ = make_datasets(cfg)
    feats = feature_columns(cfg)
    n_feat = len(feats)
    n_t = len(cfg["dataset"]["target_pollutants"])
    n_h = len(cfg["dataset"]["horizons"])
    t_feat_idx = feats.index(cfg["dataset"]["primary_target"])
    t_indices = [feats.index(p) for p in cfg["dataset"]["target_pollutants"]]

    model = build_model(name, cfg, n_feat, len(stations), n_t, n_h,
                        target_feature_idx=t_feat_idx, target_indices=t_indices)
    model.load_state_dict(torch.load(ckpt, weights_only=False)["model_state"])
    predict(model, datasets["test"], cfg)  # forward pass over the test set
    peak = peak_memory_mb()

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["peak_memory_mb"] = round(float(peak), 1) if peak is not None else None
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"{name}: peak_memory_mb = {stats['peak_memory_mb']} (ckpt {ckpt.name})")
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None,
                        help="measure a single model in this process; the "
                             "no-arg form dispatches one subprocess per model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"], cfg.get("num_threads"))

    if args.model:
        measure_one(cfg, args.model)
        return

    # Driver: one fresh subprocess per model so peak working set is per-model.
    base = [sys.executable, str(Path(__file__).resolve())]
    if args.config:
        base += ["--config", args.config]
    for name in EFFICIENCY_MODELS:
        if not (Path(cfg["paths"]["checkpoints_dir"]) / f"{name}_stats.json").exists():
            continue
        subprocess.run(base + ["--model", name], cwd=Path(__file__).resolve().parents[1])
    print("done: peak_memory_mb backfilled where checkpoints were available")


if __name__ == "__main__":
    main()
