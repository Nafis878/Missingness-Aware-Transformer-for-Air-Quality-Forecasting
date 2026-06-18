"""Run merged-imputation forecasting experiments on all datasets.

This is the one-command runner for the improved ``hybrid_top8`` forecasting
experiment. It trains models that consume ``hybrid_top8`` imputed values while
preserving the original observation mask, so mask-aware models can distinguish
measured inputs from reconstructed inputs.

Colab usage::

    %cd air-transformer
    !pip install -q pyarrow pyyaml scikit-learn statsmodels scipy matplotlib seaborn openpyxl
    !python scripts/run_hybrid8_masked_all.py

Useful quick smoke test::

    !python scripts/run_hybrid8_masked_all.py --datasets dhaka --seeds 42

Defaults:

* datasets: dhaka, beijing, delhi
* models: hybrid8_masked_proposed, hybrid8_masked_variant_B,
  hybrid8_masked_proposed_md
* seeds: 42,43,44

The runner is file-existence resumable because ``scripts/03_train_baselines.py``
skips completed seed bundles unless ``--force`` is passed here.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

DATASETS = {
    "dhaka": {
        "config": "config.yaml",
        "prepare": ["scripts/01_prepare_data.py", "--config", "config.yaml"],
        "processed_check": REPO / "data" / "processed" / "all_stations.parquet",
        "outputs_dir": REPO / "outputs",
        "processed_dir": REPO / "data" / "processed",
    },
    "beijing": {
        "config": "config_beijing.yaml",
        "prepare": [
            "scripts/01b_prepare_beijing.py", "--config", "config_beijing.yaml",
        ],
        "processed_check": (
            REPO / "data" / "processed" / "beijing" / "all_stations.parquet"
        ),
        "outputs_dir": REPO / "outputs" / "beijing",
        "processed_dir": REPO / "data" / "processed" / "beijing",
    },
    "delhi": {
        "config": "config_delhi.yaml",
        "prepare": [
            "scripts/01c_prepare_delhi.py", "--config", "config_delhi.yaml",
        ],
        "processed_check": (
            REPO / "data" / "processed" / "delhi" / "all_stations.parquet"
        ),
        "outputs_dir": REPO / "outputs" / "delhi",
        "processed_dir": REPO / "data" / "processed" / "delhi",
    },
}

DEFAULT_MODELS = (
    "hybrid8_masked_proposed,"
    "hybrid8_masked_variant_B,"
    "hybrid8_masked_proposed_md"
)
DEFAULT_SEEDS = "42,43,44"


def run_step(step: list[str], label: str) -> None:
    """Run one subprocess step, stopping the whole pipeline on failure."""
    cmd = [sys.executable, *step]
    print(f"\n=== {label}: {' '.join(step)} ===", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO)
    mins = (time.perf_counter() - t0) / 60
    if result.returncode != 0:
        raise SystemExit(
            f"{label} failed after {mins:.1f} min: {' '.join(step)}\n"
            "Fix the error, then rerun this script to resume."
        )
    print(f"=== {label}: done in {mins:.1f} min ===", flush=True)


def print_device() -> None:
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print(
            "WARNING: no GPU detected. This can take many hours on CPU. "
            "For Colab, use Runtime -> Change runtime type -> GPU.",
            flush=True,
        )


def prepare_dataset(name: str, info: dict, force_prepare: bool) -> None:
    check = Path(info["processed_check"])
    if check.exists() and not force_prepare:
        print(f"\n=== {name}: processed data exists, skipping prepare ===", flush=True)
        return
    run_step(list(info["prepare"]), f"{name} prepare")


def train_dataset(
    name: str,
    info: dict,
    models: str,
    seeds: str,
    force: bool,
) -> None:
    step = [
        "scripts/03_train_baselines.py",
        "--config", str(info["config"]),
        "--models", models,
        "--seeds", seeds,
    ]
    if force:
        step.append("--force")
    run_step(step, f"{name} train")


def evaluate_dataset(name: str, info: dict) -> None:
    run_step(["-m", "src.evaluate", "--config", str(info["config"])],
             f"{name} evaluate")


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"WARNING: missing artifact path, not packaging: {src}", flush=True)
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def package_artifacts(dataset_names: list[str]) -> Path:
    staging = REPO / "_hybrid8_masked_all_artifacts"
    if staging.exists():
        shutil.rmtree(staging)

    for name in dataset_names:
        info = DATASETS[name]
        if name == "dhaka":
            copy_if_exists(Path(info["processed_dir"]),
                           staging / "data" / "processed")
            copy_if_exists(Path(info["outputs_dir"]), staging / "outputs")
        else:
            copy_if_exists(Path(info["processed_dir"]),
                           staging / "data" / "processed" / name)
            copy_if_exists(Path(info["outputs_dir"]),
                           staging / "outputs" / name)

    zip_path = shutil.make_archive(
        str(REPO / "hybrid8_masked_all_artifacts"), "zip", root_dir=staging
    )
    shutil.rmtree(staging)
    return Path(zip_path)


def parse_list(raw: str, allowed: set[str], what: str) -> list[str]:
    values = [v.strip().lower() for v in raw.split(",") if v.strip()]
    unknown = set(values) - allowed
    if unknown:
        raise SystemExit(f"unknown {what}: {sorted(unknown)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="dhaka,beijing,delhi",
        help="comma-separated subset: dhaka,beijing,delhi",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="comma-separated model list for scripts/03_train_baselines.py",
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--force", action="store_true",
                        help="retrain even if prediction bundles exist")
    parser.add_argument("--force-prepare", action="store_true",
                        help="rebuild processed data even if parquet exists")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    dataset_names = parse_list(args.datasets, set(DATASETS), "datasets")
    print_device()

    t_total = time.perf_counter()
    for name in dataset_names:
        info = DATASETS[name]
        print(f"\n\n##### DATASET: {name.upper()} #####", flush=True)
        if not args.skip_prepare:
            prepare_dataset(name, info, args.force_prepare)
        train_dataset(name, info, args.models, args.seeds, args.force)
        if not args.skip_evaluate:
            evaluate_dataset(name, info)

    if not args.no_zip:
        print("\nZipping artifacts ...", flush=True)
        zip_path = package_artifacts(dataset_names)
        print(f"Download/use: {zip_path}", flush=True)

    print(f"TOTAL: {(time.perf_counter() - t_total) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
