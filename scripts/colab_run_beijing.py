"""One-command Beijing training runner for Google Colab (T4 GPU).

Usage (in a Colab cell, after unzipping/cloning the repo)::

    %cd air-transformer
    !pip install -q pyarrow pyyaml scikit-learn statsmodels seaborn openpyxl
    !python scripts/colab_run_beijing.py

Runs the full Beijing experiment grid (UCI download -> prep -> all learned
models x 3 seeds -> robustness suites) against ``config_beijing.yaml`` and
zips the artifacts for download. Every step resumes from existing files, so
a Colab disconnect just means re-running this script.

Model grid (per the work order — no SARIMA/LSTM/MICE on Beijing):
persistence, seasonal_naive, gru, gru_d, dlinear, patchtst, two_stage_knn,
two_stage_saits (script 03); proposed, proposed_md, variant_B (script 04);
MCAR + outage robustness at 10/30/50% (script 05).

After it finishes, download ``beijing_artifacts.zip``, unzip it into the
local repo root (it contains ``outputs/beijing/`` and
``data/processed/beijing/``), then locally run::

    python scripts/07_make_paper_assets.py --config config_beijing.yaml --skip-interpretability
    python scripts/07_make_paper_assets.py --secondary-config config_beijing.yaml --skip-interpretability

Note: GPU-trained checkpoints are saved CPU-portable, but GPU and CPU runs at
the same seed are not bit-identical; Beijing numbers are GPU-trained (this is
recorded in UPGRADE_LOG.md / RESULTS.md).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = "config_beijing.yaml"

STEPS: list[list[str]] = [
    ["scripts/01b_prepare_beijing.py", "--config", CFG],
    ["scripts/03_train_baselines.py", "--config", CFG, "--models",
     "persistence,seasonal_naive,gru,gru_d,dlinear,patchtst,"
     "two_stage_knn,two_stage_saits"],
    ["scripts/04_train_proposed.py", "--config", CFG, "--seeds", "42,43,44"],
    ["scripts/04_train_proposed.py", "--config", CFG, "--seeds", "42,43,44",
     "--miss-dropout"],
    ["scripts/04_train_proposed.py", "--config", CFG, "--seeds", "42,43,44",
     "--variant", "B", "--name", "variant_B"],
    ["scripts/05_ablations.py", "--config", CFG, "--robustness"],
]


def main() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: no GPU detected — this will take ~20 h on CPU. "
                  "In Colab: Runtime -> Change runtime type -> T4 GPU.")
    except ImportError:
        sys.exit("torch is not installed")

    t_total = time.perf_counter()
    for step in STEPS:
        cmd = [sys.executable, *step]
        print(f"\n=== {' '.join(step)} ===", flush=True)
        t0 = time.perf_counter()
        result = subprocess.run(cmd, cwd=REPO)
        mins = (time.perf_counter() - t0) / 60
        if result.returncode != 0:
            sys.exit(f"step failed after {mins:.1f} min: {' '.join(step)} "
                     f"(rerun this script to resume)")
        print(f"=== done in {mins:.1f} min ===", flush=True)

    print("\nZipping artifacts ...")
    staging = REPO / "_beijing_artifacts"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(REPO / "outputs" / "beijing",
                    staging / "outputs" / "beijing")
    shutil.copytree(REPO / "data" / "processed" / "beijing",
                    staging / "data" / "processed" / "beijing")
    zip_path = shutil.make_archive(str(REPO / "beijing_artifacts"), "zip",
                                   root_dir=staging)
    shutil.rmtree(staging)
    print(f"TOTAL: {(time.perf_counter() - t_total) / 3600:.2f} h")
    print(f"Download: {zip_path}")
    print("Unzip into the local repo root, then run scripts/07 as per the "
          "docstring.")


if __name__ == "__main__":
    main()
