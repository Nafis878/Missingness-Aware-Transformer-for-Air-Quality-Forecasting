"""One-command Delhi training runner for Google Colab (GPU).

Usage (in a Colab cell, after unzipping/cloning the repo)::

    %cd air-transformer
    !pip install -q pyarrow pyyaml scikit-learn statsmodels seaborn openpyxl
    !python scripts/colab_run_delhi.py

Runs the full Delhi experiment grid (prep -> all learned models x 3 seeds ->
robustness suites) against ``config_delhi.yaml`` and zips the artifacts for
download. The 6 CPCB station CSVs are auto-downloaded from their content-
addressed Mendeley URLs (see ``src/data/load_delhi.py: DELHI_FILES``), so this
is genuinely one command. Every step resumes from existing files, so a Colab
disconnect just means re-running this script.

Model grid (matched to the Beijing run for cross-dataset comparability —
no SARIMA/LSTM/MICE): persistence, seasonal_naive, gru, gru_d, dlinear,
patchtst, two_stage_knn, two_stage_saits (script 03); proposed, proposed_md,
variant_B (script 04); MCAR + outage robustness sweep (script 05).

After it finishes, download ``delhi_artifacts.zip``, unzip it into the local
repo root (it contains ``outputs/delhi/`` and ``data/processed/delhi/``), then
locally run the 3-dataset asset regeneration in COLAB.md.

Note: GPU-trained checkpoints are saved CPU-portable, but GPU and CPU runs at
the same seed are not bit-identical; Delhi numbers are GPU-trained (recorded in
UPGRADE_LOG.md / RESULTS.md).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = "config_delhi.yaml"

STEPS: list[list[str]] = [
    ["scripts/01c_prepare_delhi.py", "--config", CFG],
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
            print("WARNING: no GPU detected — this will be very slow on CPU. "
                  "In Colab: Runtime -> Change runtime type -> GPU.")
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
    staging = REPO / "_delhi_artifacts"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(REPO / "outputs" / "delhi",
                    staging / "outputs" / "delhi")
    shutil.copytree(REPO / "data" / "processed" / "delhi",
                    staging / "data" / "processed" / "delhi")
    zip_path = shutil.make_archive(str(REPO / "delhi_artifacts"), "zip",
                                   root_dir=staging)
    shutil.rmtree(staging)
    print(f"TOTAL: {(time.perf_counter() - t_total) / 3600:.2f} h")
    print(f"Download: {zip_path}")
    print("Unzip into the local repo root, then run scripts/07 as per COLAB.md.")


if __name__ == "__main__":
    main()
