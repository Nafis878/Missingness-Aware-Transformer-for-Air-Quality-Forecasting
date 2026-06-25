"""One-command Dhaka ablation-grid runner for Google Colab (T4 GPU).

Usage (in a Colab cell, after unzipping/cloning the repo *with* the processed
Dhaka data under ``data/processed/``)::

    %cd air-transformer
    !pip install -q pyarrow pyyaml scikit-learn statsmodels seaborn openpyxl
    !python scripts/colab_run_ablations.py

Fills in the requested architecture/hyperparameter ablation grid points whose
checkpoints were never produced locally (the manuscript marks them ``not_run``):

    no_station_embed, no_pos_enc, heads4, heads16, layers2, layers4, seq336

against ``config.yaml`` (Dhaka), seeds 42/43/44. ``05_ablations.py`` resumes per
``(variant, seed)`` from ``outputs/ablation_results.json``, so the already-done
``seq336`` seed 42 is skipped and only seeds 43/44 are trained; a Colab
disconnect just means re-running this script. Robustness is skipped here (the
robustness suite is unchanged and already cached locally).

NB: these grid points are therefore **GPU-trained**, while the reference
``full``/``variant_B`` rows already in the artifact set are CPU-trained. This is
the same CPU/GPU convention already used and disclosed for Beijing/Delhi (see
``UPGRADE_LOG.md`` / ``RESULTS.md``); the ablation table captions note it.

After it finishes, download ``ablation_artifacts.zip``, unzip it into the local
repo root (it overlays ``outputs/ablation_results.json`` and the new
``outputs/checkpoints/abl_*`` files), then locally run::

    python scripts/30_reviewer_requested_assets.py --config config.yaml

to regenerate ``requested_architecture_ablations.tex`` and
``requested_hyperparameter_ablations.tex`` with the real numbers.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = "config.yaml"

# The grid points the manuscript marks not_run (heads 4/16, layers 2/4,
# no-station-embedding, no-positional-encoding) plus the two missing seeds of
# the 336-step window. variant_setup() halves the seq336 batch to bound RAM.
VARIANTS = [
    "no_station_embed",
    "no_pos_enc",
    "heads4",
    "heads16",
    "layers2",
    "layers4",
    "seq336",
]

STEPS: list[list[str]] = [
    ["scripts/05_ablations.py", "--config", CFG, "--skip-robustness",
     "--seeds", "42,43,44", "--variants", ",".join(VARIANTS)],
]


def main() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: no GPU detected — this will take many hours on CPU. "
                  "In Colab: Runtime -> Change runtime type -> T4 GPU.")
    except ImportError:
        sys.exit("torch is not installed")

    # Fail fast with a clear message if the processed Dhaka data is absent
    # (the ablations train on data/processed/, which must be in the repo zip).
    processed = REPO / "data" / "processed" / "all_stations.parquet"
    if not processed.exists():
        sys.exit(
            f"missing {processed.relative_to(REPO)} — the Dhaka ablations train "
            "on the processed parquet. Include data/processed/ in the repo zip "
            "(or run scripts/01_prepare_data.py first)."
        )

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
    staging = REPO / "_ablation_artifacts"
    if staging.exists():
        shutil.rmtree(staging)
    out_dir = staging / "outputs"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # The results JSON carries every variant's per-seed RMSE (what script 30
    # reads); the abl_* checkpoints/stats are included so efficiency/peak-memory
    # and any per-seed bundle export can use them later.
    results_json = REPO / "outputs" / "ablation_results.json"
    if results_json.exists():
        shutil.copy2(results_json, out_dir / "ablation_results.json")
    for variant in VARIANTS:
        for f in (REPO / "outputs" / "checkpoints").glob(f"abl_{variant}_*"):
            shutil.copy2(f, ckpt_dir / f.name)

    zip_path = shutil.make_archive(str(REPO / "ablation_artifacts"), "zip",
                                   root_dir=staging)
    shutil.rmtree(staging)
    print(f"TOTAL: {(time.perf_counter() - t_total) / 3600:.2f} h")
    print(f"Download: {zip_path}")
    print("Unzip into the local repo root, then run "
          "scripts/30_reviewer_requested_assets.py --config config.yaml.")


if __name__ == "__main__":
    main()
