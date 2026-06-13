"""Delhi dataset preparation: (optional download) -> clean -> hourly parquet.

Usage::

    python scripts/01c_prepare_delhi.py --config config_delhi.yaml

Mirrors scripts/01b_prepare_beijing.py for the Delhi Multi-Site Air-Quality
dataset (CPCB; Mendeley bzhzr9b64v). Downloads the archive only when
``data.archive_url`` is set and the station CSVs are not already in
``data/raw/delhi/`` — otherwise place the CSVs there manually (see the config
comment / COLAB.md). Cleaning reuses :func:`src.data.clean.clean` unchanged.
After this script the whole pipeline runs via ``--config config_delhi.yaml``.

Outputs:

* ``data/processed/delhi/all_stations.parquet``
* ``outputs/delhi/data_cleaning_report.md``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import clean
from src.data.load_delhi import download_delhi, load_all_delhi
from src.utils import load_config, seed_everything, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_delhi.yaml",
                        help="path to config_delhi.yaml")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("01c_prepare_delhi", cfg["paths"]["logs_dir"])
    seed_everything(cfg["seed"], cfg.get("num_threads"))

    download_delhi(cfg["paths"]["raw_dir"],
                   cfg["data"].get("archive_url") or None,
                   force=args.force_download)
    df, load_rep = load_all_delhi(cfg)
    df, clean_rep = clean(df, cfg)

    processed = Path(cfg["paths"]["processed_dir"])
    combined_path = processed / "all_stations.parquet"
    df.to_parquet(combined_path, index=False)
    logger.info("wrote %s (%d rows, %d stations)", combined_path, len(df),
                df["station"].nunique())

    # cleaning report: reuse script 01's renderer (same report objects)
    m01 = __import__("01_prepare_data") if "01_prepare_data" in sys.modules else None
    if m01 is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        m01 = __import__("01_prepare_data")
    report_path = Path(cfg["paths"]["outputs_dir"]) / "data_cleaning_report.md"
    m01.write_cleaning_report(report_path, df, load_rep, clean_rep, cfg)
    logger.info("wrote cleaning report to %s", report_path)

    for col in ("PM2.5", "PM10", "O3", "CO"):
        if col in df.columns:
            logger.info("missingness %-6s: %.2f%%", col, df[col].isna().mean() * 100)


if __name__ == "__main__":
    main()
