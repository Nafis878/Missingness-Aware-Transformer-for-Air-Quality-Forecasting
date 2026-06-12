"""Shared utilities: config loading, logging, reproducibility.

Every script in this project uses these three entry points:

* :func:`load_config` -- read ``config.yaml`` into a dict and resolve paths.
* :func:`setup_logging` -- file + console logging into ``outputs/logs/``.
* :func:`seed_everything` -- fix all RNGs and enable deterministic torch.
"""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def project_root() -> Path:
    """Return the repository root (the directory containing ``config.yaml``)."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve all entries under ``paths`` to absolute paths.

    Parameters
    ----------
    config_path:
        Path to a YAML config. Defaults to ``<repo root>/config.yaml``.
    """
    path = Path(config_path) if config_path else project_root() / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    root = path.resolve().parent
    cfg["paths"] = {k: str((root / v).resolve()) for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(name: str, logs_dir: str | Path) -> logging.Logger:
    """Configure root logging to a timestamped file plus stderr console.

    Parameters
    ----------
    name:
        Logical name of the running script; used in the log filename.
    logs_dir:
        Directory for log files (created if absent).
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF-save chatter
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logger = logging.getLogger(name)
    logger.info("logging to %s", logfile)
    return logger


def seed_everything(seed: int, num_threads: int | None = None) -> None:
    """Fix Python/NumPy/torch RNGs and enable deterministic torch algorithms.

    torch is imported lazily so that data-only scripts do not pay its import cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if num_threads:
            torch.set_num_threads(num_threads)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def peak_memory_mb() -> float | None:
    """Peak working-set size of this process in MB (Windows), else None.

    Uses Win32 ``GetProcessMemoryInfo`` via ctypes to avoid extra
    dependencies; on non-Windows platforms falls back to ``resource``.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wt

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(pmc), pmc.cb
            ):
                return pmc.PeakWorkingSetSize / (1024 * 1024)
        else:  # pragma: no cover - unix path
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # pragma: no cover - measurement is best-effort
        pass
    return None


def export_table(
    df: pd.DataFrame,
    tables_dir: str | Path,
    name: str,
    caption: str,
    label: str,
    float_format: str = "%.1f",
) -> None:
    """Write a table as CSV and booktabs LaTeX into ``tables_dir``.

    Used by every script that produces a paper table so all outputs share
    one format.
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv")
    latex = df.to_latex(
        float_format=float_format, caption=caption, label=label,
        escape=True, bold_rows=False,
    )
    if "\\toprule" not in latex:  # fall back to Styler for booktabs rules
        latex = df.style.format(precision=1).to_latex(
            caption=caption, label=label, hrules=True
        )
    (tables_dir / f"{name}.tex").write_text(latex, encoding="utf-8")
    logging.getLogger(__name__).info("wrote table %s (.csv + .tex)", name)
