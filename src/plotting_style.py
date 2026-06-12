"""Single matplotlib style for every figure in the paper.

Colorblind-safe palette (Okabe-Ito), serif fonts, no chart junk.
Use :func:`apply_style` once at script start and :func:`save_figure` for every
figure so each is written as both 300-dpi PNG and vector PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Okabe-Ito colorblind-safe palette.
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]


def apply_style() -> None:
    """Apply the global paper style to matplotlib."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,  # embed TrueType in PDF (journal requirement)
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, figures_dir: str | Path, name: str) -> None:
    """Save ``fig`` as ``<name>.png`` (300 dpi) and ``<name>.pdf`` then close it."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png")
    fig.savefig(figures_dir / f"{name}.pdf")
    plt.close(fig)
