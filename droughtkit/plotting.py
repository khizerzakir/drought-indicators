"""Shared figure style so notebooks, book and slides look like one system."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

INK = "#1d2a2e"
PAPER = "#fbf8f1"
MUTED = "#6b7472"
GRID = "#e4ddcd"
DRY = "#b5542d"
WET = "#1f6f78"
OCHRE = "#d99a2b"

#: Diverging dry -> wet palette (brown/sienna to teal), colour-blind safe.
DRY_WET = LinearSegmentedColormap.from_list(
    "dry_wet",
    ["#5c1f0e", "#9c3f1d", "#d07a3c", "#efc58c", "#f4efe3",
     "#9fd0c4", "#4a9c99", "#1f6f78", "#0f3f4a"],
)

#: Sequential sand -> teal palette for amounts (precipitation, NDVI).
SAND_TEAL = LinearSegmentedColormap.from_list(
    "sand_teal", ["#f4ead2", "#d9c79c", "#9fc3a8", "#4a9c99", "#1f6f78", "#0f3f4a"])

SPI_BOUNDS = [-3.1, -2, -1.5, -1, 1, 1.5, 2, 3.1]
SPI_COLORS = ["#6b2410", "#b5542d", "#e8a65d", "#f1ece0", "#8ccac0", "#2a8a8f", "#134e5e"]
SPI_CMAP = ListedColormap(SPI_COLORS)
SPI_NORM = BoundaryNorm(SPI_BOUNDS, SPI_CMAP.N)


def use_style() -> None:
    """Apply the course matplotlib style."""
    mpl.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.labelsize": 10, "font.size": 10, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=[WET, DRY, OCHRE, "#5b6c8f", "#7a9a4a", INK]),
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def index_bars(series: pd.Series, ax=None, title: str | None = None,
               dry_is_negative: bool = True, ylim: float = 3.2):
    """Bar plot of a standardised index coloured dry (brown) / wet (teal)."""
    ax = ax or plt.gca()
    s = series.dropna()
    vals = s.to_numpy() if dry_is_negative else -s.to_numpy()
    colors = SPI_CMAP(SPI_NORM(np.clip(vals, -3.09, 3.09)))
    ax.bar(s.index, s.to_numpy(), width=31, color=colors, linewidth=0)
    for y in (-1, -2) if dry_is_negative else (1, 2):
        ax.axhline(y, color=DRY, lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-ylim, ylim)
    if title:
        ax.set_title(title)
    return ax


def source_note(fig, text: str) -> None:
    """Small provenance line at the bottom of a figure."""
    fig.text(0.01, -0.01, text, fontsize=8, color=MUTED, ha="left", va="top")
