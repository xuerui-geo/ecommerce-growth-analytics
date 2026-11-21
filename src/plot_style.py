"""Shared chart style: thin marks, recessive grid, fixed categorical order."""

from pathlib import Path

import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parents[1] / "images"
IMAGES.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CONTEXT = "#a3a29c"  # neutral gray for context / comparison marks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = SERIES[:5]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 2, "axes.prop_cycle": plt.cycler(color=SERIES),
        "legend.frameon": False,
    })


def save(fig, name):
    fig.savefig(IMAGES / f"{name}.png")
