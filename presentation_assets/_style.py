"""One visual language for every defence-presentation asset.

Slide-first, not paper-first: large type, thick lines, few gridlines, short labels.
Default canvas is 16:9 at 1920 px wide.

Detector colours are inherited from the thesis so a figure on a slide and the same
figure in the report read as the same detector:

    CA-CFAR                      classical red
    GO / OS / SO -CFAR           gold / green / violet
    TinyCNN (original synthetic) TinyCNN blue
    CARRADA-informed TinyCNN     purple
    real-fine-tuned TinyCNN      dark teal
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
DIA = os.path.join(HERE, "diagrams")
DATA = os.path.join(HERE, "data")

# ----------------------------------------------------------------- palette
INK = "#1a1a1a"
GREY = "#7a7a7a"
FAINT = "#d8d8d8"
PAPER = "#ffffff"

CA = "#c0392b"       # CA-CFAR
GO = "#d19a13"       # GO-CFAR
OS = "#2e8b57"       # OS-CFAR
SO = "#8e44ad"       # SO-CFAR
DL = "#2c6fbb"       # TinyCNN, original synthetic training
DLI = "#7b52ab"      # TinyCNN, CARRADA-informed synthetic training
DLF = "#0e5b5b"      # TinyCNN, real-domain fine-tuned
THEORY = "#1a1a1a"   # clairvoyant reference

POS = "#2c6fbb"      # an established gain
NEU = "#9aa5ad"      # a tie / sanity case
NEG = "#c0392b"      # a loss

COL = {"CA": CA, "GO": GO, "OS": OS, "SO": SO, "DL": DL,
       "A": DL, "B": DLI, "C": DLF, "D": CA}

# ----------------------------------------------------------------- type scale
BASE = 17
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": BASE,
    "axes.titlesize": BASE + 4,
    "axes.labelsize": BASE + 1,
    "xtick.labelsize": BASE - 1,
    "ytick.labelsize": BASE - 1,
    "legend.fontsize": BASE - 1,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "lines.linewidth": 3.0,
    "lines.markersize": 11,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

# 16:9 canvases. The tight bbox trims the canvas, so the export dpi is set high
# enough that even the narrowest asset clears 1920 px.
FULL = (12.8, 7.2)
WIDE = (12.8, 5.6)
SHORT = (12.8, 4.4)
DPI = 215


def clean(ax, grid="y", spines=("top", "right")):
    """Strip chart junk: no box, at most one faint grid direction."""
    ax.spines[list(spines)].set_visible(False)
    for s in ax.spines.values():
        s.set_color(INK)
    ax.tick_params(colors=INK)
    if grid:
        ax.grid(axis=grid, alpha=0.18, lw=1.2)
        ax.set_axisbelow(True)
    return ax


def blank(ax):
    """A pure drawing surface for diagrams."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def save(fig, name, where=FIG, transparent=False, svg=True):
    """Write PNG (>=1920 px wide) and SVG. Returns the paths written."""
    os.makedirs(where, exist_ok=True)
    out = []
    png = os.path.join(where, name + ".png")
    fig.savefig(png, dpi=DPI, transparent=transparent)
    out.append(png)
    if svg:
        svgp = os.path.join(where, name + ".svg")
        fig.savefig(svgp, transparent=transparent)
        out.append(svgp)
    plt.close(fig)
    for p in out:
        print("   wrote", os.path.relpath(p, ROOT))
    return out


def footnote(fig, text, colour=GREY, size=BASE - 4):
    """One short provenance/caveat line under the axes. Never a paragraph."""
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=size, color=colour)
