"""Conceptual diagrams for the defence deck (Figures 1-5, plus backups I and J).

These carry no experimental result. Every constant that appears in them (window
geometry, parameter count, dilations, NMS size, split sizes) is a frozen
configuration value, listed in presentation_assets/README.md.

    venv/Scripts/python.exe presentation_assets/build_diagrams.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BASE, CA, DIA, DL, FAINT, FULL, GREY, INK, NEU, SHORT,   # noqa: E402
                    WIDE, blank, save)

GUARD_C = "#f5c542"
REF_C = "#9ec9e8"
CUT_C = "#c0392b"


def _chain(ax, boxes, y=0.5, h=0.34, gap=0.030, fc="#ffffff", ec=INK,
           fs=BASE - 2, lw=2.2):
    """Evenly spaced rounded boxes joined by arrows, laid out across [0, 1]."""
    n = len(boxes)
    w = (1.0 - gap * (n - 1)) / n
    for i, (title, sub) in enumerate(boxes):
        x = i * (w + gap)
        face = fc[i] if isinstance(fc, list) else fc
        edge = ec[i] if isinstance(ec, list) else ec
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.02",
                                    linewidth=lw, edgecolor=edge, facecolor=face,
                                    zorder=2))
        ax.text(x + w / 2, y + (0.045 if sub else 0.0), title, ha="center",
                va="center", fontsize=fs, fontweight="bold", color=edge, zorder=3)
        if sub:
            ax.text(x + w / 2, y - 0.055, sub, ha="center", va="center",
                    fontsize=fs - 3, color=GREY, zorder=3)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.004, y), (x + w + gap - 0.004, y),
                                         arrowstyle="-|>", mutation_scale=22,
                                         linewidth=2.2, color=INK, zorder=1))
    return w


# ------------------------------------------------------------------ Figure 1
def d01_fmcw_pipeline():
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [("Transmit\nchirps", ""), ("Receive\nechoes", ""),
             ("Range\nFFT", "→ range"), ("Doppler\nFFT", "→ velocity"),
             ("RD map", "range × Doppler"), ("Detector", ""),
             ("Target\npeaks", "")]
    edges = [INK, INK, INK, INK, INK, CA, INK]
    faces = ["#ffffff", "#ffffff", "#ffffff", "#ffffff", "#eef3f8", "#fdecea", "#ffffff"]
    ax.set_xlim(-0.012, 1.012)
    _chain(ax, boxes, y=0.52, h=0.40, gap=0.022, fc=faces, ec=edges, fs=BASE - 4)

    ax.annotate("this thesis", xy=(0.795, 0.30), xytext=(0.795, 0.09),
                ha="center", fontsize=BASE, color=CA, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CA, lw=2.6))
    ax.text(0.5, 0.95, "Where the detector sits in an FMCW radar chain",
            ha="center", fontsize=BASE + 4, fontweight="bold", color=INK)
    save(fig, "d01_fmcw_pipeline", DIA)


# ------------------------------------------------------------------ Figure 2
def d02_cfar_window():
    """21 range x 9 Doppler: 1 CUT, 24 guard, 164 reference."""
    nr, nd = 21, 9
    gr, gd = 2, 2              # guard cells per side
    cr, cd = nr // 2, nd // 2

    kind = np.zeros((nr, nd), dtype=int)          # 0 reference, 1 guard, 2 CUT
    for i in range(nr):
        for j in range(nd):
            if abs(i - cr) <= gr and abs(j - cd) <= gd:
                kind[i, j] = 1
    kind[cr, cd] = 2

    n_cut = int((kind == 2).sum())
    n_guard = int((kind == 1).sum())
    n_ref = int((kind == 0).sum())
    assert (n_cut, n_guard, n_ref) == (1, 24, 164), (n_cut, n_guard, n_ref)

    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    blank(ax)
    colours = {0: REF_C, 1: GUARD_C, 2: CUT_C}
    for i in range(nr):
        for j in range(nd):
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=colours[kind[i, j]],
                                   edgecolor="white", linewidth=1.8))
    ax.set_xlim(-2.6, nr + 0.4)
    ax.set_ylim(-2.4, nd + 2.8)
    ax.set_aspect("equal")

    ax.annotate("", xy=(nr, -1.15), xytext=(0, -1.15),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=2.2))
    ax.text(nr / 2, -1.95, "Range  —  21 cells", ha="center", fontsize=BASE + 1,
            color=INK, fontweight="bold")
    ax.annotate("", xy=(-1.15, nd), xytext=(-1.15, 0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=2.2))
    ax.text(-1.95, nd / 2, "Doppler  —  9 cells", ha="center", va="center",
            rotation=90, fontsize=BASE + 1, color=INK, fontweight="bold")

    ax.annotate("CUT", xy=(cr + 0.5, cd + 0.5), xytext=(cr + 0.5, nd + 1.9),
                ha="center", fontsize=BASE, fontweight="bold", color=CUT_C,
                arrowprops=dict(arrowstyle="-|>", color=CUT_C, lw=2.4))
    ax.annotate("Guard  ±2 cells", xy=(cr + 2.4, cd + 2.4),
                xytext=(cr + 5.6, nd + 1.9), ha="center", fontsize=BASE,
                fontweight="bold", color="#a8801a",
                arrowprops=dict(arrowstyle="-|>", color="#a8801a", lw=2.4))
    ax.annotate("Reference", xy=(1.5, cd + 0.5), xytext=(3.2, nd + 1.9),
                ha="center", fontsize=BASE, fontweight="bold", color="#2a6f9e",
                arrowprops=dict(arrowstyle="-|>", color="#2a6f9e", lw=2.4))

    ax.text(nr / 2, -3.0,
            "189 cells   =   1 CUT   +   24 guard   +   164 reference",
            ha="center", fontsize=BASE + 2, fontweight="bold", color=INK)
    ax.text(nr / 2, nd + 3.1, "The CFAR footprint",
            ha="center", fontsize=BASE + 4, fontweight="bold", color=INK)
    save(fig, "d02_cfar_window", DIA)


# ------------------------------------------------------------------ Figure 3
def d03_cfar_failure_modes():
    rng = np.random.default_rng(7)
    x = np.arange(80)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))

    # --- masking -----------------------------------------------------------
    a = axes[0]
    bg = 1.0 + 0.10 * rng.standard_normal(80)
    sig = bg.copy()
    sig[30] = 3.2            # weak target
    sig[38] = 9.0            # strong interferer
    a.plot(x, sig, color=GREY, lw=2.4)
    a.plot([30], [sig[30]], "o", color=DL, ms=16, zorder=5)
    a.plot([38], [sig[38]], "o", color=INK, ms=16, zorder=5)
    a.plot(x, np.where((x > 22) & (x < 46), 4.6, 2.4), color=CA, lw=3.6, ls="--")
    a.annotate("threshold rises", xy=(34, 4.6), xytext=(46, 7.4), color=CA,
               fontsize=BASE - 1, fontweight="bold", ha="left",
               arrowprops=dict(arrowstyle="-|>", color=CA, lw=2.4))
    a.text(38, 10.1, "strong\nneighbour", ha="center", fontsize=BASE - 2, color=INK)
    a.annotate("weak target\nmissed", xy=(30, 3.2), xytext=(9, 6.2), color=DL,
               fontsize=BASE - 1, fontweight="bold", ha="center",
               arrowprops=dict(arrowstyle="-|>", color=DL, lw=2.4))
    a.set_title("Target masking", fontsize=BASE + 3, fontweight="bold", color=INK,
                pad=16)
    a.set_ylim(0, 13)

    # --- clutter edge ------------------------------------------------------
    b = axes[1]
    lvl = np.where(x < 40, 1.0, 5.0) + 0.12 * rng.standard_normal(80)
    sig2 = lvl.copy()
    sig2[28] = 4.2
    b.fill_between(x, 0, 13, where=(x >= 40), color=FAINT, alpha=0.8, zorder=0)
    b.plot(x, sig2, color=GREY, lw=2.4, zorder=2)
    b.plot([28], [sig2[28]], "o", color=DL, ms=16, zorder=5)
    b.plot(x, np.where(x < 32, 2.4, np.where(x < 48, 7.0, 9.4)), color=CA,
           lw=3.6, ls="--", zorder=3)
    b.annotate("window spans\ntwo backgrounds", xy=(36, 7.0), xytext=(52, 11.0),
               color=CA, fontsize=BASE - 1, fontweight="bold", ha="center",
               arrowprops=dict(arrowstyle="-|>", color=CA, lw=2.4))
    b.annotate("target\nmissed", xy=(28, 4.2), xytext=(8, 7.6), color=DL,
               fontsize=BASE - 1, fontweight="bold", ha="center",
               arrowprops=dict(arrowstyle="-|>", color=DL, lw=2.4))
    b.set_title("Clutter edge", fontsize=BASE + 3, fontweight="bold", color=INK,
                pad=16)
    b.set_ylim(0, 13)

    for a in axes:
        blank(a)
        a.set_xlim(0, 79)
        a.text(0.5, -0.05, "one cut through the range\u2013Doppler map",
               transform=a.transAxes, ha="center", fontsize=BASE - 4, color=GREY)

    fig.suptitle("Two ways a fixed local rule goes wrong", fontsize=BASE + 4,
                 fontweight="bold", color=INK, y=1.08)
    fig.tight_layout()
    save(fig, "d03_cfar_failure_modes", DIA)


# ------------------------------------------------------------------ Figure 4
def d04_cfar_vs_tinycnn():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
    nr, nd, gr, gd = 21, 9, 2, 2
    cr, cd = nr // 2, nd // 2

    # --- CFAR --------------------------------------------------------------
    a = axes[0]
    blank(a)
    for i in range(nr):
        for j in range(nd):
            g = abs(i - cr) <= gr and abs(j - cd) <= gd
            c = CUT_C if (i == cr and j == cd) else (GUARD_C if g else REF_C)
            a.add_patch(Rectangle((i, j), 1, 1, facecolor=c, edgecolor="white", lw=1.4))
    a.set_xlim(-0.5, nr + 0.5)
    a.set_ylim(-7.6, nd + 3.6)
    a.set_aspect("equal")
    a.text(nr / 2, nd + 2.4, "CFAR", ha="center", fontsize=BASE + 6,
           fontweight="bold", color=CA)
    a.text(nr / 2, nd + 0.8, "21 × 9 footprint", ha="center",
           fontsize=BASE, color=INK)
    for k, line in enumerate([
            "fixed mathematical rule",
            "CUT compared with an estimated",
            "local background",
            "",
            "guard cells are EXCLUDED",
            "from that estimate"]):
        a.text(nr / 2, -1.4 - 1.05 * k, line, ha="center", fontsize=BASE - 2,
               color=INK if k < 3 else CA,
               fontweight="bold" if k >= 4 else "normal")

    # --- TinyCNN -----------------------------------------------------------
    b = axes[1]
    blank(b)
    for i in range(nr):
        for j in range(nd):
            b.add_patch(Rectangle((i, j), 1, 1, facecolor="#dce8f4",
                                  edgecolor="white", lw=1.4))
    # The receptive field composed layer by layer: range half-widths 1,3,6,10
    # (dilations 1,2,3,4) and Doppler half-widths 1,2,3,4 (dilation 1 throughout).
    for hr, hd, col in zip((1, 3, 6, 10), (1, 2, 3, 4),
                           ("#9dc3e6", "#6ba3d6", "#3d80c2", "#2c6fbb")):
        b.add_patch(Rectangle((cr - hr, cd - hd), 2 * hr + 1, 2 * hd + 1,
                              facecolor="none", edgecolor=col, lw=3.0))
    b.add_patch(Rectangle((cr, cd), 1, 1, facecolor=DL, edgecolor="white", lw=1.4))
    b.set_xlim(-0.5, nr + 0.5)
    b.set_ylim(-7.6, nd + 3.6)
    b.set_aspect("equal")
    b.text(nr / 2, nd + 2.4, "TinyCNN", ha="center", fontsize=BASE + 6,
           fontweight="bold", color=DL)
    b.text(nr / 2, nd + 0.8, "21 × 9 receptive field", ha="center",
           fontsize=BASE, color=INK)
    for k, line in enumerate([
            "learned local decision rule",
            "4 conv layers, 5,041 parameters",
            "range dilation 1, 2, 3, 4  |  Doppler 1",
            "",
            "MAY use the whole receptive field,",
            "guard cells included"]):
        b.text(nr / 2, -1.4 - 1.05 * k, line, ha="center", fontsize=BASE - 2,
               color=INK if k < 3 else DL,
               fontweight="bold" if k >= 4 else "normal")

    fig.suptitle("Comparable spatial extent — different decision rule",
                 fontsize=BASE + 4, fontweight="bold", color=INK, y=1.00)
    save(fig, "d04_cfar_vs_tinycnn", DIA)


# ------------------------------------------------------------------ Figure 5
def d05_evaluation_pipeline():
    fig, ax = plt.subplots(figsize=WIDE)
    blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [("RD frame", ""), ("Detector\nscore map", ""),
             ("Local peak\nextraction", "NMS, 5 × 5"),
             ("Score\nthreshold", ""), ("One-to-one\nmatching", "Hungarian"),
             ("Metrics", "")]
    faces = ["#ffffff", "#eef3f8", "#ffffff", "#ffffff", "#ffffff", "#ffffff"]
    edges = [GREY, INK, INK, INK, INK, INK]
    _chain(ax, boxes, y=0.56, h=0.38, gap=0.028, fc=faces, ec=edges, fs=BASE - 3)

    ax.add_patch(FancyBboxPatch((0.178, 0.10), 0.822, 0.20,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=0, facecolor="#f2f6fa", zorder=0))
    ax.text(0.59, 0.20,
            "Same post-processing and matcher for both detector families",
            ha="center", va="center", fontsize=BASE, color=INK, fontweight="bold")
    ax.text(0.59, 0.135, "final NMS: radius 2  →  5 × 5 neighbourhood",
            ha="center", va="center", fontsize=BASE - 3, color=GREY)
    ax.text(0.5, 0.96, "One shared evaluation path", ha="center",
            fontsize=BASE + 4, fontweight="bold", color=INK)
    save(fig, "d05_evaluation_pipeline", DIA)


# --------------------------------------------------------------- backup I
def b09_dense_vs_box():
    """Why the real-data metric had to change: point truth vs extended support."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    for a in axes:
        blank(a)
        a.set_xlim(0, 22)
        a.set_ylim(0, 16)
        a.set_aspect("equal")
        for i in range(0, 23, 1):
            a.plot([i, i], [0, 16], color="#eeeeee", lw=1.0, zorder=0)
        for j in range(0, 17, 1):
            a.plot([0, 22], [j, j], color="#eeeeee", lw=1.0, zorder=0)

    a = axes[0]
    a.add_patch(Rectangle((10, 7), 1, 1, facecolor=DL, edgecolor="white", lw=1.5))
    a.add_patch(Rectangle((9, 6), 3, 3, facecolor="none", edgecolor=CA,
                          lw=3.0, ls="--"))
    a.text(11, 12.4, "synthetic:\none true cell, ±1-bin gate", ha="center",
           fontsize=BASE - 1, color=INK, fontweight="bold")

    b = axes[1]
    rng = np.random.default_rng(3)
    cx, cy = 10, 7
    for i in range(22):
        for j in range(16):
            d = ((i - cx) / 9.0) ** 2 + ((j - cy) / 6.5) ** 2
            if d < 1.0 and rng.random() > 0.12 * d:
                b.add_patch(Rectangle((i, j), 1, 1, facecolor="#2c6fbb",
                                      alpha=0.75, edgecolor="white", lw=1.0))
    b.add_patch(Rectangle((6, 3), 9, 8, facecolor="none", edgecolor=CA,
                          lw=3.0, ls="--"))
    b.text(11, 12.4, "CARRADA:\nextended annotated support", ha="center",
           fontsize=BASE - 1, color=INK, fontweight="bold")
    fig.suptitle("The truth definition cannot transfer", fontsize=BASE + 4,
                 fontweight="bold", color=INK, y=1.02)
    fig.text(0.5, 0.01, "CARRADA median support — pedestrian 9 × 9,  "
                        "cyclist 11 × 11,  car 18 × 13 bins",
             ha="center", fontsize=BASE - 3, color=GREY)
    save(fig, "b09_dense_vs_box_truth", DIA)


# --------------------------------------------------------------- backup J
def b10_dataset_splits():
    """Which split does what. Names, roles and frame counts never share a line."""
    fig, ax = plt.subplots(figsize=FULL)
    blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    synth = [("bench2_train", "network training", "8,400", False),
             ("bench2_val", "threshold and epoch selection", "1,600", False),
             ("bench2_tune", "CFAR window, NMS radius", "400", False),
             ("bench2_snr_sweep", "detection probability vs cell SNR", "1,300", False),
             ("holdout", "the reported numbers \u2014 read once", "3,200", True)]
    real = [("CARRADA train", "domain calibration, fine-tuning", "8,088  (18 seq)",
             False),
            ("CARRADA validation", "Stage 5 final evaluation", "2,448  (6 seq)", True),
            ("CARRADA test", "zero-shot read, then Stage 4", "2,130  (6 seq)", True)]

    def column(items, x0, x1, colour, heading):
        ax.text(x0, 0.945, heading, fontsize=BASE + 2, fontweight="bold", color=colour)
        for k, (name, role, frames, hi) in enumerate(items):
            y = 0.855 - 0.105 * k
            ax.add_patch(FancyBboxPatch(
                (x0, y - 0.042), x1 - x0, 0.086,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                linewidth=2.4 if hi else 1.5,
                edgecolor=colour if hi else "#c4c4c4",
                facecolor="#eef3f8" if hi else "#ffffff"))
            ax.text(x0 + 0.018, y + 0.019, name, va="center", fontsize=BASE - 3,
                    family="monospace", fontweight="bold" if hi else "normal",
                    color=INK)
            ax.text(x0 + 0.018, y - 0.021, role, va="center", fontsize=BASE - 5,
                    color=GREY)
            ax.text(x1 - 0.018, y + 0.019, frames, va="center", ha="right",
                    fontsize=BASE - 2, fontweight="bold" if hi else "normal",
                    color=colour if hi else INK)

    column(synth, 0.02, 0.48, DL, "SYNTHETIC")
    column(real, 0.52, 0.98, CA, "REAL  (CARRADA)")

    ax.text(0.5, 0.245, "Selection never shares a split with the arm it scores.",
            ha="center", fontsize=BASE + 2, fontweight="bold", color=INK)
    ax.text(0.5, 0.175,
            "bench2_tune sets the CFAR window and the NMS radius;  "
            "bench2_val sets the network's threshold and epoch;  "
            "holdout is read once, after both are frozen.",
            ha="center", fontsize=BASE - 4, color=GREY)
    ax.text(0.5, 0.115,
            "CARRADA train calibrates the Stage-4 background and fine-tunes in "
            "Stage 5.  The validation split is scored once, at the very end.",
            ha="center", fontsize=BASE - 4, color=GREY)
    ax.text(0.5, 1.01, "Data splits", ha="center", fontsize=BASE + 5,
            fontweight="bold", color=INK)
    save(fig, "b10_dataset_splits", DIA)


def main():
    print("diagrams ->", os.path.relpath(DIA, os.path.dirname(DIA)))
    d01_fmcw_pipeline()
    d02_cfar_window()
    d03_cfar_failure_modes()
    d04_cfar_vs_tinycnn()
    d05_evaluation_pipeline()
    b09_dense_vs_box()
    b10_dataset_splits()


if __name__ == "__main__":
    main()
