"""Report figures for Chapter 8 (domain alignment and real-data adaptation).

Regenerates the two cross-stage synthesis figures directly into the report tree,
with report-style titles (no development numbering) and the thesis-wide detector
colour identity:

    CA-CFAR                      classical red   -- as in Chapters 4-7
    original synthetic TinyCNN   TinyCNN blue    -- as in Chapters 4-7
    CARRADA-informed TinyCNN     purple          -- the learned arm, retrained
    real-fine-tuned TinyCNN      dark teal       -- the learned arm, adapted

One fixed classical baseline; one learned detector evolving through three
training domains.

    python scripts/make_ch8_figures.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "experiments", "carrada_real_finetune", "results",
                   "validation_analysis.json")
OUT = os.path.join(ROOT, "report", "figures", "Chapter8")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})

INK, GREY = "#1a1a1a", "#8a8a8a"
CFAR = "#c0392b"        # classical baseline, unchanged across the thesis
TCNN_A = "#2c6fbb"      # original synthetic TinyCNN, the Chapter 4-7 blue
TCNN_B = "#7b52ab"      # CARRADA-informed synthetic TinyCNN
TCNN_C = "#0e5b5b"      # real-fine-tuned TinyCNN

COL = {"A": TCNN_A, "B": TCNN_B, "C": TCNN_C, "D": CFAR}
FLAT = {
    "A": "original synthetic TinyCNN",
    "B": "CARRADA-informed TinyCNN",
    "C": "real-fine-tuned TinyCNN",
    "D": "CA-CFAR",
}
ORDER = ["A", "B", "C", "D"]


def load():
    with open(SRC, encoding="utf-8") as fh:
        return json.load(fh)


def fig_progression(r):
    """Left: recall rises then plateaus.  Right: burden explodes then collapses."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    stages = ["A", "B", "C"]
    names = ["original\nsynthetic", "CARRADA-informed\nsynthetic",
             "+ real\nfine-tuning"]
    rec = [r["pooled_exact_recall"][s] for s in stages]
    bur = [r["unmatched_per_frame"][s]["mean"] for s in stages]
    cols = [COL[s] for s in stages]

    a1.plot(range(3), rec, "-", color=GREY, lw=1.6, zorder=1)
    a1.scatter(range(3), rec, s=110, c=cols, zorder=3, edgecolor="white", lw=1.2)
    a1.axhline(r["pooled_exact_recall"]["D"], color=CFAR, lw=1.5, ls="--")
    a1.text(2.34, r["pooled_exact_recall"]["D"] - 0.013, "CA-CFAR",
            fontsize=8.5, color=CFAR, ha="right")
    for i, v in enumerate(rec):
        a1.text(i, v + 0.007, "%.4f" % v, ha="center", fontsize=8.5, color=INK)
    a1.set_xticks(range(3))
    a1.set_xticklabels(names, fontsize=8.5)
    a1.set_xlim(-0.45, 2.45)
    a1.set_ylabel("exact dense-mask object recall")
    a1.set_ylim(0.85, 0.99)
    a1.grid(alpha=0.22)
    a1.spines[["top", "right"]].set_visible(False)
    a1.set_title("Recall rises with domain alignment, then plateaus")

    a2.plot(range(3), bur, "-", color=GREY, lw=1.6, zorder=1)
    a2.scatter(range(3), bur, s=110, c=cols, zorder=3, edgecolor="white", lw=1.2)
    a2.axhline(r["unmatched_per_frame"]["D"]["mean"], color=CFAR, lw=1.5, ls="--")
    a2.text(1.35, r["unmatched_per_frame"]["D"]["mean"] * 1.22, "CA-CFAR",
            fontsize=8.5, color=CFAR, ha="left")
    for i, v in enumerate(bur):
        a2.annotate("%.1f" % v, (i, v), textcoords="offset points",
                    xytext=(0, -16 if i == 1 else 12), ha="center",
                    fontsize=8.5, color=INK)
    a2.set_yscale("log")
    a2.set_ylim(7.0, 260.0)
    a2.set_xticks(range(3))
    a2.set_xticklabels(names, fontsize=8.5)
    a2.set_xlim(-0.45, 2.45)
    a2.set_ylabel("unmatched detections / frame")
    a2.grid(alpha=0.22, which="both")
    a2.spines[["top", "right"]].set_visible(False)
    a2.set_title("Output burden rises, then falls below the baseline")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch8_fig1_progression.png"))
    plt.close(fig)


def fig_burden_curve(r):
    """Recall against output burden, all four arms, evaluation-only sweep."""
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for a in ORDER:
        c = sorted(r["burden_curve"][a], key=lambda x: x["burden"])
        ls = "--" if a == "D" else "-"
        ax.plot([x["burden"] for x in c], [x["recall"] for x in c], ls,
                color=COL[a], lw=1.9, label=FLAT[a])
        ax.scatter([r["unmatched_per_frame"][a]["mean"]],
                   [r["pooled_exact_recall"][a]], s=70, color=COL[a],
                   edgecolor="white", lw=1.3, zorder=5)
    bD = r["unmatched_per_frame"]["D"]["mean"]
    ax.axvline(bD, color=GREY, lw=1.0, ls=":")
    ax.text(bD * 0.80, 0.475, "CA-CFAR's frozen\nburden %.2f / frame" % bD,
            fontsize=7.6, color=GREY, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(0.06, 700.0)
    ax.set_xlabel("unmatched detections per frame  (output burden --- not $P_{fa}$)")
    ax.set_ylabel("exact dense-mask object recall")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.22, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Recall against output burden on the CARRADA validation split\n"
                 "filled markers: the frozen deployed operating points",
                 fontsize=9.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch8_fig2_burden_curve.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    r = load()
    fig_progression(r)
    fig_burden_curve(r)
    print("wrote 2 figures to", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
