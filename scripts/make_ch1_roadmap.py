"""The five-stage roadmap figure for Chapter 1.

One box per stage: the stage name, the question it answers, and the answer in
one line.  Deliberately carries no confidence intervals and no numbers — its
purpose is orientation, not evidence.

    python scripts/make_ch1_roadmap.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures", "Chapter1")

INK = "#1a1a1a"
SYNTH_F, SYNTH_E = "#e8f0f9", "#2c6fbb"      # controlled synthetic stages
TRANS_F, TRANS_E = "#fdecea", "#c0392b"      # the transfer test
INTER_F, INTER_E = "#f2ecf7", "#7b52ab"      # domain-informed intervention
ADAPT_F, ADAPT_E = "#e6f0f0", "#0e5b5b"      # real adaptation
FINAL_F, FINAL_E = "#f4f4f4", "#1a1a1a"      # the conclusion

STAGES = [
    ("STAGE 1  Controlled synthetic benchmark",
     "Can learning improve the local decision rule?",
     "Yes — gains in several difficult synthetic conditions.",
     SYNTH_F, SYNTH_E),
    ("STAGE 2  Attribution experiments",
     "What explains that behaviour?",
     "Not simple energy integration; supervision and local information matter.",
     SYNTH_F, SYNTH_E),
    ("STAGE 3  Zero-shot CARRADA transfer",
     "Does the synthetic advantage survive on real radar?",
     "No — recall parity with CA-CFAR, at a higher learned output burden.",
     TRANS_F, TRANS_E),
    ("STAGE 4  Target-domain-informed synthetic training",
     "Does reducing the measured domain mismatch recover transfer?",
     "Recall recovers — but output density becomes excessive.",
     INTER_F, INTER_E),
    ("STAGE 5  Supervised real-domain adaptation",
     "Can high recall be retained efficiently?",
     "Yes — TinyCNN exceeds CA-CFAR recall at a comparable output burden.",
     ADAPT_F, ADAPT_E),
]

CONCLUSION = ("CFAR transfers strongly without training; the learned detector reaches a higher\n"
              "observed ceiling when the deployment domain is represented during training.")


def main():
    os.makedirs(OUT, exist_ok=True)

    n = len(STAGES)
    box_h, gap = 1.00, 0.42
    concl_h = 0.80
    total = n * box_h + n * gap + concl_h

    fig, ax = plt.subplots(figsize=(6.6, 7.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total)
    ax.axis("off")

    y = total
    for title, question, answer, fc, ec in STAGES:
        y -= box_h
        ax.add_patch(FancyBboxPatch(
            (0.15, y), 9.7, box_h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(0.45, y + box_h - 0.26, title, fontsize=9.4, fontweight="bold",
                color=ec, va="center", zorder=3)
        ax.text(0.45, y + box_h - 0.55, question, fontsize=8.8, style="italic",
                color=INK, va="center", zorder=3)
        ax.text(0.45, y + box_h - 0.82, "→  " + answer, fontsize=8.8,
                color=INK, va="center", zorder=3)
        y -= gap
        ax.add_patch(FancyArrowPatch(
            (5.0, y + gap), (5.0, y + 0.06), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.2, color="#8a8a8a", zorder=1))

    ax.add_patch(FancyBboxPatch(
        (0.15, y - concl_h + 0.06), 9.7, concl_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=FINAL_E, facecolor=FINAL_F, zorder=2))
    ax.text(5.0, y - concl_h + 0.62, "CONCLUSION", fontsize=9.2,
            fontweight="bold", color=FINAL_E, ha="center", va="center", zorder=3)
    ax.text(5.0, y - concl_h + 0.30, CONCLUSION, fontsize=8.8, color=INK,
            ha="center", va="center", zorder=3, linespacing=1.45)

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "ch1_fig1_roadmap.png"), dpi=220,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(os.path.relpath(OUT, ROOT), "ch1_fig1_roadmap.png"))


if __name__ == "__main__":
    main()
