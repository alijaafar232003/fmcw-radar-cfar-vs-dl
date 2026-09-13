"""The all-detector zero-shot figure for Chapter 7.

Reads the frozen CARRADA zero-shot summary and plots every detector arm on the two axes the
chapter argues cannot be separated: exact dense-mask object recall against unmatched output
burden. Nothing is recomputed.

    venv/Scripts/python.exe scripts/make_ch7_alldetector_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "experiments", "e6_nms_audit", "results", "p4_carrada", "r1_test",
                   "r1_summary.json")
OUT = os.path.join(ROOT, "report", "figures", "Chapter7")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
INK, GREY = "#1a1a1a", "#8a8a8a"
BLUE, RED, GOLD, GREEN, PURPLE = "#2c6fbb", "#c0392b", "#d19a13", "#2e8b57", "#8e44ad"

ARMS = [
    ("TinyCNN",  "TinyCNN",  "tinycnn_seed_mean", "TinyCNN",  BLUE,   "s"),
    ("CA",       "CA-CFAR",  "CA",                "CFAR-CA",  RED,    "o"),
    ("GO",       "GO-CFAR",  "GO",                "CFAR-GO",  GOLD,   "o"),
    ("SO",       "SO-CFAR",  "SO",                "CFAR-SO",  PURPLE, "o"),
    ("OS",       "OS-CFAR",  "OS",                "CFAR-OS",  GREEN,  "o"),
]

# Label offsets in points, tuned so no annotation overlaps a marker or another label.
OFFSET = {"TinyCNN": (-4, 13), "CA-CFAR": (2, 13), "GO-CFAR": (-2, -25),
          "SO-CFAR": (-16, 11), "OS-CFAR": (14, -25)}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SRC, encoding="utf-8") as fh:
        r = json.load(fh)

    pts = []
    for _, label, rkey, bkey, colour, marker in ARMS:
        recall = (r["primary"]["tinycnn_seed_mean"] if rkey == "tinycnn_seed_mean"
                  else r["primary"]["cfar"][rkey])
        burden = r["unmatched_burden"][bkey]["mean"]
        pts.append((label, recall, burden, colour, marker))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, recall, burden, colour, marker in pts:
        ax.scatter(burden, recall, s=115, color=colour, marker=marker,
                   edgecolor="white", lw=1.2, zorder=5)
        dx, dy = OFFSET[label]
        ax.annotate("%s\n%.4f / %.2f" % (label, recall, burden),
                    (burden, recall), textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=7.8, color=colour)

    ax.set_xlabel("unmatched detections per frame  (output burden — not $P_{fa}$)")
    ax.set_ylabel("exact dense-mask object recall")
    ax.set_xlim(8, 37)
    ax.set_ylim(0.845, 0.995)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("", xy=(34.5, 0.862), xytext=(10.5, 0.862),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
    ax.text(22.5, 0.851, "more output emitted $\\rightarrow$", ha="center",
            fontsize=7.4, color=GREY)
    ax.set_title("Zero-shot on the CARRADA test split: recall against output burden",
                 fontsize=9.6)

    fig.tight_layout()
    p = os.path.join(OUT, "ch7_fig_alldetector.png")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
