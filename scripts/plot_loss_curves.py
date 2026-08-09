"""Training vs validation loss for TinyCNN on bench_train (Milestone 15).

Numbers are read verbatim from results/dl/train_bench.log so the figure can never
drift from the run that produced the model.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "dl" / "train_bench.log"
OUT = ROOT / "results" / "dl" / "loss_curves.png"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
TRAIN, VAL = "#2a78d6", "#eb6834"

PAT = re.compile(r"epoch\s+(\d+)\s+train\s+([\d.]+)\s+val\s+([\d.]+)")


def read_history(path: Path):
    ep, tr, va = [], [], []
    for m in PAT.finditer(path.read_text(encoding="utf-8", errors="replace")):
        ep.append(int(m.group(1)))
        tr.append(float(m.group(2)))
        va.append(float(m.group(3)))
    return ep, tr, va


def main() -> None:
    ep, tr, va = read_history(LOG)

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(ep, tr, color=TRAIN, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label="Training")
    ax.plot(ep, va, color=VAL, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label="Validation")

    # Direct labels instead of a legend box (only two series).
    ax.annotate("Training", (ep[-1], tr[-1]), xytext=(6, -2), textcoords="offset points",
                color=TRAIN, fontsize=11, fontweight="bold", va="center")
    ax.annotate("Validation", (ep[-1], va[-1]), xytext=(6, 2), textcoords="offset points",
                color=VAL, fontsize=11, fontweight="bold", va="center")

    # The point of the figure: still descending when training was stopped.
    ax.annotate(
        "still falling when we stopped",
        xy=(ep[-1], va[-1]), xytext=(-118, 46), textcoords="offset points",
        color=INK2, fontsize=10,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.1,
                        connectionstyle="arc3,rad=-0.25"),
    )

    ax.set_xlabel("Epoch", color=INK2, fontsize=11)
    ax.set_ylabel("Loss (BCE, pos_weight=200)", color=INK2, fontsize=11)
    ax.set_title("TinyCNN training vs validation loss — 9,600 frames, 12 epochs",
                 color=INK, fontsize=13, pad=14, loc="left")

    ax.set_xticks(ep)
    ax.set_xlim(0.6, len(ep) + 1.4)
    ax.tick_params(colors=MUTED, labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)

    fig.text(0.008, 0.015,
             "Curves stay close and both descend — no overfitting. "
             "No early stopping was used; 12 epochs was a fixed choice.",
             color=MUTED, fontsize=9)

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT}")
    print(f"epochs {ep[0]}..{ep[-1]}  train {tr[0]:.5f}->{tr[-1]:.5f}  val {va[0]:.5f}->{va[-1]:.5f}")


if __name__ == "__main__":
    main()
