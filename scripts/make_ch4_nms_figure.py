"""Figure 4.8 — how the non-maximum-suppression radius was selected.

Every value is read from the development sweep in
`experiments/e6_nms_audit/results/p1_selection/`, which was run on `bench2_tune` and confirmed on
`bench2_val`. Neither the final synthetic holdout nor CARRADA contributed anything to this figure
or to the choice it records.

Four panels, in the order the argument runs:

  A  candidate density on pure-noise frames — the mechanism a suppression radius controls
  B  isolated-target Pd — flat, so density is not itself the cost
  C  masking and unresolved Pd — where a large radius does cost detections
  D  duplicate in-gate detections — the failure that rules radius 1 out

Run:  venv/Scripts/python.exe scripts/make_ch4_nms_figure.py
Out:  report/figures/Chapter4/ch4_fig8_nms_selection.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL = os.path.join(ROOT, "experiments", "e6_nms_audit", "results", "p1_selection")
OUT = os.path.join(ROOT, "report", "figures", "Chapter4")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
INK, BLUE, RED, GREY, GOLD, GREEN = ("#1a1a1a", "#2c6fbb", "#c0392b", "#8a8a8a",
                                     "#d19a13", "#2e7d52")
CHOSEN = 2


def band(ax, r=CHOSEN):
    """Mark the selected radius identically on every panel."""
    ax.axvspan(r - 0.28, r + 0.28, color=GREEN, alpha=0.10, zorder=0, lw=0)


def main():
    v = pd.read_csv(os.path.join(SEL, "nms_selection_sweep_val.csv"))
    radii = sorted(v.nms.unique())
    x = np.array(radii, float)

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 6.6))
    (a, b), (c, d) = ax

    # ---------------------------------------------------------------- A: candidate density
    nz = v[v.condition == "noise"].sort_values("nms")
    band(a)
    a.plot(nz.nms, nz.cand_per_frame_CA, "o-", color=BLUE, lw=1.8, ms=5, label="CA-CFAR")
    a.plot(nz.nms, nz.cand_per_frame_DL, "s-", color=RED, lw=1.8, ms=5, label="TinyCNN")
    a.set_yscale("log")
    a.set_ylabel("candidate peaks per noise frame")
    a.set_title("A.  Suppression controls candidate density", loc="left")
    a.legend(frameon=False, fontsize=8)
    for _, r in nz.iterrows():
        a.annotate("%.0f" % r.cand_per_frame_CA, (r.nms, r.cand_per_frame_CA),
                   textcoords="offset points", xytext=(0, 7), ha="center",
                   fontsize=7.2, color=BLUE)
    a.text(0.97, 0.34, "the network's candidate count barely moves:\nits positive-logit floor "
                       "removes most cells\nbefore suppression is applied",
           transform=a.transAxes, ha="right", va="top", fontsize=7.2, color=GREY)

    # ---------------------------------------------------------------- B: isolated Pd
    band(b)
    for m, col, mk in (("DL", RED, "s"), ("CA", BLUE, "o"), ("OS", GOLD, "^")):
        s = v[(v.condition == "isolated") & (v.method == m)].sort_values("nms")
        b.plot(s.nms, s.pd, mk + "-", color=col, lw=1.8, ms=5,
               label="TinyCNN" if m == "DL" else m + "-CFAR")
    b.set_ylim(0.60, 0.80)
    b.set_ylabel(r"$P_d$, isolated targets")
    b.set_title("B.  Isolated detection is insensitive to the radius", loc="left")
    b.legend(frameon=False, fontsize=8, loc="lower right")
    b.text(0.03, 0.90, "total spread across radii 1 to 5: %.3f (TinyCNN)"
           % float(np.ptp(v[(v.condition == "isolated") & (v.method == "DL")]["pd"])),
           transform=b.transAxes, fontsize=7.4, color=GREY)

    # ---------------------------------------------------------------- C: the cost of suppression
    band(c)
    for cond, col, mk, lab in (("masking", RED, "s", "masking"),
                               ("unresolved", BLUE, "o", "unresolved")):
        s = v[(v.condition == cond) & (v.method == "DL")].sort_values("nms")
        c.plot(s.nms, s.pd, mk + "-", color=col, lw=1.8, ms=5, label=lab + " (TinyCNN)")
    c.axhline(0.5, color=GREY, ls=":", lw=1.1)
    c.text(4.95, 0.513, "structural ceiling for the\nunresolved stratum ($\\approx 0.5$)",
           fontsize=7.0, color=GREY, va="bottom", ha="right")
    c.set_ylabel(r"$P_d$ at matched measured $P_{fa}$")
    c.set_xlabel("suppression radius (window side $2r+1$)")
    c.set_title("C.  Larger radii merge genuinely separate peaks", loc="left")
    c.legend(frameon=False, fontsize=8, loc="center right")

    # ---------------------------------------------------------------- D: duplicates
    band(d)
    for m, col, mk in (("DL", RED, "s"), ("CA", BLUE, "o"), ("OS", GOLD, "^")):
        s = v[(v.condition == "unresolved") & (v.method == m)].sort_values("nms")
        d.plot(s.nms, 100 * s.frac_truths_two_plus, mk + "-", color=col, lw=1.8, ms=5,
               label="TinyCNN" if m == "DL" else m + "-CFAR")
    d.set_ylim(-0.4, 9.0)
    d.set_ylabel("truths with $\\geq$2 in-gate detections (%)")
    d.set_xlabel("suppression radius (window side $2r+1$)")
    d.set_title("D.  Radius 1 admits duplicate detections", loc="left")
    d.legend(frameon=False, fontsize=8, loc="upper right")
    d.annotate("exactly zero at radii 2, 3 and 5", (3, 0.0), xytext=(3.0, 2.6),
               fontsize=7.4, color=INK, ha="center",
               arrowprops=dict(arrowstyle="->", lw=0.9, color=INK))

    for k in (a, b, c, d):
        k.set_xticks(x)
        k.set_xticklabels(["%d\n($%d\\times%d$)" % (int(r), 2 * int(r) + 1, 2 * int(r) + 1)
                           for r in x], fontsize=8)
        k.grid(alpha=0.22)
        k.spines[["top", "right"]].set_visible(False)
    a.set_xlabel("")
    b.set_xlabel("")

    fig.tight_layout()
    fig.text(0.5, -0.055,
             "Development-set measurements on bench2_val (200 frames per condition), eight frozen "
             "seeds, matched measured $P_{fa}=10^{-4}$, $\\pm1$-bin gate.\nThe shaded band marks "
             "the selected radius of 2. These are protocol-selection measurements, not benchmark "
             "results: neither the final synthetic\nholdout nor any real data contributed to them.",
             ha="center", fontsize=7.4, color=GREY)
    p = os.path.join(OUT, "ch4_fig8_nms_selection.png")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
