"""Figure 6.1 — summary of the three attribution experiments.

Every value is read from the artefact that produced it, so the figure cannot drift from
Tables 6.1--6.3 or from the macros in the prose:

  E2  experiments/integrated_cfar/results/deltas.csv        mean delta_integration, gate 1
  E3  experiments/label_ablation/results/epoch_curve.csv    dv_bins and pd, gate 1, epoch 21
  E4  experiments/e4_cut_guard_mask/results/deltas.csv      mean delta_pd, gates 1 and 0

Nothing is recomputed and no experiment is re-run. All three were scored on the validation split
at the SUPERSEDED suppression radius of 5, which the figure states on its face.

Run:  venv/Scripts/python.exe scripts/make_ch6_figure.py
Out:  report/figures/Chapter6/ch6_fig1_attribution_summary.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICF = os.path.join(ROOT, "experiments", "integrated_cfar", "results")
LAB = os.path.join(ROOT, "experiments", "label_ablation", "results")
E4R = os.path.join(ROOT, "experiments", "e4_cut_guard_mask", "results")
OUT = os.path.join(ROOT, "report", "figures", "Chapter6")
os.makedirs(OUT, exist_ok=True)

# Same rcParams and palette as the Chapter 4 and 5 figure scripts.
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
INK, BLUE, RED, GREY, GOLD = "#1a1a1a", "#2c6fbb", "#c0392b", "#8a8a8a", "#d19a13"


def main():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.4, 4.1))

    # ---------------------------------------------------------------- A: E2 integrated CFAR
    d2 = pd.read_csv(os.path.join(ICF, "deltas.csv"))
    d2 = d2[d2.gate == 1]
    vals = [float(d2[d2.statistic == s].delta_integration.mean()) for s in ("uniform", "psf")]
    labs = ["Uniform\n$3\\times3$", "PSF-weighted\n$3\\times3$"]
    a.bar([0, 1], vals, 0.5, color=RED)
    a.axhline(0, color=INK, lw=1.1)
    for x, v in zip([0, 1], vals):
        a.text(x, v - 0.0022, "%+.4f" % v, ha="center", va="top", fontsize=8.4, color=INK)
    a.set_xticks([0, 1])
    a.set_xticklabels(labs, fontsize=8.4)
    a.set_ylim(min(vals) * 1.55, max(vals) * 0.30)
    a.set_ylabel("mean $\\Delta P_d$ vs standard CFAR")
    a.set_title("A.  E2 — integrating mainlobe energy", loc="left")
    a.text(0.5, 0.06, "both below zero:\nintegration makes CFAR worse",
           transform=a.transAxes, ha="center", fontsize=8, color=RED)

    # ---------------------------------------------------------------- B: E3 label ablation
    ec = pd.read_csv(os.path.join(LAB, "epoch_curve.csv"))
    g1 = ec[(ec.gate == 1) & (ec.epoch == 21)].set_index("arm")
    arms = ("flat3x3", "gaussian", "point")
    names = ["Flat $3\\times3$\n(frozen arm)", "Gaussian $3\\times3$", "Point $1\\times1$"]
    dv = [float(g1.loc[k, "dv_bins"]) for k in arms]
    pdv = [float(g1.loc[k, "pd"]) for k in arms]
    b.bar(range(3), dv, 0.5, color=[BLUE, GOLD, "#2e7d52"])
    for x, (v, p) in enumerate(zip(dv, pdv)):
        b.text(x, v + 0.008, "%.3f" % v, ha="center", fontsize=8.4, color=INK)
        b.text(x, 0.012, "$P_d$ %.4f" % p, ha="center", fontsize=7.6, color="white")
    b.set_xticks(range(3))
    b.set_xticklabels(names, fontsize=8.4)
    b.set_ylim(0, max(dv) * 1.22)
    b.set_ylabel("Doppler localisation error [bins]")
    b.set_title("B.  E3 — supervision shape", loc="left")
    b.text(0.97, 0.60, "concentrating the label\nimproves placement;\ntolerant $P_d$ barely moves",
           transform=b.transAxes, ha="right", va="top", fontsize=8, color=GREY)

    # ---------------------------------------------------------------- C: E4 guard masking
    d4 = pd.read_csv(os.path.join(E4R, "deltas.csv"))
    g = [float(d4[d4.gate == t].delta_pd.mean()) for t in (1, 0)]
    c.bar([0, 1], g, 0.5, color=[BLUE if v > 0 else RED for v in g])
    c.axhline(0, color=INK, lw=1.1)
    for x, v in zip([0, 1], g):
        off = 0.006 if v > 0 else -0.006
        c.text(x, v + off, "%+.4f" % v, ha="center",
               va="bottom" if v > 0 else "top", fontsize=8.4, color=INK)
    c.set_xticks([0, 1])
    c.set_xticklabels(["$\\pm1$ bin\n(primary gate)", "exact cell"], fontsize=8.4)
    c.set_ylim(min(g) * 1.30, max(0.055, max(g) * 3.2))
    c.set_ylabel("$\\Delta P_d$  (unmasked $-$ masked)")
    c.set_title("C.  E4 — guard-region access", loc="left")
    c.text(0.03, 0.42, "sign reverses: guard access\nhelps slightly at $\\pm1$, and\n"
                       "hurts exact centring a great deal",
           transform=c.transAxes, ha="left", va="top", fontsize=8, color=GREY)

    for ax in (a, b, c):
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.text(0.5, -0.055,
             "All three experiments were scored on the validation split at the superseded "
             "suppression radius of 5, and none has been re-scored at the final radius of 2.\n"
             "The large exact-cell effects in panels A and C are far bigger than any protocol "
             "effect measured by the radius audit; the small $+0.0150$ in panel C is not "
             "confirmed at radius 2.",
             ha="center", fontsize=7.6, color=GREY)

    p = os.path.join(OUT, "ch6_fig1_attribution_summary.png")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
