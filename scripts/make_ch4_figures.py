"""Generate the figures for Chapter 4 (Detectors and Evaluation Protocol).

Every value is read from the artefact that produced it: the window from configs/frozen_dl.yaml,
the multipliers and their audit from configs/cfar_alpha_table.json and
results/cfar_alpha_audit.csv, the geometry sweep from results/cfar_geometry_percondition_v2.csv,
and the network from src/dl/models.py via the frozen contract.

Run:  venv/Scripts/python.exe scripts/make_ch4_figures.py
Out:  report/figures/Chapter4/
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures", "Chapter4")
os.makedirs(OUT, exist_ok=True)

FZ = yaml.safe_load(open(os.path.join(ROOT, "configs", "frozen_dl.yaml"), encoding="utf-8"))
GR, TR = FZ["cfar"]["guard"], FZ["cfar"]["train"]
GD, TD = FZ["cfar"]["guard_v"], FZ["cfar"]["train_v"]
WR, WD = 1 + 2 * (GR + TR), 1 + 2 * (GD + TD)
N_GUARD = (1 + 2 * GR) * (1 + 2 * GD) - 1
N_REF = WR * WD - N_GUARD - 1

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
INK, BLUE, RED, GREY, GOLD = "#1a1a1a", "#2c6fbb", "#c0392b", "#8a8a8a", "#d19a13"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


# ----------------------------------------------------------------------- Fig 4.1
def fig_window():
    """The window actually used, with its three cell roles counted."""
    grid = np.zeros((WD, WR))
    cd, cr = WD // 2, WR // 2
    for i in range(WD):
        for j in range(WR):
            dr, dd = abs(j - cr), abs(i - cd)
            grid[i, j] = 3 if (dr == 0 and dd == 0) else (
                2 if (dr <= GR and dd <= GD) else 1)

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    cmap = matplotlib.colors.ListedColormap(["#eef2f6", "#9fc3e8", "#e8913c"])
    ax.imshow(grid, cmap=cmap, origin="lower", vmin=1, vmax=3)
    for x in range(WR + 1):
        ax.axvline(x - 0.5, color="white", lw=0.8)
    for y in range(WD + 1):
        ax.axhline(y - 0.5, color="white", lw=0.8)
    ax.set_xticks(range(0, WR, 2)); ax.set_xticklabels(range(-cr, cr + 1, 2))
    ax.set_yticks(range(WD)); ax.set_yticklabels(range(-cd, cd + 1))
    ax.set_xlabel("Range offset from the cell under test [bins]")
    ax.set_ylabel("Doppler offset [bins]")
    ax.set_title("The %d $\\times$ %d window: %d cells = 1 CUT + %d guard + %d reference"
                 % (WR, WD, WR * WD, N_GUARD, N_REF), fontsize=9.5)
    handles = [
        matplotlib.patches.Patch(facecolor="#e8913c", edgecolor="white",
                                 label="1 cell under test"),
        matplotlib.patches.Patch(facecolor="#9fc3e8", edgecolor="white",
                                 label="%d guard cells (excluded)" % N_GUARD),
        matplotlib.patches.Patch(facecolor="#eef2f6", edgecolor="#cccccc",
                                 label="%d reference cells (estimate $Z$)" % N_REF),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, frameon=False, fontsize=8)
    save(fig, "ch4_fig1_window.png")


# ----------------------------------------------------------------------- Fig 4.2
def fig_alpha_calibration():
    """Why the multipliers are calibrated rather than taken from the closed form."""
    curve = pd.read_csv(os.path.join(ROOT, "results", "cfar_alpha_curve.csv"))
    audit = pd.read_csv(os.path.join(ROOT, "results", "cfar_alpha_audit.csv"))
    tab = json.load(open(os.path.join(ROOT, "configs", "cfar_alpha_table.json"),
                         encoding="utf-8"))
    target = tab["meta"]["target_pfa"]

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0),
                           gridspec_kw={"width_ratios": [1.25, 1]})

    colours = {"CA": BLUE, "GO": GOLD, "OS": "#2e8b57", "SO": RED}
    for m, c in colours.items():
        s = curve[curve.method == m].sort_values("alpha")
        if len(s):
            ax[0].semilogy(s.alpha, s.pfa, color=c, lw=1.4, label=m)
    ax[0].axhline(target, color=INK, lw=1.0, ls="--")
    ax[0].set_xlim(curve.alpha.min(), 24)
    ax[0].set_ylim(1e-6, 2e-1)
    ax[0].text(23.5, target * 1.5, "design $P_{fa}=10^{-4}$",
               ha="right", fontsize=7.6, color=INK)
    for m, c in colours.items():
        a = audit[audit.method == m]
        if len(a):
            ax[0].plot(float(a.alpha_calibrated.iloc[0]), target, "o", color=c, ms=5)
    ax[0].set_xlabel(r"Threshold multiplier $\alpha$")
    ax[0].set_ylabel("Measured per-cell $P_{fa}$ on noise")
    ax[0].set_title("Calibration: sweep $\\alpha$, read off the design rate", fontsize=9.5)
    ax[0].legend(frameon=False, fontsize=7.6, ncol=2)
    ax[0].spines[["top", "right"]].set_visible(False)

    # What the closed form would have delivered, on a held-out seed.
    m_ord = ["CA", "GO", "OS", "SO"]
    got = [float(audit[audit.method == m].pfa_cells_verifyseed_at_current.iloc[0])
           for m in m_ord]
    cal = [float(audit[audit.method == m].pfa_cells_verifyseed_at_calibrated.iloc[0])
           for m in m_ord]
    x = np.arange(len(m_ord))
    ax[1].bar(x - 0.2, got, 0.38, color=GREY, label="at analytic reference $\\alpha$")
    ax[1].bar(x + 0.2, cal, 0.38, color=BLUE, label="at calibrated $\\alpha$")
    ax[1].axhline(target, color=INK, lw=1.0, ls="--")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels(m_ord)
    ax[1].set_ylabel("Achieved $P_{fa}$, held-out seed")
    ax[1].set_title("The analytic reference misses the target rate", fontsize=9.5)
    ax[1].legend(frameon=False, fontsize=7.4)
    ax[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.text(0.5, -0.06,
             "Left: measured false-alarm rate against threshold multiplier, on pure-noise frames "
             "through the identical processing chain.\nRight: what each variant achieves at its "
             "analytic reference multiplier versus its calibrated one, on a seed not used for\n"
             "calibration. The analytic reference is the CA expression evaluated with each "
             "statistic's cell count: exact for CA,\na half-ring approximation for GO and SO, and "
             "a first approximation only for OS. Dashed line is the design rate.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch4_fig2_alpha_calibration.png")


# ----------------------------------------------------------------------- Fig 4.3
def fig_geometry_sweep():
    """The window was selected by measurement on a dedicated split."""
    g = pd.read_csv(os.path.join(ROOT, "results", "cfar_geometry_percondition_v2.csv"))
    piv = g.groupby(["window", "kind"])["pd_at_pfa"].mean().unstack()
    order = ["21x21", "21x13", "21x11", "29x9", "21x9"]
    piv = piv.reindex([o for o in order if o in piv.index])

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(piv))
    w = 0.26
    for i, (k, c) in enumerate(zip(piv.columns, [BLUE, GOLD, "#2e8b57"])):
        ax.bar(x + (i - 1) * w, piv[k], w, label=k, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("x", r" $\times$ ") for s in piv.index])
    ax.set_ylabel("Mean $P_d$ over conditions")
    ax.set_xlabel("Window (range $\\times$ Doppler)")
    ax.set_ylim(0.30, 0.50)
    ax.set_title("Window geometry chosen on the tuning split, before any test read\n"
                 "(SO is absent: it was added to the project after this selection)",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    sel = "%dx%d" % (WR, WD)
    if sel in list(piv.index):
        ax.annotate("selected", xy=(list(piv.index).index(sel), piv.loc[sel].max()),
                    xytext=(list(piv.index).index(sel) - 0.6, 0.485),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.1),
                    color=RED, fontsize=8)
    save(fig, "ch4_fig3_geometry_sweep.png")


# ----------------------------------------------------------------------- Fig 4.4
def fig_network():
    """The frozen network, with the receptive field growing to the CFAR window."""
    dr = FZ["model"]["dilations_range"]
    dd = FZ["model"]["dilations_doppler"]
    width = FZ["model"]["width"]
    rf_r = np.cumsum([2 * d for d in dr]) + 1
    rf_d = np.cumsum([2 * d for d in dd]) + 1
    chans = [width] * (len(dr) - 1) + [1]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)

    ax.add_patch(Rectangle((1, 16), 13, 14, facecolor="white", edgecolor=GREY, lw=1.4))
    ax.text(7.5, 25, "RD map", ha="center", fontsize=8.4)
    ax.text(7.5, 20, "normalised\ndB", ha="center", fontsize=7.2, color=GREY)

    x0, w, gap = 18, 15, 4
    for i in range(len(dr)):
        x = x0 + i * (w + gap)
        last = i == len(dr) - 1
        ax.add_patch(Rectangle((x, 16), w, 14, facecolor="white",
                               edgecolor=RED if last else BLUE, lw=1.4))
        ax.text(x + w / 2, 26.5, "Conv $3\\times3$", ha="center", fontsize=7.8)
        ax.text(x + w / 2, 22.6, "%d ch,  dil (%d,%d)" % (chans[i], dr[i], dd[i]),
                ha="center", fontsize=6.9, color=GREY)
        ax.text(x + w / 2, 18.4, "BN + ReLU" if not last else "logits",
                ha="center", fontsize=6.9, color=GREY)
        ax.annotate("", xy=(x - 0.6, 23), xytext=(x - gap + 0.6, 23),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
        ax.text(x + w / 2, 11.5, "RF %d$\\times$%d" % (rf_r[i], rf_d[i]),
                ha="center", fontsize=7.4,
                color=RED if last else GREY)
        ax.annotate("", xy=(x + w / 2, 15.4), xytext=(x + w / 2, 13.4),
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

    ax.text(50, 39, "Receptive field grows with depth until it equals the CFAR window",
            ha="center", fontsize=8.8, color=INK)
    ax.text(50, 34.5, "{:,} parameters, no downsampling: the output stays at full RD resolution"
            .format(FZ["model"]["params"]), ha="center", fontsize=8.4, color=GREY)
    ax.text(50, 5.5, "final receptive field %d $\\times$ %d  =  the CFAR reference window"
            % (rf_r[-1], rf_d[-1]), ha="center", fontsize=8.6, color=RED)
    save(fig, "ch4_fig4_network.png")


# ----------------------------------------------------------------------- Fig 4.5
def fig_protocol():
    """One rulebook: both detectors reduced to points, then scored identically."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(4, 60)

    # Each arm carries its own fixed input transform: that is where the paths genuinely differ.
    ax.add_patch(Rectangle((1, 40), 27, 15, facecolor="#eaf1fa", edgecolor=BLUE, lw=1.4))
    ax.text(14.5, 50.5, "CFAR", ha="center", fontsize=9, color=BLUE)
    ax.text(14.5, 46.0, "power $\\rightarrow$ ratio CUT$/Z$", ha="center",
            fontsize=7.2, color=GREY)
    ax.text(14.5, 42.2, "own fixed transform", ha="center", fontsize=6.6, color=BLUE)

    ax.add_patch(Rectangle((1, 16), 27, 15, facecolor="#fdecea", edgecolor=RED, lw=1.4))
    ax.text(14.5, 26.5, "TinyCNN", ha="center", fontsize=9, color=RED)
    ax.text(14.5, 22.0, "log/median $\\rightarrow$ logits", ha="center",
            fontsize=7.2, color=GREY)
    ax.text(14.5, 18.2, "own fixed transform", ha="center", fontsize=6.6, color=RED)

    stages = [
        (34, "Peak\nextraction", "local maxima,\n$5\\times5$ NMS"),
        (54, "Threshold", "swept to matched\nMEASURED $P_{fa}$"),
        (74, "Matching", "one-to-one,\nHungarian, $\\pm$%d bin" % FZ["evaluation"]["gate"]),
    ]
    for x, t, s in stages:
        ax.add_patch(Rectangle((x, 28), 17, 15, facecolor="white", edgecolor=INK, lw=1.4))
        ax.text(x + 8.5, 38, t, ha="center", fontsize=8.2)
        ax.text(x + 8.5, 32, s, ha="center", fontsize=6.9, color=GREY)

    for y in (47, 24):
        ax.annotate("", xy=(33.4, 35.5), xytext=(28.4, y),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    for x in (34, 54):
        ax.annotate("", xy=(x + 19.6, 35.5), xytext=(x + 17.4, 35.5),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

    # Metrics box, connected rather than floating.
    ax.add_patch(Rectangle((74, 8), 17, 12, facecolor="white", edgecolor=INK, lw=1.4))
    ax.text(82.5, 15.5, "Metrics", ha="center", fontsize=8.2)
    ax.text(82.5, 11.5, "TP / FP / FN", ha="center", fontsize=6.9, color=GREY)
    ax.annotate("", xy=(82.5, 20.6), xytext=(82.5, 27.4),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

    ax.text(50, 58, "One shared evaluation path, applied identically to both arms",
            ha="center", fontsize=9, color=INK)
    ax.text(71, 14, "$P_d$ at matched measured $P_{fa}$,\nlocalization error,\n"
                    "false alarms per frame",
            ha="right", va="center", fontsize=7.4, color=INK)
    ax.text(50, 5.5, "The three shared stages are one code path. What differs is the score map "
                     "and the fixed transform that produced it.",
            ha="center", fontsize=7.5, color=RED)
    save(fig, "ch4_fig5_protocol.png")


def main():
    print("Chapter 4 figures ->", os.path.relpath(OUT, ROOT))
    fig_window()
    fig_alpha_calibration()
    fig_geometry_sweep()
    fig_network()
    fig_protocol()
    return 0


if __name__ == "__main__":
    sys.exit(main())
