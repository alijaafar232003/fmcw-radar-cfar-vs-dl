"""The cross-domain figure: what happened to TinyCNN and CFAR moving synthetic -> real.

Design rule this figure exists to respect: **synthetic Pd and CARRADA mask recall are different
measurements** (point truth with a ±1 gate at matched measured Pfa, versus extended segmentation
support). They must never be pooled or read as the same axis.

So the figure separates the two domains into ROWS, states each row's metric on the row itself, and
puts the LEVELS on the left and the WITHIN-DOMAIN DIFFERENCE on the right. The difference column is
what legitimately carries across: within each domain it is a like-for-like paired contrast, and the
sign flip between rows is the finding.

Reads frozen synthetic results and preserved R1 outputs. Recomputes neither.
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

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
FIGS = os.path.join(EXP, "figures", "r1_test")

C_DL, C_CF = "#1F6FB4", "#C1440E"
C_WIN_DL, C_WIN_CF = "#1F6FB4", "#C1440E"
SYN_BG, REAL_BG = "#F4F7FA", "#FBF5F2"

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 170, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.22, "axes.axisbelow": True,
                     "axes.edgecolor": "#888888"})


def main():
    os.makedirs(FIGS, exist_ok=True)

    # ---- frozen synthetic headline: READ, never recomputed (bench2_test, epoch 21) ----
    syn = pd.read_csv(os.path.join(ROOT, "results", "dl",
                                   "bootstrap_frozen_ep021_test_tol1.csv"))
    syn = syn.sort_values("diff", ascending=False).reset_index(drop=True)

    # ---- real R1 result ----
    R1 = json.load(open(os.path.join(EXP, "results", "r1_test", "r1_summary.json")))
    P, BS = R1["primary"], R1["paired_bootstrap"]

    fig = plt.figure(figsize=(13.2, 8.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.58, wspace=0.22,
                          left=0.075, right=0.985, top=0.855, bottom=0.105)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for ax in (axA, axB):
        ax.set_facecolor(SYN_BG)
    for ax in (axC, axD):
        ax.set_facecolor(REAL_BG)

    # ================= PANEL A — synthetic levels =================
    x = np.arange(len(syn))
    axA.bar(x - .2, syn["dl_pd"], .4, color=C_DL, label="frozen TinyCNN")
    axA.bar(x + .2, syn["cfar_pd"], .4, color=C_CF,
            label="best CFAR per condition\n(oracle selection)")
    axA.set_xticks(x)
    axA.set_xticklabels(["{}\n({})".format(c, m) for c, m in
                         zip(syn["condition"], syn["cfar_method"])], fontsize=7.2)
    axA.set_ylim(0, 1.0)
    axA.set_ylabel("$P_d$ at matched measured $P_{fa}$")
    axA.set_title("A · Matched synthetic domain — by condition", fontsize=10, loc="left",
                  fontweight="bold")
    axA.legend(fontsize=7.2, loc="upper right", framealpha=.9)

    # ================= PANEL B — synthetic difference =================
    d = syn["diff"].to_numpy()
    lo = d - syn["ci_lo"].to_numpy()
    hi = syn["ci_hi"].to_numpy() - d
    cols = [C_WIN_DL if v > 0 else C_WIN_CF for v in d]
    axB.barh(x, d, color=cols, height=.62)
    axB.errorbar(d, x, xerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=1.1, capsize=3)
    axB.axvline(0, color="#333333", lw=1.2)
    axB.set_yticks(x); axB.set_yticklabels(syn["condition"], fontsize=7.5)
    axB.invert_yaxis()
    axB.set_xlabel("TinyCNN − CFAR   [$P_d$]")
    axB.set_title("B · Synthetic difference — TinyCNN ahead in 6 of 7", fontsize=10, loc="left",
                  fontweight="bold")
    axB.set_xlim(-0.06, 0.38)
    axB.annotate("TinyCNN better →", xy=(0.30, 6.4), fontsize=7.5, color=C_DL, ha="center")

    # ================= PANEL C — real levels =================
    cats = [("overall", R1["eligible_test_objects"], P["tinycnn_seed_mean"], P["cfar"]["CA"]),
            ("near zero\nDoppler", R1["by_doppler"]["near_zero"]["eligible_objects"],
             R1["by_doppler"]["near_zero"]["tinycnn"], R1["by_doppler"]["near_zero"]["cfar_CA"]),
            ("away from\nzero Doppler", R1["by_doppler"]["away_zero"]["eligible_objects"],
             R1["by_doppler"]["away_zero"]["tinycnn"], R1["by_doppler"]["away_zero"]["cfar_CA"])]
    for c in ("car", "pedestrian", "cyclist"):
        v = R1["by_class"][c]
        cats.append((c, v["eligible_objects"], v["tinycnn"], v["cfar_CA"]))
    xc = np.arange(len(cats))
    axC.bar(xc - .2, [c[2] for c in cats], .4, color=C_DL, label="frozen TinyCNN (8-seed mean)")
    axC.bar(xc + .2, [c[3] for c in cats], .4, color=C_CF, label="frozen CA-CFAR (primary)")
    for i, c in enumerate(cats):
        top = max(c[2], c[3])
        axC.annotate("n={:,}".format(c[1]), (i, top + 0.018), ha="center", fontsize=6.6,
                     color="#333333")
    axC.axvline(2.5, color="#999999", lw=1, ls=":")
    axC.annotate("pre-registered strata", (1.0, 1.055), ha="center", fontsize=6.8,
                 color="#777777")
    axC.annotate("by class (mask sizes differ)", (4.0, 1.055), ha="center",
                 fontsize=6.8, color="#777777")
    axC.set_xticks(xc); axC.set_xticklabels([c[0] for c in cats], fontsize=7.0)
    axC.set_ylim(0, 1.10)
    axC.set_ylabel("exact dense-mask object recall")
    axC.set_title("C · Unseen real CARRADA domain — zero-shot", fontsize=10, loc="left",
                  fontweight="bold")
    # Direct bar labels rather than a legend box: Panel A already establishes the colours,
    # and every corner of this panel is occupied by a bar.
    axC.annotate("TinyCNN", (-0.2, cats[0][2] - 0.045), ha="center", fontsize=6.8,
                 color="white", fontweight="bold", rotation=90)
    axC.annotate("CA-CFAR", (0.2, cats[0][3] - 0.05), ha="center", fontsize=6.8,
                 color="white", fontweight="bold", rotation=90)

    # ================= PANEL D — real difference =================
    ps = R1["per_sequence"]
    y = np.arange(len(ps))
    dd = [r["delta"] for r in ps]
    axD.barh(y, dd, color=[C_WIN_DL if v > 0 else C_WIN_CF for v in dd], height=.55)
    axD.axvline(0, color="#333333", lw=1.2)
    # pooled estimate with sequence-aware CI
    axD.barh([len(ps) + .6], [BS["mean_delta"]], color="#7A1F05", height=.55)
    axD.errorbar([BS["mean_delta"]], [len(ps) + .6],
                 xerr=[[BS["mean_delta"] - BS["ci_lo"]], [BS["ci_hi"] - BS["mean_delta"]]],
                 fmt="none", ecolor="#111111", elinewidth=1.4, capsize=4)
    axD.set_yticks(list(y) + [len(ps) + .6])
    axD.set_yticklabels([r["sequence"][-8:] for r in ps] + ["POOLED\n95% CI"], fontsize=7.2)
    axD.invert_yaxis()
    axD.set_xlabel("TinyCNN − CA-CFAR   [mask recall]")
    axD.set_title("D · Real difference — CA-CFAR ahead in 6 of 6", fontsize=10, loc="left",
                  fontweight="bold")
    axD.set_xlim(-0.30, 0.06)
    axD.annotate("← CFAR better", xy=(-0.22, 6.9), fontsize=7.5, color=C_CF, ha="center")

    # ================= row banners and honest framing =================
    fig.text(0.5, 0.965, "What happened to the learned and classical detectors "
             "when the domain changed?", ha="center", fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.935,
             "The advantage reverses. Left column: levels within each domain.  "
             "Right column: the paired difference, which is the like-for-like quantity.",
             ha="center", fontsize=9.2, color="#444444")

    fig.text(0.014, 0.735, "SYNTHETIC\nDOMAIN", rotation=90, va="center", ha="center",
             fontsize=10.5, fontweight="bold", color="#1F4E79")
    fig.text(0.014, 0.30, "REAL\nCARRADA", rotation=90, va="center", ha="center",
             fontsize=10.5, fontweight="bold", color="#8A3010")

    fig.text(0.5, 0.885,
             "metric: point-target $P_d$, ±1-bin gate, matched measured $P_{fa}$  ·  "
             "bench2_test, 8 seeds, epoch 21  ·  frozen benchmark, read not recomputed",
             ha="center", fontsize=8, style="italic", color="#1F4E79")
    fig.text(0.5, 0.452,
             "metric: exact dense-mask object recall  ·  6 official test sequences, "
             "1,627 eligible objects  ·  zero-shot, no retraining or recalibration",
             ha="center", fontsize=8, style="italic", color="#8A3010")

    fig.text(0.5, 0.016,
             "⚠  The two rows measure DIFFERENT quantities and must not be pooled or compared "
             "numerically: synthetic $P_d$ scores a point target within ±1 bin, while CARRADA "
             "recall asks whether a peak fell inside an extended annotated support.\n"
             "The comparable statement is the RANKING: TinyCNN leads in the matched synthetic "
             "domain (B), CA-CFAR leads in the unseen real domain (D). Vertical scales differ "
             "between the right-hand panels; both are anchored at zero.",
             ha="center", fontsize=8.1, color="#333333",
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFFDF0", ec="#D9C89A"))

    p = os.path.join(FIGS, "r1_fig5_crossdomain_4panel.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    sys.exit(main())
