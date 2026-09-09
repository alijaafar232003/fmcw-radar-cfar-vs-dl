"""Generate the figures for Chapter 5 (Synthetic Benchmark Results).

Every panel reads a frozen result artefact. Nothing is recomputed and no detector is re-run.

Run:  venv/Scripts/python.exe scripts/make_ch5_figures.py
Out:  report/figures/Chapter5/
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The FINAL benchmark artefacts, produced under the revised post-processing protocol of
# configs/final_protocol.yaml. The superseded radius-5 tables remain in results/dl/ and are
# deliberately not read here, so no old value can survive into a figure.
DLR = os.path.join(ROOT, "experiments", "e6_nms_audit", "results", "final")
RUNTIME = os.path.join(ROOT, "results", "dl")   # runtime excludes NMS; radius-independent
OUT = os.path.join(ROOT, "report", "figures", "Chapter5")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
INK, BLUE, RED, GREY, GOLD = "#1a1a1a", "#2c6fbb", "#c0392b", "#8a8a8a", "#d19a13"
GREEN = "#2e8b57"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


def _order(df):
    """Conditions sorted by the size of the difference, so the figure reads top to bottom."""
    return df.sort_values("diff", ascending=True)


# ----------------------------------------------------------------------- Fig 5.1
def fig_headline():
    """Per-condition levels and the paired difference, at the primary gate."""
    d = _order(pd.read_csv(os.path.join(DLR, "bootstrap_final_r2_tol1.csv")))
    y = np.arange(len(d))

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4),
                           gridspec_kw={"width_ratios": [1.05, 1]})

    h = 0.36
    ax[0].barh(y + h / 2, d.dl_pd, h, color=BLUE, label="TinyCNN (8 seeds)")
    ax[0].barh(y - h / 2, d.cfar_pd, h, color=RED, label="best CFAR per condition")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(["%s\n(%s)" % (c.replace("_", " "), m)
                           for c, m in zip(d.condition, d.cfar_method)], fontsize=7.6)
    ax[0].set_xlabel("$P_d$ at matched measured $P_{fa}$")
    ax[0].set_xlim(0, 1.0)
    ax[0].set_title("Levels", fontsize=9.5, loc="left")
    ax[0].legend(frameon=False, fontsize=7.4, loc="lower right")
    ax[0].spines[["top", "right"]].set_visible(False)

    err = np.vstack([d["diff"] - d.ci_lo, d.ci_hi - d["diff"]])
    cols = [BLUE if v > 0 else RED for v in d["diff"]]
    ax[1].barh(y, d["diff"], 0.6, color=cols)
    ax[1].errorbar(d["diff"], y, xerr=err, fmt="none", ecolor=INK,
                   elinewidth=1.0, capsize=2.5)
    ax[1].axvline(0, color=INK, lw=1.0)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel("TinyCNN $-$ CFAR   [$P_d$]")
    ax[1].set_title("Paired difference, 95% CI", fontsize=9.5, loc="left")
    ax[1].spines[["top", "right"]].set_visible(False)
    # Place the marker clear of the whisker, not of the bar: with the revised protocol two
    # conditions are non-significant and their intervals straddle zero, so a bar-anchored label
    # would sit on top of the error bar.
    for i, (v, s, hi, lo) in enumerate(zip(d["diff"], d["sig (Bonf)"], d.ci_hi, d.ci_lo)):
        x = (hi if v > 0 else lo) + (0.012 if v > 0 else -0.012)
        ax[1].text(x, i, "*" if s else "n.s.", va="center",
                   ha="left" if v > 0 else "right", fontsize=8 if s else 6.8, color=INK)

    fig.tight_layout()
    fig.text(0.5, -0.05,
             "Held-out split, 400 frames per condition, read once. The classical arm is the best CFAR "
             "variant for that condition, named in the label.\n"
             "* marks significance after Bonferroni correction across the seven conditions; "
             "n.s. marks conditions that do not reach it.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch5_fig1_headline.png")


# ----------------------------------------------------------------------- Fig 5.2
def fig_pd_vs_snr():
    """The result in the currency the comparable literature uses."""
    sweep = pd.read_csv(os.path.join(DLR, "pd_vs_snr_final_r2.csv"))
    PFA = 1e-4
    theory = 10.0 * math.log10(math.log(PFA) / math.log(0.5) - 1.0)

    def snr50(sub):
        sub = sub.sort_values("snr")
        x, y = sub["snr"].to_numpy(), sub["pd"].to_numpy()
        for i in range(1, len(y)):
            if y[i - 1] < 0.5 <= y[i]:
                return x[i - 1] + (0.5 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
        return float("nan")

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    cols = {"CA": RED, "GO": GOLD, "OS": GREEN, "SO": "#8e44ad", "DL": BLUE}

    for a, gate, ttl in ((ax[0], 1, "Primary gate: $\\pm1$ bin"),
                         (ax[1], 0, "Exact cell: $\\pm0$ bins")):
        for m, c in cols.items():
            s = sweep[(sweep.gate == gate) & (sweep.method == m)].sort_values("snr")
            a.plot(s.snr, s.pd, color=c, lw=1.6 if m == "DL" else 1.1,
                   ls="-" if m == "DL" else "--",
                   label="TinyCNN" if m == "DL" else m, zorder=5 if m == "DL" else 2)
            v = snr50(s)
            if not math.isnan(v):
                a.plot(v, 0.5, "o", color=c, ms=4.5, zorder=6)
        a.axhline(0.5, color=GREY, lw=0.8, ls=":")
        a.axvline(theory, color=INK, lw=1.0, ls="-.")
        a.set_xlabel("Cell SNR [dB]")
        a.set_title(ttl, fontsize=9.5, loc="left")
        a.set_xlim(-2, 24)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("$P_d$ at matched measured $P_{fa}=10^{-4}$")
    ax[0].legend(frameon=False, fontsize=7.4, loc="lower right", ncol=2)
    ax[0].annotate("clairvoyant\nbound", xy=(theory, 0.16), xytext=(theory - 8.4, 0.20),
                   fontsize=7.2, color=INK,
                   arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    fig.tight_layout()
    fig.text(0.5, -0.06,
             "Markers show where each detector reaches $P_d=0.5$. The dash-dotted line is a "
             "detector that knows the true noise power.\nLeft: at the primary gate the network "
             "reaches the half-detection point at a lower SNR than any CFAR variant. Right: at "
             "the exact-cell\ngate it never reaches it at all — the same asymmetry the "
             "localization measurements show.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch5_fig2_pd_vs_snr.png")


# ----------------------------------------------------------------------- Fig 5.3
def fig_tolerance():
    """The same benchmark judged at two tolerances."""
    t1 = pd.read_csv(os.path.join(DLR, "bootstrap_final_r2_tol1.csv")
                     ).set_index("condition")
    t3 = pd.read_csv(os.path.join(DLR, "bootstrap_final_r2_tol3.csv")
                     ).set_index("condition")
    order = t1["diff"].sort_values().index
    y = np.arange(len(order))
    h = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.barh(y + h / 2, t1.loc[order, "diff"], h, color=BLUE, label="$\\pm1$ bin (primary)")
    ax.barh(y - h / 2, t3.loc[order, "diff"], h, color=GREY, label="$\\pm3$ bins (robustness)")
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in order], fontsize=8)
    ax.set_xlabel("TinyCNN $-$ CFAR   [$P_d$]")
    ax.set_title("Conclusions are stable across tolerance, with one exception",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # Mark the one condition where widening the gate moves the comparison against the network.
    # At the primary gate this condition is an exact tie, so "the classical arm leads" would be a
    # statement about the robustness gate only, and is not written as though it held at both.
    i = list(order).index("clutter")
    ax.annotate("the only condition where widening\nthe gate moves against the network",
                xy=(float(t3.loc["clutter", "diff"]), i - h / 2),
                xytext=(0.06, i + 1.0), fontsize=7.4, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    fig.tight_layout()
    fig.text(0.5, -0.06,
             "Widening the gate credits a detector that finds a target but places it further "
             "away. The ordering is unchanged in six of\nseven conditions; only the homogeneous "
             "clutter condition moves materially, and it moves against the network.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch5_fig3_tolerance.png")


# ----------------------------------------------------------------------- Fig 5.4
def fig_localization():
    """Where the network is worse: placing the detection it has found."""
    loc = pd.read_csv(os.path.join(DLR, "localization_final_r2.csv"))
    a = loc[(loc.gate == 1) & (loc.group == "ALL")].set_index("method")
    order = ["CA", "GO", "OS", "SO", "DL"]
    names = ["CA-CFAR", "GO-CFAR", "OS-CFAR", "SO-CFAR", "TinyCNN"]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

    x = np.arange(len(order))
    w = 0.36
    ax[0].bar(x - w / 2, [a.loc[m, "dr_bins"] for m in order], w,
              color=BLUE, label="range")
    ax[0].bar(x + w / 2, [a.loc[m, "dv_bins"] for m in order], w,
              color=GOLD, label="Doppler")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=7.4, rotation=20)
    ax[0].set_ylabel("Mean absolute error [bins]")
    ax[0].set_title("Localization error by axis", fontsize=9.5, loc="left")
    ax[0].legend(frameon=False, fontsize=7.4)
    ax[0].spines[["top", "right"]].set_visible(False)

    cols = [RED if m != "DL" else BLUE for m in order]
    ax[1].bar(x, [a.loc[m, "rmse_bins"] for m in order], 0.55, color=cols)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7.4, rotation=20)
    ax[1].set_ylabel("RMSE [bins]")
    ax[1].set_title("Combined localization RMSE", fontsize=9.5, loc="left")
    ax[1].spines[["top", "right"]].set_visible(False)
    for i, m in enumerate(order):
        ax[1].text(i, a.loc[m, "rmse_bins"] + 0.02, "%.2f" % a.loc[m, "rmse_bins"],
                   ha="center", fontsize=7.2, color=INK)

    fig.tight_layout()
    fig.text(0.5, -0.10,
             "Measured over true positives only, pooled across all conditions at the primary "
             "gate. The network finds more targets but places\nthem less precisely, and the "
             "penalty is largest on the Doppler axis. Chapter 4 records the $3\\times3$ block "
             "label as the\nleading candidate explanation; Chapter 6 tests it.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch5_fig4_localization.png")


# ----------------------------------------------------------------------- Fig 5.5
def fig_operating_point():
    """The protocol delivered what it promised: every arm sits at the design rate."""
    pf = pd.read_csv(os.path.join(DLR, "measured_pfa_final_r2_tol1.csv"))
    order = ["CA", "GO", "OS", "SO", "DL"]
    names = {"CA": "CA-CFAR", "GO": "GO-CFAR", "OS": "OS-CFAR",
             "SO": "SO-CFAR", "DL": "TinyCNN"}

    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    conds = sorted(pf.condition.unique())
    x = np.arange(len(conds))
    w = 0.16
    cols = {"CA": RED, "GO": GOLD, "OS": GREEN, "SO": "#8e44ad", "DL": BLUE}
    for i, m in enumerate(order):
        s = pf[pf.method == m].set_index("condition").reindex(conds)
        ax.bar(x + (i - 2) * w, s["ratio_to_target"], w, color=cols[m], label=names[m])
    ax.axhline(1.0, color=INK, lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in conds], fontsize=7.2, rotation=18)
    ax.set_ylabel("achieved $P_{fa}$ / design")
    ax.set_ylim(0.9, 1.1)
    ax.set_title("Every arm is scored at the same achieved false-alarm rate",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=6.9, ncol=5, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.text(0.5, -0.13,
             "Ratio of the achieved per-cell false-alarm rate to the design rate of $10^{-4}$, "
             "for every arm and condition of the held-out split.\nAll values lie within "
             "2% of unity, so the comparison is at a genuinely matched operating point "
             "rather than an assumed one.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch5_fig5_operating_point.png")


def main():
    print("Chapter 5 figures ->", os.path.relpath(OUT, ROOT))
    fig_headline()
    fig_pd_vs_snr()
    fig_tolerance()
    fig_localization()
    fig_operating_point()
    return 0


if __name__ == "__main__":
    sys.exit(main())
