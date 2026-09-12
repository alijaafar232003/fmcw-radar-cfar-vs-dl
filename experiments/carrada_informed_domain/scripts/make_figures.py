"""Result figures A-G for the CARRADA-informed domain intervention. Phase 10.

Reads only the finished artefacts. Synthetic and real metrics are kept visually distinct: the
domain panel (A) is synthetic-vs-real statistics, everything else is real-data recall or burden,
and no panel places a synthetic Pd beside a CARRADA recall.

    python make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
RES = os.path.join(EXP, "results")
OUT = os.path.join(EXP, "figures")
AUD = os.path.join(EXP, "domain_audit")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
INK, RED, BLUE, GREY, GOLD = "#1a1a1a", "#c0392b", "#2c6fbb", "#8a8a8a", "#d19a13"
CA = "#2e7d52"
LBL = {"A": "original\nTinyCNN", "B": "CARRADA-informed\nTinyCNN", "C": "CA-CFAR"}
COL = {"A": RED, "B": BLUE, "C": CA}


def load():
    r5 = json.load(open(os.path.join(RES, "test_analysis_r5.json"), encoding="utf-8"))
    r2 = json.load(open(os.path.join(RES, "test_analysis_r2.json"), encoding="utf-8"))
    bm = json.load(open(os.path.join(RES, "burden_matched.json"), encoding="utf-8"))
    gap = pd.read_csv(os.path.join(AUD, "domain_gap.csv"))
    return r5, r2, bm, gap


def fig_a(gap):
    """Domain statistics: original vs CARRADA-informed vs CARRADA train."""
    want = [("gamma_k_moment", "frame median", "Gamma shape $k$"),
            ("db_std", "frame median", "background spread [dB]"),
            ("lag1_doppler", "frame median", "lag-1 Doppler corr\n(declared residual)"),
            ("ridge_peak_over_floor_db", "frame mean", "ridge peak [dB]"),
            ("ridge_halfpower_bins", "frame mean", "ridge half-power [bins]"),
            ("range mean-power dB span", "across blocks", "range span [dB]")]
    fig, axes = plt.subplots(1, len(want), figsize=(2.35 * len(want), 3.1))
    for ax, (stat, scope, title) in zip(axes, want):
        row = gap[(gap.statistic == stat) & (gap.scope == scope)]
        if row.empty:
            ax.axis("off")
            continue
        row = row.iloc[0]
        vals = [row["sim_bench2_train"], row["newsim_pilot"], row["carrada_train"]]
        cols = [RED, BLUE, INK]
        ax.bar(range(3), vals, 0.62, color=cols)
        ax.axhline(row["carrada_train"], color=INK, lw=1.0, ls="--", alpha=0.7)
        for i, v in enumerate(vals):
            ax.text(i, v, ("%.2f" % v) if abs(v) >= 0.1 else ("%.3f" % v),
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["orig", "new", "real"], fontsize=8)
        ax.set_title(title, fontsize=8.6)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.28)
    fig.suptitle("A.  Synthetic training domain vs the real target domain "
                 "(dashed = CARRADA train)", y=1.04, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figA_domain_statistics.png"))
    plt.close(fig)


def fig_b(r5):
    """Exact CARRADA recall with sequence spread."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ps = r5["per_sequence_exact_recall"]
    for i, arm in enumerate("ABC"):
        v = r5["pooled_exact_recall"][arm]
        ax.bar(i, v, 0.55, color=COL[arm], zorder=2)
        seqv = list(ps[arm].values())
        ax.scatter(np.full(len(seqv), i) + np.linspace(-0.17, 0.17, len(seqv)), seqv,
                   s=17, color="white", edgecolor=INK, lw=0.7, zorder=3)
        ax.text(i, v + 0.006, "%.4f" % v, ha="center", fontsize=9, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LBL[a] for a in "ABC"], fontsize=8.5)
    ax.set_ylabel("exact dense-mask object recall")
    ax.set_ylim(0.80, 1.0)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    p = r5["primary_B_minus_A"]
    ax.set_title("B.  CARRADA test, 1,627 eligible objects\n"
                 "primary B$-$A $= %+.4f$, 95%% CI [%+.4f, %+.4f]"
                 % (p["mean_delta"], p["ci_lo"], p["ci_hi"]), fontsize=9.5)
    ax.text(0.5, 0.03, "open circles = the six test sequences", transform=ax.transAxes,
            ha="center", fontsize=7.5, color=GREY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figB_exact_recall.png"))
    plt.close(fig)


def fig_c(r5):
    """Tolerance ladder."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(3)
    for arm in "ABC":
        v = [r5["tolerance"]["tol%d" % t][arm] for t in (0, 1, 3)]
        ax.plot(x, v, "o-", color=COL[arm], lw=1.9, ms=6, label=LBL[arm].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(["exact", "$\\pm$1", "$\\pm$3"])
    ax.set_xlabel("support dilation (tolerance, NOT localisation error)")
    ax.set_ylabel("dense-mask object recall")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("C.  Tolerance ladder on CARRADA test")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figC_tolerance.png"))
    plt.close(fig)


def fig_d(r5):
    """Output burden."""
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    u = r5["unmatched_per_frame"]
    v = [u[a]["mean"] for a in "ABC"]
    ax.bar(range(3), v, 0.55, color=[COL[a] for a in "ABC"])
    for i, x in enumerate(v):
        ax.text(i, x + 1.5, "%.1f" % x, ha="center", fontsize=9.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LBL[a] for a in "ABC"], fontsize=8.5)
    ax.set_ylabel("unmatched detections per frame")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("D.  Output burden  (NOT false alarms, NOT $P_{fa}$)")
    ax.text(0.5, 0.86, "the CARRADA-informed arm emits %.1f$\\times$ the original arm's output"
            % (u["B"]["mean"] / u["A"]["mean"]), transform=ax.transAxes, ha="center",
            fontsize=8, color=RED)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figD_burden.png"))
    plt.close(fig)


def fig_e(r5):
    """Range-stratified recall."""
    order = ["10-60", "60-110", "110-160", "160-210", "210-246"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    w = 0.26
    x = np.arange(len(order))
    for j, arm in enumerate("ABC"):
        v = [r5["by_range"][arm].get(k, {}).get("recall", np.nan) for k in order]
        ax.bar(x + (j - 1) * w, v, w, color=COL[arm], label=LBL[arm].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(n=%d)" % (k, r5["by_range"]["A"][k]["n_objects"]) for k in order],
                       fontsize=8)
    ax.set_xlabel("range block [bins]")
    ax.set_ylabel("exact recall")
    ax.set_ylim(0.78, 1.02)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.20))
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    a10 = r5["by_range"]["A"]["10-60"]["recall"]
    b10 = r5["by_range"]["B"]["10-60"]["recall"]
    c10 = r5["by_range"]["C"]["10-60"]["recall"]
    ax.set_title("E.  Range-stratified recall — the pre-specified question was block 10–60\n"
                 "there the deficit closes: %.4f $\\to$ %.4f, against CA-CFAR's %.4f"
                 % (a10, b10, c10), fontsize=9.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figE_range_strata.png"))
    plt.close(fig)


def fig_f(r5):
    """Per-sequence paired comparison."""
    ps = r5["per_sequence_exact_recall"]
    seqs = sorted(ps["A"])
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    y = np.arange(len(seqs))
    for i, s in enumerate(seqs):
        ax.plot([ps["A"][s], ps["B"][s]], [i, i], "-", color=GREY, lw=1.4, zorder=1)
    ax.scatter([ps["A"][s] for s in seqs], y, s=52, color=RED, zorder=3, label="original")
    ax.scatter([ps["B"][s] for s in seqs], y, s=52, color=BLUE, zorder=3,
               label="CARRADA-informed")
    ax.scatter([ps["C"][s] for s in seqs], y, s=34, color=CA, marker="D", zorder=3,
               label="CA-CFAR")
    ax.set_yticks(y)
    ax.set_yticklabels(seqs, fontsize=7.5)
    ax.set_xlabel("exact dense-mask object recall")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("F.  Per-sequence paired comparison (the resampling unit)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figF_per_sequence.png"))
    plt.close(fig)


def fig_bm(bm):
    """The post-hoc burden-matched comparison -- the caveat figure."""
    rows = bm["rows"]
    labels = [r["target_label"] for r in rows if r["arm"] == "A"]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    x = np.arange(len(labels))
    w = 0.26
    for j, arm in enumerate("ABC"):
        v = [next(r["recall"] for r in rows if r["target_label"] == L and r["arm"] == arm)
             for L in labels]
        ax.bar(x + (j - 1) * w, v, w, color=COL[arm], label=LBL[arm].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(["matched to\n%s\n(%.1f/frame)"
                        % (L.replace(" level", ""),
                           next(r["target"] for r in rows if r["target_label"] == L))
                        for L in labels], fontsize=7.8)
    ax.set_ylabel("exact recall at matched output burden")
    ax.set_ylim(0.55, 1.06)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Post-hoc: recall when every arm emits the SAME number of detections\n"
                 "CA-CFAR leads at every matched burden; the original arm cannot reach "
                 "109/frame (its list saturates near 30)", fontsize=9.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figH_burden_matched.png"))
    plt.close(fig)


def fig_g(r5, bm):
    """Conceptual summary of the two chains."""
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    ax.axis("off")
    p = r5["primary_B_minus_A"]

    def chain(y, items, colour, head):
        ax.text(0.005, y + 0.115, head, fontsize=9.5, color=colour, weight="bold",
                transform=ax.transAxes)
        n = len(items)
        for i, (t, sub) in enumerate(items):
            x = 0.02 + i * (0.96 / n)
            ax.add_patch(plt.Rectangle((x, y - 0.055), 0.96 / n - 0.035, 0.115,
                                       transform=ax.transAxes, facecolor="white",
                                       edgecolor=colour, lw=1.3, zorder=2))
            ax.text(x + (0.96 / n - 0.035) / 2, y + 0.028, t, transform=ax.transAxes,
                    ha="center", fontsize=8.2, zorder=3)
            ax.text(x + (0.96 / n - 0.035) / 2, y - 0.022, sub, transform=ax.transAxes,
                    ha="center", fontsize=7.4, color=GREY, zorder=3)
            if i < n - 1:
                ax.annotate("", xy=(x + 0.96 / n - 0.031, y + 0.003),
                            xytext=(x + 0.96 / n - 0.035, y + 0.003),
                            xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", color=colour, lw=1.2))

    chain(0.76, [("original simulator", "1 look, flat, 39% ridge"),
                 ("TinyCNN", "5,041 params"),
                 ("domain shift", "measured"),
                 ("CARRADA test", "recall %.4f" % r5["pooled_exact_recall"]["A"])],
          RED, "Chapter 7 — the original chain")

    chain(0.34, [("CARRADA-train audit", "7.4 looks, ridge 23 dB, range profile"),
                 ("CARRADA-informed simulator", "background only; targets unchanged"),
                 ("the SAME TinyCNN", "same architecture, labels, optimiser"),
                 ("CARRADA test", "recall %.4f" % r5["pooled_exact_recall"]["B"])],
          BLUE, "This follow-up — the domain-intervention chain")

    ax.text(0.5, 0.10, "recall $%+.4f$  (95%% CI [%+.4f, %+.4f], excludes zero)   "
                       "but output burden $\\times%.1f$  (%.1f $\\to$ %.1f per frame)"
            % (p["mean_delta"], p["ci_lo"], p["ci_hi"],
               r5["unmatched_per_frame"]["B"]["mean"] / r5["unmatched_per_frame"]["A"]["mean"],
               r5["unmatched_per_frame"]["A"]["mean"], r5["unmatched_per_frame"]["B"]["mean"]),
            transform=ax.transAxes, ha="center", fontsize=9.6)
    ax.text(0.5, 0.025, "No real radar frame entered training or threshold calibration. "
                        "CARRADA train was used only to measure the target domain.",
            transform=ax.transAxes, ha="center", fontsize=8, color=GREY)
    fig.savefig(os.path.join(OUT, "figG_summary.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    r5, r2, bm, gap = load()
    fig_a(gap)
    fig_b(r5)
    fig_c(r5)
    fig_d(r5)
    fig_e(r5)
    fig_f(r5)
    fig_bm(bm)
    fig_g(r5, bm)
    print("wrote 8 figures to", os.path.relpath(OUT, ROOT))
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
