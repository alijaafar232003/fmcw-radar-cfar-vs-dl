"""Result figures for the real-data fine-tuning follow-up, on official CARRADA Validation.

    fig1  exact recall, four arms
    fig2  exact recall with the 95% sequence bootstrap interval
    fig3  recall vs unmatched-output burden (the evaluation-only sweep)
    fig4  tolerance ladder
    fig5  range-stratified recall
    fig6  per-sequence C vs CA-CFAR
    fig7  seed variability, A vs B vs C
    fig8  the three-stage progression

    python make_figures.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
OUT = os.path.join(EXP, "figures")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
INK, RED, BLUE, GREEN, GREY = "#1a1a1a", "#c0392b", "#2c6fbb", "#2e7d52", "#8a8a8a"
GOLD = "#d19a13"
COL = {"A": RED, "B": GOLD, "C": BLUE, "D": GREEN}
LBL = {"A": "original\nTinyCNN", "B": "CARRADA-informed\nTinyCNN",
       "C": "real-fine-tuned\nTinyCNN", "D": "CA-CFAR"}
FLAT = {k: v.replace("\n", " ") for k, v in LBL.items()}
ORDER = ["A", "B", "C", "D"]


def R():
    return json.load(open(os.path.join(EXP, "results", "validation_analysis.json"),
                          encoding="utf-8"))


def fig1(r):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    v = [r["pooled_exact_recall"][a] for a in ORDER]
    ax.bar(range(4), v, 0.58, color=[COL[a] for a in ORDER])
    for i, x in enumerate(v):
        ax.text(i, x + 0.004, "%.4f" % x, ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LBL[a] for a in ORDER], fontsize=8)
    ax.set_ylabel("exact dense-mask object recall")
    ax.set_ylim(0.82, 1.0)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("1.  Official CARRADA Validation — %d objects, %d frames"
                 % (r["eligible_objects"], r["n_frames"]))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_exact_recall.png"))
    plt.close(fig)


def fig2(r):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ps = r["per_sequence_exact_recall"]
    for i, a in enumerate(ORDER):
        v = r["pooled_exact_recall"][a]
        ax.bar(i, v, 0.55, color=COL[a], zorder=2)
        sv = list(ps[a].values())
        ax.scatter(np.full(len(sv), i) + np.linspace(-0.18, 0.18, len(sv)), sv,
                   s=18, color="white", edgecolor=INK, lw=0.7, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LBL[a] for a in ORDER], fontsize=8)
    ax.set_ylabel("exact recall")
    ax.set_ylim(0.70, 1.02)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    p = r["primary_C_minus_D"]
    ax.set_title("2.  With the six-sequence spread\nprimary C$-$D $= %+.4f$, "
                 "95%% CI [%+.4f, %+.4f]" % (p["mean_delta"], p["ci_lo"], p["ci_hi"]),
                 fontsize=9.5)
    ax.text(0.5, 0.03, "open circles = the six Validation sequences (the resampling unit)",
            transform=ax.transAxes, ha="center", fontsize=7.4, color=GREY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_recall_ci.png"))
    plt.close(fig)


def fig3(r):
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for a in ORDER:
        c = sorted(r["burden_curve"][a], key=lambda x: x["burden"])
        b = [x["burden"] for x in c]
        rc = [x["recall"] for x in c]
        ax.plot(b, rc, "-", color=COL[a], lw=1.8, label=FLAT[a])
        fb = r["unmatched_per_frame"][a]["mean"]
        fr_ = r["pooled_exact_recall"][a]
        ax.scatter([fb], [fr_], s=64, color=COL[a], edgecolor="white", lw=1.3, zorder=5)
    bD = r["unmatched_per_frame"]["D"]["mean"]
    ax.axvline(bD, color=GREY, lw=1.0, ls="--")
    ax.text(bD * 1.06, 0.60, "CA-CFAR's frozen\nburden %.2f/frame" % bD,
            fontsize=7.6, color=GREY)
    ax.set_xscale("log")
    ax.set_xlabel("unmatched detections per frame  (output burden — NOT $P_{fa}$)")
    ax.set_ylabel("exact dense-mask object recall")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.22, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("3.  Recall vs output burden (evaluation-only sweep; selects nothing)\n"
                 "filled markers = the frozen deployed operating points", fontsize=9.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_burden_curve.png"))
    plt.close(fig)


def fig4(r):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(3)
    for a in ORDER:
        v = [r["tolerance"]["tol%d" % t][a] for t in (0, 1, 3)]
        ax.plot(x, v, "o-", color=COL[a], lw=1.9, ms=6, label=FLAT[a])
    ax.set_xticks(x)
    ax.set_xticklabels(["exact", "$\\pm$1", "$\\pm$3"])
    ax.set_xlabel("support dilation (tolerance around an EXTENDED mask,\nnot localisation "
                  "accuracy)", fontsize=8.4)
    ax.set_ylabel("dense-mask object recall")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("4.  Tolerance ladder")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_tolerance.png"))
    plt.close(fig)


def fig5(r):
    order = ["10-60", "60-110", "110-160", "160-210", "210-246"]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    w = 0.2
    x = np.arange(len(order))
    for j, a in enumerate(ORDER):
        v = [r["by_range"][a].get(k, {}).get("recall", np.nan) for k in order]
        ax.bar(x + (j - 1.5) * w, v, w, color=COL[a], label=FLAT[a])
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(n=%d)" % (k, r["by_range"]["D"][k]["n_objects"]) for k in order],
                       fontsize=8)
    ax.set_xlabel("range block [bins]")
    ax.set_ylabel("exact recall")
    ax.set_ylim(0.72, 1.03)
    ax.legend(fontsize=7.8, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.20))
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("5.  Range-stratified recall")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_range.png"))
    plt.close(fig)


def fig6(r):
    ps = r["per_sequence_exact_recall"]
    seqs = sorted(ps["D"])
    d = [ps["C"][s] - ps["D"][s] for s in seqs]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = np.arange(len(seqs))
    ax.barh(y, d, 0.55, color=[BLUE if v > 0 else RED for v in d])
    for i, v in enumerate(d):
        ax.text(v + (0.004 if v > 0 else -0.004), i, "%+.4f" % v,
                va="center", ha="left" if v > 0 else "right", fontsize=8)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(seqs, fontsize=7.5)
    ax.set_xlabel("exact recall, fine-tuned TinyCNN $-$ CA-CFAR")
    ax.set_xlim(-0.06, 0.24)
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    p = r["primary_C_minus_D"]
    ax.set_title("6.  Per sequence: C $-$ CA-CFAR\nmean %+.4f, CI [%+.4f, %+.4f] — "
                 "ahead on %d of 6" % (p["mean_delta"], p["ci_lo"], p["ci_hi"],
                                       sum(1 for v in d if v > 0)), fontsize=9.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_per_sequence.png"))
    plt.close(fig)


def fig7(r):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, a in enumerate(["A", "B", "C"]):
        v = np.array(list(r["per_seed_pooled"][a].values()), dtype=float)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)), v,
                   s=34, color=COL[a])
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=COL[a], lw=2.4)
        sv = r["seed_variability"][a]
        ax.text(i, 0.836, "sd %.4f" % sv["sd"], ha="center", fontsize=8, color=COL[a])
    ax.axhline(r["pooled_exact_recall"]["D"], color=GREEN, lw=1.4, ls="--")
    ax.text(2.35, r["pooled_exact_recall"]["D"] + 0.002, "CA-CFAR", fontsize=8, color=GREEN)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LBL[a] for a in ("A", "B", "C")], fontsize=8)
    ax.set_ylabel("exact recall, per seed")
    ax.set_ylim(0.83, 0.99)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("7.  Seed variability — real adaptation shrinks it 10$\\times$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_seeds.png"))
    plt.close(fig)


def fig8(r):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    stages = ["A", "B", "C"]
    names = ["original\nsynthetic", "CARRADA-informed\nsynthetic", "real\nfine-tuning"]
    rec = [r["pooled_exact_recall"][s] for s in stages]
    bur = [r["unmatched_per_frame"][s]["mean"] for s in stages]

    a1.plot(range(3), rec, "o-", color=BLUE, lw=2.2, ms=9)
    a1.axhline(r["pooled_exact_recall"]["D"], color=GREEN, lw=1.5, ls="--")
    a1.text(2.05, r["pooled_exact_recall"]["D"] - 0.012, "CA-CFAR", fontsize=8.5, color=GREEN)
    for i, v in enumerate(rec):
        a1.text(i, v + 0.006, "%.4f" % v, ha="center", fontsize=8.5)
    a1.set_xticks(range(3)); a1.set_xticklabels(names, fontsize=8.5)
    a1.set_ylabel("exact recall"); a1.set_ylim(0.85, 0.99)
    a1.grid(alpha=0.22); a1.spines[["top", "right"]].set_visible(False)
    a1.set_title("recall rises, then plateaus")

    a2.plot(range(3), bur, "o-", color=RED, lw=2.2, ms=9)
    a2.axhline(r["unmatched_per_frame"]["D"]["mean"], color=GREEN, lw=1.5, ls="--")
    a2.text(2.05, r["unmatched_per_frame"]["D"]["mean"] * 1.25, "CA-CFAR", fontsize=8.5,
            color=GREEN)
    for i, v in enumerate(bur):
        a2.text(i, v * 1.18, "%.1f" % v, ha="center", fontsize=8.5)
    a2.set_yscale("log")
    a2.set_xticks(range(3)); a2.set_xticklabels(names, fontsize=8.5)
    a2.set_ylabel("unmatched detections / frame")
    a2.grid(alpha=0.22, which="both"); a2.spines[["top", "right"]].set_visible(False)
    a2.set_title("burden explodes, then collapses")

    fig.suptitle("8.  The three stages on real radar — what each one actually bought",
                 y=1.02, fontsize=10.5)
    fig.text(0.5, -0.04, "Stage 2 bought recall at ~5x the output. Stage 3 bought the output "
                         "back (10x lower) while keeping the recall.",
             ha="center", fontsize=8.4, color=GREY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_progression.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    r = R()
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f(r)
    print("wrote 8 figures to", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
