"""R1 result figures. Reads the preserved Test outputs; reruns no detector."""
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
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import carrada_input as ci   # noqa: E402
import eligibility as el     # noqa: E402
import metrics as mt         # noqa: E402

OUT = os.path.join(EXP, "results", "r1_test")
FIGS = os.path.join(EXP, "figures", "r1_test")
C_DL, C_CA, C_REF = "#1F6FB4", "#C1440E", "#666666"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def main():
    os.makedirs(FIGS, exist_ok=True)
    R = json.load(open(os.path.join(OUT, "r1_summary.json")))
    dfo = pd.read_csv(os.path.join(OUT, "per_object.csv"))
    dff = pd.read_csv(os.path.join(OUT, "per_frame.csv"))
    mp = pd.read_csv(os.path.join(OUT, "multipeak_test.csv"))
    seqs = [r["sequence"] for r in R["per_sequence"]]

    # ---- fig 1: headline + per-sequence ----
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    p = R["primary"]
    seed_vals = list(p["tinycnn_per_seed"].values())
    ax.bar([0], [p["tinycnn_seed_mean"]], color=C_DL, width=.55, label="frozen TinyCNN (8-seed mean)")
    ax.bar([1], [p["cfar"]["CA"]], color=C_CA, width=.55, label="frozen CA-CFAR (primary)")
    ax.scatter(np.full(len(seed_vals), 0.0) + np.linspace(-.18, .18, len(seed_vals)), seed_vals,
               s=22, facecolors="none", edgecolors="black", zorder=5, linewidths=.8,
               label="individual seeds")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["TinyCNN", "CA-CFAR"])
    ax.set_ylim(0, 1); ax.set_ylabel("exact dense-mask object recall")
    b = R["paired_bootstrap"]
    ax.set_title("Primary result — CARRADA Test ({:,} eligible objects)\n"
                 "paired Δ = {:+.3f}, 95% CI [{:+.3f}, {:+.3f}], n = 6 sequences".format(
                     R["eligible_test_objects"], b["mean_delta"], b["ci_lo"], b["ci_hi"]),
                 fontsize=9.5)
    ax.legend(fontsize=7)

    ax = axes[1]
    x = np.arange(len(seqs))
    t = [r["tinycnn"] for r in R["per_sequence"]]
    c = [r["cfar_CA"] for r in R["per_sequence"]]
    ax.plot(x, t, "o-", color=C_DL, label="TinyCNN (seed mean)")
    ax.plot(x, c, "s-", color=C_CA, label="CA-CFAR")
    for i in x:
        ax.plot([i, i], [t[i], c[i]], color="#BBBBBB", lw=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([s[-8:] for s in seqs], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.02); ax.set_ylabel("exact-mask recall")
    ax.set_title("Per test sequence — CA-CFAR ahead in all 6", fontsize=9.5)
    ax.legend(fontsize=7.5)
    save(fig, "r1_fig1_primary_and_per_sequence.png")

    # ---- fig 2: strata ----
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9))
    for ax, key, title in ((axes[0], "by_class", "by class"),
                           (axes[1], "by_doppler", "zero-Doppler stratum"),
                           (axes[2], "by_range", "by range block  [bins]")):
        d = {k: v for k, v in R[key].items() if v["eligible_objects"] > 0}
        ks = list(d)
        x = np.arange(len(ks))
        ax.bar(x - .2, [d[k]["tinycnn"] for k in ks], .4, color=C_DL, label="TinyCNN")
        ax.bar(x + .2, [d[k]["cfar_CA"] for k in ks], .4, color=C_CA, label="CA-CFAR")
        for i, k in enumerate(ks):
            ax.annotate("n={}".format(d[k]["eligible_objects"]), (i, 0.03), ha="center",
                        fontsize=6.5, color="white")
        ax.set_xticks(x); ax.set_xticklabels(ks, rotation=20, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("exact-mask recall"); axes[0].legend(fontsize=7.5)
    fig.suptitle("Pre-registered strata. Class masks differ in size (car 18×13, cyclist 11×11, "
                 "pedestrian 9×9), so class recall is not purely detector sensitivity.",
                 fontsize=8.5)
    save(fig, "r1_fig2_strata.png")

    # ---- fig 3: tolerance ladder + burden + seeds ----
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9))
    ax = axes[0]
    tol = R["tolerance"]
    xs = [0, 1, 2]
    ax.plot(xs, [tol["tol{}".format(t)]["tinycnn"] for t in (0, 1, 3)], "o-", color=C_DL,
            label="TinyCNN")
    ax.plot(xs, [tol["tol{}".format(t)]["cfar_CA"] for t in (0, 1, 3)], "s-", color=C_CA,
            label="CA-CFAR")
    ax.set_xticks(xs); ax.set_xticklabels(["exact", "±1", "±3"])
    ax.set_ylim(0, 1.02); ax.set_ylabel("recall"); ax.legend(fontsize=7.5)
    ax.set_title("Tolerance around annotated support\n(NOT localization accuracy)", fontsize=9)

    ax = axes[1]
    bl = R["unmatched_burden"]
    ks = ["TinyCNN", "CFAR-CA", "CFAR-GO", "CFAR-SO", "CFAR-OS"]
    ax.bar(range(len(ks)), [bl[k]["mean"] for k in ks],
           color=[C_DL] + [C_CA] * 4)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("unmatched detections / frame")
    ax.set_title("Unmatched-detection burden\n**not** a false-alarm rate", fontsize=9)

    ax = axes[2]
    sv = R["primary"]["tinycnn_per_seed"]
    ax.bar([int(k) for k in sv], list(sv.values()), color=C_DL)
    ax.axhline(R["primary"]["cfar"]["CA"], color=C_CA, ls="--", lw=1.3, label="CA-CFAR")
    ax.axhline(R["primary"]["tinycnn_seed_mean"], color=C_DL, ls=":", lw=1.3, label="seed mean")
    ax.set_xlabel("frozen seed"); ax.set_ylabel("exact-mask recall"); ax.set_ylim(0, 1)
    ax.set_title("Seed spread {:.3f}–{:.3f} (sd {:.3f})".format(
        min(sv.values()), max(sv.values()), R["primary"]["tinycnn_seed_sd"]), fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "r1_fig3_tolerance_burden_seeds.png")

    # ---- fig 4: multi-peak diagnostic + recall-burden curve ----
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    ax = axes[0]
    for lab, sub, col in (("TinyCNN", mp[mp.detector == "tinycnn"], C_DL),
                          ("CA-CFAR", mp[(mp.detector == "cfar") & (mp.cfar_kind == "CA")], C_CA)):
        v = sub["peaks_inside_support"].to_numpy()
        frac = [float((v == 0).mean()), float((v == 1).mean()), float((v >= 2).mean())]
        off = -.18 if lab == "TinyCNN" else .18
        ax.bar(np.arange(3) + off, frac, .36, color=col, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["0 peaks", "1 peak", "≥2 peaks"])
    ax.set_ylabel("fraction of eligible objects")
    ax.set_title("Post-NMS peaks inside one object's own support\n"
                 "≥2 inflates unmatched burden for both arms", fontsize=9)
    ax.legend(fontsize=7.5)

    # recall vs unmatched burden, swept over score (pre-registered S6)
    ax = axes[1]
    objs_by_frame = {}
    for seq in ci.sequences("Test"):
        for fr in ci.frames(seq):
            objs_by_frame[(seq, fr)] = ci.objects(seq, fr)

    def curve(tag, cuts):
        z = np.load(os.path.join(OUT, "{}_peaks.npz".format(tag)), allow_pickle=True)
        store = {(str(a), str(b)): z["peaks"][int(c):int(d)] for a, b, c, d in z["index"]}
        xs, ys = [], []
        for cut in cuts:
            n_el = n_hit = n_un = n_fr = 0
            for key, objs in objs_by_frame.items():
                pk = store.get(key, np.zeros((0, 3)))
                pk = pk[pk[:, 2] >= cut] if len(pk) else pk
                r = mt.score_frame(pk, objs, tolerance=0)
                n_el += r["n_eligible_objects"]; n_hit += r["n_matched_eligible"]
                n_un += r["n_unmatched_peaks"]; n_fr += 1
            xs.append(n_un / n_fr); ys.append(n_hit / max(1, n_el))
        return xs, ys

    thr = {int(r["seed"]): float(r["threshold"]) for r in json.load(
        open(os.path.join(EXP, "configs", "tinycnn_zero_shot_thresholds.json")))["thresholds"]}
    dl_cuts = np.quantile(np.load(os.path.join(OUT, "tinycnn_seed0_peaks.npz"),
                                  allow_pickle=True)["peaks"][:, 2],
                          [.10, .3, .5, .65, .78, .88, .95, .99])
    ca_cuts = np.quantile(np.load(os.path.join(OUT, "cfar_CA_peaks.npz"),
                                  allow_pickle=True)["peaks"][:, 2],
                          [.10, .3, .5, .65, .78, .88, .95, .99])
    xd, yd = curve("tinycnn_seed0", dl_cuts)
    xc, yc = curve("cfar_CA", ca_cuts)
    ax.plot(xd, yd, "o-", color=C_DL, ms=4, label="TinyCNN (seed 0)")
    ax.plot(xc, yc, "s-", color=C_CA, ms=4, label="CA-CFAR")
    ax.scatter([R["unmatched_burden"]["TinyCNN"]["mean"]], [R["primary"]["tinycnn_per_seed"]["0"]],
               marker="*", s=150, color=C_DL, edgecolors="black", zorder=6,
               label="frozen operating point")
    ax.scatter([R["unmatched_burden"]["CFAR-CA"]["mean"]], [R["primary"]["cfar"]["CA"]],
               marker="*", s=150, color=C_CA, edgecolors="black", zorder=6)
    ax.set_xlabel("unmatched detections / frame   (NOT Pfa)")
    ax.set_ylabel("exact-mask object recall")
    ax.set_title("Recall–unmatched-burden curve\ndescriptive; no threshold chosen from it",
                 fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "r1_fig4_multipeak_and_burden_curve.png")
    print("\ndone")


if __name__ == "__main__":
    sys.exit(main())
