"""Backup figures for defence Q&A (A-H).

Not part of the main deck. Same visual language, same final artefacts.

    venv/Scripts/python.exe presentation_assets/build_backup.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BASE, CA, DATA, DL, DLF, DLI, FULL, GO, GREY, INK,      # noqa: E402
                    NEU, OS, POS, SHORT, SO, WIDE, clean, footnote, save)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E6 = os.path.join(ROOT, "experiments", "e6_nms_audit", "results")
FIN = os.path.join(E6, "final")
SEL = os.path.join(E6, "p1_selection")
ZS = os.path.join(E6, "p4_carrada", "r1_test", "r1_summary.json")
CIR = os.path.join(ROOT, "experiments", "carrada_informed_domain", "results")
FTR = os.path.join(ROOT, "experiments", "carrada_real_finetune", "results")
R0 = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation", "results")

USED: dict = {}


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def note(k, v, src):
    USED[k] = {"value": v, "source": os.path.relpath(src, ROOT)}
    return v


# ------------------------------------------------------------------ backup A
def b01_cfar_geometry():
    src = os.path.join(ROOT, "results", "cfar_geometry_percondition_v2.csv")
    g = pd.read_csv(src)
    piv = g.groupby(["window", "kind"])["pd_at_pfa"].mean().unstack()
    order = [w for w in ["21x21", "21x13", "21x11", "29x9", "21x9"] if w in piv.index]
    piv = piv.reindex(order)
    note("cfar_geometry_mean_pd", {w: {k: round(float(v), 4) for k, v in r.items()}
                                   for w, r in piv.iterrows()}, src)

    fig, ax = plt.subplots(figsize=WIDE)
    x = np.arange(len(piv))
    w = 0.26
    for i, (k, c) in enumerate(zip(piv.columns, [CA, GO, OS])):
        ax.bar(x + (i - 1) * w, piv[k], w, label="%s-CFAR" % k, color=c, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("x", " × ") for s in piv.index])
    ax.set_ylabel("mean $P_d$ over conditions")
    ax.set_xlabel("window (range × Doppler)")
    ax.set_ylim(0.30, 0.52)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    clean(ax, grid="y")
    sel = order.index("21x9")
    ax.annotate("selected", xy=(sel, float(piv.iloc[sel].max()) + 0.006),
                xytext=(sel - 0.9, 0.495), fontsize=BASE, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.4))
    ax.set_title("Why 21 × 9", fontweight="bold", pad=14)
    footnote(fig, "Chosen on bench2_tune, before any test read. "
                  "SO-CFAR was added after this selection and does not appear.")
    save(fig, "b01_cfar_geometry_sweep")


# ------------------------------------------------------------------ backup B
def b02_nms_radius():
    src = os.path.join(SEL, "nms_selection_sweep_val.csv")
    v = pd.read_csv(src)
    dup = v.groupby(["nms", "method"])["frac_truths_two_plus"].mean().unstack()
    mask = v[v.condition == "masking"].groupby(["nms", "method"])["pd"].mean().unstack()
    note("nms_frac_truths_two_plus",
         {int(r): {k: round(float(x), 4) for k, x in row.items()}
          for r, row in dup.iterrows()}, src)
    note("nms_masking_pd",
         {int(r): {k: round(float(x), 4) for k, x in row.items()}
          for r, row in mask.iterrows()}, src)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    a = axes[0]
    for m, c in (("CA", CA), ("DL", DL)):
        if m in dup.columns:
            a.plot(dup.index, dup[m], "o-", color=c, lw=3.2,
                   label="TinyCNN" if m == "DL" else "CA-CFAR")
    a.set_xlabel("suppression radius $r$")
    a.set_ylabel("truths with 2+ in-gate detections")
    a.legend(frameon=False)
    clean(a, grid="y")
    a.set_title("Duplicates vanish at $r \\geq 2$", fontsize=BASE + 2,
                fontweight="bold", pad=12)

    b = axes[1]
    for m, c in (("CA", CA), ("DL", DL)):
        if m in mask.columns:
            b.plot(mask.index, mask[m], "o-", color=c, lw=3.2,
                   label="TinyCNN" if m == "DL" else "CA-CFAR")
    b.set_xlabel("suppression radius $r$")
    b.set_ylabel("$P_d$, masking stratum")
    b.legend(frameon=False)
    clean(b, grid="y")
    b.set_title("Larger radii merge real targets", fontsize=BASE + 2,
                fontweight="bold", pad=12)

    for a in axes:
        a.axvspan(1.6, 2.4, color="#e8eef4", zorder=0)
    fig.suptitle("Why radius 2  (5 × 5)", fontsize=BASE + 4, fontweight="bold",
                 color=INK, y=1.03)
    fig.tight_layout()
    footnote(fig, "bench2_val, development data only.")
    save(fig, "b02_nms_radius_sweep")


# ------------------------------------------------------------------ backup C
def b03_tolerance():
    s1 = os.path.join(FIN, "bootstrap_final_r2_tol1.csv")
    s3 = os.path.join(FIN, "bootstrap_final_r2_tol3.csv")
    t1 = pd.read_csv(s1).set_index("condition")["diff"]
    t3 = pd.read_csv(s3).set_index("condition")["diff"]
    order = ["clutter_edge", "mixed", "masking", "isolated", "multi",
             "unresolved", "clutter"]
    note("delta_pd_tol1", {k: round(float(t1[k]), 4) for k in order}, s1)
    note("delta_pd_tol3", {k: round(float(t3[k]), 4) for k in order}, s3)

    fig, ax = plt.subplots(figsize=FULL)
    y = np.arange(len(order))[::-1]
    h = 0.36
    ax.barh(y + h / 2, [t1[c] for c in order], h, color=DL, label="$\\pm1$ bin (primary)",
            zorder=3)
    ax.barh(y - h / 2, [t3[c] for c in order], h, color=NEU, label="$\\pm3$ bins",
            zorder=3)
    ax.axvline(0, color=INK, lw=2.0)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in order], fontsize=BASE + 1)
    ax.set_xlabel("$\\Delta P_d$  (TinyCNN $-$ best CFAR)")
    ax.legend(frameon=False, loc="lower right")
    clean(ax, grid="x")
    ax.annotate("only clutter changes sign", xy=(t3["clutter"], -0.18),
                xytext=(0.09, 0.30), fontsize=BASE - 1, color=CA, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CA, lw=2.2))
    ax.set_title("Does the conclusion depend on the matching tolerance?",
                 fontweight="bold", pad=14)
    save(fig, "b03_tolerance_pm1_pm3")


# ------------------------------------------------------------------ backup D
def b04_carrada_range_blocks():
    r = jload(ZS)
    blocks = ["10-60", "60-110", "110-160", "160-210", "210-246"]
    st = r["by_range"]
    dl = [st[b]["tinycnn"] for b in blocks]
    ca = [st[b]["cfar_CA"] for b in blocks]
    n = [st[b]["eligible_objects"] for b in blocks]
    note("zeroshot_range_recall",
         {b: {"TinyCNN": round(d, 4), "CA-CFAR": round(c, 4), "n": int(k)}
          for b, d, c, k in zip(blocks, dl, ca, n)}, ZS)

    fig, ax = plt.subplots(figsize=FULL)
    x = np.arange(len(blocks))
    w = 0.38
    ax.bar(x - w / 2, ca, w, color=CA, label="CA-CFAR", zorder=3)
    ax.bar(x + w / 2, dl, w, color=DL, label="TinyCNN", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(n=%d)" % (b, k) for b, k in zip(blocks, n)],
                       fontsize=BASE - 2)
    ax.set_xlabel("range block [bins]")
    ax.set_ylabel("exact dense-mask recall")
    ax.set_ylim(0.76, 1.03)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    clean(ax, grid="y")
    ax.annotate("near-range deficit  %+.4f" % (dl[0] - ca[0]), xy=(0.19, dl[0] - 0.004),
                xytext=(1.34, 0.902), fontsize=BASE - 1, color=CA, fontweight="bold",
                ha="center", arrowprops=dict(arrowstyle="-|>", color=CA, lw=2.2))
    ax.set_title("Zero-shot recall by range — the deficit is near, not far",
                 fontweight="bold", pad=14)
    footnote(fig, "This is the block Stage 4 was pre-specified to test, "
                  "and the one it recovers.")
    save(fig, "b04_carrada_range_blocks")


# ------------------------------------------------------------------ backup E
def b05_mask_size_by_class():
    src = os.path.join(R0, "r0_object_footprint.csv")
    fp = pd.read_csv(src)
    cls = ["pedestrian", "cyclist", "car"]
    med = fp.groupby("class")[["range_extent", "doppler_extent", "dense_cells"]].median()
    cnt = fp["class"].value_counts()
    note("carrada_mask_median",
         {c: {"range": int(med.loc[c, "range_extent"]),
              "doppler": int(med.loc[c, "doppler_extent"]),
              "cells": int(med.loc[c, "dense_cells"]),
              "n_instances": int(cnt[c])} for c in cls}, src)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    a = axes[0]
    x = np.arange(3)
    w = 0.38
    a.bar(x - w / 2, [med.loc[c, "range_extent"] for c in cls], w, color=DL,
          label="range", zorder=3)
    a.bar(x + w / 2, [med.loc[c, "doppler_extent"] for c in cls], w, color=GO,
          label="Doppler", zorder=3)
    for i, c in enumerate(cls):
        a.text(i, max(med.loc[c, "range_extent"], med.loc[c, "doppler_extent"]) + 0.5,
               "%d×%d" % (med.loc[c, "range_extent"], med.loc[c, "doppler_extent"]),
               ha="center", fontsize=BASE, fontweight="bold", color=INK)
    a.set_xticks(x)
    a.set_xticklabels(cls, fontsize=BASE)
    a.set_ylabel("median extent [bins]")
    a.set_ylim(0, 22)
    a.legend(frameon=False, ncol=2)
    clean(a, grid="y")
    a.axhline(1, color=CA, lw=2.4, ls="--")
    a.set_xlim(-0.55, 3.25)
    a.annotate("synthetic\npoint target\n= 1 × 1", xy=(2.55, 1.0),
               xytext=(2.66, 5.0), ha="left", fontsize=BASE - 3, color=CA,
               fontweight="bold",
               arrowprops=dict(arrowstyle="-|>", color=CA, lw=2.0))
    a.set_title("Annotated support by class", fontsize=BASE + 2, fontweight="bold",
                pad=12)

    b = axes[1]
    b.bar(x, [med.loc[c, "dense_cells"] for c in cls], 0.5, color=[DL, DLI, DLF],
          zorder=3)
    for i, c in enumerate(cls):
        b.text(i, med.loc[c, "dense_cells"] + 3, "%d" % med.loc[c, "dense_cells"],
               ha="center", fontsize=BASE, fontweight="bold", color=INK)
    b.set_xticks(x)
    b.set_xticklabels(cls, fontsize=BASE)
    b.set_ylabel("median dense cells")
    clean(b, grid="y")
    b.set_title("Cells per object", fontsize=BASE + 2, fontweight="bold", pad=12)

    fig.suptitle("Class recall is confounded by mask size", fontsize=BASE + 4,
                 fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "b05_mask_size_by_class")


# ------------------------------------------------------------------ backup F
def b06_per_seed_carrada():
    r = jload(ZS)
    seeds = r["primary"]["tinycnn_per_seed"]
    ca = r["primary"]["cfar"]["CA"]
    ks = sorted(seeds, key=lambda k: int(k))
    vals = [seeds[k] for k in ks]
    note("zeroshot_per_seed", {k: round(v, 4) for k, v in seeds.items()}, ZS)

    fig, ax = plt.subplots(figsize=FULL)
    x = np.arange(len(ks))
    ax.bar(x, vals, 0.55, color=[DL if v < ca else "#1f4e79" for v in vals], zorder=3)
    ax.axhline(ca, color=CA, lw=3.0, ls="--", zorder=4)
    ax.axhline(float(np.mean(vals)), color=INK, lw=2.0, ls=":", zorder=4)
    ax.set_xlim(-1.9, len(ks) - 0.4)
    ax.text(-1.8, ca + 0.0018, "CA-CFAR\n%.4f" % ca, ha="left",
            fontsize=BASE - 3, color=CA, fontweight="bold")
    ax.text(-1.8, np.mean(vals) - 0.0105, "8-seed mean\n%.4f" % np.mean(vals),
            ha="left", fontsize=BASE - 3, color=INK, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0016, "%.4f" % v, ha="center", fontsize=BASE - 4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["seed %s" % k for k in ks], fontsize=BASE - 3)
    ax.set_ylabel("exact dense-mask recall")
    ax.set_ylim(0.83, 0.91)
    clean(ax, grid="y")
    ax.set_title("Per-seed zero-shot recall — %d of 8 above CA-CFAR"
                 % sum(v > ca for v in vals), fontweight="bold", pad=14)
    footnote(fig, "No seed was selected, dropped or ensembled. "
                  "sd %.4f across the eight frozen seeds."
                  % r["primary"]["tinycnn_seed_sd"])
    save(fig, "b06_per_seed_carrada")


# ------------------------------------------------------------------ backup G
def b07_burden_matched():
    src = os.path.join(CIR, "burden_matched.json")
    bm = jload(src)
    rows = bm["rows"]
    levels = []
    for r in rows:
        if r["target_label"] not in levels:
            levels.append(r["target_label"])
    arms = {"A": ("original synthetic", DL), "B": ("CARRADA-informed", DLI),
            "C": ("CA-CFAR", CA)}
    note("stage4_burden_matched",
         {r["target_label"] + " / " + r["arm"]: [round(r["recall"], 4),
                                                 round(r["burden"], 2)]
          for r in rows}, src)

    fig, ax = plt.subplots(figsize=FULL)
    x = np.arange(len(levels))
    w = 0.26
    for i, (k, (lbl, c)) in enumerate(arms.items()):
        vals = [next(r["recall"] for r in rows
                     if r["target_label"] == L and r["arm"] == k) for L in levels]
        ax.bar(x + (i - 1) * w, vals, w, color=c, label=lbl, zorder=3)
        for xx, v in zip(x + (i - 1) * w, vals):
            ax.text(xx, v + 0.008, "%.3f" % v, ha="center", fontsize=BASE - 4,
                    color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([L.replace(" level", "\nburden level") for L in levels],
                       fontsize=BASE - 2)
    ax.set_ylabel("exact recall at matched output burden")
    ax.set_ylim(0.55, 1.04)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    clean(ax, grid="y")
    ax.set_title("CARRADA test: recall when every arm emits the same amount",
                 fontweight="bold", pad=14)
    footnote(fig, "Post-hoc diagnostic, not pre-specified. CA-CFAR leads both learned "
                  "arms at every matched burden level.", colour=INK, size=BASE - 3)
    save(fig, "b07_burden_matched")


# ------------------------------------------------------------------ backup H
def b08_stage5_burden_curve():
    src = os.path.join(FTR, "validation_analysis.json")
    v = jload(src)
    bd = v["unmatched_per_frame"]["D"]["mean"]
    style = {"A": ("original synthetic", DL), "B": ("CARRADA-informed", DLI),
             "C": ("+ real fine-tuning", DLF), "D": ("CA-CFAR", CA)}
    at = {}
    fig, ax = plt.subplots(figsize=FULL)
    for k, (lbl, c) in style.items():
        cur = sorted(v["burden_curve"][k], key=lambda z: z["burden"])
        ax.plot([z["burden"] for z in cur], [z["recall"] for z in cur],
                "--" if k == "D" else "-", color=c, lw=4.0 if k == "C" else 2.8,
                label=lbl)
        near = min(cur, key=lambda z: abs(z["burden"] - bd))
        at[lbl] = round(near["recall"], 4)
        ax.scatter([v["unmatched_per_frame"][k]["mean"]],
                   [v["pooled_exact_recall"][k]], s=250, color=c,
                   edgecolor="white", lw=2.2, zorder=6)
    note("stage5_recall_at_CA_burden", at, src)

    ax.axvline(bd, color=INK, lw=2.2, ls=":")
    ax.annotate("CA-CFAR's own burden\n%.2f / frame" % bd, xy=(bd, 0.52),
                xytext=(1.2, 0.60), fontsize=BASE - 1, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
    ax.set_xscale("log")
    ax.set_xlim(0.1, 600)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("unmatched detections per frame   (output burden — not $P_{fa}$)")
    ax.set_ylabel("exact recall, CARRADA validation")
    ax.legend(frameon=False, loc="lower right")
    clean(ax, grid="both")
    ax.set_title("At the SAME output burden, the fine-tuned arm recalls more",
                 fontweight="bold", pad=14)
    footnote(fig, "Evaluation-only sweep; it selected nothing. "
                  "At %.2f / frame:  fine-tuned %.4f  |  informed %.4f  |  CA %.4f."
                  % (bd, at["+ real fine-tuning"], at["CARRADA-informed"],
                     at["CA-CFAR"]), colour=INK, size=BASE - 3)
    save(fig, "b08_stage5_burden_curve")


def main():
    print("backup figures:")
    for f in (b01_cfar_geometry, b02_nms_radius, b03_tolerance,
              b04_carrada_range_blocks, b05_mask_size_by_class,
              b06_per_seed_carrada, b07_burden_matched, b08_stage5_burden_curve):
        f()
    out = os.path.join(DATA, "values_backup.json")
    os.makedirs(DATA, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(USED, fh, indent=1)
    print("   wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
