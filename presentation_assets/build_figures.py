"""Result figures for the defence deck (Figures 6-16).

Every value is read from a FINAL frozen artefact at build time; nothing is typed in.
The exact file behind each number is recorded in presentation_assets/README.md and
dumped to presentation_assets/data/values.json.

Deliberately NOT read: results/dl/ radius-5 tables, results/cfar_test*, the
superseded bench2_test split, PROGRESS.md, and any notebook output.

    venv/Scripts/python.exe presentation_assets/build_figures.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BASE, CA, DATA, DL, DLF, DLI, FULL, GO, GREY, INK,      # noqa: E402
                    NEU, OS, POS, SHORT, SO, THEORY, WIDE, blank, clean,
                    footnote, save)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E6 = os.path.join(ROOT, "experiments", "e6_nms_audit", "results")
FIN = os.path.join(E6, "final")
ATT = os.path.join(E6, "p10_attribution")
ZS = os.path.join(E6, "p4_carrada", "r1_test", "r1_summary.json")
CIR = os.path.join(ROOT, "experiments", "carrada_informed_domain", "results")
FTR = os.path.join(ROOT, "experiments", "carrada_real_finetune", "results")
R0 = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation", "results")

USED: dict = {}          # everything that reaches a figure, for data/values.json


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def note(key, value, source):
    USED[key] = {"value": value, "source": os.path.relpath(source, ROOT)}
    return value


# ===================================================================== Figure 6
def f06_synthetic_headline():
    src = os.path.join(FIN, "bootstrap_final_r2_tol1.csv")
    b = pd.read_csv(src).set_index("condition")
    order = ["clutter_edge", "mixed", "masking", "isolated", "multi",
             "unresolved", "clutter"]
    b = b.loc[order]
    established = b["sig (Bonf)"].to_dict()
    note("synthetic_delta_pd", {k: round(float(v), 4) for k, v in b["diff"].items()}, src)
    note("synthetic_significant", {k: bool(v) for k, v in established.items()}, src)

    fig, ax = plt.subplots(figsize=FULL)
    y = np.arange(len(b))[::-1]
    colours = [POS if established[c] else NEU for c in b.index]
    ax.barh(y, b["diff"], 0.62, color=colours, zorder=3)
    for yy, c in zip(y, b.index):
        d = b.loc[c, "diff"]
        ax.text(d + 0.006, yy, "%+.4f" % d, va="center", fontsize=BASE,
                fontweight="bold" if established[c] else "normal",
                color=INK if established[c] else GREY)
    ax.axvline(0, color=INK, lw=2.0)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in b.index], fontsize=BASE + 2)
    ax.set_xlim(-0.012, 0.325)
    ax.set_xlabel("$\\Delta P_d$   (TinyCNN $-$ best CFAR, matched measured $P_{fa}$)")
    clean(ax, grid="x")

    ax.axhspan(-0.55, 1.5, color="#f4f6f7", zorder=0)
    ax.text(0.315, 0.5, "tie / sanity", ha="right", va="center", fontsize=BASE - 1,
            color=GREY, style="italic")
    ax.text(0.315, 4.0, "established gains", ha="right", va="center",
            fontsize=BASE - 1, color=POS, style="italic", fontweight="bold")
    ax.set_title("Seven synthetic conditions, one operating point",
                 fontweight="bold", pad=16)
    footnote(fig, "5 established gains  |  clutter ≈ tie  |  "
                  "unresolved = sanity / protocol stress case "
                  "(not significant after Bonferroni)", colour=INK, size=BASE - 2)
    save(fig, "f06_synthetic_headline")


# ===================================================================== Figure 7
def f07_roc_operating_point():
    src = os.path.join(FIN, "roc_final_r2_tol1.csv")
    roc = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=FULL)
    arms = [("CA", "CA-CFAR", CA, 3.4, "-"), ("GO", "GO-CFAR", GO, 2.0, ":"),
            ("OS", "OS-CFAR", OS, 2.0, ":"), ("SO", "SO-CFAR", SO, 2.0, ":"),
            ("DL", "TinyCNN", DL, 4.2, "-")]
    at1e4 = {}
    for m, lbl, c, lw, ls in arms:
        d = roc[(roc.method == m) & (roc.pfa_per_cell > 0)].sort_values("pfa_per_cell")
        ax.plot(d.pfa_per_cell, d.pd, color=c, lw=lw, ls=ls, label=lbl,
                zorder=5 if m in ("DL", "CA") else 2)
        at1e4[m] = round(float(np.interp(np.log10(1e-4),
                                         np.log10(d.pfa_per_cell.to_numpy()),
                                         d.pd.to_numpy())), 4)
    note("roc_pd_at_pfa_1e-4_pooled", at1e4, src)

    ax.axvspan(5e-5, 2e-4, color="#e8eef4", zorder=0)
    ax.axvline(1e-4, color=INK, lw=2.0, ls="--", zorder=1)
    ax.annotate("compared HERE\nmeasured $P_{fa}=10^{-4}$", xy=(1e-4, 0.52),
                xytext=(2.2e-5, 0.795), ha="center", fontsize=BASE,
                fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.4))
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 4e-2)
    ax.set_ylim(0, 0.88)
    ax.set_xlabel("measured per-cell $P_{fa}$")
    ax.set_ylabel("$P_d$  (pooled, 7 scored conditions)")
    ax.legend(frameon=False, loc="upper left", fontsize=BASE - 1)
    clean(ax, grid="y")
    ax.set_title("Matched at the MEASURED false-alarm rate, not at a design $\\alpha$",
                 fontweight="bold", pad=16)
    footnote(fig, "Each point is a different score threshold. "
                  "TinyCNN's curve ends near $8\\times10^{-4}$: the shared extractor "
                  "keeps only positive logits.")
    save(fig, "f07_roc_operating_point")


# ===================================================================== Figure 8
def f08_pd_vs_snr():
    src = os.path.join(FIN, "pd_vs_snr_final_r2.csv")
    ssrc = os.path.join(FIN, "pd_vs_snr_final_r2_summary.csv")
    sweep = pd.read_csv(src)
    summ = pd.read_csv(ssrc).set_index(["gate", "method"])
    PFA = 1e-4
    sl = np.linspace(-2, 24, 400)
    theory = PFA ** (1.0 / (1.0 + 10.0 ** (sl / 10.0)))
    t50 = 10 * math.log10(math.log(PFA) / math.log(0.5) - 1.0)

    dl1 = float(summ.loc[(1, "DL"), "snr50"])
    ca1 = float(summ.loc[(1, "CA"), "snr50"])
    ca0 = float(summ.loc[(0, "CA"), "snr50"])
    note("snr50_gate1", {"TinyCNN": round(dl1, 2), "CA-CFAR": round(ca1, 2),
                         "margin_dB": round(ca1 - dl1, 2)}, ssrc)
    note("snr50_gate0", {"CA-CFAR": round(ca0, 2), "clairvoyant": round(t50, 2),
                         "cfar_loss_dB": round(ca0 - t50, 2)}, ssrc)

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 6.2), sharey=True)
    for a, gate, ttl in ((ax[0], 1, "Primary gate  $\\pm1$ bin"),
                         (ax[1], 0, "Exact cell")):
        a.plot(sl, theory, ls=(0, (6, 3)), color=THEORY, lw=3.0,
               label="clairvoyant reference", zorder=6)
        for m, c in (("GO", GO), ("OS", OS), ("SO", SO)):
            d = sweep[(sweep.gate == gate) & (sweep.method == m)].sort_values("snr")
            a.plot(d.snr, d.pd, color=c, lw=1.6, ls=":", alpha=0.8, zorder=2)
        for m, lbl, c in (("CA", "CA-CFAR", CA), ("DL", "TinyCNN", DL)):
            d = sweep[(sweep.gate == gate) & (sweep.method == m)].sort_values("snr")
            a.plot(d.snr, d.pd, color=c, lw=4.0, label=lbl, zorder=5)
        a.axhline(0.5, color=GREY, lw=1.4, ls=":")
        a.set_xlabel("cell SNR [dB]")
        a.set_xlim(-2, 24)
        a.set_ylim(0, 1.02)
        a.set_title(ttl, fontsize=BASE + 2, fontweight="bold", pad=12)
        clean(a, grid="y")

    ax[0].fill_betweenx([0, 0.5], dl1, ca1, color="#cfe0ee", zorder=1, alpha=0.9)
    ax[0].plot(dl1, 0.5, "o", color=DL, ms=16, mec="white", mew=2.2, zorder=8)
    ax[0].plot(ca1, 0.5, "o", color=CA, ms=16, mec="white", mew=2.2, zorder=8)
    ax[0].annotate("", xy=(ca1, 0.575), xytext=(dl1, 0.575),
                   arrowprops=dict(arrowstyle="<->", color=INK, lw=2.6,
                                   shrinkA=0, shrinkB=0))
    ax[0].text((dl1 + ca1) / 2, 0.615, "%.2f dB" % (ca1 - dl1), ha="center",
               fontsize=BASE + 3, fontweight="bold", color=INK, zorder=9)
    ax[0].text(dl1, 0.055, "%.2f" % dl1, ha="right", fontsize=BASE - 2,
               color=DL, fontweight="bold", zorder=9)
    ax[0].text(ca1 + 0.25, 0.055, "%.2f" % ca1, ha="left", fontsize=BASE - 2,
               color=CA, fontweight="bold", zorder=9)
    ax[0].set_ylabel("$P_d$ at matched measured $P_{fa}=10^{-4}$")
    ax[0].legend(frameon=False, loc="upper left", fontsize=BASE - 3)

    ax[1].plot(ca0, 0.5, "o", color=CA, ms=15, mec="white", mew=2.0, zorder=8)
    ax[1].plot(t50, 0.5, "o", color=THEORY, ms=13, mec="white", mew=2.0, zorder=8)
    ax[1].annotate("CFAR loss %.2f dB" % (ca0 - t50), xy=(ca0, 0.5),
                   xytext=(ca0 + 1.6, 0.20), fontsize=BASE - 1, color=CA,
                   fontweight="bold",
                   arrowprops=dict(arrowstyle="-|>", color=CA, lw=2.2))
    ax[1].text(1.0, 0.86, "TinyCNN never reaches\n$P_d=0.5$ here",
               fontsize=BASE - 2, color=DL, fontweight="bold")

    fig.suptitle("TinyCNN detects a %.2f dB weaker target at the same false-alarm rate"
                 % (ca1 - dl1), fontsize=BASE + 4, fontweight="bold", color=INK,
                 y=1.03)
    fig.tight_layout()
    save(fig, "f08_pd_vs_snr")


# ===================================================================== Figure 9
def f09_localisation():
    src = os.path.join(FIN, "localization_final_r2.csv")
    loc = pd.read_csv(src)
    a = loc[(loc.gate == 1) & (loc.group == "ALL")].set_index("method")
    rm_dl, rm_ca = float(a.loc["DL", "rmse_bins"]), float(a.loc["CA", "rmse_bins"])
    dv_dl, dv_ca = float(a.loc["DL", "dv_bins"]), float(a.loc["CA", "dv_bins"])
    note("localisation_rmse_bins", {"TinyCNN": round(rm_dl, 3), "CA-CFAR": round(rm_ca, 3),
                                    "ratio": round(rm_dl / rm_ca, 2)}, src)
    note("localisation_doppler_bins", {"TinyCNN": round(dv_dl, 3),
                                       "CA-CFAR": round(dv_ca, 3)}, src)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    a1 = axes[0]
    vals = [rm_ca, rm_dl]
    a1.bar([0, 1], vals, 0.5, color=[CA, DL], zorder=3)
    for i, v in enumerate(vals):
        a1.text(i, v + 0.035, "%.3f" % v, ha="center", fontsize=BASE + 4,
                fontweight="bold", color=INK)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["CA-CFAR", "TinyCNN"], fontsize=BASE + 2)
    a1.set_ylabel("combined RMSE [bins]")
    a1.set_ylim(0, 1.16)
    clean(a1, grid="y")
    a1.set_title("Localisation error", fontsize=BASE + 2, fontweight="bold", pad=12)
    a1.annotate("", xy=(1, rm_dl), xytext=(1, rm_ca),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=2.4))
    a1.text(1.12, (rm_dl + rm_ca) / 2, "%.2f× worse" % (rm_dl / rm_ca),
            fontsize=BASE + 1, fontweight="bold", color=INK, va="center")

    a2 = axes[1]
    a2.bar([0, 1], [dv_ca, dv_dl], 0.5, color=[CA, DL], zorder=3)
    for i, v in enumerate([dv_ca, dv_dl]):
        a2.text(i, v + 0.014, "%.3f" % v, ha="center", fontsize=BASE + 2,
                fontweight="bold", color=INK)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["CA-CFAR", "TinyCNN"], fontsize=BASE)
    a2.set_ylabel("Doppler error [bins]")
    a2.set_ylim(0, 0.47)
    clean(a2, grid="y")
    a2.set_title("Doppler axis", fontsize=BASE + 2, fontweight="bold", pad=12)

    fig.suptitle("Better detection, looser placement", fontsize=BASE + 4,
                 fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "f09_localisation")


# ==================================================================== Figure 10
def f10_attribution_summary():
    e2s = os.path.join(ATT, "e2", "deltas.csv")
    e3s = os.path.join(ATT, "e3", "epoch_curve.csv")
    e4s = os.path.join(ATT, "e4", "deltas.csv")

    e2 = pd.read_csv(e2s)
    e2 = e2[e2.gate == 1]
    uni = float(e2[e2.statistic == "uniform"].delta_integration.mean())
    psf = float(e2[e2.statistic == "psf"].delta_integration.mean())
    note("e2_mean_delta_pd_gate1", {"uniform": round(uni, 4), "psf": round(psf, 4)}, e2s)

    # Pooled over conditions at the frozen epoch -- the same row the thesis quotes.
    e3 = pd.read_csv(e3s)
    e3 = e3[(e3.gate == 1) & (e3.epoch == 21)].set_index("arm")["dv_bins"]
    flat, gauss, point = (float(e3["flat3x3"]), float(e3["gaussian"]),
                          float(e3["point"]))
    note("e3_pooled_doppler_error_bins_gate1_epoch21",
         {"flat 3x3": round(flat, 4), "gaussian": round(gauss, 4),
          "point": round(point, 4)}, e3s)

    e4 = pd.read_csv(e4s)
    d1 = float(e4[e4.gate == 1].delta_pd.mean())
    d0 = float(e4[e4.gate == 0].delta_pd.mean())
    note("e4_mean_delta_pd", {"gate +/-1": round(d1, 4), "exact cell": round(d0, 4)}, e4s)

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.2))

    # --- E1: integrated CFAR
    a = axes[0]
    a.barh([1, 0], [uni, psf], 0.5, color=[CA, "#d98880"], zorder=3)
    for yy, v in ((1, uni), (0, psf)):
        a.text(v + 0.0018, yy, "%+.4f" % v, va="center", ha="left",
               fontsize=BASE, fontweight="bold", color=INK)
    a.set_yticks([1, 0])
    a.set_yticklabels(["uniform 3×3", "PSF-weighted"], fontsize=BASE - 1)
    a.set_xlim(-0.048, 0.012)
    a.axvline(0, color=INK, lw=2.0)
    a.set_xlabel("mean $\\Delta P_d$ given to CFAR", fontsize=BASE - 2)
    clean(a, grid="x")
    a.text(0.010, 1.45, "1   Integrated CFAR — giving CFAR the neighbouring "
                        "target energy made it WORSE",
           fontsize=BASE, fontweight="bold", color=INK, ha="right")

    # --- E2: label ablation
    b = axes[1]
    b.barh([2, 1, 0], [flat, gauss, point], 0.5,
           color=[DL, "#6ba3d6", "#9dc3e6"], zorder=3)
    for yy, v in ((2, flat), (1, gauss), (0, point)):
        b.text(v + 0.008, yy, "%.4f" % v, va="center", fontsize=BASE,
               fontweight="bold", color=INK)
    b.set_yticks([2, 1, 0])
    b.set_yticklabels(["flat 3×3\n(frozen arm)", "Gaussian", "point"],
                      fontsize=BASE - 2)
    b.set_xlim(0, 0.50)
    b.set_xlabel("pooled Doppler error [bins]", fontsize=BASE - 2)
    clean(b, grid="x")
    b.text(0.495, 2.55, "2   Label ablation — the placement deficit came "
                        "largely from the 3×3 supervision",
           fontsize=BASE, fontweight="bold", color=INK, ha="right")

    # --- E3: guard masking
    c = axes[2]
    c.barh([1, 0], [d1, d0], 0.5, color=[DL, CA], zorder=3)
    c.text(d1 + 0.006, 1, "%+.4f" % d1, va="center", fontsize=BASE,
           fontweight="bold", color=INK)
    c.text(d0 + 0.006, 0, "%+.4f" % d0, va="center", fontsize=BASE,
           fontweight="bold", color=INK)
    c.set_yticks([1, 0])
    c.set_yticklabels(["$\\pm1$ bin", "exact cell"], fontsize=BASE - 1)
    c.set_xlim(-0.20, 0.09)
    c.axvline(0, color=INK, lw=2.0)
    c.set_xlabel("mean $\\Delta P_d$ from guard-cell access", fontsize=BASE - 2)
    clean(c, grid="x")
    c.text(0.085, 1.45, "3   Guard masking — a small tolerant-detection gain, "
                        "a large exact-cell cost",
           fontsize=BASE, fontweight="bold", color=INK, ha="right")

    fig.suptitle("Not simply more energy — learned local context "
                 "and supervision matter",
                 fontsize=BASE + 4, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout(h_pad=2.6)
    save(fig, "f10_attribution_summary")


# ==================================================================== Figure 11
def f11_domain_gap():
    a = jload(os.path.join(R0, "r0_audit.json"))
    fp = pd.read_csv(os.path.join(R0, "r0_object_footprint.csv"))
    med = fp.groupby("class")[["range_extent", "doppler_extent"]].median()
    note("carrada_median_support_bins",
         {k: [int(v.range_extent), int(v.doppler_extent)]
          for k, v in med.iterrows()}, os.path.join(R0, "r0_object_footprint.csv"))

    sim, car = a["background_simulator"], a["background_carrada"]
    rid = a["ridge"]

    def m(d, k):
        return d[k]["mean"] if isinstance(d[k], dict) else d[k]

    sup = " / ".join("%d×%d" % (med.loc[c, "range_extent"],
                                     med.loc[c, "doppler_extent"])
                     for c in ("pedestrian", "cyclist", "car"))
    rows = [
        ("Effective Gamma shape $k$", "%.3f" % m(sim, "gamma_k_moment"),
         "%.3f" % m(car, "gamma_k_moment"), "smoother"),
        ("Background spread [dB]", "%.3f" % m(sim, "db_std"),
         "%.3f" % m(car, "db_std"), "less speckle"),
        ("Lag-1 correlation, range", "%.3f" % m(sim, "lag1_range"),
         "%.3f" % m(car, "lag1_range"), "essentially the same"),
        ("Lag-1 correlation, Doppler", "%.3f" % m(sim, "lag1_doppler"),
         "%.3f" % m(car, "lag1_doppler"), "very different"),
        ("Zero-Doppler ridge",
         "%.1f dB, ±%d bins" % (rid["simulator_ridge_peak_over_floor_db"],
                                     rid["ridge_half_width_simulator"]),
         "%.1f dB, ±%d bin" % (rid["carrada_ridge_peak_over_floor_db"],
                                    rid["ridge_half_width_carrada"]), "sharper"),
        ("Target representation", "point target", "extended object", "different truth"),
        ("Median support [bins]", "1 × 1", sup, "ped / cyc / car"),
    ]
    note("domain_audit_rows", [[r[0], r[1], r[2]] for r in rows],
         os.path.join(R0, "r0_audit.json"))

    fig, ax = plt.subplots(figsize=FULL)
    blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.40, 0.925, "SYNTHETIC", ha="center", fontsize=BASE + 2,
            fontweight="bold", color=DL)
    ax.text(0.68, 0.925, "CARRADA", ha="center", fontsize=BASE + 2,
            fontweight="bold", color=CA)
    for k, (lab, sv, cv, tag) in enumerate(rows):
        y = 0.845 - 0.103 * k
        if k % 2 == 0:
            ax.add_patch(plt.Rectangle((0.01, y - 0.040), 0.98, 0.082,
                                       facecolor="#f4f6f7", edgecolor="none",
                                       zorder=0))
        ax.text(0.02, y, lab, va="center", fontsize=BASE, color=INK)
        ax.text(0.40, y, sv, va="center", ha="center", fontsize=BASE + 1,
                color=DL, fontweight="bold")
        ax.text(0.68, y, cv, va="center", ha="center", fontsize=BASE + 1,
                color=CA, fontweight="bold")
        ax.text(0.972, y, tag, va="center", ha="right", fontsize=BASE - 3,
                color=GREY, style="italic")
    ax.text(0.5, 0.045, "CARRADA is different, not simply harder.", ha="center",
            fontsize=BASE + 4, fontweight="bold", color=INK)
    ax.text(0.5, 1.0, "What the domain audit measured", ha="center",
            fontsize=BASE + 4, fontweight="bold", color=INK)
    save(fig, "f11_domain_gap")


# ==================================================================== Figure 12
def f12_zeroshot_all_detectors():
    r = jload(ZS)
    arms = [("TinyCNN", r["primary"]["tinycnn_seed_mean"],
             r["unmatched_burden"]["TinyCNN"]["mean"], DL, "s"),
            ("CA-CFAR", r["primary"]["cfar"]["CA"],
             r["unmatched_burden"]["CFAR-CA"]["mean"], CA, "o"),
            ("GO-CFAR", r["primary"]["cfar"]["GO"],
             r["unmatched_burden"]["CFAR-GO"]["mean"], GO, "o"),
            ("SO-CFAR", r["primary"]["cfar"]["SO"],
             r["unmatched_burden"]["CFAR-SO"]["mean"], SO, "o"),
            ("OS-CFAR", r["primary"]["cfar"]["OS"],
             r["unmatched_burden"]["CFAR-OS"]["mean"], OS, "o")]
    note("zeroshot_recall_burden",
         {n: [round(rc, 4), round(bd, 2)] for n, rc, bd, _, _ in arms}, ZS)

    off = {"TinyCNN": (0, -46), "CA-CFAR": (0, 26), "GO-CFAR": (0, -46),
           "SO-CFAR": (-34, 22), "OS-CFAR": (30, -46)}
    fig, ax = plt.subplots(figsize=FULL)
    for n, rc, bd, c, mk in arms:
        ax.scatter(bd, rc, s=420, color=c, marker=mk, edgecolor="white",
                   lw=2.4, zorder=5)
        dx, dy = off[n]
        ax.annotate("%s\n%.4f  |  %.2f" % (n, rc, bd), (bd, rc),
                    textcoords="offset points", xytext=(dx, dy), ha="center",
                    fontsize=BASE - 1, color=c, fontweight="bold")
    ax.set_xlim(7, 38)
    ax.set_ylim(0.843, 0.998)
    ax.set_xlabel("unmatched detections per frame   (output burden — not $P_{fa}$)")
    ax.set_ylabel("exact dense-mask recall")
    clean(ax, grid="both")
    ax.annotate("", xy=(36, 0.851), xytext=(9, 0.851),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.2))
    ax.text(22.5, 0.8455, "more output emitted", ha="center", fontsize=BASE - 3,
            color=GREY)
    ax.annotate("CA:\nlowest burden", xy=(12.28, 0.8838), xytext=(10.6, 0.930),
                fontsize=BASE - 2, color=CA, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=CA, lw=2.0))
    ax.annotate("SO / OS:\nhighest recall,\ndensest output", xy=(29.9, 0.9665),
                xytext=(20.0, 0.947), fontsize=BASE - 2, color=INK,
                fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0))
    ax.set_title("Zero-shot on CARRADA — the synthetic advantage disappears",
                 fontweight="bold", pad=16)
    footnote(fig, "TinyCNN does not dominate the classical family: "
                  "lower recall than CA-CFAR, and twice its output.",
             colour=INK, size=BASE - 2)
    save(fig, "f12_zeroshot_all_detectors")


# ==================================================================== Figure 13
def f13_zeroshot_primary():
    r = jload(ZS)
    dl = r["primary"]["tinycnn_seed_mean"]
    ca = r["primary"]["cfar"]["CA"]
    pb = r["paired_bootstrap"]
    bdl = r["unmatched_burden"]["TinyCNN"]["mean"]
    bca = r["unmatched_burden"]["CFAR-CA"]["mean"]
    note("zeroshot_primary",
         {"TinyCNN_recall": round(dl, 4), "CA_recall": round(ca, 4),
          "seq_balanced_delta": round(pb["mean_delta"], 4),
          "ci": [round(pb["ci_lo"], 4), round(pb["ci_hi"], 4)],
          "TinyCNN_burden": round(bdl, 2), "CA_burden": round(bca, 2),
          "burden_ratio": round(bdl / bca, 2)}, ZS)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    a = axes[0]
    a.bar([0, 1], [ca, dl], 0.5, color=[CA, DL], zorder=3)
    for i, v in enumerate([ca, dl]):
        a.text(i, v + 0.004, "%.4f" % v, ha="center", fontsize=BASE + 3,
               fontweight="bold", color=INK)
    a.set_xticks([0, 1])
    a.set_xticklabels(["CA-CFAR", "TinyCNN"], fontsize=BASE + 2)
    a.set_ylim(0.80, 0.93)
    a.set_ylabel("exact dense-mask recall")
    clean(a, grid="y")
    a.set_title("Recall — statistically comparable", fontsize=BASE + 2,
                fontweight="bold", pad=12)
    a.text(0.5, 0.815, "difference %+.4f\n95%% CI [%+.4f, %+.4f]"
           % (pb["mean_delta"], pb["ci_lo"], pb["ci_hi"]),
           ha="center", fontsize=BASE - 1, color=INK,
           bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2f6fa",
                     edgecolor=GREY, lw=1.4))

    b = axes[1]
    b.bar([0, 1], [bca, bdl], 0.5, color=[CA, DL], zorder=3)
    for i, v in enumerate([bca, bdl]):
        b.text(i, v + 0.45, "%.2f" % v, ha="center", fontsize=BASE + 3,
               fontweight="bold", color=INK)
    b.set_xticks([0, 1])
    b.set_xticklabels(["CA-CFAR", "TinyCNN"], fontsize=BASE + 2)
    b.set_ylim(0, 28)
    b.set_ylabel("unmatched detections / frame")
    clean(b, grid="y")
    b.set_title("Output burden — %.1f× more" % (bdl / bca),
                fontsize=BASE + 2, fontweight="bold", pad=12)

    fig.suptitle("Comparable recall, ~%.1f× more unmatched output"
                 % (bdl / bca), fontsize=BASE + 4, fontweight="bold",
                 color=INK, y=1.03)
    fig.tight_layout()
    footnote(fig, "Unmatched detections are an output-burden measure. "
                  "CARRADA cannot support a false-alarm rate.")
    save(fig, "f13_zeroshot_primary")


# ==================================================================== Figure 14
def f14_domain_alignment():
    src = os.path.join(CIR, "test_analysis_r5.json")     # the thesis headline set
    t = jload(src)
    A, B, C = (t["pooled_exact_recall"]["A"], t["pooled_exact_recall"]["B"],
               t["pooled_exact_recall"]["C"])
    bA, bB, bC = (t["unmatched_per_frame"]["A"]["mean"],
                  t["unmatched_per_frame"]["B"]["mean"],
                  t["unmatched_per_frame"]["C"]["mean"])
    p = t["primary_B_minus_A"]
    nr = {k: t["by_range"][k]["10-60"]["recall"] for k in ("A", "B", "C")}
    note("stage4_test_recall_burden",
         {"original synthetic": [round(A, 4), round(bA, 2)],
          "CARRADA-informed": [round(B, 4), round(bB, 2)],
          "CA-CFAR": [round(C, 4), round(bC, 2)],
          "B_minus_A": round(p["mean_delta"], 4),
          "ci": [round(p["ci_lo"], 4), round(p["ci_hi"], 4)]}, src)
    note("stage4_near_range_10_60", {k: round(v, 4) for k, v in nr.items()}, src)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    a = axes[0]
    labels = ["original\nsynthetic", "CARRADA-informed\nsynthetic", "CA-CFAR"]
    cols = [DL, DLI, CA]
    a.bar([0, 1, 2], [A, B, C], 0.52, color=cols, zorder=3)
    for i, v in enumerate([A, B, C]):
        a.text(i, v + 0.004, "%.4f" % v, ha="center", fontsize=BASE + 2,
               fontweight="bold", color=INK)
    for i, v in enumerate([bA, bB, bC]):
        a.text(i, 0.832, "burden %.2f" % v, ha="center", fontsize=BASE - 2,
               color=INK if i != 1 else CA,
               fontweight="bold" if i == 1 else "normal")
    a.set_xticks([0, 1, 2])
    a.set_xticklabels(labels, fontsize=BASE - 1)
    a.set_ylim(0.82, 0.975)
    a.set_ylabel("exact recall, CARRADA test")
    clean(a, grid="y")
    a.text(-0.42, 0.9665, "%+.4f\nCI [%+.4f, %+.4f]"
           % (p["mean_delta"], p["ci_lo"], p["ci_hi"]),
           ha="left", va="top", fontsize=BASE - 4, color=DLI, fontweight="bold")
    a.annotate("", xy=(0.86, 0.9425), xytext=(0.14, 0.8775),
               arrowprops=dict(arrowstyle="-|>", color=DLI, lw=3.0))
    a.set_title("Recall recovers", fontsize=BASE + 2, fontweight="bold", pad=12)

    b = axes[1]
    b.bar([0, 1, 2], [bA, bB, bC], 0.52, color=cols, zorder=3)
    for i, v in enumerate([bA, bB, bC]):
        b.text(i, v * 1.06, "%.1f" % v, ha="center", fontsize=BASE + 1,
               fontweight="bold", color=INK)
    b.set_yscale("log")
    b.set_ylim(7, 260)
    b.set_xticks([0, 1, 2])
    b.set_xticklabels(["orig.", "informed", "CA"], fontsize=BASE - 1)
    b.set_ylabel("unmatched det. / frame")
    clean(b, grid="y")
    b.set_title("… output density explodes", fontsize=BASE + 2,
                fontweight="bold", pad=12)

    fig.suptitle("Matching the training background recovered recall — "
                 "at a heavy output cost",
                 fontsize=BASE + 3, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    footnote(fig, "Near range (10–60 bins) recovers too: %.4f → %.4f, "
                  "against CA-CFAR's %.4f." % (nr["A"], nr["B"], nr["C"]))
    save(fig, "f14_domain_alignment")


# ==================================================================== Figure 15
def f15_real_adaptation():
    src = os.path.join(FTR, "validation_analysis.json")
    v = jload(src)
    R = v["pooled_exact_recall"]
    Bd = v["unmatched_per_frame"]
    pts = {k: (Bd[k]["mean"], R[k]) for k in ("A", "B", "C", "D")}
    p = v["primary_C_minus_D"]
    ad = v["adaptation_C_minus_B"]
    note("stage5_validation_recall_burden",
         {"original synthetic": [round(R["A"], 4), round(Bd["A"]["mean"], 2)],
          "CARRADA-informed": [round(R["B"], 4), round(Bd["B"]["mean"], 2)],
          "fine-tuned": [round(R["C"], 4), round(Bd["C"]["mean"], 2)],
          "CA-CFAR": [round(R["D"], 4), round(Bd["D"]["mean"], 2)]}, src)
    note("stage5_deltas",
         {"C_minus_D": [round(p["mean_delta"], 4), round(p["ci_lo"], 4),
                        round(p["ci_hi"], 4)],
          "C_minus_B": [round(ad["mean_delta"], 4), round(ad["ci_lo"], 4),
                        round(ad["ci_hi"], 4)]}, src)

    fig, ax = plt.subplots(figsize=FULL)
    style = {"A": ("original synthetic", DL, "o"),
             "B": ("CARRADA-informed", DLI, "o"),
             "C": ("+ real fine-tuning", DLF, "s"),
             "D": ("CA-CFAR", CA, "D")}
    for k, (lbl, c, mk) in style.items():
        x, y = pts[k]
        ax.scatter(x, y, s=470 if k == "C" else 380, color=c, marker=mk,
                   edgecolor="white", lw=2.6, zorder=6)
    for a, b_ in (("A", "B"), ("B", "C")):
        ax.add_patch(FancyArrowPatch(pts[a], pts[b_], arrowstyle="-|>",
                                     mutation_scale=26, lw=2.6, color=GREY,
                                     connectionstyle="arc3,rad=0.12",
                                     shrinkA=22, shrinkB=22, zorder=3))
    ax.annotate("original synthetic\n%.4f  |  %.2f" % (R["A"], Bd["A"]["mean"]),
                pts["A"], textcoords="offset points", xytext=(4, -52),
                ha="center", fontsize=BASE - 1, color=DL, fontweight="bold")
    ax.annotate("CARRADA-informed\n%.4f  |  %.2f" % (R["B"], Bd["B"]["mean"]),
                pts["B"], textcoords="offset points", xytext=(-6, 30),
                ha="center", fontsize=BASE - 1, color=DLI, fontweight="bold")
    ax.annotate("+ real fine-tuning\n%.4f  |  %.2f" % (R["C"], Bd["C"]["mean"]),
                pts["C"], textcoords="offset points", xytext=(30, 30),
                ha="center", fontsize=BASE - 1, color=DLF, fontweight="bold")
    ax.annotate("CA-CFAR\n%.4f  |  %.2f" % (R["D"], Bd["D"]["mean"]),
                pts["D"], textcoords="offset points", xytext=(96, -4),
                ha="center", va="center", fontsize=BASE - 1, color=CA,
                fontweight="bold")
    ax.axhline(R["D"], color=CA, lw=1.8, ls="--", alpha=0.65, zorder=1)
    ax.axvline(Bd["D"]["mean"], color=CA, lw=1.8, ls="--", alpha=0.65, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(7, 260)
    ax.set_ylim(0.855, 0.995)
    ax.set_xlabel("unmatched detections per frame   (output burden — not $P_{fa}$)")
    ax.set_ylabel("exact recall, CARRADA validation")
    clean(ax, grid="both")
    ax.text(0.985, 0.03, "better → up and to the LEFT", transform=ax.transAxes,
            ha="right", fontsize=BASE - 2, color=GREY, style="italic")
    ax.set_title("Real-domain adaptation improved selectivity",
                 fontweight="bold", pad=16)
    footnote(fig, "The final operating point was also selected on held-out CARRADA "
                  "TRAIN sequences under a burden constraint —\nweight fine-tuning "
                  "alone does not account for the whole burden reduction. "
                  "C $-$ B recall: %+.4f, CI [%+.4f, %+.4f]."
                  % (ad["mean_delta"], ad["ci_lo"], ad["ci_hi"]),
             colour=INK, size=BASE - 4)
    save(fig, "f15_real_adaptation")


# ==================================================================== Figure 16
def f16_five_stage_story():
    fig, ax = plt.subplots(figsize=FULL)
    blank(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    stages = [
        ("1  Controlled synthetic", "TinyCNN wins", DL),
        ("2  Attribution", "the gain is not simple energy integration", DL),
        ("3  Zero-shot real data", "the advantage disappears", CA),
        ("4  Domain-informed synthetic", "recall returns — burden becomes very high",
         DLI),
        ("5  Real fine-tuning + target-domain calibration",
         "better recall / burden trade-off than CA-CFAR", DLF),
    ]
    for k, (title, sub, c) in enumerate(stages):
        y = 0.845 - 0.145 * k
        ax.add_patch(plt.Rectangle((0.03, y - 0.052), 0.012, 0.104, facecolor=c,
                                   edgecolor="none"))
        ax.text(0.065, y + 0.024, title, va="center", fontsize=BASE + 2,
                fontweight="bold", color=c)
        ax.text(0.065, y - 0.026, sub, va="center", fontsize=BASE, color=INK)
        if k < 4:
            ax.add_patch(FancyArrowPatch((0.036, y - 0.056), (0.036, y - 0.088),
                                         arrowstyle="-|>", mutation_scale=18,
                                         lw=2.0, color=GREY))
    ax.add_patch(plt.Rectangle((0.02, 0.015), 0.96, 0.155, facecolor="#f2f4f6",
                               edgecolor=INK, lw=2.0))
    ax.text(0.5, 0.130, "CFAR  =  stronger training-free baseline", ha="center",
            fontsize=BASE + 1, fontweight="bold", color=CA)
    ax.text(0.5, 0.078, "Learned detector  =  higher observed ceiling",
            ha="center", fontsize=BASE + 1, fontweight="bold", color=DLF)
    ax.text(0.5, 0.036, "when the deployment domain is represented in learning",
            ha="center", fontsize=BASE - 2, color=INK)
    ax.text(0.5, 1.0, "Five stages, five answers", ha="center",
            fontsize=BASE + 5, fontweight="bold", color=INK)
    save(fig, "f16_five_stage_story")


def main():
    print("figures:")
    for f in (f06_synthetic_headline, f07_roc_operating_point, f08_pd_vs_snr,
              f09_localisation, f10_attribution_summary, f11_domain_gap,
              f12_zeroshot_all_detectors, f13_zeroshot_primary,
              f14_domain_alignment, f15_real_adaptation, f16_five_stage_story):
        f()
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "values.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(USED, fh, indent=1)
    print("   wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
