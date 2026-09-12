"""Side-by-side domain comparison + plots, from the audit artefacts written by `audit_domain.py`.

Reads only the audit CSV/JSON already produced, plus a deterministic frame sample it needs for the
two PROFILE plots (mean power per Doppler bin, mean power per range bin) which are shapes rather
than scalars and so cannot be summarised in the per-frame tables.

Emits, into domain_audit/:
    domain_gap.csv            one row per statistic per domain, plus the gap
    domain_gap.md             the same as a readable table
    plots/*.png

Any number of domains can be compared; `--tags` names them in order. With `--domain carrada_train`
among them the profile sampler opens ONLY the 18 official Train sequences, same guard as the audit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
sys.path.insert(0, os.path.join(R_EXP, "src"))
sys.path.insert(0, HERE)

import r0_background as bg                                              # noqa: E402
from audit_domain import (CARRADA, META, OUT, RANGE_BLOCKS, Guard,      # noqa: E402
                          carrada_object_cells, carrada_power, sim_object_cells, sim_power)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
INK, BLUE, RED, GREY, GOLD, GREEN = "#1a1a1a", "#2c6fbb", "#c0392b", "#8a8a8a", "#d19a13", "#2e7d52"
COLOUR = {"carrada_train": INK, "sim_bench2_train": RED, "sim_bench2_val": "#e08a7a",
          "newsim": BLUE, "newsim_pilot": BLUE}
LABEL = {"carrada_train": "CARRADA train (real)", "sim_bench2_train": "original simulator",
         "sim_bench2_val": "original simulator (val)", "newsim": "CARRADA-informed simulator",
         "newsim_pilot": "CARRADA-informed simulator"}

STATS = [
    ("gamma_k_moment", "Gamma shape k (moment)", "higher = smoother"),
    ("cv", "coefficient of variation", "lower = smoother"),
    ("db_std", "background spread [dB]", "lower = smoother"),
    ("lag1_range", "lag-1 range correlation", ""),
    ("lag1_doppler", "lag-1 Doppler correlation", ""),
]


def lab(t):
    return LABEL.get(t, t)


def col(t):
    return COLOUR.get(t, GOLD)


# ------------------------------------------------------------------ profiles (shapes, not scalars)
def profiles(tag, n=200, seed=20260912):
    """Mean background power per Doppler bin and per range bin, averaged over a frame sample."""
    rng = np.random.default_rng(seed)
    dop, rng_prof = [], []

    if tag == "carrada_train":
        splits = json.load(open(os.path.join(
            R_EXP, "configs", "official_splits.json")))["splits"]
        allowed = splits["Train"]
        guard = Guard(allowed)
        light = json.load(open(os.path.join(META, "light_dataset_frame_oriented.json")))
        fo = json.load(open(os.path.join(META, "annotations_frame_oriented.json")))
        pool = [(s, f) for s in sorted(allowed) for f in sorted(light.get(s, {}))]
        pick = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
        for seq, fr in pick:
            p, _ = carrada_power(guard, seq, fr)
            cells = carrada_object_cells(fo, seq, fr)
            dop.append(bg.ridge_profile(p, cells))
            rng_prof.append(_range_profile(p, cells))
    else:
        fdir = (os.path.join(ROOT, "data", tag.replace("sim_", ""), "frames")
                if tag.startswith("sim_") else os.path.join(EXP, "pilot", "frames"))
        names = sorted(os.listdir(fdir))
        pick = [names[i] for i in rng.choice(len(names), size=min(n, len(names)), replace=False)]
        for nm in pick:
            p = sim_power(fdir, nm)
            cells = sim_object_cells(fdir, nm)
            dop.append(bg.ridge_profile(p, cells))
            rng_prof.append(_range_profile(p, cells))

    D = np.nanmean(np.vstack(dop), axis=0)
    R = np.nanmean(np.vstack(rng_prof), axis=0)
    return D, R


def _range_profile(power, cells, shape=(256, 64)):
    """Mean power per RANGE bin over object-free, border-valid, ridge-excluded cells."""
    keep = bg.border_mask(shape) & ~bg.ridge_mask(4, shape)
    obj = bg.object_mask_from_cells(cells, shape)
    if obj.any():
        keep &= ~bg.dilate(obj, bg.OBJECT_DILATION)
    prof = np.full(shape[0], np.nan)
    for r in range(shape[0]):
        row = keep[r, :]
        if row.sum() > 4:
            prof[r] = power[r, row].mean()
    return prof


# ------------------------------------------------------------------ table
def build_table(tags, ref="carrada_train"):
    rows = []
    frames = {t: pd.read_csv(os.path.join(OUT, "%s_frames.csv" % t)) for t in tags}
    ridge = {}
    for t in tags:
        p = os.path.join(OUT, "%s_ridge.csv" % t)
        if os.path.exists(p):
            ridge[t] = pd.read_csv(p)
    blocks = {t: pd.read_csv(os.path.join(OUT, "%s_blocks.csv" % t)) for t in tags
              if os.path.exists(os.path.join(OUT, "%s_blocks.csv" % t))}

    def add(stat, scope, vals):
        r = {"statistic": stat, "scope": scope}
        for t in tags:
            r[t] = vals.get(t, np.nan)
        if ref in vals and np.isfinite(vals.get(ref, np.nan)):
            for t in tags:
                if t == ref or not np.isfinite(vals.get(t, np.nan)):
                    continue
                r["gap_" + t] = vals[t] - vals[ref]
                r["ratio_" + t] = (vals[t] / vals[ref]) if vals[ref] else np.nan
        rows.append(r)

    # frame-level medians (robust) and means (comparable with the published R0 audit)
    for key, _, _ in STATS:
        add(key, "frame median", {t: float(frames[t][key].median()) for t in tags
                                  if key in frames[t]})
        add(key, "frame mean", {t: float(frames[t][key].mean()) for t in tags
                               if key in frames[t]})

    # within-range-block medians: local speckle, free of range non-stationarity
    for key, _, _ in STATS:
        for b0, b1 in RANGE_BLOCKS:
            b = "%d-%d" % (b0, b1)
            vals = {}
            for t in blocks:
                sub = blocks[t][blocks[t].block == b]
                if len(sub) and key in sub:
                    vals[t] = float(sub[key].median())
            if vals:
                add(key, "block %s median" % b, vals)

    # ridge
    for key in ("ridge_peak_over_floor_db", "ridge_bins_above_3db", "ridge_halfpower_bins"):
        add(key, "frame mean", {t: float(ridge[t][key].mean()) for t in ridge if key in ridge[t]})
    add("frac_frames_ridge_over_10db", "fraction",
        {t: float((ridge[t].ridge_peak_over_floor_db > 10).mean()) for t in ridge})

    # range non-stationarity: dB span of the block mean power, and the k spread across blocks
    for key, nm in (("mean_power", "range mean-power dB span"),
                    ("gamma_k_moment", "range k span (max-min)"),
                    ("db_std", "range dB-spread span (max-min)")):
        vals = {}
        for t in blocks:
            g = blocks[t].groupby("block")[key].median()
            if len(g) < 2:
                continue
            vals[t] = (10 * np.log10(g.max() / g.min())) if key == "mean_power" \
                else float(g.max() - g.min())
        if vals:
            add(nm, "across blocks", vals)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots
def plot_all(tags, pdir):
    os.makedirs(pdir, exist_ok=True)
    frames = {t: pd.read_csv(os.path.join(OUT, "%s_frames.csv" % t)) for t in tags}
    blocks = {t: pd.read_csv(os.path.join(OUT, "%s_blocks.csv" % t)) for t in tags
              if os.path.exists(os.path.join(OUT, "%s_blocks.csv" % t))}

    # 1. distributions
    fig, axes = plt.subplots(1, len(STATS), figsize=(3.0 * len(STATS), 3.0))
    for ax, (key, title, note) in zip(axes, STATS):
        lo = min(frames[t][key].quantile(0.01) for t in tags if key in frames[t])
        hi = max(frames[t][key].quantile(0.99) for t in tags if key in frames[t])
        bins = np.linspace(lo, hi, 55)
        for t in tags:
            if key not in frames[t]:
                continue
            ax.hist(frames[t][key].clip(lo, hi), bins=bins, histtype="step", lw=1.6,
                    density=True, color=col(t), label=lab(t))
        ax.set_title(title + (("\n(" + note + ")") if note else ""))
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Per-frame background statistics, identical selection protocol", y=1.04,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "fig1_distributions.png"))
    plt.close(fig)

    # 2. Doppler + range profiles
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for t in tags:
        try:
            D, R = profiles(t)
        except Exception as e:                                   # pragma: no cover
            print("  profile skipped for %s: %s" % (t, e))
            continue
        z = bg.ZERO_DOPPLER_BIN
        outside = np.ones(len(D), bool)
        outside[max(0, z - 10):z + 11] = False
        floor = np.nanmedian(D[outside])
        a.plot(np.arange(len(D)) - z, 10 * np.log10(D / floor), lw=1.6, color=col(t), label=lab(t))
        b.plot(np.arange(len(R)), 10 * np.log10(R / np.nanmax(R)), lw=1.4, color=col(t),
               label=lab(t))
    a.set_xlabel("Doppler bin relative to zero"); a.set_ylabel("mean power over floor [dB]")
    a.set_title("A.  Zero-Doppler ridge"); a.set_xlim(-16, 16)
    a.axhline(0, color=GREY, lw=0.7, ls=":")
    a.legend(fontsize=7.5, frameon=False)
    b.set_xlabel("range bin"); b.set_ylabel("mean power, relative to max [dB]")
    b.set_title("B.  Range profile (range non-stationarity)")
    for r0, r1 in RANGE_BLOCKS:
        b.axvline(r0, color=GREY, lw=0.5, ls=":")
    b.legend(fontsize=7.5, frameon=False)
    for ax in (a, b):
        ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "fig2_profiles.png"))
    plt.close(fig)

    # 3. per-range-block statistics
    keys = ("gamma_k_moment", "db_std", "lag1_range", "lag1_doppler")
    titles = ("Gamma shape k", "background spread [dB]", "lag-1 range corr", "lag-1 Doppler corr")
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2))
    order = ["%d-%d" % b for b in RANGE_BLOCKS]
    x = np.arange(len(order))
    for ax, key, title in zip(axes, keys, titles):
        for t in blocks:
            g = blocks[t].groupby("block")[key].median().reindex(order)
            ax.plot(x, g.values, "o-", lw=1.6, ms=4, color=col(t), label=lab(t))
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_title(title); ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Background statistics by range block (frozen Chapter 7 blocks)", y=1.03,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "fig3_range_blocks.png"))
    plt.close(fig)

    # 4. per-sequence / per-condition variability of the headline statistics
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2))
    for ax, key, title in zip(axes, ("gamma_k_moment", "db_std", "lag1_doppler"),
                              ("Gamma shape k", "background spread [dB]", "lag-1 Doppler corr")):
        pos = 0
        ticks, names = [], []
        for t in tags:
            if key not in frames[t]:
                continue
            g = frames[t].groupby("group")[key].mean().sort_values()
            ax.scatter(np.full(len(g), pos) + np.linspace(-0.22, 0.22, len(g)), g.values,
                       s=16, color=col(t), alpha=0.85)
            ax.hlines(g.median(), pos - 0.3, pos + 0.3, color=col(t), lw=2)
            ticks.append(pos); names.append(lab(t).replace(" (real)", "").replace(" ", "\n"))
            pos += 1
        ax.set_xticks(ticks); ax.set_xticklabels(names, fontsize=7)
        ax.set_title(title + "\nper sequence / per condition", fontsize=8.5)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "fig4_group_variability.png"))
    plt.close(fig)
    print("wrote 4 plots to", os.path.relpath(pdir, ROOT))


def _md(df):
    """Markdown table without pulling in `tabulate`; the environment stays as frozen."""
    def cell(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else "%.4f" % v
        return "" if v is None else str(v)
    head = list(df.columns)
    body = [[cell(v) for v in row] for row in df.itertuples(index=False)]
    w = [max(len(head[i]), *(len(r[i]) for r in body)) if body else len(head[i])
         for i in range(len(head))]
    out = ["| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(head)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(head))) + "|"]
    for r in body:
        out.append("| " + " | ".join(r[i].ljust(w[i]) for i in range(len(head))) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["carrada_train", "sim_bench2_train", "sim_bench2_val"])
    ap.add_argument("--ref", default="carrada_train")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    T = build_table(a.tags, a.ref)
    T.to_csv(os.path.join(OUT, "domain_gap.csv"), index=False)

    cols = ["statistic", "scope"] + list(a.tags) + \
           [c for c in T.columns if c.startswith(("gap_", "ratio_"))]
    cols = [c for c in cols if c in T.columns]
    with open(os.path.join(OUT, "domain_gap.md"), "w", encoding="utf-8") as fh:
        fh.write("# Domain gap: original simulator vs CARRADA train\n\n")
        fh.write("Reference domain: `%s`. Gap = domain - reference.\n\n" % a.ref)
        fh.write(_md(T[cols]))
    print("wrote domain_gap.csv / domain_gap.md")

    if not a.no_plots:
        plot_all(a.tags, os.path.join(OUT, "plots"))


if __name__ == "__main__":
    main()
