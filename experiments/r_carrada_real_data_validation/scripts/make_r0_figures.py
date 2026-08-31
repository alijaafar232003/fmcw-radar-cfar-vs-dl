"""R0 figures. Reads only the cached statistics written by run_r0_audit.py, plus a small
deterministic sample of RD frames for the illustrative panels.

READ-ONLY with respect to CARRADA, the simulator and every frozen asset.
Writes only into experiments/r_carrada_real_data_validation/figures/.
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
sys.path.insert(0, os.path.join(EXP, "src"))
import r0_background as bg  # noqa: E402

CARRADA = r"C:/Users/USER/Desktop/CARRADA/Carrada"
META = os.path.join(ROOT, "data", "carrada_meta")
RESULTS = os.path.join(EXP, "results")
FIGS = os.path.join(EXP, "figures")
SIM_SPLIT = "bench2_val"
SEED = 20260831

C_CAR, C_SIM, C_REF = "#C1440E", "#1F6FB4", "#555555"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def load():
    rep = json.load(open(os.path.join(RESULTS, "r0_audit.json")))
    dfc = pd.read_csv(os.path.join(RESULTS, "r0_background_carrada_frames.csv"))
    dfs = pd.read_csv(os.path.join(RESULTS, "r0_background_simulator_frames.csv"))
    fp = pd.read_csv(os.path.join(RESULTS, "r0_object_footprint.csv"))
    ridge = np.load(os.path.join(RESULTS, "r0_ridge_profiles.npz"))
    return rep, dfc, dfs, fp, ridge


# ---------------------------------------------------------------- fig 1: example RD maps
def fig_examples(rep):
    fo = json.load(open(os.path.join(META, "annotations_frame_oriented.json")))
    seq = "2019-09-16-13-18-33"
    ann = sorted(f for f in fo[seq] if len(fo[seq][f]) > 0)
    frame = ann[len(ann) // 2]
    db_c = np.load("{}/{}/range_doppler_numpy/{}.npy".format(CARRADA, seq, frame))

    fdir = os.path.join(ROOT, "data", SIM_SPLIT, "frames")
    md = pd.read_csv(os.path.join(ROOT, "data", SIM_SPLIT, "metadata.csv"))
    sname = md[md.condition == "multi"].frame_id.iloc[0]
    p_s = np.load(os.path.join(fdir, sname, "rd_map_mag.npy")).astype(np.float64) ** 2
    db_s = 10 * np.log10(np.maximum(p_s, 1e-300))

    bc = rep["background_carrada"]
    bs = rep["background_simulator"]
    notes = ("background std {:.2f} dB,  k = {:.1f}\n"
             "zero-Doppler ridge {:.0f} dB".format(
                 bc["db_std"]["mean"], bc["gamma_k_moment"]["mean"],
                 rep["ridge"]["carrada_ridge_peak_over_floor_db"]),
             "background std {:.2f} dB,  k = {:.1f}\n"
             "zero-Doppler ridge {:.0f} dB".format(
                 bs["db_std"]["mean"], bs["gamma_k_moment"]["mean"],
                 rep["ridge"]["simulator_ridge_peak_over_floor_db"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, db, title, note in (
            (axes[0], db_c, "CARRADA — real measurement\n{}  frame {}".format(seq, frame), notes[0]),
            (axes[1], db_s, "simulator — this project\n{}  (condition: multi)".format(sname), notes[1])):
        # identical display normalization: dB relative to that frame's own median
        rel = db - np.median(db)
        im = ax.imshow(rel, aspect="auto", origin="lower", cmap="magma",
                       vmin=-5, vmax=35, extent=[-13.44, 13.44, 0, 50])
        ax.set_xlabel("velocity  [m/s]")
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
        ax.text(0.02, 0.985, note, transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
                color="white",
                bbox=dict(boxstyle="round,pad=0.35", fc="black", ec="none", alpha=0.55))
        fig.colorbar(im, ax=ax, label="power above frame median  [dB]")
    axes[0].set_ylabel("range  [m]")

    for v in fo[seq][frame].values():
        c = np.array(v["range_doppler"]["dense"])
        r0, r1 = c[:, 0].min(), c[:, 0].max()
        d0, d1 = c[:, 1].min(), c[:, 1].max()
        axes[0].add_patch(plt.Rectangle(
            ((d0 - 32.5) * 0.42013, r0 * 0.1952),
            (d1 - d0 + 1) * 0.42013, (r1 - r0 + 1) * 0.1952,
            fill=False, edgecolor="#39FF14", lw=1.2, label="annotated object support"))
    h, l = axes[0].get_legend_handles_labels()
    if h:
        axes[0].legend(h[:1], l[:1], loc="lower right", fontsize=7.5,
                       facecolor="black", edgecolor="none", labelcolor="white", framealpha=0.55)

    fig.suptitle("Range-Doppler maps, identical axes and identical display normalization "
                 "(dB above each frame's own median)", fontsize=10.5, y=1.02)
    fig.text(0.5, 0.955, "ILLUSTRATIVE examples, manually selected — not statistical evidence",
             ha="center", fontsize=8, style="italic", color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "r0_fig1_rd_examples.png")


# -------------------------------------- fig 1b: same comparison, dense-cell overlay variant
def fig_examples_dense_overlay(rep):
    """Companion to fig 1. Same two frames, but the CARRADA annotation is drawn cell-by-cell
    rather than as a bounding box, so the true shape of the dense segmentation support is
    visible. Both variants are kept: the box version reads more cleanly, this one shows that
    the support is an irregular blob rather than a rectangle."""
    fo = json.load(open(os.path.join(META, "annotations_frame_oriented.json")))
    seq = "2019-09-16-13-18-33"
    ann = sorted(f for f in fo[seq] if len(fo[seq][f]) > 0)
    frame = ann[len(ann) // 2]
    db_c = np.load("{}/{}/range_doppler_numpy/{}.npy".format(CARRADA, seq, frame))

    fdir = os.path.join(ROOT, "data", SIM_SPLIT, "frames")
    md = pd.read_csv(os.path.join(ROOT, "data", SIM_SPLIT, "metadata.csv"))
    sname = md[md.condition == "multi"].frame_id.iloc[0]
    p_s = np.load(os.path.join(fdir, sname, "rd_map_mag.npy")).astype(np.float64) ** 2
    db_s = 10 * np.log10(np.maximum(p_s, 1e-300))
    lab = np.load(os.path.join(fdir, sname, "label_mask.npy"))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, db, title in (
            (axes[0], db_c, "CARRADA — real measurement\n{}  frame {}".format(seq, frame)),
            (axes[1], db_s, "simulator — this project\n{}  (condition: multi)".format(sname))):
        im = ax.imshow(db - np.median(db), aspect="auto", origin="lower", cmap="magma",
                       vmin=-5, vmax=35, extent=[-13.44, 13.44, 0, 50])
        ax.set_xlabel("velocity  [m/s]")
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="power above frame median  [dB]")
    axes[0].set_ylabel("range  [m]")

    for v in fo[seq][frame].values():
        c = np.array(v["range_doppler"]["dense"])
        axes[0].scatter((c[:, 1] - 32) * 0.42013, c[:, 0] * 0.1952, s=2.5,
                        marker="s", facecolors="none", edgecolors="#39FF14", linewidths=0.45,
                        label="dense annotated support (per cell)")
    cells = np.argwhere(lab > 0)
    if len(cells):
        axes[1].scatter((cells[:, 1] - 32) * 0.42013, cells[:, 0] * 0.1952, s=2.5,
                        marker="s", facecolors="none", edgecolors="#39FF14", linewidths=0.45,
                        label="3x3 point-target label")
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h[:1], l[:1], loc="lower right", fontsize=7.5, facecolor="black",
                      edgecolor="none", labelcolor="white", framealpha=0.55)

    fig.suptitle("Annotation support drawn cell-by-cell: an irregular CARRADA blob "
                 "versus a 3x3 synthetic point label", fontsize=10.5, y=1.02)
    fig.text(0.5, 0.955, "ILLUSTRATIVE examples, manually selected — not statistical evidence",
             ha="center", fontsize=8, style="italic", color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "r0_fig1b_rd_examples_dense_overlay.png")


# ------------------------------------------------- fig 2: background power / dB distributions
def fig_distributions(rep):
    fo = json.load(open(os.path.join(META, "annotations_frame_oriented.json")))
    light = json.load(open(os.path.join(META, "light_dataset_frame_oriented.json")))
    half = rep["background_protocol"]["ridge_half_width_applied"]
    rng = np.random.default_rng(SEED)

    pc = []
    for s in sorted(light):
        for f in rng.choice(light[s], size=3, replace=False):
            f = str(f)
            db = np.load("{}/{}/range_doppler_numpy/{}.npy".format(CARRADA, s, f))
            p = 10.0 ** (db / 10.0)
            cells = [tuple(c) for v in fo[s].get(f, {}).values()
                     for c in v["range_doppler"]["dense"]]
            k = bg.background_mask(cells, half)
            pc.append(p[k] / p[k].mean())
    pc = np.concatenate(pc)

    fdir = os.path.join(ROOT, "data", SIM_SPLIT, "frames")
    names = sorted(os.listdir(fdir))
    ps = []
    for n in rng.choice(names, size=90, replace=False):
        p = np.load(os.path.join(fdir, n, "rd_map_mag.npy")).astype(np.float64) ** 2
        lab = np.load(os.path.join(fdir, n, "label_mask.npy"))
        k = bg.background_mask([tuple(c) for c in np.argwhere(lab > 0)], half)
        ps.append(p[k] / p[k].mean())
    ps = np.concatenate(ps)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    grid = np.linspace(0, 6, 240)
    ax.hist(pc, bins=grid, density=True, histtype="step", color=C_CAR, lw=1.6,
            label="CARRADA  ({:,} cells)".format(len(pc)))
    ax.hist(ps, bins=grid, density=True, histtype="step", color=C_SIM, lw=1.6,
            label="simulator  ({:,} cells)".format(len(ps)))
    ax.plot(grid, np.exp(-grid), color=C_REF, ls="--", lw=1.3,
            label="exponential (single-look, k = 1)")
    ax.set_xlabel("background power / frame mean   [dimensionless]")
    ax.set_ylabel("probability density")
    ax.set_title("Background power distribution", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 6)

    ax = axes[1]
    for v, c, lab in ((pc, C_CAR, "CARRADA"), (ps, C_SIM, "simulator")):
        d = 10 * np.log10(v[v > 0])
        xs = np.sort(d)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, lw=1.6, label=lab)
    e = np.random.default_rng(0).exponential(1.0, 400000)
    xs = np.sort(10 * np.log10(e[e > 0]))
    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=C_REF, ls="--", lw=1.3,
            label="exponential (k = 1)")
    ax.set_xlabel("background power relative to frame mean   [dB]")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Background in dB - the simulator sits on the reference,\n"
                 "CARRADA is visibly narrower", fontsize=9.5)
    ax.set_xlim(-25, 12)
    ax.legend(fontsize=7.5)
    fig.suptitle("Eligible background only: object support dilated {} bins, zero-Doppler band "
                 "+-{} bins and the 21x9 detector border excluded. No trimming."
                 .format(bg.OBJECT_DILATION, half), fontsize=8.5)
    save(fig, "r0_fig2_background_distributions.png")


# ------------------------------------------------------------- fig 3: domain statistics
def fig_statistics(rep):
    keys = [("cv", "CV(power)", 1.0),
            ("gamma_k_moment", "Gamma shape $k$", 1.0),
            ("db_std", "background std [dB]", 5.57),
            ("lag1_range", "lag-1 range corr.", None),
            ("lag1_doppler", "lag-1 Doppler corr.", None)]
    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.4))
    for ax, (k, title, ref) in zip(axes, keys):
        a, b = rep["background_carrada"][k], rep["background_simulator"][k]
        ax.errorbar([0], [a["mean"]], yerr=[a["se"]], fmt="o", color=C_CAR, capsize=4, ms=7)
        ax.errorbar([1], [b["mean"]], yerr=[b["se"]], fmt="s", color=C_SIM, capsize=4, ms=7)
        if ref is not None:
            ax.axhline(ref, color=C_REF, ls="--", lw=1.2)
            ax.text(-0.55, ref, "single-look reference", fontsize=6.5, color=C_REF,
                    va="bottom", ha="left")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["CARRADA", "simulator"], fontsize=8)
        ax.set_xlim(-0.6, 1.6)
        # Correlations share one honest 0..0.8 scale. Auto-scaling zoomed the range panel to a
        # 0.014-wide window, which made two statistically indistinguishable values look far
        # apart - the opposite of what the numbers say.
        if k.startswith("lag1"):
            ax.set_ylim(0, 0.8)
        else:
            lo = min(a["mean"] - a["se"], b["mean"] - b["se"], ref if ref else np.inf)
            hi = max(a["mean"] + a["se"], b["mean"] + b["se"], ref if ref else -np.inf)
            pad = 0.12 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(max(0, lo - pad), hi + pad)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("value  (mean $\\pm$ SE)")
    fig.suptitle("R0 background statistics. Uncertainty is taken ACROSS groups - {} CARRADA "
                 "sequences, {} simulator conditions - never across correlated frames."
                 .format(rep["background_carrada"]["cv"]["n_groups"],
                         rep["background_simulator"]["cv"]["n_groups"]), fontsize=9)
    save(fig, "r0_fig3_domain_statistics.png")


# ------------------------------------------------------------------ fig 4: ridge profile
def fig_ridge(rep, ridge):
    v = (np.arange(64) - bg.ZERO_DOPPLER_BIN) * 0.42013
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for prof, floor, c, lab in ((ridge["carrada"], float(ridge["carrada_floor"]), C_CAR, "CARRADA"),
                                (ridge["simulator"], float(ridge["simulator_floor"]), C_SIM,
                                 "simulator")):
        ax.plot(v, 10 * np.log10(prof / floor), color=c, lw=1.7, label=lab)
    half = rep["background_protocol"]["ridge_half_width_applied"]
    ax.axvspan(-(half + 0.5) * 0.42013, (half + 0.5) * 0.42013, color="#999999", alpha=0.18,
               label="excluded ridge band ($\\pm${} bins)".format(half))
    ax.axhline(3.0, color=C_REF, ls=":", lw=1.1, label="+3 dB band-selection threshold")
    ax.set_xlabel("velocity  [m/s]")
    ax.set_ylabel("mean background power above off-ridge floor  [dB]")
    ax.set_title("Zero-Doppler ridge: CARRADA {:.1f} dB vs simulator {:.1f} dB"
                 .format(rep["ridge"]["carrada_ridge_peak_over_floor_db"],
                         rep["ridge"]["simulator_ridge_peak_over_floor_db"]), fontsize=10)
    ax.legend(fontsize=7.5)
    save(fig, "r0_fig4_zero_doppler_ridge.png")


# ------------------------------------------------------------- fig 5: frame-median audit
def fig_frame_median(rep, dfc, dfs):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axes[0]
    ax.hist(dfc.frame_median_db - dfc.frame_median_db.mean(), bins=60, color=C_CAR, alpha=0.75,
            density=True, label="CARRADA ({} frames)".format(len(dfc)))
    ax.hist(dfs.frame_median_db - dfs.frame_median_db.mean(), bins=60, color=C_SIM, alpha=0.65,
            density=True, label="simulator ({} frames)".format(len(dfs)))
    ax.set_xlabel("frame median power, centred on each domain's own mean  [dB]")
    ax.set_ylabel("density")
    ax.set_title("Global scale varies far more in the simulator\n"
                 "(sd {:.2f} dB) than in CARRADA (sd {:.2f} dB)"
                 .format(dfs.frame_median_db.std(), dfc.frame_median_db.std()), fontsize=9)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    order = dfc.groupby("sequence").frame_median_db.median().sort_values().index
    ax.boxplot([dfc[dfc.sequence == s].frame_median_db.values for s in order],
               showfliers=False, widths=0.6)
    ax.set_xticks([]); ax.set_xlabel("CARRADA sequence  (sorted by median)")
    ax.set_ylabel("frame median power  [dB, arbitrary offset]")
    ax.set_title("Per-sequence spread of the normalizing statistic", fontsize=9)
    fig.suptitle("TinyCNN input is  10log10(P/median P) = dB - median(dB); the adapter is exact "
                 "in dB and the unknown CARRADA constant cancels.", fontsize=8.5)
    save(fig, "r0_fig5_frame_median.png")


# --------------------------------------------------------------- fig 6: object footprint
def fig_footprint(fp):
    classes = ["pedestrian", "cyclist", "car"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, col, title in ((axes[0], "range_extent", "range extent  [bins]"),
                           (axes[1], "doppler_extent", "Doppler extent  [bins]"),
                           (axes[2], "dense_cells", "annotated cells per object")):
        data = [fp[fp["class"] == c][col].values for c in classes]
        bpl = ax.boxplot(data, tick_labels=classes, showfliers=False, patch_artist=True, widths=0.55)
        for patch, c in zip(bpl["boxes"], ["#7FB069", "#E6AA68", "#C1440E"]):
            patch.set_facecolor(c); patch.set_alpha(0.65)
        ax.set_title(title, fontsize=9.5)
        for i, c in enumerate(classes):
            med = fp[fp["class"] == c][col].median()
            ax.annotate("{:.0f}".format(med), (i + 1, med), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=7.5)
    axes[0].set_ylabel("bins")
    axes[0].axhline(1, color=C_REF, ls="--", lw=1.2)
    axes[1].axhline(1, color=C_REF, ls="--", lw=1.2)
    axes[0].text(3.4, 1.4, "a synthetic point target\noccupies 1 bin", fontsize=6.8,
                 color=C_REF, ha="right")
    fig.suptitle("CARRADA annotates extended object support, not point targets "
                 "({:,} frame-level object instances)".format(len(fp)), fontsize=9.5)
    save(fig, "r0_fig6_object_footprint.png")


def main():
    os.makedirs(FIGS, exist_ok=True)
    rep, dfc, dfs, fp, ridge = load()
    fig_examples(rep)
    fig_examples_dense_overlay(rep)
    fig_distributions(rep)
    fig_statistics(rep)
    fig_ridge(rep, ridge)
    fig_frame_median(rep, dfc, dfs)
    fig_footprint(fp)
    print("all figures written to", os.path.relpath(FIGS, ROOT))


if __name__ == "__main__":
    sys.exit(main())
