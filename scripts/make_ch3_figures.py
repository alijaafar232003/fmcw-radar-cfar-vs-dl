"""Generate the figures for Chapter 3 (Simulator, Radar Specification and Dataset Design).

Unlike Chapter 2, this chapter IS the specification, so concrete values belong here. Every
quantity is read from the project's own configuration or measured data.

Run:  venv/Scripts/python.exe scripts/make_ch3_figures.py
Out:  report/figures/Chapter3/
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from radar.config import RadarConfig, derive_radar_params  # noqa: E402

OUT = os.path.join(ROOT, "report", "figures", "Chapter3")
os.makedirs(OUT, exist_ok=True)

CFG = RadarConfig()
DER = derive_radar_params(CFG)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})

INK, BLUE, RED, GREY, GOLD = "#1a1a1a", "#2c6fbb", "#c0392b", "#8a8a8a", "#d19a13"

# The eight strata, in config order.
STRATA = ["isolated", "masking", "clutter", "clutter_edge",
          "multi", "noise", "mixed", "unresolved"]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


def load_meta(split="bench2_test"):
    return pd.read_csv(os.path.join(ROOT, "data", split, "metadata.csv"))


# ----------------------------------------------------------------------- Fig 3.1
def fig_simulator_chain():
    """What the simulator adds beyond the ideal chain of Chapter 2."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)

    main = [
        (3,  "Scene\ndraw", "targets, ranges,\nvelocities"),
        (26, "Chirp\nsynthesis", "TX ramp, echo\nat delay $\\tau$"),
        (49, "Dechirp\n+ 2 FFTs", "range then\nDoppler"),
        (72, "RD map", "labels and\ncell SNR"),
    ]
    w, h = 21, 16
    for x, t, s in main:
        ax.add_patch(Rectangle((x, 30), w, h, facecolor="white", edgecolor=BLUE, lw=1.5))
        ax.text(x + w / 2, 41, t, ha="center", va="center", fontsize=9, color=INK)
        ax.text(x + w / 2, 34.5, s, ha="center", va="center", fontsize=7.3, color=GREY)
    for x, _, _ in main[:-1]:
        ax.annotate("", xy=(x + w + 1.4, 38), xytext=(x + w + 0.4, 38),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))

    inject = [
        (14, "Radar equation\n$1/R^4$"),
        (37, "Thermal noise\n(calibrated)"),
        (60, "Clutter ridge\n(Weibull)"),
        (83, "Multipath\nechoes"),
    ]
    for x, t in inject:
        ax.add_patch(Rectangle((x - 9, 6), 18, 13, facecolor="#fdf3e3",
                               edgecolor=GOLD, lw=1.2))
        ax.text(x, 12.5, t, ha="center", va="center", fontsize=7.4, color=INK)
        ax.annotate("", xy=(x, 29), xytext=(x, 19.5),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))

    ax.text(50, 55, "Chapter 2 described the upper row. The lower row is what makes the "
                    "data a benchmark:",
            ha="center", fontsize=8.6, color=INK)
    ax.text(50, 50, "each effect is switched on per stratum, so a detector can be tested "
                    "against one difficulty at a time.",
            ha="center", fontsize=8.6, color=RED)
    save(fig, "ch3_fig1_simulator_chain.png")


# ----------------------------------------------------------------------- Fig 3.2
def fig_strata_montage():
    """One representative frame from each of the eight strata."""
    meta = load_meta()
    fdir = os.path.join(ROOT, "data", "bench2_test", "frames")

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.4))
    ext = [-DER.vel_max, DER.vel_max, 0, DER.range_max]

    for ax, name in zip(axes.ravel(), STRATA):
        sub = meta[meta["condition"] == name]
        if len(sub) == 0:
            ax.axis("off")
            ax.set_title(name, fontsize=8)
            continue
        # Deterministic: the median frame id of the stratum, never chosen on appearance.
        fid = sorted(sub["frame_id"])[len(sub) // 2]
        p = os.path.join(fdir, str(fid), "rd_map_mag.npy")
        if not os.path.exists(p):
            ax.axis("off"); ax.set_title(name, fontsize=8); continue
        db = 20 * np.log10(np.maximum(np.load(p), 1e-12))
        db = db - np.median(db)
        ax.imshow(db, aspect="auto", origin="lower", cmap="viridis",
                  extent=ext, vmin=-6, vmax=28)

        # Overlay the annotated truth. Most targets sit near the detection transition and are
        # not obvious by eye -- which is the point, but it makes an unmarked montage unreadable.
        lp = os.path.join(fdir, str(fid), "labels.csv")
        if os.path.exists(lp):
            try:
                lab = pd.read_csv(lp)
                if len(lab):
                    ax.scatter(lab["velocity_mps"], lab["range_m"], s=52,
                               facecolors="none", edgecolors=RED, linewidths=1.1)
            except Exception:
                pass

        n_t = int(sub[sub["frame_id"] == fid]["num_targets"].iloc[0])
        ax.set_title("{}  ({} target{})".format(
            name.replace("_", " "), n_t, "" if n_t == 1 else "s"), fontsize=8.2)
        ax.tick_params(labelsize=6.5)

    for ax in axes[1, :]:
        ax.set_xlabel("Velocity [m/s]", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Range [m]", fontsize=7.5)

    fig.tight_layout()
    fig.text(0.5, -0.03,
             "One frame per stratum, chosen deterministically as the median frame index of that "
             "stratum, never on appearance. Red circles mark\nthe annotated targets. Common "
             "colour scale, dB above each frame's own noise floor. Most targets are close to the "
             "detection\nthreshold by design, so several are not obvious by eye.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch3_fig2_strata_montage.png")


# ----------------------------------------------------------------------- Fig 3.3
def fig_psf():
    """Measured point-spread function: what one target actually occupies."""
    p = os.path.join(ROOT, "results", "psf_profile.csv")
    if not os.path.exists(p):
        print("  [skip] no psf_profile.csv")
        return
    psf = pd.read_csv(p)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for a, axis in zip(ax, ["range", "doppler"]):
        for case, ls in (("on-bin", "-"), ("off-bin", "--")):
            s = psf[(psf.axis == axis) & (psf.case == case)]
            if len(s) == 0:
                continue
            a.plot(s.offset_bins, s.rel_db, ls, lw=1.4,
                   color=BLUE if case == "on-bin" else RED, label=case)
        a.axhline(-3, color=GREY, lw=0.9, ls=":")
        a.text(0.02, 0.06, "$-3$ dB", transform=a.transAxes, fontsize=7.4, color=GREY)
        a.set_xlim(-8, 8)
        a.set_ylim(-45, 3)
        a.set_xlabel("Offset from the peak [bins]")
        a.set_title(axis.capitalize() + " axis")
        a.legend(frameon=False, fontsize=7.5)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Relative power [dB]")

    fig.tight_layout()
    fig.text(0.5, -0.05,
             "Measured by placing an isolated target and reading the response. "
             "The mainlobe spans about two bins, which is what sets the guard-cell count.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch3_fig3_psf.png")


# ----------------------------------------------------------------------- Fig 3.4
def fig_snr_distribution():
    """Where the difficulty sits: cell SNR across the test split."""
    lab = []
    fdir = os.path.join(ROOT, "data", "bench2_test", "frames")
    for nm in sorted(os.listdir(fdir)):
        p = os.path.join(fdir, nm, "labels.csv")
        if os.path.exists(p):
            try:
                lab.append(pd.read_csv(p))
            except Exception:
                pass
    if not lab:
        print("  [skip] no labels for fig 3.4")
        return
    L = pd.concat(lab, ignore_index=True)
    col = "cell_snr_db" if "cell_snr_db" in L.columns else None
    if col is None:
        print("  [skip] no cell_snr_db column")
        return
    # The `noise` stratum has empty label files, so the concatenated column can be object dtype.
    v = pd.to_numeric(L[col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.hist(v, bins=60, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    for q, c in ((10, GREY), (50, RED), (90, GREY)):
        x = np.percentile(v, q)
        ax.axvline(x, color=c, lw=1.2, ls="--" if q != 50 else "-")
        ax.text(x, ax.get_ylim()[1] * 0.94, " {}th\n {:.1f} dB".format(q, x),
                fontsize=7.2, color=c, va="top")
    ax.set_xlabel("Cell SNR at the detector [dB]")
    ax.set_ylabel("Targets")
    ax.set_title("Cell SNR spans the detection transition rather than sitting above it")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.text(0.5, -0.13,
             "{:,} target instances across the test split. Cell SNR is measured per target after "
             "propagation, not requested at\nthe scene level, so the spread reflects range and "
             "amplitude as well as the configured scene SNR.".format(len(v)),
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch3_fig4_snr_distribution.png")


def main():
    print("Chapter 3 figures ->", os.path.relpath(OUT, ROOT))
    fig_simulator_chain()
    fig_strata_montage()
    fig_psf()
    fig_snr_distribution()
    return 0


if __name__ == "__main__":
    sys.exit(main())
