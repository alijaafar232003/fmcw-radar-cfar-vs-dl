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


# --------------------------------------------------------------------------- frame selection
# The rule below is the one already used in notebooks/dataset_overview.ipynb (`pick`): the frame
# of a condition whose target count is closest to that condition's median target count, ties
# broken by the first in sorted order. It depends only on target count, never on appearance or
# on how any detector performed, and Figures 3.2 and 3.3 both use it so they show the same
# frames.
def pick_frame(meta, condition):
    sub = meta[meta["condition"] == condition]
    if len(sub) == 0:
        return None
    typical = int(np.median(sub["num_targets"]))
    sub = sub.assign(_d=(sub["num_targets"] - typical).abs())
    return sub.sort_values(["_d", "frame_id"]).iloc[0]["frame_id"]


def rd_db(fdir, fid):
    """RD power in dB above that frame's own median (noise-floor) level."""
    mag = np.load(os.path.join(fdir, str(fid), "rd_map_mag.npy"))
    p = mag ** 2
    return 10.0 * np.log10(p / np.median(p) + 1e-12)


def labels_of(fdir, fid):
    p = os.path.join(fdir, str(fid), "labels.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


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
        fid = pick_frame(meta, name)
        if fid is None or not os.path.exists(
                os.path.join(fdir, str(fid), "rd_map_mag.npy")):
            ax.axis("off"); ax.set_title(name, fontsize=8); continue

        db = rd_db(fdir, fid)
        ax.imshow(db, aspect="auto", origin="lower", cmap="viridis",
                  extent=ext, vmin=0, vmax=30)

        # Overlay the annotated truth. Most targets sit near the detection transition and are
        # not obvious by eye -- which is the point, but it makes an unmarked montage unreadable.
        lab = labels_of(fdir, fid)
        if len(lab):
            ax.scatter(lab["velocity_mps"], lab["range_m"], s=52,
                       facecolors="none", edgecolors=RED, linewidths=1.1)

        n_t = int(meta[meta["frame_id"] == fid]["num_targets"].iloc[0])
        ax.set_title("{}  ({} target{})".format(
            name.replace("_", " "), n_t, "" if n_t == 1 else "s"), fontsize=8.2)
        ax.tick_params(labelsize=6.5)

    for ax in axes[1, :]:
        ax.set_xlabel("Velocity [m/s]", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Range [m]", fontsize=7.5)

    fig.tight_layout()
    fig.text(0.5, -0.03,
             "One frame per stratum. Red circles mark the annotated targets. Common colour "
             "scale, dB above each frame's own\nnoise floor. Most targets sit near the detection "
             "transition by design, so several are not obvious at this scale.",
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch3_fig2_strata_montage.png")


# ----------------------------------------------------------------------- Fig 3.3 (zoomed)
def fig_strata_zoom():
    """Zoomed companion to the montage: what actually distinguishes the hard strata.

    Ported from notebooks/dataset_overview.ipynb section 4 ("The four hard cases, zoomed"),
    extended to seven strata and using the same frames as Figure 3.2.
    """
    meta = load_meta()
    fdir = os.path.join(ROOT, "data", "bench2_test", "frames")

    r_ax = np.linspace(0, DER.range_max, CFG.N // 2)
    v_ax = np.linspace(-DER.vel_max, DER.vel_max, CFG.M)

    # (stratum, half-window in range [m], half-window in velocity [m/s], caption line)
    panels = [
        ("isolated",     3.0, 3.0, "one target, plain noise"),
        ("masking",      2.6, 2.2, "strong + weak partner"),
        ("unresolved",   1.2, 1.2, "both in one resolution cell"),
        ("multi",        9.0, 9.0, "several, all well separated"),
        ("clutter",      6.0, 6.0, "target INSIDE the ridge"),
        ("clutter_edge", 6.0, 7.0, "target BESIDE the ridge"),
        ("mixed",        6.0, 6.0, "clutter + multipath"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.6))
    flat = axes.ravel()
    im = None

    for ax, (name, hr, hv, sub) in zip(flat, panels):
        fid = pick_frame(meta, name)
        lab = labels_of(fdir, fid)
        db = rd_db(fdir, fid)

        # Centre on the STRONGEST target, not the centroid: strata with several targets can
        # have a centroid that contains no target at all.
        if len(lab):
            a = lab.loc[lab["cell_snr_db"].idxmax()]
            rc, vc = float(a["range_m"]), float(a["velocity_mps"])
        else:
            rc, vc = r_ax[len(r_ax) // 2], 0.0

        ri = np.clip(np.searchsorted(r_ax, [rc - hr, rc + hr]), 0, len(r_ax) - 1)
        vi = np.clip(np.searchsorted(v_ax, [vc - hv, vc + hv]), 0, len(v_ax) - 1)
        ext = [v_ax[vi[0]], v_ax[vi[1]], r_ax[ri[0]], r_ax[ri[1]]]

        im = ax.imshow(db[ri[0]:ri[1] + 1, vi[0]:vi[1] + 1], aspect="auto", origin="lower",
                       cmap="viridis", vmin=0, vmax=30, extent=ext, interpolation="nearest")
        for _, L in lab.iterrows():
            if ext[0] <= L["velocity_mps"] <= ext[1] and ext[2] <= L["range_m"] <= ext[3]:
                ax.scatter(L["velocity_mps"], L["range_m"], s=150, facecolors="none",
                           edgecolors="white", linewidths=1.6)
        ax.set_title(name.replace("_", " "), fontsize=8.6, loc="left")
        ax.text(0.03, 0.03, sub, transform=ax.transAxes, fontsize=6.9,
                color="white", va="bottom")
        ax.tick_params(labelsize=6.5)

    flat[-1].axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=flat[-1], fraction=0.5, pad=0.02)
        cb.set_label("dB above noise floor", fontsize=7.5)
        cb.ax.tick_params(labelsize=6.5)

    for ax in list(axes[0, :]) + list(axes[1, :3]):
        ax.set_xlabel("Velocity [m/s]", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Range [m]", fontsize=7.5)

    # No in-image footer: figure numbers are assigned by LaTeX and a hard-coded one drifts as
    # soon as a float moves. The caption in the .tex carries the explanation.
    fig.tight_layout()
    save(fig, "ch3_fig3_strata_zoom.png")


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
    save(fig, "ch3_fig4_psf.png")


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
    save(fig, "ch3_fig5_snr_distribution.png")


def main():
    print("Chapter 3 figures ->", os.path.relpath(OUT, ROOT))
    fig_simulator_chain()
    fig_strata_montage()
    fig_strata_zoom()
    fig_psf()
    fig_snr_distribution()
    return 0


if __name__ == "__main__":
    sys.exit(main())
