"""Generate the explanatory figures for Chapter 2 (Radar Background and State of the Art).

Every quantity is read from the project's own configuration or data -- nothing here is a
hand-drawn illustration with invented numbers. The chirp geometry comes from src/radar/config.py,
the CFAR window from src/cfar/cfar.py defaults, the alphas from configs/cfar_alpha_table.json,
and the example RD map from a real frame of bench2_test.

Run:  venv/Scripts/python.exe scripts/make_ch2_figures.py
Out:  report/figures/Chapter2/
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.radar.config import RadarConfig, derive_radar_params  # noqa: E402

OUT = os.path.join(ROOT, "report", "figures", "Chapter2")
os.makedirs(OUT, exist_ok=True)

CFG = RadarConfig()
DER = derive_radar_params(CFG)

# CFAR window actually used (src/cfar/cfar.py defaults, selected on the tuning split).
GUARD_R, TRAIN_R = 2, 8
GUARD_D, TRAIN_D = 2, 2
WIN_R = 1 + 2 * (GUARD_R + TRAIN_R)      # 21
WIN_D = 1 + 2 * (GUARD_D + TRAIN_D)      # 9

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

INK = "#1a1a1a"
BLUE = "#2c6fbb"
RED = "#c0392b"
GREY = "#8a8a8a"
GOLD = "#d19a13"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


# ----------------------------------------------------------------------- Fig 2.1
def fig_chirp():
    """Transmitted and received chirp: the beat frequency is the range measurement."""
    Tc_us = CFG.Tc * 1e6
    B_MHz = CFG.B / 1e6

    R = 20.0                                  # a target at 20 m
    tau = 2 * R / 2.998e8                     # round-trip delay [s]
    tau_us = tau * 1e6
    fb = DER.slope * tau                      # beat frequency [Hz]

    # The true delay is invisible at chirp scale (0.13 us against 72.4 us), so the ramps would
    # lie exactly on top of one another. The delay is drawn magnified, and the magnification is
    # stated on the figure; every printed number is the true one.
    EXAG = 60.0
    tau_plot = tau_us * EXAG

    fig, ax = plt.subplots(2, 1, figsize=(6.8, 4.6),
                           gridspec_kw={"height_ratios": [2.1, 1]})

    # --- frequency-vs-time ramps
    a = ax[0]
    n = 400
    t = np.linspace(0, Tc_us, n)
    a.plot(t, t / Tc_us * B_MHz, color=BLUE, lw=1.9, label="Transmitted chirp")
    tr = np.linspace(tau_plot, Tc_us + tau_plot, n)
    a.plot(tr, (tr - tau_plot) / Tc_us * B_MHz, color=RED, lw=1.9, ls="--",
           label="Received echo, delayed by $\\tau$")

    # delay marker
    a.annotate("", xy=(0, -0.06 * B_MHz), xytext=(tau_plot, -0.06 * B_MHz),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    a.text(tau_plot / 2, -0.155 * B_MHz, "$\\tau = 2R/c$", ha="center", color=INK, fontsize=8.5)

    # beat frequency marker at a readable instant
    t0 = 0.60 * Tc_us
    f_tx = t0 / Tc_us * B_MHz
    f_rx = (t0 - tau_plot) / Tc_us * B_MHz
    a.plot([t0, t0], [f_rx, f_tx], color=GOLD, lw=3.0, solid_capstyle="butt")
    a.annotate("the two differ by a constant\nfrequency $f_b$ (gap NOT to scale)",
               xy=(t0, (f_rx + f_tx) / 2),
               xytext=(t0 + 0.07 * Tc_us, 0.26 * B_MHz),
               arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.1),
               color=GOLD, fontsize=8.2)

    a.set_ylabel("Frequency offset [MHz]")
    a.set_xlabel("Time within one chirp [$\\mu$s]")
    a.set_ylim(-0.20 * B_MHz, 1.10 * B_MHz)
    a.set_xlim(-0.03 * Tc_us, 1.20 * Tc_us)
    a.legend(loc="upper left", frameon=False, fontsize=8)
    a.set_title("The echo is a delayed copy of the chirp, so the delay becomes a frequency")
    a.spines[["top", "right"]].set_visible(False)
    a.text(0.985, 0.06,
           "Delay drawn $\\times$%d for visibility, so the vertical\n"
           "gap is exaggerated by the same factor.\n"
           "True values at $R$ = 20 m: $\\tau$ = %.3f $\\mu$s, $f_b$ = %.2f MHz."
           % (EXAG, tau_us, fb / 1e6),
           transform=a.transAxes, ha="right", va="bottom", fontsize=7.4, color=GREY)

    # --- beat tone, over a short window so individual cycles are visible
    b = ax[1]
    span = 4.0                                   # microseconds
    tb = np.linspace(20.0, 20.0 + span, 3000)
    b.plot(tb, np.cos(2 * np.pi * fb * (tb * 1e-6)), color=GOLD, lw=1.1)
    b.set_xlabel("Time within one chirp [$\\mu$s]   (%.0f $\\mu$s window)" % span)
    b.set_ylabel("Beat signal")
    b.set_yticks([])
    b.set_xlim(20.0, 20.0 + span)
    b.spines[["top", "right", "left"]].set_visible(False)
    b.set_title("Mixing the transmitted chirp with its delayed echo produces a beat "
                "frequency proportional to range", fontsize=9)
    b.text(0.985, 0.94,
           "$f_b = 2BR / (c\\,T_c)$;    $R = 20$ m $\\Rightarrow f_b = %.2f$ MHz "
           "(shown here at true scale)" % (fb / 1e6),
           transform=b.transAxes, ha="right", va="top", fontsize=8.4, color=INK)

    fig.tight_layout()
    fig.text(0.5, -0.03,
             "Radar of this work: $B$ = %.0f MHz, $T_c$ = %.1f $\\mu$s, "
             "slope = %.3f MHz/$\\mu$s, $N$ = %d samples per chirp"
             % (B_MHz, Tc_us, DER.slope / 1e12, CFG.N),
             ha="center", fontsize=8, color=GREY)
    save(fig, "ch2_fig1_fmcw_chirp.png")


# ----------------------------------------------------------------------- Fig 2.2
def fig_chain():
    """From received samples to the range-Doppler map: two FFTs."""
    # Five boxes of width w with gap g: the row spans 5w + 4g. Lay it out symmetrically
    # inside the axes so the final box is never clipped at the right edge.
    w, h, g = 18, 15, 5
    n_box = 5
    span = n_box * w + (n_box - 1) * g          # 99
    margin = 2.0
    x_lim = span + 2 * margin                   # 103

    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    ax.axis("off")
    ax.set_xlim(0, x_lim)
    ax.set_ylim(3, 34)

    # NB: matplotlib is not LaTeX -- "--" would render as two hyphens, so use the
    # en-dash character directly.
    labels = [
        ("Received\nsamples", "$N \\times M$\n%d $\\times$ %d" % (CFG.N, CFG.M), 8.2),
        ("Range FFT\n(fast time)", "along $N$", 8.2),
        ("Doppler FFT\n(slow time)", "along $M$", 8.2),
        ("Range–Doppler\nmap", "%d $\\times$ %d bins" % (CFG.N // 2, CFG.M), 7.5),
        ("Detector", "CFAR or CNN", 8.2),
    ]
    boxes = [(margin + i * (w + g), t, s, fs) for i, (t, s, fs) in enumerate(labels)]
    for x, title, sub, fs in boxes:
        edge = BLUE if "Detector" not in title else RED
        ax.add_patch(Rectangle((x, 11), w, h, fill=True, facecolor="white",
                               edgecolor=edge, lw=1.5, zorder=2))
        ax.text(x + w / 2, 21.5, title, ha="center", va="center", fontsize=fs,
                color=INK, zorder=3)
        ax.text(x + w / 2, 15.0, sub, ha="center", va="center", fontsize=7.4,
                color=GREY, zorder=3)

    for x, _, _, _ in boxes[:-1]:
        ax.annotate("", xy=(x + w + g - 0.6, 18.5), xytext=(x + w + 0.6, 18.5),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

    ax.text(boxes[1][0] + w / 2, 7.0, "range resolution\n$c/2B$ = %.3f m" % DER.range_res,
            ha="center", fontsize=7.4, color=BLUE)
    ax.text(boxes[2][0] + w / 2, 7.0,
            "velocity resolution\n$\\lambda/2MT_r$ = %.3f m/s" % DER.vel_res,
            ha="center", fontsize=7.4, color=BLUE)
    ax.text(x_lim / 2, 30, "This chapter concerns the final stage only.",
            ha="center", fontsize=8.4, color=RED)
    save(fig, "ch2_fig2_processing_chain.png")


# ----------------------------------------------------------------------- Fig 2.3
def fig_rd_example():
    """A real simulated RD map, in the units the detectors actually see."""
    fdir = os.path.join(ROOT, "data", "bench2_test", "frames")
    names = sorted(os.listdir(fdir))
    mag = None
    for nm in names[:400]:                       # first frame with a clear target
        p = os.path.join(fdir, nm, "rd_map_mag.npy")
        if not os.path.exists(p):
            continue
        m = np.load(p)
        db = 20 * np.log10(np.maximum(m, 1e-12))
        if db.max() - np.median(db) > 25:
            mag, frame_name = m, nm
            break
    if mag is None:
        print("  [skip] no suitable frame for fig 2.3")
        return

    db = 20 * np.log10(np.maximum(mag, 1e-12))
    db = db - np.median(db)                      # dB above the frame's own noise floor
    r0, d0 = np.unravel_index(np.argmax(db), db.shape)

    # Noise-free counterpart: the target response alone, on the same axes. This is the
    # detection problem stated visually -- the left panel is what we want, the middle is
    # what the sensor delivers.
    rr, dd = np.meshgrid(np.arange(db.shape[0]), np.arange(db.shape[1]), indexing="ij")
    clean = db.max() * np.exp(-(((rr - r0) / 1.1) ** 2 + ((dd - d0) / 1.1) ** 2) / 2.0)

    ext = [-DER.vel_max, DER.vel_max, 0, DER.range_max]
    vmax = max(28, db.max())

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9),
                           gridspec_kw={"width_ratios": [1, 1, 1.05]})

    ax[0].imshow(clean, aspect="auto", origin="lower", cmap="viridis",
                 extent=ext, vmin=-6, vmax=vmax)
    ax[0].set_title("Target response alone")
    ax[0].set_ylabel("Range [m]")

    im = ax[1].imshow(db, aspect="auto", origin="lower", cmap="viridis",
                      extent=ext, vmin=-6, vmax=vmax)
    ax[1].set_title("What the sensor delivers")
    ax[1].set_yticklabels([])

    for a in ax[:2]:
        a.set_xlabel("Radial velocity [m/s]")
    cb = fig.colorbar(im, ax=ax[1], pad=0.02)
    cb.set_label("dB above noise floor", fontsize=7.5)

    prof = db[r0, :]
    ax[2].plot(np.linspace(-DER.vel_max, DER.vel_max, db.shape[1]), prof,
               color=BLUE, lw=1.3)
    ax[2].axhline(0, color=GREY, lw=0.9, ls=":")
    ax[2].set_xlabel("Radial velocity [m/s]")
    ax[2].set_ylabel("dB above noise floor")
    ax[2].set_title("Cut through the target's range bin")
    ax[2].spines[["top", "right"]].set_visible(False)
    ax[2].text(0.03, 0.03,
               "The detector scores each cell;\nlocal peaks are then extracted\n"
               "as target candidates.",
               transform=ax[2].transAxes, va="bottom", fontsize=7.4, color=GREY)

    fig.tight_layout()
    fig.text(0.5, -0.04,
             "One frame of the simulated benchmark (%s), %d $\\times$ %d bins, "
             "0--%.0f m and $\\pm$%.1f m/s."
             % (frame_name.replace("_", "\\_"), db.shape[0], db.shape[1],
                DER.range_max, DER.vel_max),
             ha="center", fontsize=7.6, color=GREY)
    save(fig, "ch2_fig3_rd_map_example.png")


# ----------------------------------------------------------------------- Fig 2.4
def fig_cfar_window():
    """The 21x9 window actually used: CUT, guard cells, reference cells."""
    grid = np.zeros((WIN_D, WIN_R))              # rows = Doppler, cols = range
    cd, cr = WIN_D // 2, WIN_R // 2
    for i in range(WIN_D):
        for j in range(WIN_R):
            dr, dd = abs(j - cr), abs(i - cd)
            if dr == 0 and dd == 0:
                grid[i, j] = 3                   # CUT
            elif dr <= GUARD_R and dd <= GUARD_D:
                grid[i, j] = 2                   # guard
            else:
                grid[i, j] = 1                   # reference

    n_ref = int((grid == 1).sum())
    n_guard = int((grid == 2).sum())

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    cmap = matplotlib.colors.ListedColormap(["#eef2f6", "#9fc3e8", "#e8913c"])
    ax.imshow(grid, cmap=cmap, origin="lower", vmin=1, vmax=3)

    for x in range(WIN_R + 1):
        ax.axvline(x - 0.5, color="white", lw=0.8)
    for y in range(WIN_D + 1):
        ax.axhline(y - 0.5, color="white", lw=0.8)

    ax.set_xticks(range(0, WIN_R, 2))
    ax.set_xticklabels(range(-cr, cr + 1, 2))
    ax.set_yticks(range(WIN_D))
    ax.set_yticklabels(range(-cd, cd + 1))
    ax.set_xlabel("Range offset from the cell under test [bins]")
    ax.set_ylabel("Doppler offset [bins]")
    ax.set_title("CFAR reference window: %d bins in range $\\times$ %d in Doppler = %d cells\n"
                 "(%d guard and %d reference cells per side in range; %d and %d in Doppler)"
                 % (WIN_R, WIN_D, WIN_R * WIN_D,
                    GUARD_R, TRAIN_R, GUARD_D, TRAIN_D), fontsize=9)

    handles = [
        matplotlib.patches.Patch(facecolor="#e8913c", edgecolor="white",
                                 label="1 cell under test (supplies $x$)"),
        matplotlib.patches.Patch(facecolor="#9fc3e8", edgecolor="white",
                                 label="%d guard cells (excluded)" % n_guard),
        matplotlib.patches.Patch(facecolor="#eef2f6", edgecolor="#cccccc",
                                 label="%d reference cells (estimate $Z$)" % n_ref),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, frameon=False, fontsize=8)
    save(fig, "ch2_fig4_cfar_window.png")
    return n_ref


# ----------------------------------------------------------------------- Fig 2.5
def fig_failure_modes():
    """The two documented CA-CFAR failure modes, simulated on the real window."""
    rng = np.random.default_rng(7)
    n = 220
    half = GUARD_R + TRAIN_R

    def ca_threshold(x, alpha):
        t = np.full_like(x, np.nan)
        for i in range(half, len(x) - half):
            lead = x[i - half:i - GUARD_R]
            lag = x[i + GUARD_R + 1:i + half + 1]
            t[i] = alpha * np.concatenate([lead, lag]).mean()
        return t

    alpha = json.load(open(os.path.join(ROOT, "configs", "cfar_alpha_table.json")))
    a_ca = alpha["entries"]["CA|g2-2|t8-2|pfa0.0001|os0.75|ax1"]
    # 1-D illustration: rescale the 2-D alpha to the 1-D reference count so the picture is honest.
    a1 = a_ca / 4.0

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    edge = n // 2

    # --- clutter edge -------------------------------------------------------
    p = np.ones(n)
    p[edge:] = 15.0                                # background step up
    x = rng.exponential(p)
    tgt = edge - 6                                 # target just BEFORE the edge, low side
    x[tgt] = 22.0
    t = ca_threshold(x, a1)

    ax[0].plot(x, color=GREY, lw=0.8, label="Cell power")
    ax[0].plot(t, color=RED, lw=1.6, label="CA-CFAR threshold")
    ax[0].axvline(edge, color=BLUE, lw=1.0, ls=":")

    # false alarms on the high side, where the threshold has not yet caught up
    fa = [i for i in range(edge, edge + half + 1)
          if np.isfinite(t[i]) and x[i] > t[i]]
    if fa:
        ax[0].plot(fa, x[fa], "o", mfc="none", mec=RED, ms=7, mew=1.3,
                   label="False alarms")
    ax[0].plot(tgt, x[tgt], "v", color=BLUE, ms=8,
               label="Target (missed)" if x[tgt] < t[tgt] else "Target")

    ax[0].set_title("Clutter edge")
    ax[0].set_xlabel("Range bin")
    ax[0].set_ylabel("Power [linear]")
    ax[0].set_yscale("log")
    ax[0].set_xlim(edge - 60, edge + 60)
    ax[0].legend(loc="upper left", frameon=False, fontsize=7.2)
    ax[0].text(0.98, 0.03,
               "The threshold cannot step as sharply as the\n"
               "background does: too high before the edge,\n"
               "too low just after it.",
               transform=ax[0].transAxes, ha="right", fontsize=7.2, color=INK)

    # --- target masking -----------------------------------------------------
    x2 = rng.exponential(np.ones(n))
    t_of_interest, interferer = 110, 118
    x2[t_of_interest] = 26.0
    x2[interferer] = 260.0                         # strong neighbour inside the window
    t2 = ca_threshold(x2, a1)

    ax[1].plot(x2, color=GREY, lw=0.8)
    ax[1].plot(t2, color=RED, lw=1.6)
    ax[1].plot(t_of_interest, x2[t_of_interest], "v", color=BLUE, ms=8)
    ax[1].plot(interferer, x2[interferer], "v", color=GOLD, ms=8)
    ax[1].annotate("interferer", xy=(interferer, x2[interferer]),
                   xytext=(interferer + 14, x2[interferer] * 1.1),
                   color=GOLD, fontsize=7.6,
                   arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.0))
    miss = x2[t_of_interest] < t2[t_of_interest]
    ax[1].annotate("target, %s" % ("missed" if miss else "detected"),
                   xy=(t_of_interest, x2[t_of_interest]),
                   xytext=(t_of_interest - 62, x2[t_of_interest] * 2.2),
                   color=BLUE, fontsize=7.6,
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))

    ax[1].set_title("Target masking")
    ax[1].set_xlabel("Range bin")
    ax[1].set_yscale("log")
    ax[1].set_xlim(t_of_interest - 60, t_of_interest + 60)
    ax[1].text(0.98, 0.03,
               "The interferer lies inside the target's own\n"
               "reference window. It raises $Z$, and with it the\n"
               "threshold, at the very cell being tested.",
               transform=ax[1].transAxes, ha="right", fontsize=7.2, color=INK)

    fig.tight_layout()
    fig.text(0.5, -0.04,
             "Simulated on the same window as the rest of this work "
             "(guard %d, training %d per side, one-dimensional cut). "
             "Exponential background." % (GUARD_R, TRAIN_R),
             ha="center", fontsize=7.4, color=GREY)
    save(fig, "ch2_fig5_cfar_failure_modes.png")


# ----------------------------------------------------------------------- Fig 2.6
def fig_cfar_as_convolution(n_ref):
    """Two decision rules over the same local footprint: fixed versus learned.

    The earlier version of this figure showed a random 21x9 "learned kernel" beside the CFAR
    kernel, which wrongly suggested TinyCNN is a single learned 21x9 convolution. It is a stack
    of four dilated 3x3 layers whose composed receptive field is 21x9, and the figure now shows
    that structure instead.
    """
    cd, cr = WIN_D // 2, WIN_R // 2
    k = np.zeros((WIN_D, WIN_R))
    for i in range(WIN_D):
        for j in range(WIN_R):
            dr, dd = abs(j - cr), abs(i - cd)
            if dr == 0 and dd == 0:
                k[i, j] = 1.0
            elif dr <= GUARD_R and dd <= GUARD_D:
                k[i, j] = 0.0
            else:
                k[i, j] = -1.0 / n_ref            # -alpha/N, alpha folded into the threshold

    # TinyCNN layer stack, from src/dl/models.py. RF = 1 + sum(2*d) per axis, cumulative.
    dil_r, dil_d = (1, 2, 3, 4), (1, 1, 1, 1)
    chans = (16, 16, 16, 1)
    rf_r = np.cumsum([2 * d for d in dil_r]) + 1
    rf_d = np.cumsum([2 * d for d in dil_d]) + 1

    fig = plt.figure(figsize=(7.3, 5.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.34, 1.5, 1.35], hspace=0.30, wspace=0.22)

    # ---- shared header
    hdr = fig.add_subplot(gs[0, :])
    hdr.axis("off")
    hdr.add_patch(Rectangle((0.24, 0.18), 0.52, 0.62, facecolor="#eef2f6",
                            edgecolor=GREY, lw=1.0))
    hdr.text(0.5, 0.49, "Same local range–Doppler footprint:  %d $\\times$ %d bins"
             % (WIN_R, WIN_D), ha="center", va="center", fontsize=10, color=INK)
    for x in (0.38, 0.62):
        hdr.annotate("", xy=(x, -0.30), xytext=(x, 0.14),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    hdr.set_xlim(0, 1)
    hdr.set_ylim(-0.35, 1)

    # ---- left: fixed rule
    axl = fig.add_subplot(gs[1, 0])
    # The CUT weight (+1) is ~164x larger than each reference weight (-1/164). Scaling to the
    # CUT would render every other cell white, so the scale is set by the reference weights and
    # the CUT deliberately saturates; it is labelled instead.
    v = 2.0 / n_ref
    axl.imshow(k, cmap="coolwarm", origin="lower", vmin=-v, vmax=v)
    axl.text(cr, cd + 1.6, "CUT: $+1$\n(off scale)", ha="center", va="bottom",
             fontsize=7.2, color=RED)
    axl.set_title("CA-CFAR: fixed, hand-designed rule", fontsize=9.5, color=BLUE)
    axl.set_xticks([0, cr, WIN_R - 1]); axl.set_xticklabels([-cr, 0, cr])
    axl.set_yticks([0, cd, WIN_D - 1]); axl.set_yticklabels([-cd, 0, cd])
    axl.set_xlabel("Range offset [bins]", fontsize=8)
    axl.set_ylabel("Doppler offset [bins]", fontsize=8)
    axl.tick_params(labelsize=7.5)

    axl2 = fig.add_subplot(gs[2, 0])
    axl2.axis("off"); axl2.set_xlim(0, 1); axl2.set_ylim(0, 1)
    axl2.text(0.5, 0.92, "One fixed weight per cell:", ha="center", fontsize=8.4, color=INK)
    axl2.text(0.5, 0.70, "$+1$ on the CUT,   $0$ on the %d guard cells,\n"
                         "$-\\alpha/N$ on each of the $N=%d$ reference cells"
              % (int((k == 0).sum()) - 0, n_ref),
              ha="center", va="center", fontsize=8.0, color=GREY)
    axl2.annotate("", xy=(0.5, 0.30), xytext=(0.5, 0.52),
                  arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axl2.add_patch(Rectangle((0.22, 0.05), 0.56, 0.22, facecolor="white",
                             edgecolor=BLUE, lw=1.4))
    axl2.text(0.5, 0.16, "detection score", ha="center", va="center",
              fontsize=9, color=INK)

    # ---- right: learned rule
    axr = fig.add_subplot(gs[1, 1])
    axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
    axr.set_title("TinyCNN: learned, multi-layer rule", fontsize=9.5, color=RED)
    n_l = len(dil_r)
    top, bh, gap = 0.94, 0.17, 0.055
    for i in range(n_l):
        y = top - i * (bh + gap) - bh
        axr.add_patch(Rectangle((0.06, y), 0.60, bh, facecolor="white",
                                edgecolor=RED, lw=1.3))
        axr.text(0.36, y + bh / 2,
                 "Conv $3\\times3$,  %d ch,\ndilation (%d, %d)"
                 % (chans[i], dil_r[i], dil_d[i]),
                 ha="center", va="center", fontsize=7.6, color=INK)
        axr.text(0.70, y + bh / 2, "%d$\\times$%d" % (rf_r[i], rf_d[i]),
                 ha="left", va="center", fontsize=7.8, color=GREY)
        if i < n_l - 1:
            axr.annotate("", xy=(0.36, y - 0.005), xytext=(0.36, y - gap + 0.005),
                         arrowprops=dict(arrowstyle="<-", color=RED, lw=1.1))
    axr.text(0.70, top + 0.05, "receptive field", ha="left", va="bottom",
             fontsize=7.2, color=GREY)

    axr2 = fig.add_subplot(gs[2, 1])
    axr2.axis("off"); axr2.set_xlim(0, 1); axr2.set_ylim(0, 1)
    axr2.text(0.5, 0.92, "Weights fitted to data, not derived:", ha="center",
              fontsize=8.4, color=INK)
    axr2.text(0.5, 0.70, "5,041 parameters over four layers.\n"
                         "The %d $\\times$ %d footprint is the COMPOSED receptive\n"
                         "field of the stack, not a single %d $\\times$ %d kernel."
              % (rf_r[-1], rf_d[-1], rf_r[-1], rf_d[-1]),
              ha="center", va="center", fontsize=8.0, color=GREY)
    axr2.annotate("", xy=(0.5, 0.30), xytext=(0.5, 0.52),
                  arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axr2.add_patch(Rectangle((0.22, 0.05), 0.56, 0.22, facecolor="white",
                             edgecolor=RED, lw=1.4))
    axr2.text(0.5, 0.16, "detection score", ha="center", va="center",
              fontsize=9, color=INK)

    save(fig, "ch2_fig6_cfar_as_convolution.png")


def main():
    print("Chapter 2 figures ->", os.path.relpath(OUT, ROOT))
    fig_chirp()
    fig_chain()
    fig_rd_example()
    n_ref = fig_cfar_window()
    fig_failure_modes()
    fig_cfar_as_convolution(n_ref)
    print("done. reference cells N =", n_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
