"""Three diagrams for the attribution experiments. Design only: no results on any of them.

    d06_label_ablation      the three label representations
    d07_masked_cnn          frozen dense / unmasked patch / guard-masked patch
    d08_integrated_cfar     single-cell CFAR vs a 3x3 integrated numerator

Every constant drawn here is read from the experiment's own pre-registration, derived-values
file or cached kernel, never typed from memory:

    experiments/label_ablation/DERIVED_VALUES.md      sigma, weights, pos_weight
    experiments/e4_cut_guard_mask/PREREGISTRATION.md  arms, mask rule, RF
    experiments/integrated_cfar/results/kernels.npz   the two 3x3 kernels
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch         # noqa: E402

import _style as S                                                # noqa: E402

DIA = S.DIA

# ----------------------------------------------------------------- frozen constants
SIGMA_R, SIGMA_D = 0.6895, 0.7007          # DERIVED_VALUES.md, from the measured PSF
POS_WEIGHT = {"point": 899.37, "flat3x3": 100.00, "gaussian": 307.51}

GUARD_R = GUARD_D = 2                      # CfarConfig
TRAIN_R, TRAIN_D = 8, 2                    # -> 21 x 9 window
HALF_R = HALF_D = TRAIN_R + GUARD_R, TRAIN_D + GUARD_D
NR, ND = 2 * (TRAIN_R + GUARD_R) + 1, 2 * (TRAIN_D + GUARD_D) + 1   # 21 x 9

TARGET = "#c0392b"
CUTC = "#c0392b"
GUARDC = "#f0c8c2"
REFC = "#cfe0f0"
HIDDEN = "#9aa5ad"
VIS = "#e8eef4"


def gauss_3x3():
    r = np.arange(-1, 2)[:, None]
    d = np.arange(-1, 2)[None, :]
    w = np.exp(-(r ** 2) / (2 * SIGMA_R ** 2) - (d ** 2) / (2 * SIGMA_D ** 2))
    return w / w.max()


# ================================================================= d06  label ablation
def label_ablation():
    point = np.zeros((3, 3)); point[1, 1] = 1.0
    flat = np.ones((3, 3))
    gauss = gauss_3x3()

    arms = [("point", point, "1.0 at the truth bin only",
             "precedent: RadCNN (Yavuz 2021)"),
            ("flat 3x3", flat, "1.0 on the whole 3x3 block",
             "the incumbent — the frozen model's supervision"),
            ("Gaussian", gauss, "3x3 weighted by the MEASURED PSF",
             "precedent: CFARNet (Liang 2025)")]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.4))
    for ax, (name, w, what, note) in zip(axes, arms):
        pad = np.zeros((5, 5))
        pad[1:4, 1:4] = w
        ax.imshow(pad, cmap="Blues", vmin=0, vmax=1.35, origin="upper")
        for i in range(5):
            for j in range(5):
                inner = 1 <= i <= 3 and 1 <= j <= 3
                v = pad[i, j]
                ax.text(j, i, ("%.2f" % v) if inner else "0",
                        ha="center", va="center", fontsize=13 if inner else 11,
                        color="white" if v > 0.75 else S.INK,
                        fontweight="bold" if inner else "normal",
                        alpha=1.0 if inner else 0.45)
        ax.add_patch(Rectangle((0.5, 0.5), 3, 3, fill=False, ec=S.INK, lw=2.6))
        # the truth sits off-grid, so draw it between cell centres
        ax.plot(2.30, 2.26, marker="x", ms=16, mew=3.4, color=TARGET, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(name, fontsize=21, pad=14)
        ax.text(0.5, -0.06, what, transform=ax.transAxes, ha="center", va="top",
                fontsize=14, color=S.INK)
        ax.text(0.5, -0.145, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=11.5, color=S.GREY, style="italic")
        ax.text(0.5, -0.235, "pos_weight  %.0f" % POS_WEIGHT[
            {"point": "point", "flat 3x3": "flat3x3", "Gaussian": "gaussian"}[name]],
            transform=ax.transAxes, ha="center", va="top", fontsize=14,
            color=S.DL, fontweight="bold")

    fig.suptitle("What does one output cell mean?  Three supervision targets",
                 fontsize=23, y=0.99)
    fig.text(0.5, 0.905,
             "red × = the true target, which lies off-grid.   Same network, same data, "
             "same schedule — only the label changes.",
             ha="center", fontsize=14, color=S.GREY)
    fig.text(0.5, 0.045,
             "pos_weight is scaled so that  pos_weight × Σy  is identical across arms: "
             "label SHAPE is the only thing that varies.",
             ha="center", fontsize=14.5, color=S.INK)
    fig.subplots_adjust(top=0.80, bottom=0.22, wspace=0.16)
    S.save(fig, "d06_label_ablation", where=DIA)


# ================================================================= d07  masked CNN
def _patch_grid(ax, mask_guard, title, subtitle, show_rf_note=True):
    """21 x 9 patch: range vertical, Doppler horizontal. CUT at the centre."""
    cr, cd = NR // 2, ND // 2
    for i in range(NR):
        for j in range(ND):
            dr, dd = abs(i - cr), abs(j - cd)
            in_guard = dr <= GUARD_R and dd <= GUARD_D
            is_cut = dr == 0 and dd == 0
            if is_cut:
                fc = CUTC
            elif in_guard and mask_guard:
                fc = HIDDEN
            else:
                fc = VIS
            ax.add_patch(Rectangle((j, NR - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="white", lw=0.8))
    ax.add_patch(Rectangle((cd - GUARD_D, NR - 1 - cr - GUARD_R),
                           2 * GUARD_D + 1, 2 * GUARD_R + 1,
                           fill=False, ec=S.INK, lw=2.4, ls="--"))
    ax.set_xlim(-0.4, ND + 0.4); ax.set_ylim(-0.4, NR + 0.4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=18.5, pad=12)
    ax.text(0.5, -0.045, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=13.5, color=S.INK)
    if show_rf_note:
        ax.text(0.5, -0.115, "21 range × 9 Doppler", transform=ax.transAxes,
                ha="center", va="top", fontsize=12, color=S.GREY, style="italic")


def masked_cnn():
    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.30)

    # --- left: the frozen dense model, for reference
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((0, 0), 64, 256, facecolor=VIS, ec=S.INK, lw=2.0))
    cr, cd = 150, 30
    ax.add_patch(Rectangle((cd - 4.5, cr - 10.5), 9, 21, fill=False, ec=S.DL, lw=3.0))
    ax.plot(cd, cr, marker="s", ms=7, color=CUTC)
    ax.annotate("receptive field\nof one output cell",
                xy=(cd + 4.5, cr + 10.5), xytext=(46, 215),
                fontsize=12.5, color=S.DL, ha="center",
                arrowprops=dict(arrowstyle="->", color=S.DL, lw=2.0))
    ax.set_xlim(-6, 70); ax.set_ylim(-10, 300)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("frozen dense TinyCNN", fontsize=18.5, pad=12)
    ax.text(0.5, -0.045, "full 256 × 64 map, one pass", transform=ax.transAxes,
            ha="center", va="top", fontsize=13.5, color=S.INK)
    ax.text(0.5, -0.115, "read-only reference — not retrained", transform=ax.transAxes,
            ha="center", va="top", fontsize=12, color=S.GREY, style="italic")

    _patch_grid(fig.add_subplot(gs[0, 1]), False,
                "unmasked patch", "nothing hidden")
    _patch_grid(fig.add_subplot(gs[0, 2]), True,
                "guard-masked patch", "24 guard cells set to the noise floor")

    fig.suptitle("Does TinyCNN win by reading the cells CFAR throws away?",
                 fontsize=23, y=0.985)
    fig.text(0.5, 0.905,
             "Dense masking is impossible — cell (r+1,v) is inside CUT (r,v)'s guard while "
             "being its own CUT. So the contrast is patch vs patch.",
             ha="center", fontsize=13.5, color=S.GREY)

    handles = [Rectangle((0, 0), 1, 1, fc=CUTC), Rectangle((0, 0), 1, 1, fc=HIDDEN),
               Rectangle((0, 0), 1, 1, fc=VIS, ec=S.GREY)]
    fig.legend(handles, ["cell under test — always visible",
                         "guard cell, hidden in the masked arm",
                         "visible input"],
               loc="lower center", ncol=3, frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.80, bottom=0.17)
    S.save(fig, "d07_masked_cnn", where=DIA)


# ================================================================= d08  integrated CFAR
def integrated_cfar():
    z = np.load(os.path.join(ROOT, "experiments", "integrated_cfar",
                             "results", "kernels.npz"))
    kern = {"uniform": z["uniform"], "psf": z["psf"]}

    fig = plt.figure(figsize=(14.0, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                          wspace=0.14, hspace=0.42)

    # --- the window
    ax = fig.add_subplot(gs[:, 0])
    cr, cd = NR // 2, ND // 2
    for i in range(NR):
        for j in range(ND):
            dr, dd = abs(i - cr), abs(j - cd)
            if dr <= 1 and dd <= 1:
                fc = CUTC
            elif dr <= GUARD_R and dd <= GUARD_D:
                fc = GUARDC
            else:
                fc = REFC
            ax.add_patch(Rectangle((j, NR - 1 - i), 1, 1, facecolor=fc,
                                   edgecolor="white", lw=0.8))
    ax.add_patch(Rectangle((cd - 1, NR - 1 - cr - 1), 3, 3,
                           fill=False, ec=S.INK, lw=3.0))
    ax.add_patch(Rectangle((cd, NR - 1 - cr), 1, 1,
                           fill=False, ec="white", lw=2.6))
    ax.set_xlim(-9.8, ND + 0.4); ax.set_ylim(-0.4, NR + 0.4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("the same 21 × 9 CFAR window", fontsize=18.5, pad=12)
    ax.annotate("integrated numerator:\nthis 3 × 3",
                xy=(cd + 2, NR - 1 - cr + 1.5), xytext=(ND + 2.6, NR * 0.60),
                fontsize=13.5, color=CUTC, va="center",
                arrowprops=dict(arrowstyle="->", color=CUTC, lw=2.2))
    ax.annotate("classical CFAR\ntests this ONE cell",
                xy=(cd - 0.05, NR - 1 - cr + 0.5), xytext=(-5.2, NR * 0.30),
                fontsize=13.5, color=S.INK, va="center", ha="center",
                arrowprops=dict(arrowstyle="->", color=S.INK, lw=2.2))
    ax.annotate("guard, excluded",
                xy=(cd + GUARD_D + 1, NR - 1 - cr - GUARD_R),
                xytext=(ND + 2.6, NR * 0.40),
                fontsize=13.5, color="#b07d74", va="center",
                arrowprops=dict(arrowstyle="->", color="#b07d74", lw=2.2))
    ax.annotate("reference ring →\nnoise estimate\n(unchanged)",
                xy=(ND - 0.6, NR - 2), xytext=(ND + 2.6, NR * 0.86),
                fontsize=13.5, color="#4a7fae", va="center",
                arrowprops=dict(arrowstyle="->", color="#4a7fae", lw=2.2))

    # --- the two kernels
    for row, (key, name, why) in enumerate(
            [("uniform", "uniform", "plain sum — adds noise as well as signal"),
             ("psf", "PSF-weighted", "matched filter for the measured response")]):
        axk = fig.add_subplot(gs[row, 1])
        k = kern[key]
        axk.imshow(k, cmap="Reds", vmin=0, vmax=0.34, origin="upper", aspect="equal")
        for i in range(3):
            for j in range(3):
                axk.text(j, i, "%.3f" % k[i, j], ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if k[i, j] > 0.20 else S.INK)
        axk.set_xticks([]); axk.set_yticks([])
        for s in axk.spines.values():
            s.set_visible(False)
        axk.set_title(name, fontsize=18, pad=10)
        axk.text(0.5, -0.09, why, transform=axk.transAxes, ha="center", va="top",
                 fontsize=12.5, color=S.GREY, style="italic")

    fig.suptitle("How much of the gain is just integrating mainlobe energy?",
                 fontsize=23, y=0.985)
    fig.text(0.5, 0.905,
             "Give classical CFAR the CNN's integration and nothing else — the ring, "
             "the guard, the matcher and the protocol are all unchanged.",
             ha="center", fontsize=13.5, color=S.GREY)
    fig.text(0.5, 0.035,
             "ratio  =  Σ (kernel × power) over the 3 × 3   ÷   noise "
             "estimate from the ring."
             "     The 3 × 3 fits inside the guard, so numerator and denominator "
             "stay disjoint.",
             ha="center", fontsize=14, color=S.INK)
    fig.subplots_adjust(top=0.80, bottom=0.13)
    S.save(fig, "d08_integrated_cfar", where=DIA)


if __name__ == "__main__":
    os.makedirs(DIA, exist_ok=True)
    label_ablation()
    masked_cnn()
    integrated_cfar()
