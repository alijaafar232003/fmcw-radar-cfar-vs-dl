"""Port two explanatory figures from notebooks/evaluation_metrics_overview.ipynb into Chapter 4.

Sources, kept faithful rather than re-derived:
  * Figure "pipeline stages"  <- notebook section 1.1, cells 11-13
        ("The four stages, in the order the code applies them")
  * Figure "matching / tolerance" <- notebook section 3, cells 31-32

Both reuse the project's own evaluation primitives (all_peaks, evaluate_frame), the same frame,
the same frozen checkpoint, the same matched thresholds and the same invented toy coordinates as
the notebook. Nothing about the detections or the scoring is altered here.

ONE DELIBERATE DEPARTURE. The notebook calls the near-truth exclusion region the "guard", and its
title says the gate and the guard "cover the same cells". That is true of the EVALUATION guard,
which `_guard_mask` builds from the MatchConfig -- but the report already uses "guard cells" for
the CFAR window's 24 excluded cells, which are a different thing entirely with different geometry.
To avoid that collision the report version calls it the "near-target exclusion region" and drops
the claim about shared geometry.

Run:  venv/Scripts/python.exe scripts/make_ch4_eval_figures.py
Out:  report/figures/Chapter4/
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
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map            # noqa: E402
from cfar.detect import all_peaks, _collapse_plateaus       # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap       # noqa: E402
from dl.models import TinyCNN                               # noqa: E402
from eval.detection import MatchConfig, evaluate_frame, _guard_mask  # noqa: E402

OUT = os.path.join(ROOT, "report", "figures", "Chapter4")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})

INK, BLUE, RED, GREY = "#1a1a1a", "#2c6fbb", "#c0392b", "#8a8a8a"
AQUA, ORANGE = "#00b4d8", "#e8913c"
# Truth must NOT share the candidate colour, or the legend shows two cyan rings that differ
# only in size. Distinct hue AND distinct marker.
TRUTH = "#00b140"
NMS = 5
CFAR_CFG = CfarConfig()
GATE, GATE3 = MatchConfig(1, 1), MatchConfig(3, 3)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


# ===================================================== Figure: the four pipeline stages
def fig_stages():
    """Notebook section 1.1: NMS fixes the candidate set, THEN the score cut is applied."""
    fdir = os.path.join(ROOT, "data", "bench2_val", "frames")
    # Same selection rule as the notebook: first masking frame of bench2_val.
    meta = pd.read_csv(os.path.join(ROOT, "data", "bench2_val", "metadata.csv"))
    fid = meta[meta.condition == "masking"].frame_id.iloc[0]
    fd = os.path.join(fdir, str(fid))

    mag = np.load(os.path.join(fd, "rd_map_mag.npy"))
    lab = pd.read_csv(os.path.join(fd, "labels.csv"))
    truths = lab[["range_bin", "velocity_bin"]].to_numpy(float)
    shape = mag.shape

    valid = cfar_valid_mask(shape, CFAR_CFG)
    ratio, _, _ = cfar_ratio_map(mag ** 2, CFAR_CFG)

    ck = torch.load(os.path.join(ROOT, "results", "dl", "tinycnn_frozen_s0_ep021.pt"),
                    weights_only=False)
    model = TinyCNN(width=ck["config"].get("width", 16))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    logit = predict_heatmap(model, mag)

    opp = pd.read_csv(os.path.join(ROOT, "results", "dl",
                                   "roc_operating_points_val_tol1.csv"))

    def cut(method):
        return float(opp[(opp.condition == "masking") &
                         (opp.method == method)].threshold.iloc[0])

    def stages(score, thr):
        """Stage 2 (NMS, before the cut) then stage 3 (the cut) -- the real order."""
        pts = all_peaks(score, valid, nms_size=NMS)
        keep = pts[:, 2] >= thr if len(pts) else np.zeros(0, dtype=bool)
        # Reproduce all_peaks' primitives, as the notebook does, to assert the order.
        peak = score == maximum_filter(score, size=2 * NMS + 1, mode="nearest")
        survivors = _collapse_plateaus(valid & peak & (score > 0))
        assert ({(int(r), int(c)) for r, c in np.argwhere(survivors)} ==
                {(int(q[0]), int(q[1])) for q in pts}), "stage 2 must reproduce all_peaks()"
        return pts, keep

    thr_ca, thr_dl = cut("CA"), cut("DL")
    cf_pts, cf_keep = stages(ratio, thr_ca)
    dl_pts, dl_keep = stages(logit, thr_dl)

    fig, axes = plt.subplots(2, 4, figsize=(11.4, 6.4))
    rows = [("CA-CFAR ratio map", ratio, cf_pts, cf_keep, thr_ca),
            ("TinyCNN logit map", logit, dl_pts, dl_keep, thr_dl)]

    for row, (name, score, pts, keep, thr) in zip(axes, rows):
        # Clip the display at a high percentile: a few very bright cells otherwise crush the
        # whole ratio map to black. Display only -- the scores themselves are untouched.
        row[0].imshow(score, aspect="auto", origin="lower", cmap="magma",
                      vmin=float(np.percentile(score, 1)),
                      vmax=float(np.percentile(score, 99.5)))
        row[0].set_title("1 — %s" % name, loc="left", fontsize=9)

        row[1].scatter(pts[:, 1], pts[:, 0], s=13, facecolor="none",
                       edgecolor=AQUA, linewidths=0.8)
        row[1].set_title("2 — NMS candidates: %d" % len(pts), loc="left", fontsize=9)

        row[2].scatter(pts[~keep, 1], pts[~keep, 0], s=13, facecolor="none",
                       edgecolor=GREY, linewidths=0.8)
        row[2].scatter(pts[keep, 1], pts[keep, 0], s=38, color=ORANGE,
                       edgecolor=INK, linewidths=0.6)
        row[2].set_title("3 — cut at %.2f: %d pass" % (thr, int(keep.sum())),
                         loc="left", fontsize=9)

        row[3].scatter(pts[keep, 1], pts[keep, 0], s=50, color=ORANGE,
                       edgecolor=INK, linewidths=0.7)
        row[3].set_title("4 — detections: %d" % int(keep.sum()), loc="left", fontsize=9)

        for k, ax in enumerate(row):
            if k:
                ax.set_facecolor("white")
            ax.scatter(truths[:, 1], truths[:, 0], s=190, marker="s", facecolors="none",
                       edgecolors=TRUTH, linewidths=1.6, zorder=6)
            ax.set_xlim(-0.5, shape[1] - 0.5)
            ax.set_ylim(-0.5, shape[0] - 0.5)
            ax.grid(False)
            ax.tick_params(labelsize=7)
        row[0].set_ylabel("Range bin", fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("Doppler bin", fontsize=8)

    handles = [
        Line2D([0], [0], marker="o", color=AQUA, lw=0, mfc="none", ms=7,
               label="NMS candidate"),
        Line2D([0], [0], marker="o", color=GREY, lw=0, mfc="none", ms=7,
               label="candidate below the cut"),
        Line2D([0], [0], marker="o", color=ORANGE, lw=0, mec=INK, ms=8,
               label="detection"),
        Line2D([0], [0], marker="s", color=TRUTH, lw=0, mfc="none", ms=11,
               label="labelled target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8.4, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("NMS fixes the candidate set first; the score cut is applied afterwards",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "ch4_fig6_pipeline_stages.png")


# ===================================================== Figure: matching and tolerance
def fig_matching():
    """Notebook section 3: the same toy frame judged at two tolerances."""
    # INVENTED coordinates, copied verbatim from the notebook. Not measurements.
    shape = (18, 18)
    truths = np.array([[4.0, 4.0],     # A - clean hit
                       [4.0, 12.0],    # B - hit, plus a second detection nearby
                       [12.0, 4.0],    # C - detection just outside the gate
                       [12.0, 12.0]])  # D - nothing near it
    dets = np.array([[5.0, 4.0],       # -> A, offset (+1, 0)
                     [4.0, 12.0],      # -> B, exact
                     [3.0, 11.0],      # near B, but B is taken
                     [14.0, 6.0],      # near C but 2 bins away
                     [9.0, 16.0]])     # nowhere near anything
    names = ["A", "B", "C", "D"]

    def draw(ax, cfg, title):
        near = _guard_mask(truths, shape, cfg)
        ev = evaluate_frame(dets, truths, shape, cfg)

        ax.imshow(np.ma.masked_where(~near, near.astype(float)), origin="lower",
                  cmap="autumn_r", alpha=0.20, vmin=0, vmax=1.4)

        for (r, v), nm in zip(truths, names):
            ax.add_patch(Rectangle((v - cfg.tol_v - 0.5, r - cfg.tol_r - 0.5),
                                   2 * cfg.tol_v + 1, 2 * cfg.tol_r + 1,
                                   facecolor="none", edgecolor=BLUE,
                                   linewidth=1.3, linestyle="--"))
            ax.scatter(v, r, marker="x", s=95, color=BLUE, linewidths=2.0, zorder=4)
            ax.text(v + cfg.tol_v + 0.7, r + cfg.tol_r + 0.4, nm, fontsize=9, color=BLUE)

        matched_det = set()
        d = np.asarray(dets, float)
        for j, k in enumerate(ev.matched_truth_idx):
            t = truths[k]
            hit = np.flatnonzero((d[:, 0] == t[0] + ev.dr[j]) & (d[:, 1] == t[1] + ev.dv[j]))
            matched_det.add(int(hit[0]))

        for i, (r, v) in enumerate(d):
            inside = bool(near[int(r), int(v)])
            if i in matched_det:
                c, lab = AQUA, "TP"
            elif inside:
                c, lab = GREY, "ignored"
            else:
                c, lab = RED, "FP"
            ax.scatter(v, r, marker="o", s=85, facecolor=c, edgecolor=INK,
                       linewidths=0.7, zorder=5)
            ax.text(v + 0.45, r - 1.05, lab, fontsize=7.5, color=c, zorder=6)

        ax.set_title("%s      TP=%d   FP=%d   FN=%d" % (title, ev.tp, ev.fp, ev.fn),
                     loc="left", fontsize=10)
        ax.set_xlim(-0.5, shape[1] - 0.5)
        ax.set_ylim(-0.5, shape[0] - 0.5)
        ax.set_xlabel("Doppler bin")
        ax.grid(False)
        return ev

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    ev1 = draw(axes[0], GATE, "primary gate $\\pm$1 bin")
    ev3 = draw(axes[1], GATE3, "robustness gate $\\pm$3 bins")
    axes[0].set_ylabel("Range bin")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Line2D([0], [0], marker="x", color=BLUE, lw=0, ms=8, label="truth"),
        Line2D([0], [0], color=BLUE, ls="--", lw=1.3, label="matching gate"),
        Patch(facecolor="#ffb347", alpha=0.45,
              label="near-target exclusion region (leaves the $P_{fa}$ denominator)"),
        Line2D([0], [0], marker="o", color=AQUA, lw=0, mec=INK, ms=8,
               label="matched (TP)"),
        Line2D([0], [0], marker="o", color=GREY, lw=0, mec=INK, ms=8,
               label="ignored: inside a gate but not matched"),
        Line2D([0], [0], marker="o", color=RED, lw=0, mec=INK, ms=8,
               label="false alarm (FP)"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=7.8,
        bbox_to_anchor=(0.5, -0.10))

    fig.suptitle("Toy example — the circles are invented detections, not measurements",
                 fontsize=10.5)
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, "ch4_fig7_matching.png")
    print("    gate +/-1 : TP=%d FP=%d FN=%d" % (ev1.tp, ev1.fp, ev1.fn))
    print("    gate +/-3 : TP=%d FP=%d FN=%d" % (ev3.tp, ev3.fp, ev3.fn))


def main():
    print("Chapter 4 evaluation figures ->", os.path.relpath(OUT, ROOT))
    fig_stages()
    fig_matching()
    return 0


if __name__ == "__main__":
    sys.exit(main())
