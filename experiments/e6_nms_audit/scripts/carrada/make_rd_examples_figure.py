"""Audience-facing RD-map examples: what the two domains look like, and where each detector fires.

Frame selection is DETERMINISTIC and made without reference to which detector looks better:
  * synthetic  — the first `isolated` and first `clutter_edge` frame of bench2_val, in file order
  * CARRADA    — for each Doppler stratum, the first annotated Test frame in sequence/frame order

CARRADA peaks are read from the PRESERVED Phase 5 outputs (no detector re-run). Synthetic peaks are
computed with the same frozen models purely for illustration; no synthetic result is recomputed or
overwritten.
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

# --------------------------------------------------- E6 PATCH (auto-generated)
# Isolated copy; the original is untouched. Only output/result paths change.
E6_DIR = "C:/Users/USER/Desktop/Radar-fyp/My work/Simulations/experiments/e6_nms_audit"
EXP = "C:/Users/USER/Desktop/Radar-fyp/My work/Simulations/experiments/r_carrada_real_data_validation"
ROOT = "C:/Users/USER/Desktop/Radar-fyp/My work/Simulations"
# ------------------------------------------------------------------------------

sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import carrada_input as ci   # noqa: E402
import detectors as det      # noqa: E402
det.NMS_SIZE = 2          # E6 PATCH
import eligibility as el     # noqa: E402

FIGS = os.path.join(E6_DIR, "figures", "r1_test")
OUT = os.path.join(E6_DIR, "results", "p4_carrada", "r1_test")
V_BIN, R_BIN = 0.42013, 0.1952
C_DL, C_CF, C_TRUTH = "#00E5FF", "#FFD166", "#39FF14"


def load_arm(tag):
    z = np.load(os.path.join(OUT, "{}_peaks.npz".format(tag)), allow_pickle=True)
    pk, ix = z["peaks"], z["index"]          # E6 PATCH: read each array ONCE
    return {(str(a), str(b)): pk[int(c):int(d)] for a, b, c, d in ix}


def draw(ax, db, title, subtitle):
    im = ax.imshow(db - np.median(db), aspect="auto", origin="lower", cmap="magma",
                   vmin=-5, vmax=35, extent=[-13.44, 13.44, 0, 50])
    ax.grid(False)
    ax.set_xlabel("velocity  [m/s]", fontsize=8)
    ax.set_title(title + "\n" + subtitle, fontsize=9.2, fontweight="bold", linespacing=1.5)
    return im


def peaks(ax, pk, colour, marker, label, size=42):
    if len(pk):
        ax.scatter((np.asarray(pk)[:, 1] - 32) * V_BIN, np.asarray(pk)[:, 0] * R_BIN,
                   s=size, marker=marker, facecolors="none", edgecolors=colour,
                   linewidths=1.3, label=label)


def main():
    os.makedirs(FIGS, exist_ok=True)
    thr = {int(r["seed"]): float(r["threshold"]) for r in json.load(
        open(os.path.join(E6_DIR, "configs", "tinycnn_zero_shot_thresholds.json")))["thresholds"]}
    model = det.load_frozen_tinycnn(0)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))

    # ---------------- synthetic panels (bench2_val, illustrative) ----------------
    md = pd.read_csv(os.path.join(ROOT, "data", "bench2_val", "metadata.csv"))
    fdir = os.path.join(ROOT, "data", "bench2_val", "frames")
    for ax, cond, tag in ((axes[0, 0], "isolated", "A"), (axes[0, 1], "clutter_edge", "B")):
        name = md[md.condition == cond].frame_id.iloc[0]      # deterministic: first in file order
        mag = np.load(os.path.join(fdir, name, "rd_map_mag.npy"))
        db = 10 * np.log10(np.maximum(mag.astype(float) ** 2, 1e-300))
        lab = pd.read_csv(os.path.join(fdir, name, "labels.csv"))
        pk_dl = det.apply_threshold(
            det.predict_points_synthetic(model, mag) if hasattr(det, "predict_points_synthetic")
            else __import__("dl.infer", fromlist=["predict_points"]).predict_points(
                model, mag, nms_size=det.NMS_SIZE), thr[0])
        from cfar.cfar import CfarConfig, cfar_ratio_map
        from cfar.detect import all_peaks
        from dl.infer import cfar_valid_mask
        ratio, _, _ = cfar_ratio_map(mag.astype(float) ** 2, CfarConfig(kind="CA"))
        pk_ca = det.apply_threshold(all_peaks(ratio, cfar_valid_mask(db.shape),
                                              nms_size=det.NMS_SIZE), det.cfar_alpha("CA"))
        draw(ax, db, "{} · SYNTHETIC — {}".format(tag, cond),
             "point truth (✕) · single-look speckle · frozen detectors")
        ax.scatter(lab["velocity_bin"].to_numpy() * 0 +
                   (lab["velocity_bin"].to_numpy() - 32) * V_BIN,
                   lab["range_bin"].to_numpy() * R_BIN, s=90, marker="x", c=C_TRUTH,
                   linewidths=1.8, label="true target (point)")
        peaks(ax, pk_dl, C_DL, "o", "TinyCNN peak")
        peaks(ax, pk_ca, C_CF, "s", "CA-CFAR peak", size=58)

    # ---------------- CARRADA panels (Test, preserved peaks) ----------------
    dl_store, ca_store = load_arm("tinycnn_seed0"), load_arm("cfar_CA")
    want = {"away_zero": (axes[1, 0], "C"), "near_zero": (axes[1, 1], "D")}
    found = {}
    for seq in ci.sequences("Test"):
        for fr in ci.frames(seq):
            objs = ci.objects(seq, fr)
            if len(objs) != 1:
                continue
            vs = el.valid_support(objs[0]["dense"])
            st = el.doppler_stratum(vs)
            # Selection is on ANNOTATION GEOMETRY only, never on detector output:
            #   near_zero  -> support actually covers the ridge bin (32)
            #   away_zero  -> support is at least 8 bins clear of it
            if len(vs) <= 40:
                continue
            dmin, dmax = vs[:, 1].min(), vs[:, 1].max()
            ok = ((st == "near_zero" and dmin <= 32 <= dmax)
                  or (st == "away_zero" and min(abs(dmin - 32), abs(dmax - 32)) >= 8))
            if st in want and st not in found and ok:
                found[st] = (seq, fr, objs)
        if len(found) == len(want):
            break

    for st, (ax, tag) in want.items():
        seq, fr, objs = found[st]
        db = ci.load_rd_db(seq, fr)
        pk_dl = det.apply_threshold(dl_store[(seq, fr)], thr[0])
        pk_ca = det.apply_threshold(ca_store[(seq, fr)], det.cfar_alpha("CA"))
        nice = "away from the ridge" if st == "away_zero" else "on the zero-Doppler ridge"
        vs = el.valid_support(objs[0]["dense"])
        draw(ax, db, "{} · REAL CARRADA — object {}".format(tag, nice),
             "{} {} · support Doppler bins {}–{} (ridge = 32) · zero-shot".format(
                 seq, fr, int(vs[:, 1].min()), int(vs[:, 1].max())))
        lo, hi = el.ZERO_DOPPLER_BAND
        ax.axvspan((lo - 32.5) * V_BIN, (hi - 32.5) * V_BIN, color="white", alpha=0.10)
        for o in objs:
            c = o["dense"]
            ax.scatter((c[:, 1] - 32) * V_BIN, c[:, 0] * R_BIN, s=2.5, marker="s",
                       facecolors="none", edgecolors=C_TRUTH, linewidths=0.45,
                       label="annotated dense support")
        peaks(ax, pk_dl, C_DL, "o", "TinyCNN peak")
        peaks(ax, pk_ca, C_CF, "s", "CA-CFAR peak", size=58)

    for ax in axes.ravel():
        ax.set_ylabel("range  [m]", fontsize=8)
        h, l = ax.get_legend_handles_labels()
        u = dict(zip(l, h))
        ax.legend(u.values(), u.keys(), fontsize=6.6, loc="upper right", facecolor="black",
                  edgecolor="none", labelcolor="white", framealpha=0.6)

    fig.suptitle("Same display convention throughout: dB above each frame's own median, "
                 "identical 256×64 axes\n"
                 "Top row synthetic (bench2_val) · bottom row real CARRADA Test — "
                 "DETERMINISTICALLY selected, not chosen to favour either detector",
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.955])
    fig.text(0.5, 0.005,
             "Synthetic frames show coarse single-look speckle and a point target. CARRADA shows a "
             "smoother background, a narrow intense zero-Doppler ridge, and objects with real "
             "spatial extent.",
             ha="center", fontsize=8, color="#333333")
    p = os.path.join(FIGS, "r1_fig6_rd_examples.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))
    for st, (seq, fr, _) in found.items():
        print("  {}: {} {}".format(st, seq, fr))


if __name__ == "__main__":
    sys.exit(main())
