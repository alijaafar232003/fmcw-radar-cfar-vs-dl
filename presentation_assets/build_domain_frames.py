"""Side-by-side RD frames: synthetic vs real CARRADA, no annotations.

A texture comparison, not a result. Every frame is displayed under one convention --
dB above that frame's own median, magma, -5..35 dB, identical 256x64 axes -- which is the
same convention used by `make_r0_figures.py` and by every RD map printed in the thesis.
Frames are selected by a fixed deterministic rule, stated in the JSON sidecar, not picked
to flatter either domain.

READ-ONLY. Reads cached dataset frames and the CARRADA archive; writes only into
presentation_assets/figures and presentation_assets/data.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")

CARRADA = r"C:/Users/USER/Desktop/CARRADA/Carrada"
META = os.path.join(ROOT, "data", "carrada_meta")

SYNTH_SPLIT = "bench2_val"          # original synthetic, the Stage 1-3 training domain
INFORMED_SPLIT = "carrada_informed_val"   # Stage 4, after the domain intervention

# Six conditions, in the order they appear on the grid. Deliberately includes both
# clutter conditions so the zero-Doppler ridge is visible in the synthetic row too.
CONDITIONS = ["isolated", "multi", "masking", "mixed", "clutter", "clutter_edge"]

EXTENT = [-13.44, 13.44, 0.0, 50.0]
VMIN, VMAX = -5.0, 35.0
CMAP = "magma"
DPI = 215


# ------------------------------------------------------------------ frame loading
def db_above_median(db):
    """The thesis display convention: dB relative to this frame's own median."""
    return db - np.median(db)


def synthetic_frames(split):
    """First frame_id of each condition in sorted order -- a fixed rule, not a choice."""
    md = pd.read_csv(os.path.join(ROOT, "data", split, "metadata.csv"))
    fdir = os.path.join(ROOT, "data", split, "frames")
    out = []
    for cond in CONDITIONS:
        name = sorted(md[md.condition == cond].frame_id)[0]
        mag = np.load(os.path.join(fdir, name, "rd_map_mag.npy")).astype(np.float64)
        db = 10.0 * np.log10(np.maximum(mag ** 2, 1e-300))
        out.append((name, cond, db_above_median(db)))
    return out


def carrada_frames(n):
    """Middle annotated frame of the first n annotated sequences, both recording days."""
    fo = json.load(open(os.path.join(META, "annotations_frame_oriented.json")))
    seqs = [s for s in sorted(fo)
            if os.path.isdir(os.path.join(CARRADA, s, "range_doppler_numpy"))
            and any(len(fo[s][f]) > 0 for f in fo[s])]
    # interleave the two recording days so the row is not all one afternoon
    day1 = [s for s in seqs if s.startswith("2019")]
    day2 = [s for s in seqs if s.startswith("2020")]
    picked = [s for pair in zip(day1, day2) for s in pair][:n]

    out = []
    for seq in picked:
        ann = sorted(f for f in fo[seq] if len(fo[seq][f]) > 0)
        frame = ann[len(ann) // 2]
        db = np.load(os.path.join(CARRADA, seq, "range_doppler_numpy", frame + ".npy"))
        out.append((seq + "/" + frame, "annotated", db_above_median(db.astype(np.float64))))
    return out


# ------------------------------------------------------------------ plotting
def grid(rows, name, subtitle):
    """rows = [(row_label, [(frame_name, tag, rel_db), ...]), ...]"""
    ncol = max(len(r[1]) for r in rows)
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol + 1.6, 3.1 * nrow + 0.9),
                             squeeze=False)
    im = None
    for i, (label, frames) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            if j < len(frames):
                im = ax.imshow(frames[j][2], aspect="auto", origin="lower", cmap=CMAP,
                               vmin=VMIN, vmax=VMAX, extent=EXTENT,
                               interpolation="nearest")
            else:
                ax.axis("off")
                continue
            ax.grid(False)
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([0, 25, 50])
            if j > 0:
                ax.set_yticklabels([])
            if i < nrow - 1:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=9)
        axes[i][0].set_ylabel(label + "\n\nrange  [m]", fontsize=11)
    for j in range(ncol):
        axes[-1][j].set_xlabel("velocity  [m/s]", fontsize=10)

    fig.suptitle(subtitle, fontsize=12.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 0.93, 0.965])
    cax = fig.add_axes([0.945, 0.10, 0.013, 0.78])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("power above that frame's own median  [dB]", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    for ext in ("png", "svg"):
        p = os.path.join(FIG, name + "." + ext)
        fig.savefig(p, dpi=DPI, facecolor="white")
        print("wrote", os.path.relpath(p, ROOT))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    syn = synthetic_frames(SYNTH_SPLIT)
    car = carrada_frames(len(CONDITIONS))
    inf = synthetic_frames(INFORMED_SPLIT)

    sub = ("Range-Doppler frames under one display convention and identical axes\n"
           "no target markers, no annotation — background texture only")

    grid([("synthetic\n(this project)", syn), ("CARRADA\n(real)", car)],
         "s04_domain_frames", sub)

    grid([("synthetic\noriginal", syn),
          ("synthetic\nCARRADA-informed", inf),
          ("CARRADA\nreal", car)],
         "s05_domain_frames_three_rows", sub)

    meta = {
        "display": {"convention": "10*log10(power) minus the frame median",
                    "cmap": CMAP, "vmin": VMIN, "vmax": VMAX, "extent": EXTENT,
                    "interpolation": "nearest", "shape": "256 range x 64 velocity"},
        "selection_rule": {
            "synthetic": "first frame_id in sorted order for each of "
                         + ", ".join(CONDITIONS),
            "carrada": "middle annotated frame of the first sequences of each "
                       "recording day, interleaved"},
        "sources": {"synthetic": "data/" + SYNTH_SPLIT + "/frames/<id>/rd_map_mag.npy",
                    "informed": "data/" + INFORMED_SPLIT + "/frames/<id>/rd_map_mag.npy",
                    "carrada": CARRADA + "/<seq>/range_doppler_numpy/<frame>.npy"},
        "frames": {
            "synthetic_original": [{"id": a, "condition": b} for a, b, _ in syn],
            "synthetic_informed": [{"id": a, "condition": b} for a, b, _ in inf],
            "carrada": [{"id": a} for a, _, _ in car]},
        "measured_context": {
            "note": "quantified in experiments/r_carrada_real_data_validation/results/"
                    "r0_audit.json; repeated here only as a reading aid",
            "background_db_std": {"synthetic": 5.573, "carrada": 2.211},
            "gamma_k": {"synthetic": 0.990, "carrada": 4.771},
            "zero_doppler_ridge_db": {"synthetic": 14.8, "carrada": 24.0},
            "ridge_width_bins": {"synthetic": 4, "carrada": 1}},
    }
    p = os.path.join(DATA, "s04_domain_frames.json")
    json.dump(meta, open(p, "w"), indent=2)
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
