"""Phase 6a - cache detector peaks at the FINAL suppression radius.

This is the only inference pass in the finalisation. It runs each frozen detector once per frame,
extracts candidate peaks at the selected radius, and caches them. Every downstream statistic
(bootstrap, measured Pfa, localisation, Pd-vs-SNR) then reads these caches, so nothing is scored
twice and no model is ever loaded again.

Nothing is retrained and nothing outside this experiment folder is written. The frozen epoch-21
checkpoints and the original radius-5 caches in results/dl/ are read-only inputs here.

Two splits are handled:

  holdout_nms       the fresh 3,200-frame synthetic holdout generated in Phase 2, indexed from its
                    own metadata.csv and per-frame labels.csv.
  bench2_snr_sweep  the existing 1,300-frame SNR-sweep split. Its frame index and truth table
                    (which carry the per-target measured cell SNR) are read from the original
                    results/cfar_snr_sweep/ artefacts; only the peaks are recomputed, because the
                    cached ones there were extracted at radius 5.

    python p6_final_peaks.py --split holdout_nms --radius 2
    python p6_final_peaks.py --split bench2_snr_sweep --radius 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map                            # noqa: E402
from cfar.detect import all_peaks                                           # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                       # noqa: E402
from dl.models import TinyCNN                                               # noqa: E402

SHAPE = (256, 64)
CFAR_METHODS = ("CA", "GO", "OS", "SO")
N_SEEDS = 8
EPOCH = 21
OUT = os.path.join(EXP, "results", "final", "peaks")


def load_models():
    ms = []
    for s in range(N_SEEDS):
        ck = torch.load(os.path.join(ROOT, "results", "dl",
                                     "tinycnn_frozen_s%d_ep%03d.pt" % (s, EPOCH)),
                        weights_only=False)
        m = TinyCNN(width=ck["config"].get("width", 16))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        ms.append(m)
    return ms


def index_holdout():
    """(frame ids, condition per frame, truth table, frame directory) for the fresh holdout."""
    root = os.path.join(EXP, "data", "holdout_nms")
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    fdir = os.path.join(root, "frames")
    ids, cond, tb = [], {}, {}
    for _, r in meta.iterrows():
        fid = int(str(r["frame_id"]).split("_")[-1])
        ids.append(fid)
        cond[fid] = r["condition"]
        lp = os.path.join(fdir, str(r["frame_id"]), "labels.csv")
        pts = np.zeros((0, 2))
        if os.path.exists(lp):
            lab = pd.read_csv(lp)
            if len(lab):
                pts = lab[["range_bin", "velocity_bin"]].to_numpy(float)
        tb[fid] = pts
    return np.array(sorted(ids)), cond, tb, fdir


def index_cached(split):
    """Frame index and truths for a split whose CFAR baseline index already exists.

    Used for `bench2_snr_sweep` (whose truth table carries the per-target cell SNR) and for
    `bench2_val` (needed as a radius-matched baseline by the attribution re-scoring). Both are read
    from the original results/cfar_<split>/ artefacts, which are not modified.
    """
    cv = os.path.join(ROOT, "results", "cfar_" + split.replace("bench2_", ""))
    fm = pd.read_csv(os.path.join(cv, "frames.csv"))
    tr = pd.read_csv(os.path.join(cv, "truths.csv"))
    tb = {int(f): g[["range_bin", "velocity_bin"]].to_numpy(float)
          for f, g in tr.groupby("frame")}
    fdir = os.path.join(ROOT, "data", split, "frames")
    ids = fm["frame"].to_numpy(int)
    cond = ({int(r.frame): r.condition for r in fm.itertuples()} if "condition" in fm.columns
            else {int(i): split for i in ids})
    return ids, cond, tb, fdir


def save(path, pk):
    rows = [np.column_stack([np.full(len(p), i, float), p]) for i, p in sorted(pk.items())
            if len(p)]
    np.savez_compressed(path, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    choices=("holdout_nms", "bench2_snr_sweep", "bench2_val"))
    ap.add_argument("--radius", type=int, required=True)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    valid = cfar_valid_mask(SHAPE, CfarConfig())
    models = load_models()
    ids, cond, tb, fdir = (index_holdout() if a.split == "holdout_nms"
                           else index_cached(a.split))
    print("%s: %d frames, %d seeds, radius %d" % (a.split, len(ids), len(models), a.radius),
          flush=True)

    cfar_pk = {m: {} for m in CFAR_METHODS}
    dl_pk = [{} for _ in models]
    t0 = time.time()
    for k, i in enumerate(ids):
        mag = np.load(os.path.join(fdir, "frame_%06d" % int(i), "rd_map_mag.npy"))
        power = mag.astype(np.float64) ** 2
        for m in CFAR_METHODS:
            ratio = cfar_ratio_map(power, CfarConfig(kind=m))[0]
            cfar_pk[m][int(i)] = all_peaks(ratio, valid, nms_size=a.radius)
        for j, mdl in enumerate(models):
            dl_pk[j][int(i)] = all_peaks(predict_heatmap(mdl, mag), valid, nms_size=a.radius)
        if (k + 1) % 200 == 0:
            el = time.time() - t0
            print("  %5d/%d  %.0f s elapsed, %.0f s remaining"
                  % (k + 1, len(ids), el, el / (k + 1) * (len(ids) - k - 1)), flush=True)

    for m in CFAR_METHODS:
        save(os.path.join(OUT, "%s_r%d_%s.npz" % (a.split, a.radius, m)), cfar_pk[m])
    for j in range(len(models)):
        save(os.path.join(OUT, "%s_r%d_DL_s%d.npz" % (a.split, a.radius, j)), dl_pk[j])

    # Frame index and truths, so downstream scripts need no simulator or dataset access at all.
    pd.DataFrame(dict(frame=ids, condition=[cond[int(i)] for i in ids],
                      n_targets=[len(tb.get(int(i), [])) for i in ids])).to_csv(
        os.path.join(OUT, "%s_index.csv" % a.split), index=False)
    rows = []
    for i in ids:
        for j, (r, v) in enumerate(tb.get(int(i), np.zeros((0, 2)))):
            rows.append(dict(frame=int(i), target=j, range_bin=r, velocity_bin=v))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "%s_truths.csv" % a.split), index=False)

    with open(os.path.join(OUT, "%s_r%d_run_info.json" % (a.split, a.radius)), "w") as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(), split=a.split,
                       radius=a.radius, n_frames=int(len(ids)), n_seeds=len(models),
                       epoch=EPOCH, retrained=False, seconds=round(time.time() - t0, 1)), f,
                  indent=2)
    print("done in %.0f s -> %s" % (time.time() - t0, os.path.relpath(OUT, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
