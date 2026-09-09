"""Phase 3 - fast synthetic rebenchmark on the fresh holdout.

Scores the fresh holdout at TWO radii: the radius selected in Phase 1, and radius 5 as a control
on the SAME frames. The control is what separates a protocol change from a sampling change.

Score maps are computed once per frame and both radii applied afterwards, so the control is nearly
free.

Core Chapter 5 outputs only: Pd per condition at matched measured Pfa, achieved Pfa, the +/-1 and
+/-3 gates, and a localisation summary. NO bootstrap, NO figures.

Writes only inside the experiment folder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map                              # noqa: E402
from cfar.detect import all_peaks                                            # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                        # noqa: E402
from dl.models import TinyCNN                                                # noqa: E402
from eval.detection import (MatchConfig, evaluate_frame, roc_curve,          # noqa: E402
                            roc_thresholds, aggregate)
from dl.stratified_eval import _at_measured_pfa                              # noqa: E402

SHAPE = (256, 64)
TARGET_PFA = 1e-4
CFAR_METHODS = ("CA", "GO", "OS", "SO")
SCORED = ("isolated", "masking", "clutter", "clutter_edge", "multi", "mixed", "unresolved")
OUT = os.path.join(EXP, "results", "p3_synth")


def load_models():
    ms = []
    for s in range(8):
        ck = torch.load(os.path.join(ROOT, "results", "dl",
                                     "tinycnn_frozen_s%d_ep021.pt" % s), weights_only=False)
        m = TinyCNN(width=ck["config"].get("width", 16))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        ms.append(m)
    return ms


def build_index(data_root, split):
    meta = pd.read_csv(os.path.join(data_root, split, "metadata.csv"))
    fdir = os.path.join(data_root, split, "frames")
    by_cond, tb = {}, {}
    for _, r in meta.iterrows():
        fid = int(str(r["frame_id"]).split("_")[-1])
        by_cond.setdefault(r["condition"], []).append(fid)
        lp = os.path.join(fdir, str(r["frame_id"]), "labels.csv")
        pts = np.zeros((0, 2))
        if os.path.exists(lp):
            try:
                lab = pd.read_csv(lp)
                if len(lab):
                    pts = lab[["range_bin", "velocity_bin"]].to_numpy(float)
            except Exception:
                pass
        tb[fid] = pts
    return {k: np.array(sorted(v)) for k, v in by_cond.items()}, tb, fdir


def score_at(frames, mc):
    """Pd, achieved Pfa, threshold and localisation, at the matched-Pfa threshold for `mc`."""
    roc = roc_curve(frames, roc_thresholds(frames, mc), mc)
    _, thr = _at_measured_pfa(roc, TARGET_PFA)
    evs = []
    for p, t, s, v in frames:
        det = p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2))
        evs.append(evaluate_frame(det, t, s, mc, valid_mask=v))
    a = aggregate(evs)
    dr = np.abs(np.concatenate([e.dr for e in evs if e.dr])) if any(e.dr for e in evs) else np.array([np.nan])
    dv = np.abs(np.concatenate([e.dv for e in evs if e.dv])) if any(e.dv for e in evs) else np.array([np.nan])
    rmse = float(np.sqrt(np.nanmean(dr ** 2 + dv ** 2))) if dr.size else np.nan
    return dict(pd=a["pd"], pfa=a["pfa_per_cell"], threshold=float(thr),
                dr_bins=float(np.nanmean(dr)), dv_bins=float(np.nanmean(dv)), rmse_bins=rmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout_nms")
    ap.add_argument("--data-root", default=os.path.join(EXP, "data"))
    ap.add_argument("--radii", type=int, nargs="+", required=True,
                    help="selected radius first, then the control (e.g. --radii 2 5)")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    valid = cfar_valid_mask(SHAPE, CfarConfig())
    models = load_models()
    by_cond, tb, fdir = build_index(a.data_root, a.split)
    print("%d seeds; conditions: %s" % (len(models), ", ".join(sorted(by_cond))), flush=True)

    rows = []
    for cond in SCORED:
        ids = by_cond.get(cond, np.array([], dtype=int))
        if not len(ids):
            continue
        print("  %-13s %d frames" % (cond, len(ids)), flush=True)

        ratio_maps = {m: {} for m in CFAR_METHODS}
        logit_maps = [{} for _ in models]
        for i in ids:
            mag = np.load(os.path.join(fdir, "frame_%06d" % int(i), "rd_map_mag.npy"))
            power = mag.astype(np.float64) ** 2
            for m in CFAR_METHODS:
                ratio_maps[m][int(i)] = cfar_ratio_map(power, CfarConfig(kind=m))[0]
            for k, mdl in enumerate(models):
                logit_maps[k][int(i)] = predict_heatmap(mdl, mag)

        for nms in a.radii:
            def fr_of(maps):
                return [(all_peaks(maps[int(i)], valid, nms_size=nms),
                         tb.get(int(i), np.zeros((0, 2))), SHAPE, valid) for i in ids]

            for gate in (1, 3):
                mc = MatchConfig(gate, gate)
                for m in CFAR_METHODS:
                    rows.append(dict(condition=cond, nms=nms, gate=gate, method=m,
                                     **score_at(fr_of(ratio_maps[m]), mc)))
                per = [score_at(fr_of(lm), mc) for lm in logit_maps]
                agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
                agg["pd_se"] = float(np.std([p["pd"] for p in per], ddof=1) / np.sqrt(len(per)))
                rows.append(dict(condition=cond, nms=nms, gate=gate, method="DL", **agg))
            print("    nms %d done" % nms, flush=True)

    df = pd.DataFrame(rows)
    p = os.path.join(OUT, "holdout_scores.csv")
    df.to_csv(p, index=False)
    print("\nwrote", os.path.relpath(p, ROOT))
    with open(os.path.join(OUT, "run_info.json"), "w") as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(), split=a.split,
                       radii=a.radii, n_seeds=len(models), target_pfa=TARGET_PFA), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
