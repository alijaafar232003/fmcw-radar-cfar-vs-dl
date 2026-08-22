"""Export ROC / FROC curves and their matched operating points, for the methodology notebook.

`notebooks/evaluation_metrics_overview.ipynb` explains how a threshold sweep becomes a ROC and how
the matched-measured-Pfa operating point is picked off it. That explanation needs a real curve, but
recomputing one inside the notebook costs ~6 s per method per condition, which would make `Run All`
unusable. So the curves are exported here and the notebook reads them.

**Runs on `bench2_val` by default, deliberately.** These curves are an illustration of the
*procedure*, not a result. `bench2_val` is the split reserved for anything still being decided, so
using it keeps a methodology figure from reading as a second look at the reported split.

Two artefacts per run:

* `roc_curves_<split>_tol<k>.csv`   -- the full swept curve, one row per (condition, method,
  threshold): Pd, measured per-cell Pfa, false alarms per frame, TP/FP/FN.
* `roc_operating_points_<split>_tol<k>.csv` -- one row per (condition, method): the threshold the
  matched-Pfa rule selects, the Pd **interpolated** at the target rate, the Pd **measured** by
  re-scoring every frame at that threshold, and the Pfa actually achieved there.

The interpolated-vs-measured pair is the point of the second file: the threshold is found by
interpolating against log Pfa, but Pd is then *measured* at it, because interpolating Pd would
report a number no run of the detector ever produced.

Uses cached peaks only -- no frame data is read.

    python scripts/roc_curve_export.py --split bench2_val --conditions isolated masking clutter_edge
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig                                     # noqa: E402
from dl.infer import cfar_valid_mask                                 # noqa: E402
from dl.stratified_eval import _at_measured_pfa                      # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,            # noqa: E402
                  roc_curve, roc_thresholds)

TARGET_PFA = 1e-4
SHAPE = (256, 64)
CFAR_METHODS = ("CA", "GO", "OS", "SO")
DLR = ROOT / "results" / "dl"


def load_peaks(p: Path) -> dict:
    a = np.load(p)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="bench2_val")
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--epoch", type=int, default=21)
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("--conditions", nargs="+",
                    default=["isolated", "masking", "clutter_edge"])
    ap.add_argument("--dl-seed", type=int, default=0,
                    help="ONE seed only. The notebook illustrates the procedure; averaging seeds "
                         "would blur the curve without changing what it demonstrates.")
    args = ap.parse_args()

    PK = args.split.replace("bench2_", "")
    CV = ROOT / "results" / f"cfar_{PK}"
    mc = MatchConfig(args.tolerance, args.tolerance)
    valid = cfar_valid_mask(SHAPE, CfarConfig())

    fm = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}

    srcs = {m: load_peaks(CV / f"peaks_{m}.npz") for m in CFAR_METHODS
            if (CV / f"peaks_{m}.npz").exists()}
    dl_path = DLR / f"peaks_{PK}_{args.tag}_s{args.dl_seed}_ep{args.epoch:03d}.npz"
    if dl_path.exists():
        srcs["DL"] = load_peaks(dl_path)
    else:
        print(f"WARNING: no DL cache at {dl_path.name} -- exporting CFAR only")
    print(f"split {args.split} | methods {list(srcs)} | gate +/-{args.tolerance}")

    curves, points = [], []

    for cond in args.conditions:
        ids = fm.loc[fm.condition == cond, "frame"].to_numpy()
        if not len(ids):
            print(f"  {cond}: no frames, skipped")
            continue

        for m, pk in srcs.items():
            fr = [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))), SHAPE, valid)
                  for i in ids]

            roc = roc_curve(fr, roc_thresholds(fr, mc), mc)
            roc.insert(0, "method", m)
            roc.insert(0, "condition", cond)
            curves.append(roc)

            pd_interp, thr = _at_measured_pfa(roc, TARGET_PFA)

            # Pd MEASURED at the selected threshold: re-score every frame there.
            evs = [evaluate_frame(
                p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2)),
                t, s, mc, valid_mask=v) for p, t, s, v in fr]
            a = aggregate(evs)

            points.append(dict(condition=cond, method=m, n_frames=len(ids),
                               threshold=thr, pd_interp=pd_interp, pd_measured=a["pd"],
                               measured_pfa=a["pfa_per_cell"],
                               ratio_to_target=a["pfa_per_cell"] / TARGET_PFA,
                               fa_per_frame=a["fa_per_frame"],
                               tp=a["tp"], fp=a["fp"], fn=a["fn"]))
            print(f"  {cond:<14} {m:<3} thr {thr:9.4f}  pd_interp {pd_interp:.4f}  "
                  f"pd_measured {a['pd']:.4f}  pfa {a['pfa_per_cell']:.3e}")

    C = pd.concat(curves, ignore_index=True)
    P = pd.DataFrame(points)

    c_out = DLR / f"roc_curves_{PK}_tol{args.tolerance}.csv"
    p_out = DLR / f"roc_operating_points_{PK}_tol{args.tolerance}.csv"
    C.to_csv(c_out, index=False)
    P.to_csv(p_out, index=False)

    d = (P.pd_measured - P.pd_interp).abs()
    print(f"\n|pd_measured - pd_interp|: max {d.max():.4f}, median {d.median():.4f}")
    print(f"measured Pfa / target: {P.ratio_to_target.min():.3f} to {P.ratio_to_target.max():.3f}")
    print(f"\nwrote {c_out}\nwrote {p_out}")


if __name__ == "__main__":
    main()
