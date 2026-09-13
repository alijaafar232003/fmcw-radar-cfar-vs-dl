"""Export a pooled ROC for the FINAL synthetic benchmark, under the FINAL protocol.

Reads the frozen radius-\\ProtocolNms candidate caches produced by
`experiments/e6_nms_audit/scripts/p6_final_peaks.py` and sweeps the score threshold through the
same `fast_eval.FrameScorer` path that produced every number in Chapter 5. Nothing is re-run and no
detector is re-inferred; this is a different view of the sweep the matched-operating-point rule
already performs.

Pooling: the seven scored conditions are pooled into one representative curve (the `noise` stratum
carries no targets and so contributes no Pd). TinyCNN is the mean over its eight frozen seeds,
averaged on a common measured-Pfa grid because the seeds do not share a threshold scale.

Out: experiments/e6_nms_audit/results/final/roc_final_r2_tol1.csv

    venv/Scripts/python.exe scripts/export_roc_final.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments", "e6_nms_audit")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(EXP, "scripts"))

from cfar.cfar import CfarConfig                                            # noqa: E402
from dl.infer import cfar_valid_mask                                       # noqa: E402
from eval import MatchConfig, roc_thresholds                               # noqa: E402
from fast_eval import FrameScorer, verify                                  # noqa: E402

SHAPE = (256, 64)
TOL = 1
SPLIT = "holdout_nms"
RADIUS = 2
SEEDS = range(8)
SCORED = ("clutter", "clutter_edge", "isolated", "masking", "mixed", "multi", "unresolved")
PK = os.path.join(EXP, "results", "final", "peaks")
OUT = os.path.join(EXP, "results", "final", "roc_final_r%d_tol%d.csv" % (RADIUS, TOL))


def load(path):
    a = np.load(path)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def sweep(scorers, thresholds):
    rows = []
    for thr in thresholds:
        tp = fp = fn = cells = 0
        for s in scorers:
            a, b, c, d, _, _, _ = s.at(thr)
            tp += a
            fp += b
            fn += c
            cells += d
        rows.append(dict(threshold=float(thr),
                         pd=tp / (tp + fn) if (tp + fn) else np.nan,
                         pfa_per_cell=fp / cells if cells else np.nan))
    return pd.DataFrame(rows)


def curve_for(pk, ids, tb, valid, check=False):
    frames = [(pk.get(int(i), np.zeros((0, 3))), tb.get(int(i), np.zeros((0, 2))), SHAPE, valid)
              for i in ids]
    grid = roc_thresholds(frames, MatchConfig(TOL, TOL))
    if check:
        n = verify(frames, list(grid[:: max(1, len(grid) // 6)]), TOL, TOL, SHAPE, valid, n=25)
        print("  FrameScorer verified against evaluate_frame on %d (frame, threshold) pairs" % n)
    scorers = [FrameScorer(f[0], f[1], SHAPE, TOL, TOL, valid) for f in frames]
    return sweep(scorers, grid)


def main():
    valid = cfar_valid_mask(SHAPE, CfarConfig())
    idx = pd.read_csv(os.path.join(PK, "%s_index.csv" % SPLIT))
    tr = pd.read_csv(os.path.join(PK, "%s_truths.csv" % SPLIT))
    tb = {int(f): g[["range_bin", "velocity_bin"]].to_numpy(float)
          for f, g in tr.groupby("frame")}
    ids = idx.loc[idx.condition.isin(SCORED), "frame"].to_numpy()
    print("%s | %d scored frames | radius %d | gate +/-%d" % (SPLIT, len(ids), RADIUS, TOL))

    out = []
    first = True
    for m in ("CA", "GO", "OS", "SO"):
        c = curve_for(load(os.path.join(PK, "%s_r%d_%s.npz" % (SPLIT, RADIUS, m))),
                      ids, tb, valid, check=first)
        first = False
        c.insert(0, "method", m)
        out.append(c[["method", "threshold", "pd", "pfa_per_cell"]])
        print("  %-3s %4d points, Pfa %.2e..%.2e" % (m, len(c), c.pfa_per_cell.min(),
                                                     c.pfa_per_cell.max()))

    # TinyCNN: average Pd across seeds on a common measured-Pfa grid.
    per_seed = []
    for s in SEEDS:
        c = curve_for(load(os.path.join(PK, "%s_r%d_DL_s%d.npz" % (SPLIT, RADIUS, s))),
                      ids, tb, valid)
        per_seed.append(c.dropna(subset=["pfa_per_cell", "pd"]))
    lo = max(c.pfa_per_cell[c.pfa_per_cell > 0].min() for c in per_seed)
    hi = min(c.pfa_per_cell.max() for c in per_seed)
    gridp = np.geomspace(lo, hi, 240)
    stack = []
    for c in per_seed:
        c = c[c.pfa_per_cell > 0].sort_values("pfa_per_cell")
        stack.append(np.interp(np.log10(gridp), np.log10(c.pfa_per_cell.to_numpy()),
                               c.pd.to_numpy()))
    dl = pd.DataFrame(dict(method="DL", threshold=np.nan, pd=np.mean(stack, axis=0),
                           pfa_per_cell=gridp))
    out.append(dl)
    print("  DL  %4d points over %d seeds, Pfa %.2e..%.2e" % (len(dl), len(per_seed), lo, hi))

    df = pd.concat(out, ignore_index=True)
    df.to_csv(OUT, index=False)
    print("wrote", os.path.relpath(OUT, ROOT), "(%d rows)" % len(df))


if __name__ == "__main__":
    main()
