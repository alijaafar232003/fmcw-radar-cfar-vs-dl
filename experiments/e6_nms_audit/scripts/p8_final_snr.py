"""Phase 6c - Pd against measured cell SNR at the final suppression radius.

Reuses the existing 1,300-frame `bench2_snr_sweep` split unchanged; only candidate extraction and
the operating point are recomputed, from the peaks cached by `p6_final_peaks.py`. No dataset is
generated and no model is retrained.

Mirrors scripts/pd_vs_snr_figure.py exactly -- same 2 dB bins, same minimum bin count, same
matched-measured-Pfa thresholds, same seed averaging -- and writes the same two schemas so the
figure and macro generators can read them:

  pd_vs_snr_final_r{R}.csv          n, snr, pd, method, gate, se     (the curves)
  pd_vs_snr_final_r{R}_summary.csv  snr50, loss_db, pd_max           (the headline crossings)

    python p8_final_snr.py --radius 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cfar.cfar import CfarConfig                                            # noqa: E402
from dl.infer import cfar_valid_mask                                        # noqa: E402
from dl.stratified_eval import _at_measured_pfa                             # noqa: E402
from eval import (MatchConfig, evaluate_frame, roc_curve,                   # noqa: E402
                  roc_thresholds)

SHAPE = (256, 64)
TARGET_PFA = 1e-4
SPLIT = "bench2_snr_sweep"
METHODS = ("CA", "GO", "OS", "SO")
PK = os.path.join(EXP, "results", "final", "peaks")
OUT = os.path.join(EXP, "results", "final")


def theory_pd(snr_db, pfa=TARGET_PFA):
    """Clairvoyant square-law detector, told the true noise power."""
    return pfa ** (1.0 / (1.0 + 10 ** (np.asarray(snr_db, float) / 10)))


def load(path):
    a = np.load(path)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def frames_of(pk, ids, tb, valid):
    return [(pk.get(int(i), np.zeros((0, 3))), tb.get(int(i), np.zeros((0, 2))), SHAPE, valid)
            for i in ids]


def hits_by_target(pk, ids, tb, valid, mc, thr):
    out = []
    for i in ids:
        det = pk.get(int(i), np.zeros((0, 3)))
        det = det[det[:, 2] >= thr][:, :2] if (len(det) and thr == thr) else np.zeros((0, 2))
        tru = tb.get(int(i), np.zeros((0, 2)))
        ev = evaluate_frame(det, tru, SHAPE, mc, valid_mask=valid)
        matched = set(ev.matched_truth_idx)
        out.extend(k in matched for k in range(len(tru)))
    return np.array(out, bool)


def curve(snr, hit, edges, min_n=20):
    d = pd.DataFrame(dict(snr=snr, hit=hit))
    d["bin"] = pd.cut(d.snr, edges)
    g = d.groupby("bin", observed=True).agg(n=("hit", "size"), snr=("snr", "mean"),
                                            pd=("hit", "mean")).reset_index(drop=True)
    return g[g.n >= min_n]


def snr_at_pd(c, target=0.5):
    c = c.sort_values("snr")
    if c.pd.min() > target or c.pd.max() < target:
        return np.nan
    return float(np.interp(target, c.pd.to_numpy(), c.snr.to_numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=2)
    a = ap.parse_args()
    R = a.radius
    os.makedirs(OUT, exist_ok=True)

    valid = cfar_valid_mask(SHAPE, CfarConfig())
    idx = pd.read_csv(os.path.join(PK, "%s_index.csv" % SPLIT))
    ids = idx.frame.to_numpy()
    # cell_snr_db lives only in the original CFAR baseline artefact, which is read-only here.
    truths = pd.read_csv(os.path.join(ROOT, "results", "cfar_snr_sweep",
                                      "truths.csv")).sort_values(["frame", "target"])
    tb = {int(f): g[["range_bin", "velocity_bin"]].to_numpy(float)
          for f, g in truths.groupby("frame")}
    snr = truths.cell_snr_db.to_numpy(float)
    edges = np.arange(np.floor(snr.min()), np.ceil(snr.max()) + 2, 2.0)
    grid = np.linspace(-5, 30, 400)
    th50 = float(np.interp(0.5, theory_pd(grid), grid))
    print("%d frames, %d targets, SNR %.1f..%.1f dB; theory Pd 0.5 at %.2f dB"
          % (len(ids), len(truths), snr.min(), snr.max(), th50), flush=True)

    dl_files = sorted(p for p in os.listdir(PK) if p.startswith("%s_r%d_DL_s" % (SPLIT, R)))
    out, rows = [], []
    for tol in (0, 1):
        mc = MatchConfig(tol, tol)
        for m in METHODS:
            pk = load(os.path.join(PK, "%s_r%d_%s.npz" % (SPLIT, R, m)))
            thr = _at_measured_pfa(roc_curve(fr := frames_of(pk, ids, tb, valid),
                                             roc_thresholds(fr, mc), mc), TARGET_PFA)[1]
            c = curve(snr, hits_by_target(pk, ids, tb, valid, mc, thr), edges)
            out.append(c.assign(method=m, gate=tol, se=np.nan))
            rows.append(dict(gate=tol, method=m, snr50=snr_at_pd(c),
                             loss_db=snr_at_pd(c) - th50, pd_max=float(c.pd.max()),
                             max_above_theory=float((c.pd - theory_pd(c.snr)).max())))
        per_seed = []
        for fn in dl_files:
            pk = load(os.path.join(PK, fn))
            thr = _at_measured_pfa(roc_curve(fr := frames_of(pk, ids, tb, valid),
                                             roc_thresholds(fr, mc), mc), TARGET_PFA)[1]
            per_seed.append(curve(snr, hits_by_target(pk, ids, tb, valid, mc, thr), edges))
        d = per_seed[0][["snr", "n"]].copy()
        P = np.vstack([c.pd.to_numpy() for c in per_seed])
        d["pd"] = P.mean(axis=0)
        d["se"] = P.std(axis=0, ddof=1) / np.sqrt(len(per_seed))
        out.append(d.assign(method="DL", gate=tol))
        rows.append(dict(gate=tol, method="DL", snr50=snr_at_pd(d),
                         loss_db=snr_at_pd(d) - th50, pd_max=float(d.pd.max()),
                         max_above_theory=float((d.pd - theory_pd(d.snr)).max())))
        print("  gate %d done" % tol, flush=True)

    pd.concat(out)[["n", "snr", "pd", "method", "gate", "se"]].to_csv(
        os.path.join(OUT, "pd_vs_snr_final_r%d.csv" % R), index=False)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "pd_vs_snr_final_r%d_summary.csv" % R), index=False)
    print("\n" + s.round(3).to_string(index=False))
    print("\ntheory (clairvoyant) reaches Pd 0.5 at %.2f dB" % th50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
