"""Phase 6f - complete E2 (integrated CFAR) at the final suppression radius.

`p10_patch_attribution.py` built an isolated copy of `run_integrated_cfar.py` at radius 2 and that
copy extracted the integrated detections, but its analysis stage never finished: it sweeps a ROC per
arm per condition per gate through `evaluate_frame`, which on this split is a multi-hour job because
the integrated arms carry ~285 candidates per frame.

This script does ONLY that analysis stage. Nothing is re-extracted and nothing is re-calibrated:

  * integrated peaks      <- results/p10_attribution/e2/peaks_int_{kind}_{stat}.npz  (radius 2)
  * plain-CFAR baselines  <- results/final/peaks/bench2_val_r2_{kind}.npz            (radius 2)
  * TinyCNN baseline      <- results/final/peaks/bench2_val_r2_DL_s*.npz             (radius 2)
  * alpha calibration     <- results/p10_attribution/e2/calibration.csv
  * PSF energy capture    <- results/p10_attribution/e2/psf_energy_capture.csv

The last two are properties of the kernel and of the noise-only false-alarm rate per CELL. Neither
passes through non-maximum suppression, so neither depends on the radius.

The arithmetic below is the original's, in the same order, with one substitution: the threshold
sweep and the scoring at the operating point go through `fast_eval.FrameScorer` instead of calling
`evaluate_frame` once per threshold per frame. `verify()` asserts the two agree exactly before any
result is computed.

Writes into results/p10_attribution/e2/ only. The original experiment folder is not touched.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfar.cfar import CfarConfig                                        # noqa: E402
from dl.infer import cfar_valid_mask                                    # noqa: E402
from dl.stratified_eval import _at_measured_pfa                         # noqa: E402
from eval import MatchConfig, roc_thresholds                            # noqa: E402

from fast_eval import FrameScorer, verify                               # noqa: E402

SPLIT = "bench2_val"
SHAPE = (256, 64)
RADIUS = 2
TARGET_PFA = 1e-4
N_BOOT = 2000
KINDS = ("CA", "GO", "OS", "SO")
STATS = ("uniform", "psf")
RANGE_RES, VEL_RES = 0.19518, 0.42013
VALID = cfar_valid_mask(SHAPE, CfarConfig())

RES = os.path.join(EXP, "results", "p10_attribution", "e2")
R2PK = os.path.join(EXP, "results", "final", "peaks")


def load_peaks(path):
    a = np.load(path)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def frames_of(pk, ids, tb):
    return [(pk.get(int(i), np.zeros((0, 3))), tb.get(int(i), np.zeros((0, 2))), SHAPE, VALID)
            for i in ids]


def sweep(scorers, thresholds):
    """ROC over a threshold grid, in the columns `_at_measured_pfa` expects."""
    rows = []
    for thr in thresholds:
        tp = fp = fn = cells = 0
        for s in scorers:
            a, b, c, d, _, _, _ = s.at(thr)
            tp += a
            fp += b
            fn += c
            cells += d
        rows.append(dict(threshold=float(thr), pd=tp / (tp + fn) if (tp + fn) else np.nan,
                         pfa_per_cell=fp / cells if cells else np.nan,
                         fa_per_frame=fp / len(scorers), tp=tp, fp=fp, fn=fn))
    return pd.DataFrame(rows)


def score_arm(pk, ids, tb, tol):
    """Sweep to the matched operating point and score there, exactly as the original does."""
    fr = frames_of(pk, ids, tb)
    grid = roc_thresholds(fr, MatchConfig(tol, tol))
    scorers = [FrameScorer(f[0], f[1], SHAPE, tol, tol, VALID) for f in fr]
    _, thr = _at_measured_pfa(sweep(scorers, grid), TARGET_PFA)

    hits, ntru, fp, cells, dr, dv = [], [], 0, 0, [], []
    for s in scorers:
        a, b, c, d, x, y, _ = s.at(thr)
        hits.append(a)
        ntru.append(a + c)
        fp += b
        cells += d
        dr.extend(x)
        dv.extend(y)
    hits = np.array(hits, float)
    ntru = np.array(ntru, float)
    dr = np.abs(np.asarray(dr, float))
    dv = np.abs(np.asarray(dv, float))
    return dict(threshold=float(thr), hits=hits, n=ntru,
                pfa=fp / cells if cells else np.nan,
                fa_per_frame=fp / len(scorers),
                dr=float(dr.mean()) if dr.size else np.nan,
                dv=float(dv.mean()) if dv.size else np.nan)


def main():
    t_start = time.perf_counter()
    print("=" * 78)
    print("E2 INTEGRATED CFAR - analysis at radius %d on %s" % (RADIUS, SPLIT))
    print("=" * 78)

    fm = pd.read_csv(os.path.join(ROOT, "results", "cfar_val", "frames.csv"))
    truths = pd.read_csv(os.path.join(ROOT, "results", "cfar_val", "truths.csv"))
    TB = {int(f): g[["range_bin", "velocity_bin"]].to_numpy(float)
          for f, g in truths.groupby("frame")}
    conds = sorted(c for c in fm.condition.unique() if c != "noise")

    # ---------------- load every arm's radius-2 candidates
    peaks = {(k, s): load_peaks(os.path.join(RES, "peaks_int_%s_%s.npz" % (k, s)))
             for k in KINDS for s in STATS}
    base = {k: load_peaks(os.path.join(R2PK, "bench2_val_r2_%s.npz" % k)) for k in KINDS}
    dl = [load_peaks(os.path.join(R2PK, "bench2_val_r2_DL_s%d.npz" % s)) for s in range(8)]
    print("loaded %d integrated arms, %d CFAR baselines, %d DL seeds (all radius %d)"
          % (len(peaks), len(base), len(dl), RADIUS))

    # ---------------- prove the fast path is the same protocol
    ids_chk = fm.loc[fm.condition == "multi", "frame"].to_numpy()
    fr_chk = frames_of(peaks[("CA", "uniform")], ids_chk, TB)
    g = roc_thresholds(fr_chk, MatchConfig(1, 1))
    n = verify(fr_chk, g[::15], 1, 1, SHAPE, VALID, n=6)
    n += verify(fr_chk, g[::15], 3, 3, SHAPE, VALID, n=6)
    print("fast_eval verified bit-identical to evaluate_frame on %d frame-threshold pairs" % n)

    rng = np.random.default_rng(0)
    summary, boots, locs = [], [], []

    for tol in (1, 3):
        for cond in conds:
            t0 = time.perf_counter()
            cid = fm.loc[fm.condition == cond, "frame"].to_numpy()

            arms = {}
            for k in KINDS:
                arms["CFAR-%s" % k] = base[k]
                for s in STATS:
                    arms["INT-%s-%s" % (s, k)] = peaks[(k, s)]

            hits, n_t = {}, None
            for name, pk in arms.items():
                r = score_arm(pk, cid, TB, tol)
                hits[name] = r["hits"]
                n_t = r["n"]
                summary.append(dict(gate=tol, condition=cond, arm=name, threshold=r["threshold"],
                                    pd=r["hits"].sum() / r["n"].sum(), measured_pfa=r["pfa"],
                                    fa_per_frame=r["fa_per_frame"]))
                locs.append(dict(gate=tol, condition=cond, arm=name,
                                 dr_bins=r["dr"], dv_bins=r["dv"]))

            dl_h = np.mean([score_arm(pk, cid, TB, tol)["hits"] for pk in dl], axis=0)
            hits["DL"] = dl_h
            summary.append(dict(gate=tol, condition=cond, arm="DL", threshold=np.nan,
                                pd=dl_h.sum() / n_t.sum(), measured_pfa=np.nan,
                                fa_per_frame=np.nan))

            best_cfar = max(KINDS, key=lambda k: hits["CFAR-%s" % k].sum())
            k_idx = rng.integers(0, len(cid), size=(N_BOOT, len(cid)))
            tot = n_t[k_idx].sum(axis=1)

            def boot(a, b):
                d = (a[k_idx].sum(axis=1) - b[k_idx].sum(axis=1)) / tot
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                return float(lo), float(hi), float(p)

            for s in STATS:
                best_int = max(KINDS, key=lambda k: hits["INT-%s-%s" % (s, k)].sum())
                hi_ = hits["INT-%s-%s" % (s, best_int)]
                hc = hits["CFAR-%s" % best_cfar]
                hd = hits["DL"]
                lo1, hi1, p1 = boot(hi_, hc)
                lo2, hi2, p2 = boot(hd, hi_)
                boots.append(dict(
                    gate=tol, condition=cond, statistic=s,
                    best_cfar=best_cfar, best_int=best_int,
                    pd_cfar=hc.sum() / n_t.sum(), pd_int=hi_.sum() / n_t.sum(),
                    pd_dl=hd.sum() / n_t.sum(),
                    delta_integration=(hi_.sum() - hc.sum()) / n_t.sum(),
                    di_lo=lo1, di_hi=hi1, di_p=p1,
                    delta_remaining=(hd.sum() - hi_.sum()) / n_t.sum(),
                    dr_lo=lo2, dr_hi=hi2, dr_p=p2))
            print("  gate +/-%d  %-14s done  (%.1f s)" % (tol, cond, time.perf_counter() - t0),
                  flush=True)

    pd.DataFrame(summary).to_csv(os.path.join(RES, "summary_pd.csv"), index=False)
    pd.DataFrame(locs).to_csv(os.path.join(RES, "localization.csv"), index=False)
    B = pd.DataFrame(boots)
    B["alpha_bonf"] = 0.05 / len(conds)
    B["sig_integration"] = B.di_p < B.alpha_bonf
    B["sig_remaining"] = B.dr_p < B.alpha_bonf
    B.to_csv(os.path.join(RES, "deltas.csv"), index=False)

    d1 = B[B.gate == 1]
    print("")
    for s in STATS:
        x = d1[d1.statistic == s]
        print("gate +/-1  %-8s  mean dPd %+.4f   negative on %d/%d   significant on %d/%d"
              % (s, x.delta_integration.mean(), int((x.delta_integration < 0).sum()), len(x),
                 int(x.sig_integration.sum()), len(x)))
    print("\ntotal %.1f min -> %s" % ((time.perf_counter() - t_start) / 60, RES))


if __name__ == "__main__":
    main()
