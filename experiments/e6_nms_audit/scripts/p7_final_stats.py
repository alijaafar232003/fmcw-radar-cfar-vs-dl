"""Phase 6b - the final synthetic benchmark statistics, from the cached radius-2 peaks.

Reproduces the four Chapter 5 artefacts, with identical schemas to the originals so the existing
figure and macro generators can read them:

  bootstrap_final_r{R}_tol{T}.csv    paired frame bootstrap, per condition, Bonferroni-corrected
                                     (mirrors scripts/bootstrap_comparison.py)
  measured_pfa_final_r{R}_tol1.csv   achieved per-cell Pfa at each matched operating point
                                     (mirrors scripts/measured_pfa_table.py)
  localization_final_r{R}.csv        localisation over true positives
                                     (mirrors scripts/localization_table.py)

No inference happens here: `p6_final_peaks.py` already ran every detector once and cached the
candidate lists. Nothing outside the experiment folder is written, and the originals in
results/dl/ are neither read as inputs nor touched.

Two efficiency measures, neither of which changes a result:

  * Each (method, condition, gate) is swept and scored ONCE. The original scripts are three
    separate programs that each re-derive the same matched threshold, so running them back to back
    would repeat the most expensive step three times.
  * Per-frame scoring goes through `fast_eval.FrameScorer`, which hoists the threshold-independent
    part of `evaluate_frame` out of the sweep. `verify()` asserts bit-identical agreement with
    `eval.detection.evaluate_frame` before any of it is used; the assertion is part of the run.

A smaller suppression radius leaves roughly four times as many CFAR candidates per frame as the
superseded radius did, so without both measures this analysis would take hours rather than minutes.

    python p7_final_stats.py --radius 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfar.cfar import CfarConfig                                            # noqa: E402
from dl.infer import cfar_valid_mask                                        # noqa: E402
from dl.stratified_eval import _at_measured_pfa                             # noqa: E402
from eval import MatchConfig, roc_thresholds                                # noqa: E402

from fast_eval import FrameScorer, verify                                   # noqa: E402

SHAPE = (256, 64)
TARGET_PFA = 1e-4
N_BOOT = 2000
BOOT_SEED = 0
RANGE_RES, VEL_RES = 0.19518, 0.42013
SCORED = ("clutter", "clutter_edge", "isolated", "masking", "mixed", "multi", "unresolved")
METHODS = ("CA", "GO", "OS", "SO", "DL")
PK = os.path.join(EXP, "results", "final", "peaks")
OUT = os.path.join(EXP, "results", "final")


def load(path):
    a = np.load(path)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def sweep(scorers, thresholds):
    """ROC over a threshold grid, in the same columns `_at_measured_pfa` expects."""
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


def score_group(pk, ids, tb, valid, tol):
    """Sweep to the matched operating point and score there. Returns everything downstream needs."""
    frames = [(pk.get(int(i), np.zeros((0, 3))), tb.get(int(i), np.zeros((0, 2))), SHAPE, valid)
              for i in ids]
    grid = roc_thresholds(frames, MatchConfig(tol, tol))
    scorers = [FrameScorer(f[0], f[1], SHAPE, tol, tol, valid) for f in frames]
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
    return dict(
        threshold=float(thr), hits=hits, n=ntru,
        pd=float(hits.sum() / ntru.sum()) if ntru.sum() else np.nan,
        pfa=fp / cells if cells else np.nan,
        dr=float(dr.mean()) if dr.size else np.nan,
        dv=float(dv.mean()) if dv.size else np.nan,
        rmse=float(np.sqrt((dr ** 2 + dv ** 2).mean())) if dr.size else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--split", default="holdout_nms")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    R, SP = a.radius, a.split
    t00 = time.time()

    os.makedirs(OUT, exist_ok=True)
    valid = cfar_valid_mask(SHAPE, CfarConfig())
    idx = pd.read_csv(os.path.join(PK, "%s_index.csv" % SP))
    tr = pd.read_csv(os.path.join(PK, "%s_truths.csv" % SP))
    tb = {int(f): g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in tr.groupby("frame")}

    srcs = {m: [load(os.path.join(PK, "%s_r%d_%s.npz" % (SP, R, m)))]
            for m in ("CA", "GO", "OS", "SO")}
    dl = sorted(p for p in os.listdir(PK) if p.startswith("%s_r%d_DL_s" % (SP, R)))
    srcs["DL"] = [load(os.path.join(PK, p)) for p in dl]
    print("radius %d, %d frames, %d seeds  (%.0f s to load)"
          % (R, len(idx), len(srcs["DL"]), time.time() - t00), flush=True)

    # ------------------------------------------------------------------ correctness gate
    ids0 = idx.loc[idx.condition == "masking", "frame"].to_numpy()[:80]
    fr0 = [(srcs["CA"][0].get(int(i), np.zeros((0, 3))), tb.get(int(i), np.zeros((0, 2))),
            SHAPE, valid) for i in ids0]
    grid0 = roc_thresholds(fr0, MatchConfig(1, 1))
    n = verify(fr0, grid0[::12], 1, 1, SHAPE, valid, n=8)
    n += verify(fr0, grid0[::12], 3, 3, SHAPE, valid, n=8)
    print("fast scorer verified against eval.detection on %d frame-threshold pairs" % n,
          flush=True)

    # ------------------------------------------------------------------ one scoring pass
    # (condition, gate, method, seed) -> the matched operating point and everything measured there.
    S = {}
    for gate in (1, 3):
        for cond in sorted(idx.condition.unique()):
            ids = idx.loc[idx.condition == cond, "frame"].to_numpy()
            for m in METHODS:
                S[(cond, gate, m)] = [score_group(pk, ids, tb, valid, gate) for pk in srcs[m]]
            print("  scored %-13s gate %d   (%.0f s)" % (cond, gate, time.time() - t00),
                  flush=True)

    # ------------------------------------------------------------------ bootstrap, tol 1 and 3
    rng = np.random.default_rng(BOOT_SEED)
    for tol in (1, 3):
        rows = []
        for cond in SCORED:
            ids = idx.loc[idx.condition == cond, "frame"].to_numpy()
            # The classical arm is the BEST variant on this condition, chosen afresh at this
            # radius rather than inherited from the superseded benchmark.
            best_m = max(("CA", "GO", "OS", "SO"), key=lambda m: S[(cond, tol, m)][0]["pd"])
            best = S[(cond, tol, best_m)][0]
            dl_h = np.mean([s["hits"] for s in S[(cond, tol, "DL")]], axis=0)

            diffs = np.empty(a.n_boot)
            for b in range(a.n_boot):
                k = rng.integers(0, len(ids), len(ids))
                tot = best["n"][k].sum()
                diffs[b] = (dl_h[k].sum() - best["hits"][k].sum()) / tot if tot else np.nan
            obs = (dl_h.sum() - best["hits"].sum()) / best["n"].sum()
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            rows.append(dict(condition=cond, n_frames=len(ids), targets=int(best["n"].sum()),
                             cfar_method=best_m, cfar_pd=best["pd"],
                             dl_pd=dl_h.sum() / best["n"].sum(), diff=obs, ci_lo=lo, ci_hi=hi,
                             p_boot=2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
        df = pd.DataFrame(rows)
        alpha = 0.05 / len(df)
        df["sig (Bonf)"] = df.p_boot < alpha
        df["crosses 0"] = (df.ci_lo <= 0) & (df.ci_hi >= 0)
        df.to_csv(os.path.join(OUT, "bootstrap_final_r%d_tol%d.csv" % (R, tol)), index=False)
        print("\n=== gate +/-%d  (Bonferroni alpha %.4f) ===" % (tol, alpha))
        print(df[["condition", "cfar_method", "cfar_pd", "dl_pd", "diff", "ci_lo", "ci_hi",
                  "p_boot", "sig (Bonf)"]].round(4).to_string(index=False))

    # ------------------------------------------------------------------ measured Pfa (tol 1)
    rows = []
    for cond in sorted(idx.condition.unique()):
        for m in METHODS:
            v = float(np.mean([s["pfa"] for s in S[(cond, 1, m)]]))
            rows.append(dict(condition=cond, method=m, n_seeds=len(S[(cond, 1, m)]),
                             measured_pfa=v, ratio_to_target=v / TARGET_PFA))
    pf = pd.DataFrame(rows)
    pf.to_csv(os.path.join(OUT, "measured_pfa_final_r%d_tol1.csv" % R), index=False)
    print("\nmeasured Pfa / target: min %.3f  max %.3f"
          % (pf.ratio_to_target.min(), pf.ratio_to_target.max()))

    # ------------------------------------------------------------------ localisation
    # Per-condition rows come from the pass above. The pooled ALL row needs its own sweep, because
    # the operating point is defined by the frames being scored.
    rows = []
    for tol in (1, 3):
        for cond in SCORED:
            for m in METHODS:
                st = S[(cond, tol, m)]
                rows.append(_loc_row(tol, cond, m, st))
        for m in METHODS:
            st = [score_group(pk, idx.frame.to_numpy(), tb, valid, tol) for pk in srcs[m]]
            rows.append(_loc_row(tol, "ALL", m, st))
            print("  pooled localisation %-3s gate %d   (%.0f s)" % (m, tol, time.time() - t00),
                  flush=True)
    loc = pd.DataFrame(rows)
    loc.to_csv(os.path.join(OUT, "localization_final_r%d.csv" % R), index=False)
    print("\n=== localisation, all conditions pooled, gate +/-1 ===")
    print(loc[(loc.gate == 1) & (loc.group == "ALL")]
          [["method", "pd", "dr_bins", "dv_bins", "rmse_bins", "range_m", "vel_mps"]]
          .round(4).to_string(index=False))
    print("\ntotal %.0f s -> %s" % (time.time() - t00, os.path.relpath(OUT, ROOT)))
    return 0


def _loc_row(tol, group, method, stats):
    def ms(key):
        v = np.array([s[key] for s in stats], float)
        return (float(v.mean()),
                float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))

    pd_m, pd_e = ms("pd")
    dr_m, dr_e = ms("dr")
    dv_m, dv_e = ms("dv")
    rm_m, rm_e = ms("rmse")
    return dict(gate=tol, group=group, method=method, n_seeds=len(stats),
                pd=pd_m, pd_se=pd_e, dr_bins=dr_m, dr_bins_se=dr_e,
                dv_bins=dv_m, dv_bins_se=dv_e, rmse_bins=rm_m, rmse_bins_se=rm_e,
                range_m=dr_m * RANGE_RES, vel_mps=dv_m * VEL_RES)


if __name__ == "__main__":
    sys.exit(main())
