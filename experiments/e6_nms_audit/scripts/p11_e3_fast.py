"""Phase 6f - E3 (label ablation) headline at the revised radius, from the cached peaks.

`e3_run_eval.py` already extracted candidates from all 24 frozen E3 checkpoints at the revised
radius; that inference is done and cached. Its own analysis stage then sweeps a ROC per arm per
seed per gate over the full split, which at this radius means roughly four times as many candidates
and turns a minutes-long job into an hours-long one.

This reproduces the one output the audit needs -- `deltas.csv`, the paired bootstrap of each label
arm against the flat 3x3 control -- from those cached peaks, using the same verified fast scorer as
`p7_final_stats.py`. Same operating point, same matching, same bootstrap seed, same Bonferroni
correction; only the redundant re-derivation is removed.

Nothing is retrained and nothing outside the experiment folder is written. The original E3 results
keep their radius-5 values.

    python p11_e3_fast.py --radius 2
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
ARMS = ("flat3x3", "point", "gaussian")
SEEDS = range(8)
GATES = (0, 1, 3)
EPOCH = 21
E3 = os.path.join(EXP, "results", "p10_attribution", "e3")


def load(path):
    a = np.load(path)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def sweep(scorers, grid):
    rows = []
    for thr in grid:
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


def hits(pk, ids, tb, valid, gate):
    frames = [(pk.get(int(i), np.zeros((0, 3))), tb.get(int(i), np.zeros((0, 2))), SHAPE, valid)
              for i in ids]
    grid = roc_thresholds(frames, MatchConfig(gate, gate))
    sc = [FrameScorer(f[0], f[1], SHAPE, gate, gate, valid) for f in frames]
    _, thr = _at_measured_pfa(sweep(sc, grid), TARGET_PFA)
    h = np.array([s.at(thr)[0] for s in sc], float)
    n = np.array([len(f[1]) for f in frames], float)
    return h, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--gates", type=int, nargs="+", default=list(GATES),
                    help="which gates to score. E3's headline claim is a gate-0 "
                         "claim; the tolerant gates are far denser and much slower.")
    a = ap.parse_args()
    t0 = time.time()

    valid = cfar_valid_mask(SHAPE, CfarConfig())
    cv = os.path.join(ROOT, "results", "cfar_val")
    fm = pd.read_csv(os.path.join(cv, "frames.csv"))
    tb = {int(f): g[["range_bin", "velocity_bin"]].to_numpy(float)
          for f, g in pd.read_csv(os.path.join(cv, "truths.csv")).groupby("frame")}
    conds = sorted(c for c in fm.condition.unique() if c != "noise")

    pk = {(arm, s): load(os.path.join(E3, "peaks", "peaks_%s_s%d_ep%03d.npz"
                                      % (arm, s, EPOCH)))
          for arm in ARMS for s in SEEDS}
    print("loaded %d peak sets in %.0f s" % (len(pk), time.time() - t0), flush=True)

    ids0 = fm.loc[fm.condition == "isolated", "frame"].to_numpy()[:80]
    fr0 = [(pk[("flat3x3", 0)].get(int(i), np.zeros((0, 3))), tb.get(int(i), np.zeros((0, 2))),
            SHAPE, valid) for i in ids0]
    g0 = roc_thresholds(fr0, MatchConfig(0, 0))
    n = verify(fr0, g0[::12], 0, 0, SHAPE, valid, n=8)
    print("fast scorer verified on %d frame-threshold pairs" % n, flush=True)

    store = {}
    for gate in a.gates:
        for cond in conds:
            ids = fm.loc[fm.condition == cond, "frame"].to_numpy()
            for arm in ARMS:
                hh = [hits(pk[(arm, s)], ids, tb, valid, gate) for s in SEEDS]
                store[(gate, cond, arm)] = (np.mean([h for h, _ in hh], axis=0), hh[0][1])
        print("  gate %d done  (%.0f s)" % (gate, time.time() - t0), flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for gate in a.gates:
        for cond in conds:
            base, nn = store[(gate, cond, "flat3x3")]
            k = rng.integers(0, len(nn), size=(N_BOOT, len(nn)))
            tot = nn[k].sum(axis=1)
            for arm in ARMS:
                if arm == "flat3x3":
                    continue
                h, _ = store[(gate, cond, arm)]
                d = (h[k].sum(axis=1) - base[k].sum(axis=1)) / tot
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                rows.append(dict(gate=gate, condition=cond, arm=arm,
                                 pd_flat=base.sum() / nn.sum(), pd_arm=h.sum() / nn.sum(),
                                 delta=(h.sum() - base.sum()) / nn.sum(),
                                 ci_lo=float(lo), ci_hi=float(hi),
                                 p_boot=float(2 * min((d <= 0).mean(), (d >= 0).mean()))))
    B = pd.DataFrame(rows)
    B["alpha_bonf"] = 0.05 / len(conds)
    B["sig"] = B.p_boot < B.alpha_bonf
    out = os.path.join(E3, "deltas_gate%s.csv" % "".join(str(g) for g in a.gates))
    B.to_csv(out, index=False)
    print("\n" + B[B.gate == 0][["condition", "arm", "pd_flat", "pd_arm", "delta", "ci_lo",
                                 "ci_hi", "p_boot", "sig"]].round(4).to_string(index=False))
    print("\ntotal %.0f s -> %s" % (time.time() - t0, os.path.relpath(out, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
