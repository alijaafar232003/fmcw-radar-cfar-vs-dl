"""Does GO win at a TRUE clutter edge? Split `clutter_edge` by whether the window reaches the ridge.

`diagnose_go_clutter_edge.py` measured the ridge at +/-4 Doppler bins and found that 41% of
`clutter_edge` targets sit far enough out that the 21x9 window's training columns never touch it.
Those frames are plain-noise frames wearing the stratum's label, and averaging them in dilutes
whatever the edge actually does — to every method.

This is a RE-ANALYSIS of the already-run test experiment, sliced by a physical covariate fixed by
the geometry (ridge extent and window reach), not a re-tuning: no threshold, window or parameter is
chosen using anything measured here. It reads the cached detections from `cfar_test_baseline.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig                                       # noqa: E402
from eval import MatchConfig, aggregate, evaluate_frame, roc_curve     # noqa: E402
from radar.config import RadarConfig                                   # noqa: E402
from radar.simulator import make_axes                                  # noqa: E402

RES = ROOT / "results" / "cfar_test"
METHODS = ("CA", "GO", "OS")
TARGET_PFA = 1e-4
SHAPE = (256, 64)
MC = MatchConfig(1, 1)
RIDGE_HALF = 4          # measured: >3 dB above floor spans bins 28..36 about v0 = 32

QUANTILES = np.clip(np.concatenate([
    np.linspace(0.0, 0.99, 40),
    1.0 - np.logspace(np.log10(0.01), np.log10(2e-5), 40)]), 0.0, 1.0)


def pd_at(roc, target):
    d = roc.dropna(subset=["pfa_per_cell"])
    d = (d.groupby("pfa_per_cell", as_index=False)
           .agg(pd=("pd", "max"), threshold=("threshold", "min"), fp=("fp", "max"))
           .sort_values("pfa_per_cell"))
    pfa, fp = d.pfa_per_cell.to_numpy(float), d.fp.to_numpy(float)
    obs = (fp > 0) & (pfa > 0)
    if not obs.any():
        return np.nan, np.nan
    cells = float(np.round(np.median(fp[obs] / pfa[obs])))
    pfa = np.where(pfa > 0, pfa, 0.5 / cells)
    x, t = np.log10(pfa), np.log10(target)
    if len(x) < 2 or t < x.min() or t > x.max():
        return np.nan, np.nan
    return float(np.interp(t, x, d.pd.to_numpy())), float(np.interp(t, x, d.threshold.to_numpy()))


def main() -> None:
    cfg = CfarConfig()
    hw_r, hw_d, g_d = cfg.half_width_r, cfg.half_width_d, cfg.guard_d

    valid = np.zeros(SHAPE, bool)
    valid[hw_r:-hw_r, hw_d:-hw_d] = True

    _, _, _, vax = make_axes(RadarConfig())
    v0 = int(np.argmin(np.abs(vax)))
    lo, hi = v0 - RIDGE_HALF, v0 + RIDGE_HALF

    frames = pd.read_csv(RES / "frames.csv")
    truths = pd.read_csv(RES / "truths.csv")
    TB = {f: g for f, g in truths.groupby("frame")}

    PEAKS = {}
    for m in METHODS:
        a = np.load(RES / f"peaks_{m}.npz")["peaks"]
        PEAKS[m] = {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}

    ids = frames.loc[frames.condition == "clutter_edge", "frame"].to_numpy()

    # A frame is a TRUE edge if any target's training columns fall inside the ridge.
    true_edge, no_edge = [], []
    for i in ids:
        t = TB.get(i)
        if t is None:
            continue
        overlap = 0
        for vb in t.velocity_bin:
            overlap += sum(1 for c in range(int(vb) - hw_d, int(vb) + hw_d + 1)
                           if abs(c - int(vb)) > g_d and lo <= c <= hi)
        (true_edge if overlap > 0 else no_edge).append(i)

    print(f"clutter_edge frames: {len(ids)}")
    print(f"   window REACHES the ridge : {len(true_edge)}  ({100*len(true_edge)/len(ids):.1f}%)")
    print(f"   window never reaches it  : {len(no_edge)}  ({100*len(no_edge)/len(ids):.1f}%)")
    print(f"   (ridge measured at +/-{RIDGE_HALF} bins; window training columns at "
          f"+/-{g_d+1}..{hw_d})")

    def score(subset, label):
        print(f"\n{'=' * 70}\n{label}  ({len(subset)} frames)\n{'=' * 70}")
        print(f"   {'method':<8}{'Pd':>8}{'meas Pfa':>12}{'FA/frame':>10}{'threshold':>11}")
        out = {}
        for m in METHODS:
            fr = []
            for i in subset:
                t = TB.get(i)
                tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
                       if t is not None else np.zeros((0, 2)))
                fr.append((PEAKS[m].get(int(i), np.zeros((0, 3))), tru, SHAPE, valid))

            pool = np.concatenate([f[0][:, 2] for f in fr if len(f[0])])
            roc = roc_curve(fr, np.unique(np.quantile(pool, QUANTILES)), MC)
            _, thr = pd_at(roc, TARGET_PFA)

            evs = [evaluate_frame(f[0][f[0][:, 2] >= thr][:, :2]
                                  if (len(f[0]) and thr == thr) else np.zeros((0, 2)),
                                  f[1], f[2], MC, valid_mask=f[3]) for f in fr]
            a = aggregate(evs)
            out[m] = a["pd"]
            print(f"   {m:<8}{a['pd']:>8.3f}{a['pfa_per_cell']:>12.2e}"
                  f"{a['fa_per_frame']:>10.2f}{thr:>11.2f}")
        best = max(out, key=out.get)
        print(f"   -> best: {best} ({out[best]:.3f}), "
              f"margin over next {out[best] - sorted(out.values())[-2]:+.3f}")
        return out

    a = score(true_edge, "TRUE EDGE — the window straddles the ridge")
    b = score(no_edge, "NO EDGE — the stratum's label is wrong for these frames")

    print(f"\n{'=' * 70}\nWHAT THE DILUTION COSTS\n{'=' * 70}")
    print(f"   {'method':<8}{'true edge':>11}{'no edge':>10}{'pooled(reported)':>18}")
    for m in METHODS:
        pooled = (a[m] * len(true_edge) + b[m] * len(no_edge)) / len(ids)
        print(f"   {m:<8}{a[m]:>11.3f}{b[m]:>10.3f}{pooled:>18.3f}")


if __name__ == "__main__":
    main()
