"""DL vs CFAR with error bars on BOTH sides, and a multiplicity correction.

Fixes a real defect in every DL-vs-CFAR number reported so far. They were produced with
`ttest_1samp(dl_seeds, cfar_value)` — which treats CFAR's Pd as an exact constant. It is not: it is
one draw over 200 frames per condition, and carries sampling error of the same order as the effects
being claimed. Every p-value computed that way is overconfident.

Two changes here:

1. **Bootstrap over FRAMES, paired.** Resample frames with replacement; within each resample
   recompute Pd for CFAR and for each DL seed on exactly the same frames, and take the difference.
   The frames are shared, so frame difficulty cancels rather than adding noise. Reported as the mean
   difference with a percentile CI and a bootstrap p-value.
2. **Bonferroni across the 7 scored conditions** — alpha 0.05 becomes 0.0071.

Detections are taken at each method's matched-measured-Pfa threshold, computed once on the full
condition. The bootstrap therefore quantifies sampling error in Pd at a fixed operating point; it
does not propagate the variability of *choosing* that threshold, which would need the sweep redone
per resample (~200x the compute). That makes these intervals mildly optimistic, and it is stated
rather than hidden.

Run on `bench2_val`. Uses cached CFAR peaks and existing DL checkpoints — no retraining.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig                                      # noqa: E402
from cfar.detect import all_peaks                                     # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                 # noqa: E402
from dl.models import TinyCNN                                         # noqa: E402
from dl.stratified_eval import NMS, _at_measured_pfa                  # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,             # noqa: E402
                  roc_curve, roc_thresholds)

SPLIT = "bench2_val"
TARGET_PFA = 1e-4
MC = MatchConfig(1, 1)
SHAPE = (256, 64)
N_BOOT = 2000
CV = ROOT / "results" / "cfar_val"
DLR = ROOT / "results" / "dl"


def per_frame_hits(frames, thr, mc):
    """(hits, targets) per frame at a fixed threshold — the unit the bootstrap resamples."""
    h, n = [], []
    for pk, tru, shape, valid in frames:
        det = pk[pk[:, 2] >= thr][:, :2] if (len(pk) and thr == thr) else np.zeros((0, 2))
        ev = evaluate_frame(det, tru, shape, mc, valid_mask=valid)
        h.append(ev.tp)
        n.append(len(tru))
    return np.array(h, float), np.array(n, float)


def matched_threshold(frames, mc):
    roc = roc_curve(frames, roc_thresholds(frames, mc), mc)
    return _at_measured_pfa(roc, TARGET_PFA)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-weight", type=int, default=100,
                    help="which DL arm to compare (default 100, the chosen value)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--tolerance", type=int, default=1,
                    help="matching gate in bins. The tolerance sweep showed `clutter` is the one "
                         "condition whose margin depends on it, so that claim needs its own CI.")
    args = ap.parse_args()
    mc = MatchConfig(args.tolerance, args.tolerance)

    valid = cfar_valid_mask(SHAPE, CfarConfig())
    frames_meta = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}
    conds = sorted(c for c in frames_meta.condition.unique() if c != "noise")

    # ---------------------------------------------------------------- CFAR, cached
    cfar_peaks = {}
    for m in ("CA", "GO", "OS", "SO"):
        a = np.load(CV / f"peaks_{m}.npz")["peaks"]
        cfar_peaks[m] = {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}

    # ---------------------------------------------------------------- DL, the chosen arm
    tag = "n" if args.pos_weight == 974 else f"pw{args.pos_weight}_s"
    ckpts = sorted(DLR.glob(f"tinycnn_{tag}*_ep050.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints for pos_weight={args.pos_weight}")
    print(f"DL arm: pos_weight={args.pos_weight}, {len(ckpts)} seeds")

    dl_peaks = []
    for cp in ckpts:
        ck = torch.load(cp, weights_only=False)
        seed = ck["config"]["seed"]
        # Peaks do not depend on the matching gate, so the cache built by tolerance_sweep.py is
        # reusable here -- this is the expensive step (8 seeds x 1,600 frames of inference).
        cache = DLR / f"peaks_val_pw{args.pos_weight}_s{seed}.npz"
        if cache.exists():
            a = np.load(cache)["peaks"]
            dl_peaks.append({int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])})
            print(f"  loaded {cache.name}")
            continue
        model = TinyCNN(width=ck["config"].get("width", 16))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        pk = {}
        for i in frames_meta.frame:
            mag = np.load(ROOT / "data" / SPLIT / "frames" / f"frame_{int(i):06d}" /
                          "rd_map_mag.npy")
            pk[int(i)] = all_peaks(predict_heatmap(model, mag), valid, nms_size=NMS)
        dl_peaks.append(pk)
        print(f"  scored {cp.name}")

    rng = np.random.default_rng(0)
    rows = []

    for cond in conds:
        ids = frames_meta.loc[frames_meta.condition == cond, "frame"].to_numpy()
        fr_of = lambda pk: [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))),
                             SHAPE, valid) for i in ids]

        # Best CFAR method on this condition, at its own matched threshold.
        best_m, best_pd, best_hits, best_n = None, -1, None, None
        for m in ("CA", "GO", "OS", "SO"):
            f = fr_of(cfar_peaks[m])
            h, n = per_frame_hits(f, matched_threshold(f, mc), mc)
            p = h.sum() / n.sum()
            if p > best_pd:
                best_m, best_pd, best_hits, best_n = m, p, h, n

        # DL: hits per frame, averaged over seeds (same frames, so pairing holds).
        dl_h = []
        for pk in dl_peaks:
            f = fr_of(pk)
            h, n = per_frame_hits(f, matched_threshold(f, mc), mc)
            dl_h.append(h)
        dl_h = np.mean(dl_h, axis=0)

        # Paired frame bootstrap on the DIFFERENCE.
        diffs = np.empty(args.n_boot)
        for b in range(args.n_boot):
            k = rng.integers(0, len(ids), len(ids))
            tot = best_n[k].sum()
            diffs[b] = (dl_h[k].sum() - best_hits[k].sum()) / tot if tot else np.nan

        obs = (dl_h.sum() - best_hits.sum()) / best_n.sum()
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        p_boot = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())

        rows.append(dict(condition=cond, n_frames=len(ids), targets=int(best_n.sum()),
                         cfar_method=best_m, cfar_pd=best_pd,
                         dl_pd=dl_h.sum() / best_n.sum(), diff=obs,
                         ci_lo=lo, ci_hi=hi, p_boot=p_boot))
        print(f"  {cond:<14} done")

    df = pd.DataFrame(rows).set_index("condition")
    alpha = 0.05 / len(df)
    df["sig (Bonf)"] = df.p_boot < alpha
    df["crosses 0"] = (df.ci_lo <= 0) & (df.ci_hi >= 0)

    out = DLR / f"bootstrap_pw{args.pos_weight}_tol{args.tolerance}.csv"
    df.to_csv(out)

    print("\n" + "=" * 100)
    print(f"DL (pos_weight={args.pos_weight}, {len(ckpts)} seeds) vs best CFAR, {SPLIT}, "
          f"gate +/-{args.tolerance}")
    print(f"paired frame bootstrap, {args.n_boot} resamples · Bonferroni alpha = {alpha:.4f}")
    print("=" * 100)
    print(df.round(4).to_string())
    print(f"\nsignificant after correction: {int(df['sig (Bonf)'].sum())} of {len(df)}")
    print("intervals that cross zero:", list(df.index[df["crosses 0"]]) or "none")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
