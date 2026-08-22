"""Paired frame bootstrap that RE-SELECTS the best CFAR variant inside every replicate.

THE PROBLEM THIS ADDRESSES. `bootstrap_comparison.py` picks the best CFAR variant once, on the full
condition, and then bootstraps DL against that fixed winner. The choice is therefore treated as
certain. It is not: on several conditions the top two variants are within a few thousandths of Pd,
which is well inside the frame-sampling noise the bootstrap is measuring on everything else. The
reported interval answers "how uncertain is the margin against OS?" when the question is "how
uncertain is the margin against whichever classical detector is actually strongest here?".

THE FIX. Inside each replicate, re-select the best CFAR variant on that replicate's frames and take
the difference against it. Uncertainty about *which* classical method is strongest is then carried
by the interval instead of being assumed away.

BOTH PROCEDURES ARE RUN ON THE SAME RESAMPLES (one rng, one set of frame indices), so the two
intervals are directly comparable and any widening is attributable to the re-selection alone rather
than to bootstrap noise.

⚠️ RE-SELECTION IS CONSERVATIVE FOR DL, BY CONSTRUCTION. Taking the maximum over four variants on
the same frames used to score them is a winner's-curse estimator: it biases the CFAR arm slightly
UP, so DL's margin comes out slightly smaller than a bias-free procedure would give. That is the
right direction for a claim of DL superiority — it cannot manufacture a win — but the resulting
interval is a fair reflection of the selection uncertainty, not an unbiased estimate of the margin
against any one named variant.

⚠️ STILL NOT PROPAGATED, in either procedure: the variability of *choosing the threshold*. Each
method's matched-measured-Pfa threshold is computed once on the full condition, as in the original
script; re-sweeping it per replicate would cost ~200x the compute. Stated, not hidden.

⚠️ POST-HOC. This is not part of `docs/test_preregistration.md`; it is a methodology fix applied
after the pre-registered read, and the pre-registered result stands as reported. Its role is to say
whether that result is robust to a source of uncertainty the original procedure ignored.

Uses cached peaks only -- no frame of the split is read.

    python scripts/bootstrap_best_cfar_reselect.py --split bench2_test --tolerance 1
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
from eval import (MatchConfig, evaluate_frame,                       # noqa: E402
                  roc_curve, roc_thresholds)

TARGET_PFA = 1e-4
SHAPE = (256, 64)
CFAR_METHODS = ("CA", "GO", "OS", "SO")
DLR = ROOT / "results" / "dl"


def load_peaks(p: Path) -> dict:
    a = np.load(p)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def per_frame_hits(frames, thr, mc):
    """(hits, targets) per frame at a fixed threshold -- the unit the bootstrap resamples."""
    h, n = [], []
    for pk, tru, shape, valid in frames:
        det = pk[pk[:, 2] >= thr][:, :2] if (len(pk) and thr == thr) else np.zeros((0, 2))
        ev = evaluate_frame(det, tru, shape, mc, valid_mask=valid)
        h.append(ev.tp)
        n.append(len(tru))
    return np.array(h, float), np.array(n, float)


def matched_threshold(frames, mc):
    return _at_measured_pfa(roc_curve(frames, roc_thresholds(frames, mc), mc), TARGET_PFA)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--epoch", type=int, default=21)
    ap.add_argument("--split", default="bench2_test")
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    PK = args.split.replace("bench2_", "")
    CV = ROOT / "results" / f"cfar_{PK}"
    mc = MatchConfig(args.tolerance, args.tolerance)
    valid = cfar_valid_mask(SHAPE, CfarConfig())

    fm = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}

    cf_peaks = {m: load_peaks(CV / f"peaks_{m}.npz") for m in CFAR_METHODS
                if (CV / f"peaks_{m}.npz").exists()}
    pat = re.compile(rf"^peaks_{PK}_{args.tag}_s\d+_ep{args.epoch:03d}\.npz$")
    dl_files = sorted(p for p in DLR.glob(f"peaks_{PK}_{args.tag}_s*.npz") if pat.match(p.name))
    if not dl_files:
        raise SystemExit(f"no DL peaks caches matching peaks_{PK}_{args.tag}_s*_ep{args.epoch:03d}")
    dl_peaks = [load_peaks(p) for p in dl_files]
    print(f"split {args.split} | CFAR {list(cf_peaks)} | {len(dl_peaks)} DL seeds | "
          f"gate +/-{args.tolerance} | {args.n_boot} resamples")

    conds = sorted(c for c in fm.condition.unique() if c != "noise")
    rng = np.random.default_rng(0)
    rows, diff_store = [], {}

    for cond in conds:
        ids = fm.loc[fm.condition == cond, "frame"].to_numpy()
        fr_of = lambda pk: [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))),
                             SHAPE, valid) for i in ids]

        # Per-frame hits for EVERY CFAR variant, each at its own matched threshold.
        cf_hits, n = {}, None
        for m, pk in cf_peaks.items():
            h, n = per_frame_hits(fr_of(pk), matched_threshold(fr_of(pk), mc), mc)
            cf_hits[m] = h

        names = list(cf_hits)
        H = np.vstack([cf_hits[m] for m in names])           # (n_methods, n_frames)

        dl_h = np.mean([per_frame_hits(fr_of(pk), matched_threshold(fr_of(pk), mc), mc)[0]
                        for pk in dl_peaks], axis=0)

        # Winner on the full sample -- the fixed choice the original procedure makes.
        full_pd = H.sum(axis=1) / n.sum()
        fixed_i = int(np.argmax(full_pd))
        fixed_m = names[fixed_i]

        # ONE set of resamples, used by both procedures.
        k = rng.integers(0, len(ids), size=(args.n_boot, len(ids)))
        tot = n[k].sum(axis=1)                                # targets in each replicate
        dl_s = dl_h[k].sum(axis=1)
        cf_s = np.stack([h[k].sum(axis=1) for h in H])        # (n_methods, n_boot)

        with np.errstate(invalid="ignore", divide="ignore"):
            d_fixed = (dl_s - cf_s[fixed_i]) / tot
            pick = np.argmax(cf_s, axis=0)                    # re-selected winner per replicate
            d_resel = (dl_s - cf_s.max(axis=0)) / tot

        def summarise(d):
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            return float(lo), float(hi), float(p)

        lo_f, hi_f, p_f = summarise(d_fixed)
        lo_r, hi_r, p_r = summarise(d_resel)
        share = {m: float((pick == i).mean()) for i, m in enumerate(names)}

        rows.append(dict(
            condition=cond, n_frames=len(ids), targets=int(n.sum()),
            cfar_method_fixed=fixed_m,
            cfar_pd_fixed=float(full_pd[fixed_i]),
            dl_pd=float(dl_h.sum() / n.sum()),
            diff_observed=float((dl_h.sum() - H[fixed_i].sum()) / n.sum()),
            ci_lo_fixed=lo_f, ci_hi_fixed=hi_f, p_fixed=p_f,
            ci_lo_reselect=lo_r, ci_hi_reselect=hi_r, p_reselect=p_r,
            width_fixed=hi_f - lo_f, width_reselect=hi_r - lo_r,
            width_ratio=(hi_r - lo_r) / (hi_f - lo_f) if hi_f > lo_f else np.nan,
            mean_diff_shift=float(np.nanmean(d_resel) - np.nanmean(d_fixed)),
            reselect_share_winner=share[fixed_m],
            reselect_share=" ".join(f"{m}:{share[m]:.2f}" for m in names),
            n_variants_ever_picked=int(len({int(i) for i in np.unique(pick)})),
        ))
        diff_store[f"{cond}__fixed"] = d_fixed
        diff_store[f"{cond}__reselect"] = d_resel
        print(f"  {cond:<14} fixed {fixed_m} [{lo_f:+.4f},{hi_f:+.4f}]  "
              f"reselect [{lo_r:+.4f},{hi_r:+.4f}]  "
              f"width x{(hi_r - lo_r) / (hi_f - lo_f):.2f}  "
              f"winner held {share[fixed_m]:.0%}")

    df = pd.DataFrame(rows).set_index("condition")
    alpha = 0.05 / len(df)
    df["sig_fixed"] = df.p_fixed < alpha
    df["sig_reselect"] = df.p_reselect < alpha
    df["verdict_changed"] = df.sig_fixed != df.sig_reselect

    stem = f"bootstrap_reselect_{args.tag}_ep{args.epoch:03d}_{PK}_tol{args.tolerance}"
    out = DLR / f"{stem}.csv"
    df.to_csv(out)
    np.savez_compressed(DLR / f"{stem}_diffs.npz", n_boot=args.n_boot, **diff_store)

    print("\n" + "=" * 104)
    print(f"BEST-CFAR RE-SELECTION INSIDE THE BOOTSTRAP -- {args.split}, gate +/-{args.tolerance}")
    print(f"Bonferroni alpha = {alpha:.4f} over {len(df)} conditions · "
          f"p resolution floor = {2 / args.n_boot:.4f}")
    print("=" * 104)
    cols = ["cfar_method_fixed", "diff_observed", "ci_lo_fixed", "ci_hi_fixed",
            "ci_lo_reselect", "ci_hi_reselect", "width_ratio", "reselect_share_winner",
            "sig_fixed", "sig_reselect", "verdict_changed"]
    print(df[cols].round(4).to_string())
    print()
    print(f"verdicts changed: {int(df.verdict_changed.sum())} of {len(df)}")
    print(f"intervals widened on {int((df.width_ratio > 1).sum())} of {len(df)} conditions "
          f"(median x{df.width_ratio.median():.2f}, max x{df.width_ratio.max():.2f})")
    print(f"\nwrote {out}")
    print(f"wrote {DLR / f'{stem}_diffs.npz'}")


if __name__ == "__main__":
    main()
