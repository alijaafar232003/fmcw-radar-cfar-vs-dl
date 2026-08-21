"""Localization on the COMMON DETECTED SUBSET — targets found by both arms.

THE PROBLEM THIS ADDRESSES. Localization error is averaged over *detected* targets only, so it
flatters whichever method detects less: a detector that only finds strong, clean targets is
measuring precision on the easy ones. CFAR detects fewer targets than the network on six of seven
conditions, so part of its 3.83x Doppler advantage could be that selection effect rather than
precision.

THE FIX. Restrict both arms to the targets **both** detected, per condition, at the same matched
operating point, and recompute. Every target then contributes to both averages or to neither.

PER SEED, NOT POOLED-THEN-INTERSECTED. Each DL seed detects a slightly different set, so the
intersection is formed per seed against CFAR's set, the error computed there, and the eight results
averaged. Intersecting a pooled DL set would silently change the population per seed.

⚠️ THIS IS A POST-HOC ANALYSIS. It was **not** in `docs/test_preregistration.md`; it is listed there
and in the comparison notebook (§9.4) as a known limitation that was deliberately not run, because
it would have been invented after seeing the numbers. It is run now at the supervisor's request and
must be reported as post-hoc, never as part of the pre-registered result.

Uses only cached peaks — no frame of `bench2_test` is read.

    python scripts/localization_common_subset.py
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
RANGE_RES_M, VEL_RES_MPS = 0.19518, 0.42013
DLR = ROOT / "results" / "dl"


def load_peaks(p: Path) -> dict:
    a = np.load(p)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def matched_threshold(frames, mc):
    roc = roc_curve(frames, roc_thresholds(frames, mc), mc)
    return _at_measured_pfa(roc, TARGET_PFA)[1]


def detected_and_errors(pk, ids, tb, valid, mc, thr):
    """Per frame: which truth indices were detected, and the |error| of each match.

    Returns {(frame, truth_idx): (abs_dr_bins, abs_dv_bins)}.
    """
    out = {}
    for i in ids:
        det = pk.get(int(i), np.zeros((0, 3)))
        det = det[det[:, 2] >= thr][:, :2] if (len(det) and thr == thr) else np.zeros((0, 2))
        tru = tb.get(i, np.zeros((0, 2)))
        ev = evaluate_frame(det, tru, SHAPE, mc, valid_mask=valid)
        # ev.dr / ev.dv are built from zip(det_idx, tru_idx) in the same order as
        # matched_truth_idx, so index j of each refers to the same matched pair.
        for j, k in enumerate(ev.matched_truth_idx):
            out[(int(i), int(k))] = (abs(float(ev.dr[j])), abs(float(ev.dv[j])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--epoch", type=int, default=21)
    ap.add_argument("--split", default="bench2_test")
    ap.add_argument("--tolerance", type=int, default=1)
    args = ap.parse_args()

    PK = args.split.replace("bench2_", "")
    CV = ROOT / "results" / f"cfar_{PK}"
    mc = MatchConfig(args.tolerance, args.tolerance)
    valid = cfar_valid_mask(SHAPE, CfarConfig())

    fm = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}

    # The best CFAR method per condition, as the headline table selected it.
    boot = pd.read_csv(DLR / f"bootstrap_{args.tag}_ep{args.epoch:03d}_{PK}_tol{args.tolerance}.csv")
    best = boot.set_index("condition").cfar_method.to_dict()

    cf_peaks = {m: load_peaks(CV / f"peaks_{m}.npz") for m in ("CA", "GO", "OS", "SO")}
    pat = re.compile(rf"^peaks_{PK}_{args.tag}_s\d+_ep{args.epoch:03d}\.npz$")
    dl_files = sorted(p for p in DLR.glob(f"peaks_{PK}_{args.tag}_s*.npz") if pat.match(p.name))
    dl_peaks = [load_peaks(p) for p in dl_files]
    print(f"{len(dl_peaks)} DL seeds | CFAR methods {list(cf_peaks)} | split {args.split}")

    conds = sorted(c for c in fm.condition.unique() if c != "noise")
    rows = []

    for cond in conds + ["ALL"]:
        ids = (fm.frame.to_numpy() if cond == "ALL"
               else fm.loc[fm.condition == cond, "frame"].to_numpy())

        # CFAR arm: for pooled, use each condition's own best method, matching the headline.
        per_seed = []
        for si, dpk in enumerate(dl_peaks):
            cf_err, dl_err, both = {}, {}, None

            groups = ([(c, fm.loc[fm.condition == c, "frame"].to_numpy()) for c in conds]
                      if cond == "ALL" else [(cond, ids)])

            cf_all, dl_all = {}, {}
            for c, cids in groups:
                m = best[c]
                cfr = [(cf_peaks[m].get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))),
                        SHAPE, valid) for i in cids]
                dfr = [(dpk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))),
                        SHAPE, valid) for i in cids]
                cf_thr = matched_threshold(cfr, mc)
                dl_thr = matched_threshold(dfr, mc)
                cf_all.update(detected_and_errors(cf_peaks[m], cids, tb, valid, mc, cf_thr))
                dl_all.update(detected_and_errors(dpk, cids, tb, valid, mc, dl_thr))

            common = set(cf_all) & set(dl_all)
            if not common:
                continue
            cf_dr = np.mean([cf_all[k][0] for k in common])
            cf_dv = np.mean([cf_all[k][1] for k in common])
            dl_dr = np.mean([dl_all[k][0] for k in common])
            dl_dv = np.mean([dl_all[k][1] for k in common])
            per_seed.append(dict(seed=si, n_common=len(common),
                                 n_cfar=len(cf_all), n_dl=len(dl_all),
                                 cf_dr=cf_dr, cf_dv=cf_dv, dl_dr=dl_dr, dl_dv=dl_dv))

        if not per_seed:
            continue
        d = pd.DataFrame(per_seed)
        se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        rows.append(dict(
            condition=cond, cfar_method=("per-condition" if cond == "ALL" else best[cond]),
            n_cfar=float(d.n_cfar.mean()), n_dl=float(d.n_dl.mean()),
            n_common=float(d.n_common.mean()),
            cfar_dr=float(d.cf_dr.mean()), cfar_dv=float(d.cf_dv.mean()),
            dl_dr=float(d.dl_dr.mean()), dl_dr_se=se(d.dl_dr),
            dl_dv=float(d.dl_dv.mean()), dl_dv_se=se(d.dl_dv),
            ratio_dr=float(d.dl_dr.mean() / d.cf_dr.mean()),
            ratio_dv=float(d.dl_dv.mean() / d.cf_dv.mean()),
        ))
        print(f"  {cond:<14} common {d.n_common.mean():6.0f} of "
              f"CFAR {d.n_cfar.mean():.0f} / DL {d.n_dl.mean():.0f}   "
              f"Doppler {d.dl_dv.mean() / d.cf_dv.mean():.2f}x")

    out = pd.DataFrame(rows).set_index("condition")
    path = DLR / f"localization_common_{args.tag}_ep{args.epoch:03d}_{PK}.csv"
    out.to_csv(path)

    # The all-detected comparison this is being contrasted with.
    full = pd.read_csv(DLR / f"localization_{args.tag}_ep{args.epoch:03d}_{PK}.csv")
    f1 = full[(full.gate == args.tolerance)]

    print("\n" + "=" * 92)
    print(f"LOCALIZATION: all-detected vs COMMON detected subset  (gate +/-{args.tolerance})")
    print("=" * 92)
    comp = []
    for cond in out.index:
        g = "ALL" if cond == "ALL" else cond
        s = f1[f1.group == g].set_index("method")
        if "DL" not in s.index:
            continue
        cfm = out.loc[cond, "cfar_method"]
        cf_row = s.drop("DL")
        full_dv = s.loc["DL", "dv_bins"] / (cf_row.dv_bins.min() if cond == "ALL"
                                            else cf_row.loc[cfm, "dv_bins"])
        full_dr = s.loc["DL", "dr_bins"] / (cf_row.dr_bins.min() if cond == "ALL"
                                            else cf_row.loc[cfm, "dr_bins"])
        comp.append(dict(condition=cond,
                         dv_all=full_dv, dv_common=out.loc[cond, "ratio_dv"],
                         dr_all=full_dr, dr_common=out.loc[cond, "ratio_dr"],
                         common_frac=out.loc[cond, "n_common"] / out.loc[cond, "n_cfar"]))
    C = pd.DataFrame(comp).set_index("condition")
    print(C.round(3).to_string())
    print("\ndv_all / dr_all use each arm's OWN detected set (the headline numbers).")
    print("dv_common / dr_common use only targets BOTH detected.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
