"""Localization error for every method — the one metric where CFAR clearly beats the network.

`aggregate()` has always computed localization (mean |Δrange|, |Δdoppler|, RMSE, and the same in
metres and m/s), and `cfar_baseline.ipynb` reports it for CFAR. **It was never extracted for the DL
side** — every script computed it and then read only `pd`. It is also a proposal-required metric.

Extracting it changes the headline. DL detects more targets in interference and clutter, but places
them substantially less precisely: at the ±1 gate its Doppler error is ~5x CFAR's. The likely cause
is the **3×3 label blob** — the network is trained that all nine cells around a target are "target",
so its peak may land anywhere inside that block, while CFAR responds to the physical peak, which is
sharper. That is Milestone 16's asymmetry #5 (`the ±1 gate may measure localisation rather than
detection`) showing up directly in the localisation metric rather than by inference.

Reported at ±1 (the pre-registered gate) and ±3. Note the metric is bounded by the gate: a detection
displaced further than the gate is a miss plus a false alarm, not a large error, so severe
displacement shows up as lost Pd instead. Both gates are therefore reported.

Run on `bench2_val`; `bench2_test` is untouched.
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

SPLIT = "bench2_val"
TARGET_PFA = 1e-4
SHAPE = (256, 64)
RANGE_RES, VEL_RES = 0.19518, 0.42013
CV = ROOT / "results" / "cfar_val"
DLR = ROOT / "results" / "dl"


def load_peaks(path: Path) -> dict:
    a = np.load(path)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def measure(pk, ids, tb, valid, mc, per_condition=None):
    fr = [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))), SHAPE, valid)
          for i in ids]
    roc = roc_curve(fr, roc_thresholds(fr, mc), mc)
    _, thr = _at_measured_pfa(roc, TARGET_PFA)
    evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2)),
                          t, s, mc, valid_mask=v) for p, t, s, v in fr]
    return aggregate(evs, range_res_m=RANGE_RES, vel_res_mps=VEL_RES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-weight", type=int, default=100)
    ap.add_argument("--tolerances", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--per-condition", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="DL arm to score, e.g. 'frozen'. Default keeps the original "
                         "`pw{pos_weight}` naming so the epoch-50 table still reproduces.")
    ap.add_argument("--epoch", type=int, default=None,
                    help="epoch of the checkpoints, used only to name the peaks caches and the "
                         "output. Omit for the original (epoch-50) files.")
    args = ap.parse_args()

    # Superseded results are never overwritten: a new arm writes a new file. `localization_pw100.csv`
    # is the epoch-50 table and stays on disk as the record of what was reported before.
    arm = args.tag or f"pw{args.pos_weight}"
    suffix = "" if args.epoch is None else f"_ep{args.epoch:03d}"

    valid = cfar_valid_mask(SHAPE, CfarConfig())
    frames_meta = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}

    sources = {}
    for m in ("CA", "GO", "OS", "SO"):
        p = CV / f"peaks_{m}.npz"
        if p.exists():
            sources[m] = [load_peaks(p)]
    # Match the seed number EXACTLY. A `_s*` glob is too loose: it also swallows epoch-suffixed
    # caches, so once `peaks_val_pw100_s0_ep050.npz` existed beside `peaks_val_pw100_s0.npz` the
    # pooled DL row silently became a 9-"seed" mean with seed 0 counted twice. Caught by the
    # regression check against the saved epoch-50 table; it moved Doppler error by 0.017 bins.
    pat = re.compile(rf"^peaks_val_{re.escape(arm)}_s\d+{re.escape(suffix)}\.npz$")
    dl = sorted(p for p in DLR.glob(f"peaks_val_{arm}_s*.npz") if pat.match(p.name))
    if dl:
        sources["DL"] = [load_peaks(p) for p in dl]
    print("methods:", {k: len(v) for k, v in sources.items()})
    print("DL peaks:", [p.name for p in dl] or "NONE FOUND")

    all_ids = frames_meta.frame.to_numpy()
    groups = [("ALL", all_ids)]
    if args.per_condition:
        groups += [(c, frames_meta.loc[frames_meta.condition == c, "frame"].to_numpy())
                   for c in sorted(frames_meta.condition.unique()) if c != "noise"]

    rows = []
    for tol in args.tolerances:
        mc = MatchConfig(tol, tol)
        for gname, ids in groups:
            for method, pks in sources.items():
                stats = [measure(pk, ids, tb, valid, mc) for pk in pks]

                # SE of the seed mean. CFAR is one deterministic run, so its SE is NaN rather than
                # 0 -- it has sampling error over frames (that is what bootstrap_comparison.py
                # measures), just not seed-to-seed error, and writing 0 would imply otherwise.
                def ms(key):
                    v = np.array([s[key] for s in stats], dtype=float)
                    return (float(v.mean()),
                            float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))

                (pd_m, pd_e) = ms("pd")
                (dr_m, dr_e) = ms("mean_abs_dr_bins")
                (dv_m, dv_e) = ms("mean_abs_dv_bins")
                (rm_m, rm_e) = ms("rmse_bins")
                rows.append(dict(
                    gate=tol, group=gname, method=method, n_seeds=len(pks),
                    pd=pd_m, pd_se=pd_e,
                    dr_bins=dr_m, dr_bins_se=dr_e,
                    dv_bins=dv_m, dv_bins_se=dv_e,
                    rmse_bins=rm_m, rmse_bins_se=rm_e,
                    range_m=np.mean([s["mean_abs_range_err_m"] for s in stats]),
                    vel_mps=np.mean([s["mean_abs_vel_err_mps"] for s in stats]),
                ))
            print(f"  gate +/-{tol}  {gname} done")

    df = pd.DataFrame(rows)
    out = DLR / f"localization_{arm}{suffix}.csv"
    df.to_csv(out, index=False)

    for tol in args.tolerances:
        d = df[(df.gate == tol) & (df.group == "ALL")].set_index("method")
        print("\n" + "=" * 78)
        print(f"Localization error, all conditions pooled, gate +/-{tol}   "
              f"[DL arm: {arm}{suffix or ' (epoch 50)'}]")
        print("=" * 78)
        print(d[["n_seeds", "pd", "pd_se", "dr_bins", "dr_bins_se", "dv_bins", "dv_bins_se",
                 "rmse_bins", "range_m", "vel_mps"]].round(4).to_string())

    d1 = df[(df.gate == 1) & (df.group == "ALL")].set_index("method")
    if "DL" in d1.index:
        best = d1.drop("DL").dv_bins.idxmin()
        ratio = d1.loc["DL", "dv_bins"] / d1.loc[best, "dv_bins"]
        se = d1.loc["DL", "dv_bins_se"] / d1.loc[best, "dv_bins"]   # CFAR treated as fixed here
        print(f"\nDL Doppler error is {ratio:.1f}x (+/- {se:.1f} from seed noise alone) "
              f"the best CFAR ({best}), and range error "
              f"{d1.loc['DL','dr_bins'] / d1.drop('DL').dr_bins.min():.1f}x.")
        print("NOTE: this ratio divides by a CFAR value carrying its own frame-sampling error,")
        print("which is NOT propagated here -- see bootstrap_comparison.py for intervals that are.")
        print("The likely cause is the 3x3 label blob: the network is trained that all nine")
        print("cells around a target are 'target', so its peak may land anywhere inside it.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
