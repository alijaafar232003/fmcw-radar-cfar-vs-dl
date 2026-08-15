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
    args = ap.parse_args()

    valid = cfar_valid_mask(SHAPE, CfarConfig())
    frames_meta = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}

    sources = {}
    for m in ("CA", "GO", "OS", "SO"):
        p = CV / f"peaks_{m}.npz"
        if p.exists():
            sources[m] = [load_peaks(p)]
    dl = sorted(DLR.glob(f"peaks_val_pw{args.pos_weight}_s*.npz"))
    if dl:
        sources["DL"] = [load_peaks(p) for p in dl]
    print("methods:", {k: len(v) for k, v in sources.items()})

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
                rows.append(dict(
                    gate=tol, group=gname, method=method, n_seeds=len(pks),
                    pd=np.mean([s["pd"] for s in stats]),
                    dr_bins=np.mean([s["mean_abs_dr_bins"] for s in stats]),
                    dv_bins=np.mean([s["mean_abs_dv_bins"] for s in stats]),
                    rmse_bins=np.mean([s["rmse_bins"] for s in stats]),
                    range_m=np.mean([s["mean_abs_range_err_m"] for s in stats]),
                    vel_mps=np.mean([s["mean_abs_vel_err_mps"] for s in stats]),
                ))
            print(f"  gate +/-{tol}  {gname} done")

    df = pd.DataFrame(rows)
    out = DLR / f"localization_pw{args.pos_weight}.csv"
    df.to_csv(out, index=False)

    for tol in args.tolerances:
        d = df[(df.gate == tol) & (df.group == "ALL")].set_index("method")
        print("\n" + "=" * 78)
        print(f"Localization error, all conditions pooled, gate +/-{tol}")
        print("=" * 78)
        print(d[["pd", "dr_bins", "dv_bins", "rmse_bins", "range_m", "vel_mps"]]
              .round(4).to_string())

    d1 = df[(df.gate == 1) & (df.group == "ALL")].set_index("method")
    if "DL" in d1.index:
        best = d1.drop("DL").dv_bins.idxmin()
        print(f"\nDL Doppler error is {d1.loc['DL','dv_bins'] / d1.loc[best,'dv_bins']:.1f}x "
              f"the best CFAR ({best}), and range error "
              f"{d1.loc['DL','dr_bins'] / d1.drop('DL').dr_bins.min():.1f}x.")
        print("The likely cause is the 3x3 label blob: the network is trained that all nine")
        print("cells around a target are 'target', so its peak may land anywhere inside it.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
