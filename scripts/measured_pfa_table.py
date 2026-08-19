"""T4 — the measured Pfa actually achieved, per condition per method.

The whole comparison rests on the claim that both arms were placed at the SAME false-alarm rate.
Every other table assumes it; this one checks it. If a method's achieved Pfa is far from the 1e-4
target on some condition, its Pd on that condition is not comparable to anyone else's and the
difference reported elsewhere is an artefact of a mismatched operating point.

Reported as achieved Pfa per cell, over the CFAR-testable interior only (the same `valid` mask both
arms are scored on), at each method's own matched threshold on that condition. DL is the 8-seed
mean.

    python scripts/measured_pfa_table.py --tag frozen --epoch 21 --split bench2_test
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
DLR = ROOT / "results" / "dl"


def load_peaks(p: Path) -> dict:
    a = np.load(p)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--epoch", type=int, default=21)
    ap.add_argument("--split", default="bench2_test")
    ap.add_argument("--tolerance", type=int, default=1)
    args = ap.parse_args()

    PK = args.split.replace("bench2_", "")
    CV = ROOT / "results" / f"cfar_{PK}"
    valid = cfar_valid_mask(SHAPE, CfarConfig())
    mc = MatchConfig(args.tolerance, args.tolerance)

    fm = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}

    srcs = {m: [load_peaks(CV / f"peaks_{m}.npz")] for m in ("CA", "GO", "OS", "SO")
            if (CV / f"peaks_{m}.npz").exists()}
    pat = re.compile(rf"^peaks_{PK}_{args.tag}_s\d+_ep{args.epoch:03d}\.npz$")
    dl = sorted(p for p in DLR.glob(f"peaks_{PK}_{args.tag}_s*.npz") if pat.match(p.name))
    if dl:
        srcs["DL"] = [load_peaks(p) for p in dl]
    print("methods:", {k: len(v) for k, v in srcs.items()})

    rows = []
    for cond, g in fm.groupby("condition"):
        ids = g.frame.to_numpy()
        for m, pks in srcs.items():
            got = []
            for pk in pks:
                fr = [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))), SHAPE, valid)
                      for i in ids]
                _, thr = _at_measured_pfa(roc_curve(fr, roc_thresholds(fr, mc), mc), TARGET_PFA)
                evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr)
                                      else np.zeros((0, 2)), t, s, mc, valid_mask=v)
                       for p, t, s, v in fr]
                got.append(aggregate(evs)["pfa_per_cell"])
            rows.append(dict(condition=cond, method=m, n_seeds=len(pks),
                             measured_pfa=float(np.mean(got)),
                             ratio_to_target=float(np.mean(got)) / TARGET_PFA))
        print(f"  {cond} done")

    df = pd.DataFrame(rows)
    out = DLR / f"measured_pfa_{args.tag}_ep{args.epoch:03d}_{PK}_tol{args.tolerance}.csv"
    df.to_csv(out, index=False)

    piv = df.pivot(index="condition", columns="method", values="measured_pfa")
    print("\n" + "=" * 78)
    print(f"T4 — measured Pfa per cell at the matched operating point ({args.split}, "
          f"target {TARGET_PFA:.0e})")
    print("=" * 78)
    print(piv.map(lambda v: f"{v:.2e}").to_string())
    print("\nratio to target (1.0 = exactly matched):")
    print(df.pivot(index="condition", columns="method",
                   values="ratio_to_target").round(2).to_string())
    worst = df.ratio_to_target.abs()
    print(f"\nworst deviation: {df.loc[worst.idxmax(), 'method']} on "
          f"{df.loc[worst.idxmax(), 'condition']} at {worst.max():.2f}x target")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
