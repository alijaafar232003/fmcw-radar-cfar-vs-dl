"""Re-sweep the CFAR Doppler window on the TUNING split — the step-6 sweep stopped too early.

Step 6 swept 21x21, 21x13, 21x11, 21x9 and 29x9, found Pd rising monotonically as the Doppler
window narrowed, and selected the narrowest it had tried. `diagnose_go_clutter_edge.py` shows Pd is
still rising at 21x7 on `clutter_edge` — so the sweep terminated at the edge of its own range
rather than at an optimum, and 21x9 was the boundary of the search, not its winner.

This extends the range downward and scores EVERY condition, so the choice cannot be made on the one
stratum that motivated the question.

Run on `bench2_tune` only. The test split is not read here, and the choice this produces must be
fixed before CFAR is re-run on test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map                      # noqa: E402
from cfar.detect import all_peaks                                     # noqa: E402
from eval import MatchConfig, aggregate, evaluate_frame, roc_curve    # noqa: E402

SPLIT = "bench2_tune"
KINDS = ("CA", "GO", "OS")
TARGET_PFA = 1e-4
MC = MatchConfig(1, 1)
OUT = ROOT / "results" / "cfar_window_resweep.csv"

# (guard_v, train_v) -> Doppler window width 2*(g+t)+1
GEOMS = {
    "21x5":  (1, 1),      # guard 1: below the measured 2-bin mainlobe half-width -- expected to hurt
    "21x7":  (2, 1),
    "21x9":  (2, 2),      # the incumbent
    "21x11": (2, 3),
    "21x13": (2, 4),
}

QUANTILES = np.clip(np.concatenate([
    np.linspace(0.0, 0.99, 40),
    1.0 - np.logspace(np.log10(0.01), np.log10(2e-5), 40)]), 0.0, 1.0)


def load():
    out = []
    for fd in sorted((ROOT / "data" / SPLIT / "frames").iterdir()):
        md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
        lab = (pd.read_csv(fd / "labels.csv")
               if (fd / "labels.csv").stat().st_size > 5 else pd.DataFrame())
        tru = (lab[["range_bin", "velocity_bin"]].to_numpy(float)
               if len(lab) else np.zeros((0, 2)))
        out.append((np.load(fd / "rd_map_mag.npy").astype(np.float64) ** 2,
                    tru, md.get("condition")))
    return out


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
    frames = load()
    conds = sorted({f[2] for f in frames} - {"noise"})
    print(f"{len(frames)} frames, conditions: {', '.join(conds)}")

    rows = []
    for label, (gv, tv) in GEOMS.items():
        for kind in KINDS:
            cfg = CfarConfig(kind=kind, guard_v=gv, train_v=tv)
            scored = {c: [] for c in conds}

            for power, tru, cond in frames:
                if cond not in scored:
                    continue
                ratio, valid, _ = cfar_ratio_map(power, cfg)
                scored[cond].append((all_peaks(ratio, valid, nms_size=5),
                                     tru, power.shape, valid))

            for cond, s in scored.items():
                pool = np.concatenate([d[0][:, 2] for d in s if len(d[0])])
                roc = roc_curve(s, np.unique(np.quantile(pool, QUANTILES)), MC)
                _, thr = pd_at(roc, TARGET_PFA)

                evs = [evaluate_frame(d[0][d[0][:, 2] >= thr][:, :2]
                                      if (len(d[0]) and thr == thr) else np.zeros((0, 2)),
                                      d[1], d[2], MC, valid_mask=d[3]) for d in s]
                a = aggregate(evs)
                rows.append(dict(window=label, kind=kind, condition=cond, pd=a["pd"],
                                 measured_pfa=a["pfa_per_cell"], threshold=thr,
                                 ref_cells=cfg.num_ref_cells,
                                 usable_doppler=64 - 2 * cfg.half_width_d))
        print(f"  done {label}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\nPd at matched MEASURED Pfa 1e-4, mean over CA/GO/OS")
    piv = df.pivot_table(index="condition", columns="window", values="pd")[list(GEOMS)]
    piv.loc["MEAN"] = piv.mean()
    print(piv.round(3).to_string())

    print("\nPer method, incumbent 21x9 vs the candidate 21x7")
    p2 = df[df.window.isin(["21x9", "21x7"])].pivot_table(
        index=["condition", "kind"], columns="window", values="pd")
    p2["gain"] = p2["21x7"] - p2["21x9"]
    print(p2.round(3).to_string())

    print("\nWho wins each condition, at each window")
    for w in GEOMS:
        sub = df[df.window == w].pivot_table(index="condition", columns="kind", values="pd")
        print(f"   {w:<7} " + "  ".join(f"{c}:{sub.loc[c].idxmax()}" for c in sub.index))

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
