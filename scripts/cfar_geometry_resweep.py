"""Re-run the step-6 geometry sweeps under the CURRENT scoring protocol, on `bench2_tune`.

WHY. `results/cfar_geometry_percondition.csv` disagrees with the step-6 table in `PROGRESS.md` on
the SIGN for two conditions (`isolated` and `multi`), not merely on absolute values. The cause is
that the saved CSV was produced under a scoring path this project has since replaced, in three
ways:

1. **NMS size.** `cfar_geometry_percondition.py` calls `all_peaks(ratio, valid)` with no explicit
   size, and at the time that defaulted to **2** (a transient state during the step-6d NMS
   investigation). It now defaults to **5**, the project standard. So the script no longer
   reproduces its own artefact.
2. **Interpolated Pd.** It reads Pd off the ROC by interpolation. The project rejected that: the
   interpolation finds the *threshold*, and Pd is then MEASURED at it, because interpolated Pd
   carries up to 0.017 of error and is biased optimistic on exactly the clutter strata this sweep
   is about.
3. **Threshold grid.** It uses uniform quantiles; the project moved to the false-alarm-anchored
   grid after Milestone 20b, because a quantile grid steps straight over the operating point.

This script re-runs both sweeps with the same scoring path as `cfar_test_baseline.py`, so the
notebook's geometry figures are consistent with every other number in the project.

It also adds the **train-cell sweep** that justified `train = 8`, which was recorded in Milestone 9
but never had its data persisted: Pd against the number of training cells, showing the knee where
more reference cells stop buying anything.

`bench2_tune` only (400 frames, seed 91001) -- disjoint from every evaluation split.

    python scripts/cfar_geometry_resweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map                     # noqa: E402
from cfar.detect import all_peaks                                    # noqa: E402
from dl.stratified_eval import NMS, _at_measured_pfa                 # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,            # noqa: E402
                  roc_curve, roc_thresholds)

DATA = ROOT / "data" / "bench2_tune" / "frames"
TARGET_PFA = 1e-4
KINDS = ("CA", "GO", "OS")
MC = MatchConfig(1, 1)

# (guard_r, train_r, guard_v, train_v). 21x21 is the inherited square window.
GEOMETRIES = {
    "21x21": (2, 8, None, None),
    "21x13": (2, 8, 2, 4),
    "21x11": (2, 8, 2, 3),
    "21x9":  (2, 8, 2, 2),
    "29x9":  (2, 12, 2, 2),
}

# The knee that set train = 8. Doppler half-width held at the selected 4 (train_v = 2).
TRAIN_R = [2, 4, 6, 8, 10, 12, 16]


def load():
    out = []
    for fd in sorted(DATA.iterdir()):
        md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
        mag = np.load(fd / "rd_map_mag.npy")
        lab = (pd.read_csv(fd / "labels.csv")
               if (fd / "labels.csv").stat().st_size > 5 else pd.DataFrame())
        truths = (lab[["range_bin", "velocity_bin"]].to_numpy(float)
                  if len(lab) else np.empty((0, 2)))
        out.append((mag ** 2, truths, mag.shape, md.get("condition", "?")))
    return out


def score(frames, cfg) -> dict:
    """Pd per condition, MEASURED at the matched-measured-Pfa threshold. Mirrors cfar_test_baseline."""
    by_cond: dict[str, list] = {}
    for power, truths, shape, cond in frames:
        if cond == "noise":
            continue
        ratio, valid, _ = cfar_ratio_map(power, cfg)
        by_cond.setdefault(cond, []).append(
            (all_peaks(ratio, valid, nms_size=NMS), truths, shape, valid))

    out = {}
    for cond, s in by_cond.items():
        roc = roc_curve(s, roc_thresholds(s, MC), MC)
        _, thr = _at_measured_pfa(roc, TARGET_PFA)
        evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr)
                              else np.zeros((0, 2)), t, sh, MC, valid_mask=v)
               for p, t, sh, v in s]
        out[cond] = aggregate(evs)["pd"]
    return out


def main() -> None:
    frames = load()
    print(f"{len(frames)} frames from bench2_tune")

    # ------------------------------------------------ (a) geometry x condition
    rows = []
    for label, (g_r, t_r, g_v, t_v) in GEOMETRIES.items():
        for kind in KINDS:
            cfg = CfarConfig(kind=kind, guard=g_r, train=t_r,
                             guard_v=g_v, train_v=t_v, pfa=TARGET_PFA)
            for cond, pdv in score(frames, cfg).items():
                rows.append(dict(window=label, kind=kind, condition=cond,
                                 n_ref=cfg.num_ref_cells, pd_at_pfa=pdv))
        print(f"  geometry {label} done")
    g = pd.DataFrame(rows)
    g.to_csv(ROOT / "results" / "cfar_geometry_percondition_v2.csv", index=False)

    # ------------------------------------------------ (b) train-cell sweep
    rows = []
    for t_r in TRAIN_R:
        for kind in KINDS:
            cfg = CfarConfig(kind=kind, guard=2, train=t_r, guard_v=2, train_v=2,
                             pfa=TARGET_PFA)
            per = score(frames, cfg)
            for cond, pdv in per.items():
                rows.append(dict(train_r=t_r, kind=kind, condition=cond,
                                 n_ref=cfg.num_ref_cells,
                                 window=f"{2 * (2 + t_r) + 1}x9", pd_at_pfa=pdv))
        print(f"  train_r={t_r} done")
    tr = pd.DataFrame(rows)
    tr.to_csv(ROOT / "results" / "cfar_train_sweep.csv", index=False)

    # ------------------------------------------------ report
    gm = g.groupby(["window", "condition"]).pd_at_pfa.mean().unstack()
    order = [w for w in GEOMETRIES if w in gm.index]
    print("\nGEOMETRY x CONDITION (mean over CA/GO/OS), current protocol:")
    print(gm.loc[order].round(3).to_string())
    print("\n21x21 -> 21x9 change:")
    print((gm.loc["21x9"] - gm.loc["21x21"]).round(3).sort_values(ascending=False).to_string())
    print("\n29x9 minus 21x9:")
    print((gm.loc["29x9"] - gm.loc["21x9"]).round(3).sort_values().to_string())

    tm = tr.groupby(["train_r", "condition"]).pd_at_pfa.mean().unstack()
    tm["POOLED"] = tm.mean(axis=1)
    print("\nTRAIN-CELL SWEEP (mean over CA/GO/OS), pooled column is the mean over conditions:")
    print(tm.round(3).to_string())
    d = tm["POOLED"].diff()
    print("\nchange in pooled Pd per step of train_r:")
    print(d.round(4).to_string())
    print("\nwrote results/cfar_geometry_percondition_v2.csv and results/cfar_train_sweep.csv")


if __name__ == "__main__":
    main()
