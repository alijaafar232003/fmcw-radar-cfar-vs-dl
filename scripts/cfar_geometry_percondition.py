"""Step 6b — is the narrow-Doppler win real, or an artefact of pooling?

The pooled sweep showed Pd tracking usable-Doppler-bins almost perfectly, which is exactly what a
measurement artefact would look like. Two competing explanations:

  (a) REAL: a narrow Doppler window keeps the training ring clear of the clutter ridge, so
      thresholds are better wherever clutter is present. Step 4 measured that contamination
      directly. If this is the cause, the advantage should be CONCENTRATED in clutter, clutter_edge
      and mixed, and largely absent from isolated, masking and multi.

  (b) ARTEFACT: pooling every condition into one ROC lets clutter frames consume the whole
      false-alarm budget, forcing the threshold up and suppressing Pd everywhere. Then the
      advantage would appear UNIFORMLY across conditions, including clutter-free ones.

The two predictions differ, so the per-condition breakdown separates them. Evaluating per condition
is also what the stratified dataset was built for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map                # noqa: E402
from cfar.detect import all_peaks                               # noqa: E402
from eval.detection import MatchConfig, roc_curve               # noqa: E402
from radar.config import RadarConfig                            # noqa: E402

DATA = ROOT / "data" / "bench2_tune" / "frames"
OUT = ROOT / "results" / "cfar_geometry_percondition.csv"

CFG = RadarConfig()
KINDS = ("CA", "GO", "OS")
TARGET_PFA = 1e-4

GEOMETRIES = {
    "21x21": (2, 8, None, None),   # inherited
    "21x13": (2, 8, 2, 4),
    "21x11": (2, 8, 2, 3),
    "21x9":  (2, 8, 2, 2),         # narrow Doppler, range half-width UNCHANGED at 10
    "29x9":  (2, 12, 2, 2),        # pooled-sweep winner, but range half-width widened to 14
}


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


def pd_at(roc: pd.DataFrame, target: float) -> float:
    d = roc.dropna(subset=["pfa_per_cell", "pd"]).sort_values("pfa_per_cell")
    x, y = d["pfa_per_cell"].to_numpy(), d["pd"].to_numpy()
    if len(x) < 2 or target < x.min() or target > x.max():
        return float("nan")
    return float(np.interp(target, x, y))


def main() -> None:
    frames = load()
    conds = sorted({f[3] for f in frames} - {"noise"})   # noise has no targets -> no Pd
    rows = []

    for label, (g_r, t_r, g_v, t_v) in GEOMETRIES.items():
        for kind in KINDS:
            c = CfarConfig(kind=kind, guard=g_r, train=t_r,
                           guard_v=g_v, train_v=t_v, pfa=TARGET_PFA)

            scored = {k: [] for k in conds}
            for power, truths, shape, cond in frames:
                if cond not in scored:
                    continue
                ratio, valid, _ = cfar_ratio_map(power, c)
                # nms_size now defaults to 2 (mainlobe-matched)
                scored[cond].append((all_peaks(ratio, valid), truths, shape, valid))

            for cond in conds:
                s = scored[cond]
                pool = np.concatenate([d[0][:, 2] for d in s if len(d[0])])
                thr = np.quantile(pool, np.linspace(0.0, 0.9995, 60))
                roc = roc_curve(s, thr, MatchConfig(tol_r=1, tol_v=1))
                rows.append(dict(window=label, kind=kind, condition=cond,
                                 pd_at_pfa=pd_at(roc, TARGET_PFA)))
        print(f"  done {label}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\nPd at MEASURED Pfa 1e-4, per condition (mean over CA/GO/OS)")
    piv = df.pivot_table(index="condition", columns="window", values="pd_at_pfa")
    piv = piv[["21x21", "21x13", "21x11", "21x9", "29x9"]]
    piv["gain ->21x9"] = piv["21x9"] - piv["21x21"]
    piv["gain ->29x9"] = piv["29x9"] - piv["21x21"]
    print(piv.round(3).to_string())

    print("\nSame, per method, at the two extremes")
    p2 = df[df.window.isin(["21x21", "21x9"])].pivot_table(
        index=["condition", "kind"], columns="window", values="pd_at_pfa")
    p2["gain"] = p2["21x9"] - p2["21x21"]
    print(p2.round(3).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
