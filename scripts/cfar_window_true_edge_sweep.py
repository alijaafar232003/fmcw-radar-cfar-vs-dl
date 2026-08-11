"""Was 21x9 chosen for a real mechanism, or for an artefact?

Step 6 selected the window largely on a +0.296 gain at `clutter_edge`. But how much of
`clutter_edge` IS an edge depends on the window: narrowing it shortens its Doppler reach, so
targets stop being straddled and the frames quietly become ordinary noise frames still carrying the
label. Different windows were therefore scored on different experiments.

The fix is to stratify by something the window cannot change: **the target's Doppler offset from
the ridge**, in bins. The ridge is measured at +/-4 bins and `clutter_edge` targets sit at offsets
6-10, so a target at offset 6 is 2 bins clear of the ridge and one at offset 10 is 6 bins clear --
regardless of which window is being tested. Scoring each geometry within each offset band compares
like with like.

Run on `bench2_tune` + `bench2_val`. The test split is not read.
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
from radar.config import RadarConfig                                  # noqa: E402
from radar.simulator import make_axes                                 # noqa: E402

SPLITS = ["bench2_tune", "bench2_val"]
KINDS = ("CA", "GO", "OS")
TARGET_PFA = 1e-4
MC = MatchConfig(1, 1)
RIDGE_HALF = 4
OUT = ROOT / "results" / "cfar_window_true_edge_sweep.csv"

GEOMS = {"21x7": (2, 1), "21x9": (2, 2), "21x11": (2, 3), "21x13": (2, 4), "21x21": (2, 8)}

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
        return np.nan
    cells = float(np.round(np.median(fp[obs] / pfa[obs])))
    pfa = np.where(pfa > 0, pfa, 0.5 / cells)
    x, t = np.log10(pfa), np.log10(target)
    if len(x) < 2 or t < x.min() or t > x.max():
        return np.nan
    return float(np.interp(t, x, d.threshold.to_numpy()))


def main() -> None:
    _, _, _, vax = make_axes(RadarConfig())
    v0 = int(np.argmin(np.abs(vax)))

    frames = []
    for s in SPLITS:
        for fd in sorted((ROOT / "data" / s / "frames").iterdir()):
            md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
            if md.get("condition") != "clutter_edge":
                continue
            lab = pd.read_csv(fd / "labels.csv")
            tru = lab[["range_bin", "velocity_bin"]].to_numpy(float)
            # Distance, in bins, from the RIDGE EDGE to the nearest target. Window-independent.
            clear = int(min(abs(int(v) - v0) for v in lab.velocity_bin) - RIDGE_HALF)
            frames.append((np.load(fd / "rd_map_mag.npy").astype(np.float64) ** 2, tru, clear))

    band = {f[2] for f in frames}
    print(f"{len(frames)} clutter_edge frames from {' + '.join(SPLITS)}")
    print(f"ridge half-width {RIDGE_HALF} bins; target clearance from ridge edge: "
          f"{min(band)}..{max(band)} bins")
    for b in sorted(band):
        print(f"   clearance {b} bins ({0.42 * b:.2f} m/s): {sum(1 for f in frames if f[2] == b)} frames")

    BANDS = {"2 (closest)": [2], "3": [3], "4": [4], "5-6 (farthest)": [5, 6]}

    rows = []
    for label, (gv, tv) in GEOMS.items():
        hw_d = gv + tv
        for kind in KINDS:
            cfg = CfarConfig(kind=kind, guard_v=gv, train_v=tv)
            scored = {}
            for power, tru, clear in frames:
                ratio, valid, _ = cfar_ratio_map(power, cfg)
                scored.setdefault(clear, []).append(
                    (all_peaks(ratio, valid, nms_size=5), tru, power.shape, valid))

            for bname, bvals in BANDS.items():
                s = [x for b in bvals for x in scored.get(b, [])]
                if len(s) < 20:
                    continue
                pool = np.concatenate([d[0][:, 2] for d in s if len(d[0])])
                thr = pd_at(roc_curve(s, np.unique(np.quantile(pool, QUANTILES)), MC), TARGET_PFA)
                evs = [evaluate_frame(d[0][d[0][:, 2] >= thr][:, :2]
                                      if (len(d[0]) and thr == thr) else np.zeros((0, 2)),
                                      d[1], d[2], MC, valid_mask=d[3]) for d in s]
                a = aggregate(evs)
                # Does THIS window's training ring actually reach the ridge for this band?
                straddles = min(bvals) <= hw_d - gv - 1 + 1 + (hw_d - min(bvals))  # informational
                rows.append(dict(window=label, kind=kind, band=bname, frames=len(s),
                                 pd=a["pd"], reach_bins=hw_d,
                                 straddles=bool(hw_d >= min(bvals) + 1)))
        print(f"  done {label}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\nPd at matched measured Pfa, mean over CA/GO/OS — by target clearance from the ridge")
    piv = df.pivot_table(index="band", columns="window", values="pd")[list(GEOMS)]
    print(piv.round(3).to_string())
    print("\n(reading down a COLUMN: does that window cope as targets get closer to the ridge?)")
    print("(reading across a ROW: at a FIXED physical difficulty, which window is better?)")

    print("\nPer method, closest band only (clearance 2 bins = the hardest true edge)")
    p2 = df[df.band == "2 (closest)"].pivot_table(index="kind", columns="window", values="pd")
    print(p2[list(GEOMS)].round(3).to_string())

    print("\nWhich window each method's training ring actually reaches the ridge at:")
    for label, (gv, tv) in GEOMS.items():
        print(f"   {label:<7} Doppler reach {gv + tv} bins -> straddles clearance <= {gv + tv - 1}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
