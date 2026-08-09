"""Step 6 — choose the CFAR window on the tuning split, by Pd at matched MEASURED Pfa.

Why measured rather than design: step 5 showed clutter drives the achieved false-alarm rate 60-125x
above design, and by different factors for each method. Comparing at "design 1e-4" would in fact
compare CA at 1.2e-2 against GO at 8.7e-3 against OS at 3.6e-2 -- three different operating points.
So each geometry gets its full ROC and Pd is read off at a common measured Pfa.

Square and rectangular windows are both swept. The Doppler axis is 64 bins, so a window sized for
the 256-bin range axis spans a third of it and costs testable map to border exclusion; the two axes
have no reason to want the same window.

Selection is made on this split alone, which the detectors are never evaluated on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.calibrate import calibrate_alpha                      # noqa: E402
from cfar.cfar import CfarConfig, cfar_ratio_map                # noqa: E402
from cfar.detect import all_peaks                               # noqa: E402
from eval.detection import MatchConfig, roc_curve               # noqa: E402
from radar.config import RadarConfig                            # noqa: E402

DATA = ROOT / "data" / "bench2_tune" / "frames"
OUT = ROOT / "results" / "cfar_geometry_sweep.csv"

CFG = RadarConfig()
KINDS = ("CA", "GO", "OS")
TARGET_PFA = 1e-4

# (guard_r, train_r, guard_v, train_v). guard stays at 2: the PSF measurement put the -30 dB
# mainlobe half-width at exactly 2 bins on both axes, and it is set by the Blackman window rather
# than by the sensor, so it does not move.
GEOMETRIES = [
    (2, 4, None, None),    # square, small
    (2, 6, None, None),    # square
    (2, 8, None, None),    # square, the inherited default
    (2, 8, 2, 2),          # rectangular, narrow Doppler
    (2, 8, 2, 3),
    (2, 8, 2, 4),
    (2, 8, 2, 6),
    (2, 12, 2, 3),         # wider in range, narrow in Doppler
    (2, 16, 2, 3),
    (2, 12, 2, 2),
]


def load_frames():
    """(power map, truths, shape, condition) for every tuning frame."""
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


def pd_at_measured_pfa(roc: pd.DataFrame, target: float) -> float:
    """Interpolate Pd where the MEASURED per-cell Pfa equals `target`.

    The ROC is traced by descending threshold, so Pfa increases down the table. numpy.interp needs
    ascending x, hence the sort. Returns NaN if the target lies outside the achieved range, which
    is itself informative -- a geometry that cannot reach 1e-4 at any threshold is unusable.
    """
    d = roc.dropna(subset=["pfa_per_cell", "pd"]).sort_values("pfa_per_cell")
    x, y = d["pfa_per_cell"].to_numpy(), d["pd"].to_numpy()
    if target < x.min() or target > x.max():
        return float("nan")
    return float(np.interp(target, x, y))


def main() -> None:
    frames = load_frames()
    print(f"tuning split: {len(frames)} frames, "
          f"{sum(len(f[1]) for f in frames)} targets\n")

    rows = []
    for g_r, t_r, g_v, t_v in GEOMETRIES:
        label = (f"{2*(g_r+t_r)+1}x{2*(g_v+t_v)+1}" if g_v is not None
                 else f"{2*(g_r+t_r)+1}x{2*(g_r+t_r)+1}")

        for kind in KINDS:
            c = CfarConfig(kind=kind, guard=g_r, train=t_r,
                           guard_v=g_v, train_v=t_v, pfa=TARGET_PFA)

            # Threshold is irrelevant to the ROC (all_peaks keeps every peak and the evaluator
            # sweeps the score), but alpha is recorded so the operating point is reproducible.
            alpha, _ = calibrate_alpha(c, CFG, TARGET_PFA, n_frames=60, seed=0)

            scored = []
            for power, truths, shape, cond in frames:
                ratio, valid, _ = cfar_ratio_map(power, c)
                det = all_peaks(ratio, valid, nms_size=5)
                scored.append((det, truths, shape, valid))

            thr = np.quantile(
                np.concatenate([s[0][:, 2] for s in scored if len(s[0])]),
                np.linspace(0.0, 0.9995, 60),
            )
            roc = roc_curve(scored, thr, MatchConfig(tol_r=1, tol_v=1))

            rows.append(dict(
                window=label, guard_r=g_r, train_r=t_r,
                guard_d=g_v if g_v is not None else g_r,
                train_d=t_v if t_v is not None else t_r,
                kind=kind, n_ref=c.num_ref_cells, alpha=alpha,
                usable_doppler=CFG.M - 2 * c.half_width_d,
                pd_at_pfa=pd_at_measured_pfa(roc, TARGET_PFA),
                pfa_min=roc["pfa_per_cell"].min(),
                pfa_max=roc["pfa_per_cell"].max(),
            ))
            r = rows[-1]
            print(f"  {label:>7} {kind:<3} ref{c.num_ref_cells:>5}  "
                  f"usableD {r['usable_doppler']:>3}  Pd@1e-4 {r['pd_at_pfa']:.3f}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\nPd at MEASURED Pfa = 1e-4")
    piv = df.pivot_table(index=["window", "usable_doppler"], columns="kind", values="pd_at_pfa")
    piv["mean"] = piv.mean(axis=1)
    print(piv.sort_values("mean", ascending=False).round(3).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
