"""Is each CFAR variant's threshold multiplier actually calibrated? Direct test on pure noise.

WHY THIS EXISTS. `PROGRESS.md` Milestone 9 records empirically calibrated alphas for CA (9.74),
GO (9.09) and OS (6.98), verified against an independent noise seed. SO-CFAR was added on 15 August,
long afterwards, and `calibrate_all()` still defaults to `kinds=("CA", "OS", "GO")` — SO is not in
the list. This checks what each variant is actually using now.

THE TEST. On pure noise every variant should achieve the same per-cell Pfa at its own alpha, because
that is what the alpha is *for*. Run the real processing chain on noise-only CPIs, apply each
method's current alpha, and measure the achieved rate. A variant that lands at 1e-4 is calibrated;
one that does not, is not.

TWO EFFECTS ARE SEPARATED, because they are easy to confuse:

* **per-cell Pfa on the raw ratio map** — this tests the ALPHA, and nothing else;
* **post-NMS Pfa**, counted over surviving local maxima — this is what the notebook's
  measured/design threshold ratio sees. NMS removes almost all candidate cells, so the two differ
  by a large factor *for every method*. A uniform offset is NMS; a method-specific one is alpha.

Calibration seed 0, verification seed 9000 — the same pair Milestone 9 used.

    python scripts/cfar_alpha_audit.py            ->  results/cfar_alpha_audit.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.calibrate import make_noise_power_frames                   # noqa: E402
from cfar.cfar import CfarConfig, cfar_ratio_map                     # noqa: E402
from cfar.detect import all_peaks                                    # noqa: E402
from dl.stratified_eval import NMS                                   # noqa: E402
from radar.config import RadarConfig                                 # noqa: E402

METHODS = ("CA", "GO", "OS", "SO")
TARGET_PFA = 1e-4


def current_alpha(kind: str) -> float:
    """The alpha `cfar_ratio_map` actually applies for this kind — closed form, per cfar.py."""
    cfg = CfarConfig(kind=kind)
    return cfg.alpha_go() if kind in ("GO", "SO") else cfg.alpha()


def pooled_ratios(kind: str, radar_cfg: RadarConfig, n_frames: int, seed: int):
    """Pooled per-cell ratio values, and the post-NMS peak scores, over pure-noise frames."""
    cfg = CfarConfig(kind=kind)
    cells, peaks, n_valid, n_frames_done = [], [], 0, 0
    for power in make_noise_power_frames(radar_cfg, n_frames, seed):
        ratio, valid, _ = cfar_ratio_map(power, cfg)
        cells.append(ratio[valid])
        n_valid += int(valid.sum())
        pk = all_peaks(ratio, valid, nms_size=NMS)
        if len(pk):
            peaks.append(pk[:, 2])
        n_frames_done += 1
    return (np.concatenate(cells),
            np.concatenate(peaks) if peaks else np.zeros(0),
            n_valid, n_frames_done)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--verify-seed", type=int, default=9000)
    args = ap.parse_args()

    radar = RadarConfig()
    print(f"{args.frames} pure-noise frames per seed; calibration seed {args.calib_seed}, "
          f"verification seed {args.verify_seed}")

    rows = []
    for kind in METHODS:
        a_now = current_alpha(kind)

        c_cells, c_peaks, n_valid, nf = pooled_ratios(kind, radar, args.frames, args.calib_seed)
        v_cells, v_peaks, n_valid_v, _ = pooled_ratios(kind, radar, args.frames, args.verify_seed)

        # Calibrated alpha = the (1 - Pfa) quantile of the pooled noise ratio distribution.
        a_cal = float(np.quantile(c_cells, 1.0 - TARGET_PFA))

        rows.append(dict(
            method=kind,
            alpha_closed_form=a_now,
            alpha_calibrated=a_cal,
            ratio_cal_over_closed=a_cal / a_now,
            # Achieved per-cell Pfa at the alpha the pipeline USES, on both seeds.
            pfa_cells_calibseed_at_current=float((c_cells >= a_now).mean()),
            pfa_cells_verifyseed_at_current=float((v_cells >= a_now).mean()),
            # ... and at the calibrated alpha, on the INDEPENDENT seed. This is the verification.
            pfa_cells_verifyseed_at_calibrated=float((v_cells >= a_cal).mean()),
            # Post-NMS, at the alpha the pipeline uses -- what the notebook's ratios see.
            pfa_peaks_verifyseed_at_current=float((v_peaks >= a_now).sum()) / max(n_valid_v, 1),
            n_cells=int(c_cells.size), n_frames=nf,
        ))
        r = rows[-1]
        print(f"  {kind}: alpha now {a_now:.3f}, calibrated {a_cal:.3f} "
              f"({a_cal / a_now:.2f}x) | per-cell Pfa at current alpha "
              f"{r['pfa_cells_verifyseed_at_current']:.2e}")

    df = pd.DataFrame(rows)

    # Achieved Pfa as a function of alpha, per method, on the VERIFICATION seed -- the curve the
    # calibration reads its answer off. Saved so the notebook can plot it without re-running.
    grid = np.geomspace(3.0, 40.0, 120)
    curve = []
    for kind in METHODS:
        v_cells, _, _, _ = pooled_ratios(kind, radar, args.frames, args.verify_seed)
        for a in grid:
            curve.append(dict(method=kind, alpha=float(a),
                              pfa=float((v_cells >= a).mean())))
    pd.DataFrame(curve).to_csv(ROOT / "results" / "cfar_alpha_curve.csv", index=False)
    print("wrote results/cfar_alpha_curve.csv")

    out = ROOT / "results" / "cfar_alpha_audit.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 96)
    print("ACHIEVED PER-CELL Pfa ON INDEPENDENT NOISE, at the alpha the pipeline currently uses")
    print("=" * 96)
    show = df.set_index("method")[["alpha_closed_form", "alpha_calibrated",
                                   "ratio_cal_over_closed",
                                   "pfa_cells_calibseed_at_current",
                                   "pfa_cells_verifyseed_at_current",
                                   "pfa_cells_verifyseed_at_calibrated"]]
    print(show.to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\ntarget = {TARGET_PFA:.0e}")

    dev = (df.set_index("method").pfa_cells_verifyseed_at_current / TARGET_PFA)
    print("\nachieved / target, per-cell, at the CURRENT alpha:")
    print(dev.round(2).to_string())
    print("\nA method near 1.0 is calibrated. A method far from 1.0 is not.")

    nmsr = (df.set_index("method").pfa_peaks_verifyseed_at_current
            / df.set_index("method").pfa_cells_verifyseed_at_current)
    print("\npost-NMS Pfa / per-cell Pfa (the NMS survival fraction), per method:")
    print(nmsr.round(4).to_string())
    print("If this factor is similar across methods it is an NMS effect, shared by all four;")
    print("only a METHOD-SPECIFIC deviation in the per-cell column above is an alpha problem.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
