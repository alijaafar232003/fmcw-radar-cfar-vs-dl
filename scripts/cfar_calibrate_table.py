"""Calibrate every CFAR variant by Monte Carlo and write the table `cfar.py` reads.

Replaces the closed-form threshold multipliers, which the 19 Aug audit measured at 0.13x-6.55x of
the target rate depending on variant (`scripts/cfar_alpha_audit.py`). The closed form assumes the
arithmetic mean of N *independent* exponential cells; the Blackman window correlates adjacent
reference cells, and only CA is a mean at all.

PROCEDURE, per variant, at the selected 21x9 geometry:

1. Pool the per-cell ratio values over `--frames` pure-noise frames through the real processing
   chain, on the CALIBRATION seed.
2. alpha = the (1 - Pfa) empirical quantile of that pool -- the level exceeded by a fraction Pfa of
   noise cells.
3. Apply it to a HELD-OUT seed and measure the achieved rate. That is the test that the value
   generalises rather than fitting the calibration draw.

Only a variant that passes step 3 within tolerance is written to the table.

WHAT THIS DOES NOT CHANGE. alpha is not an input to `cfar_ratio_map`'s output -- the ratio map is
`power / noise`, and alpha is the threshold applied to it afterwards. Every result in this project
is scored at matched MEASURED Pfa, which sweeps the threshold and discards alpha. So this changes
the DESIGN operating point and nothing else. `--verify-invariance` asserts exactly that.

    python scripts/cfar_calibrate_table.py --frames 300
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.calibrate import ALL_KINDS, make_noise_power_frames        # noqa: E402
from cfar.cfar import CfarConfig, alpha_key, cfar_ratio_map          # noqa: E402
from radar.config import RadarConfig                                 # noqa: E402

TARGET_PFA = 1e-4
OUT = ROOT / "configs" / "cfar_alpha_table.json"


def pooled_cells(cfg: CfarConfig, radar: RadarConfig, n_frames: int, seed: int) -> np.ndarray:
    out = []
    for power in make_noise_power_frames(radar, n_frames, seed):
        ratio, valid, _ = cfar_ratio_map(power, cfg)
        out.append(ratio[valid])
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--verify-seed", type=int, default=9000)
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="max |achieved/target - 1| on the HELD-OUT seed to accept a value")
    ap.add_argument("--verify-invariance", action="store_true",
                    help="assert the ratio map does not depend on alpha, then exit")
    args = ap.parse_args()

    radar = RadarConfig()

    if args.verify_invariance:
        # The claim the whole re-run rests on: alpha is metadata, not an input to the map.
        power = next(iter(make_noise_power_frames(radar, 1, 12345)))
        for kind in ALL_KINDS:
            c = CfarConfig(kind=kind)
            r1, v1, a1 = cfar_ratio_map(power, c)
            r2, v2, a2 = cfar_ratio_map(power, CfarConfig(kind=kind, pfa=1e-2))
            same = np.array_equal(r1, r2) and np.array_equal(v1, v2)
            print(f"  {kind}: ratio map identical under alpha {a1:.3f} vs {a2:.3f} -> {same}")
            assert same, f"{kind}: ratio map DID change with alpha -- the premise is wrong"
        print("VERIFIED: the ratio map is independent of alpha for all four variants.")
        return

    print(f"{args.frames} pure-noise frames per seed | calibrate on seed {args.calib_seed}, "
          f"verify on seed {args.verify_seed}")

    entries, report = {}, []
    for kind in ALL_KINDS:
        cfg = CfarConfig(kind=kind, pfa=TARGET_PFA)
        closed = cfg.closed_form_alpha()

        cal_cells = pooled_cells(cfg, radar, args.frames, args.calib_seed)
        alpha = float(np.quantile(cal_cells, 1.0 - TARGET_PFA))

        ver_cells = pooled_cells(cfg, radar, args.frames, args.verify_seed)
        achieved_cal = float((ver_cells >= alpha).mean())
        achieved_closed = float((ver_cells >= closed).mean())

        dev = abs(achieved_cal / TARGET_PFA - 1.0)
        ok = dev <= args.tolerance
        if ok:
            entries[alpha_key(cfg)] = alpha

        report.append(dict(kind=kind, closed=closed, calibrated=alpha,
                           achieved_at_closed=achieved_closed, achieved_at_calibrated=achieved_cal,
                           accepted=ok))
        print(f"  {kind}: closed {closed:7.3f} -> calibrated {alpha:7.3f} "
              f"({alpha / closed:.2f}x) | held-out Pfa {achieved_cal:.3e} "
              f"({achieved_cal / TARGET_PFA:.2f}x target) {'ACCEPT' if ok else 'REJECT'}")

    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {
            "generated": str(date.today()),
            "generated_at_commit": commit,
            "method": "Monte-Carlo (1 - Pfa) quantile of pooled per-cell ratio on pure noise",
            "target_pfa": TARGET_PFA,
            "frames_per_seed": args.frames,
            "calibration_seed": args.calib_seed,
            "verification_seed": args.verify_seed,
            "acceptance": f"|achieved/target - 1| <= {args.tolerance} on the held-out seed",
            "geometry": "21x9 (guard 2/2, train 8/2) -- keys are geometry-specific",
            "note": ("alpha is NOT an input to cfar_ratio_map; it is the threshold applied to it. "
                     "Every result scored at matched MEASURED Pfa is unaffected by this table."),
            "python": platform.python_version(),
        },
        "entries": entries,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT} with {len(entries)} of {len(ALL_KINDS)} variants")
    for r in report:
        print(f"  {r['kind']:3s} achieved/target: closed {r['achieved_at_closed'] / TARGET_PFA:6.2f}x"
              f"  ->  calibrated {r['achieved_at_calibrated'] / TARGET_PFA:5.2f}x")


if __name__ == "__main__":
    main()
