"""Export the measured point-spread function as a profile, for the CFAR-geometry figure.

`measure_psf.py` prints the widths that justify `guard = 2`; it does not persist the underlying
profile, so the notebook had no way to *show* the reasoning without pasting numbers. This writes the
profile itself.

The worst case is the off-bin target (offset 0.5): a target sitting exactly on a bin centre gives an
artificially narrow response, so the geometry must be justified against the wider one.

    python scripts/export_psf_profile.py   ->  results/psf_profile.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.config import RadarConfig, derive_radar_params               # noqa: E402
from radar.rd_processing import compute_rd_map                          # noqa: E402
from radar.simulator import Target, make_axes, simulate_cpi             # noqa: E402


def profile(cfg: RadarConfig, offset_bins: float) -> pd.DataFrame:
    derive_radar_params(cfg)
    _, _, range_axis, vel_axis = make_axes(cfg)
    dr = range_axis[1] - range_axis[0]
    dv = vel_axis[1] - vel_axis[0]

    r0 = range_axis[len(range_axis) // 2] + offset_bins * dr
    v0 = vel_axis[len(vel_axis) // 2] + offset_bins * dv

    cpi = simulate_cpi([Target(r0, v0, 0.0)], cfg, noise_var=0.0, seed=0)
    _, mag, _, _, _ = compute_rd_map(cpi, cfg)

    power_db = 20 * np.log10(mag + 1e-30)
    pr, pv = np.unravel_index(int(np.argmax(power_db)), power_db.shape)
    peak = float(power_db[pr, pv])

    rows = []
    for axis, prof, pk in (("range", power_db[:, pv], pr), ("doppler", power_db[pr, :], pv)):
        for i, v in enumerate(prof):
            off = i - pk
            if abs(off) <= 20:
                rows.append(dict(axis=axis, offset_bins=int(off), rel_db=float(v) - peak,
                                 offset_bins_frac=offset_bins))
    return pd.DataFrame(rows)


def main() -> None:
    cfg = RadarConfig()
    df = pd.concat([profile(cfg, 0.0).assign(case="on-bin"),
                    profile(cfg, 0.5).assign(case="off-bin (worst case)")], ignore_index=True)
    out = ROOT / "results" / "psf_profile.csv"
    df.to_csv(out, index=False)

    for case, g in df.groupby("case"):
        for axis, gg in g.groupby("axis"):
            s = gg.set_index("offset_bins").rel_db
            for floor in (-30.0, -40.0):
                # first offset at which the response is below `floor` and stays below
                out_bins = [abs(o) for o, v in s.items() if v < floor]
                print(f"  {case:22s} {axis:8s} below {floor:.0f} dB beyond "
                      f"{min(out_bins) if out_bins else float('nan'):.0f} bins")
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
