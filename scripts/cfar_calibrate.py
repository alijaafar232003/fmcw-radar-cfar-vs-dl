"""Step 5 — calibrate the CFAR thresholds at the new sensor, and see what clutter does to them.

Four parts:

  1. Monte-Carlo alpha for CA / OS / GO. Only CA has a closed form; OS uses an order statistic
     and GO the max of two half-means, so reusing CA's alpha leaves them at the wrong operating
     point entirely.
  2. Cross-check empirical CA against the closed form. They should agree to a few percent; the
     residual is the window's inter-cell correlation, which the formula does not model, so the
     empirical value is the more accurate one for this radar.
  3. Verify the achieved false-alarm rate on INDEPENDENT noise (different seed from calibration).
  4. Measure the achieved rate under heavy-tailed clutter. CA's alpha assumes exponentially
     distributed cell power; heavy tails break that assumption, so this quantifies how far the
     rate drifts from design as clutter gets spikier. This is the first half of the clutter-sweep
     analysis the config was built to support.

The alpha values here are for guard 2 / train 8. Step 6 may change the geometry, in which case
they are recomputed -- what is being established now is that the PROCEDURE works at this sensor,
and the shape of the clutter-drift relationship.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.calibrate import calibrate_all, make_noise_power_frames    # noqa: E402
from cfar.cfar import CfarConfig, cfar_ratio_map                     # noqa: E402
from radar.calibration import (                                      # noqa: E402
    noise_variance_for_reference_snr,
    processing_gains,
)
from radar.config import RadarConfig                                 # noqa: E402
from radar.rd_processing import compute_rd_map                       # noqa: E402
from radar.simulator import ClutterConfig, simulate_cpi              # noqa: E402

CFG = RadarConfig()
GUARD, TRAIN = 2, 8
KINDS = ("CA", "GO", "OS")

N_CAL = 150      # frames for calibration
N_VER = 150      # frames for verification (disjoint seed)
N_CLU = 60       # frames per clutter setting


def measured_pfa(power_frames, cfg_cfar: CfarConfig, alpha: float) -> float:
    """Fraction of validly-tested cells whose ratio exceeds alpha."""
    hits = tested = 0
    for p in power_frames:
        ratio, valid, _ = cfar_ratio_map(p, cfg_cfar)
        hits += int(np.count_nonzero(ratio[valid] > alpha))
        tested += int(valid.sum())
    return hits / max(tested, 1)


def clutter_power_frames(n: int, cnr_db: float, distribution: str, shape: float, seed: int):
    """Target-free frames with clutter + thermal noise, through the real chain."""
    noise_var = noise_variance_for_reference_snr(15.0, CFG)
    cl = ClutterConfig(enabled=True, cnr_db=cnr_db, width_bins=3,
                       distribution=distribution, shape=shape)
    for k in range(n):
        cpi = simulate_cpi([], CFG, noise_var=noise_var, seed=seed + k, clutter=cl)
        _, mag, _, _, _ = compute_rd_map(cpi, CFG)
        yield mag ** 2


def main() -> None:
    print(f"geometry: guard {GUARD}, train {TRAIN} -> "
          f"{CfarConfig(guard=GUARD, train=TRAIN).num_ref_cells} reference cells\n")

    # ---- 1 & 2 -----------------------------------------------------------------------
    print("1-2. CALIBRATED ALPHA, AND CA AGAINST ITS CLOSED FORM")
    print(f"   {'Pfa':>8}{'CA':>9}{'GO':>9}{'OS':>9}{'CA analytic':>14}{'diff':>8}")
    alphas: dict[float, dict[str, float]] = {}
    for pfa in (1e-3, 1e-4):
        a = calibrate_all(CFG, pfa, guard=GUARD, train=TRAIN, n_frames=N_CAL, seed=0)
        alphas[pfa] = a
        analytic = CfarConfig(kind="CA", guard=GUARD, train=TRAIN, pfa=pfa).alpha()
        diff = 100 * (a["CA"] - analytic) / analytic
        print(f"   {pfa:>8.0e}{a['CA']:>9.3f}{a['GO']:>9.3f}{a['OS']:>9.3f}"
              f"{analytic:>14.3f}{diff:>7.1f}%")
    print("   (diff is the window's inter-cell correlation, absent from the formula)")

    # ---- 3 ---------------------------------------------------------------------------
    print("\n3. ACHIEVED Pfa ON INDEPENDENT NOISE (calibration seed 0, verification seed 9000)")
    print(f"   {'design':>8}{'CA':>11}{'GO':>11}{'OS':>11}")
    for pfa in (1e-3, 1e-4):
        row = {}
        for kind in KINDS:
            c = CfarConfig(kind=kind, guard=GUARD, train=TRAIN, pfa=pfa)
            frames = make_noise_power_frames(CFG, N_VER, seed=9000)
            row[kind] = measured_pfa(frames, c, alphas[pfa][kind])
        print(f"   {pfa:>8.0e}" + "".join(f"{row[k]:>11.2e}" for k in KINDS))

    # ---- 4 ---------------------------------------------------------------------------
    print("\n4. ACHIEVED Pfa UNDER CLUTTER, using the noise-calibrated alpha (design 1e-4)")
    print("   CA's alpha assumes exponential cell power; heavy tails break that assumption.")
    print(f"\n   {'clutter':<22}{'CA':>11}{'GO':>11}{'OS':>11}{'CA x design':>13}")
    settings = [
        ("none (thermal only)", None, None, None),
        ("rayleigh, 15 dB", 15.0, "rayleigh", 2.0),
        ("weibull c=2.0, 15 dB", 15.0, "weibull", 2.0),
        ("weibull c=1.5, 15 dB", 15.0, "weibull", 1.5),
        ("weibull c=1.1, 15 dB", 15.0, "weibull", 1.1),
        ("weibull c=1.5, 25 dB", 25.0, "weibull", 1.5),
        ("k nu=1.5, 15 dB", 15.0, "k", 1.5),
    ]
    for label, cnr, dist, shp in settings:
        row = {}
        for kind in KINDS:
            c = CfarConfig(kind=kind, guard=GUARD, train=TRAIN, pfa=1e-4)
            if cnr is None:
                frames = make_noise_power_frames(CFG, N_CLU, seed=12000)
            else:
                frames = clutter_power_frames(N_CLU, cnr, dist, shp, seed=12000)
            row[kind] = measured_pfa(frames, c, alphas[1e-4][kind])
        mult = row["CA"] / 1e-4
        print(f"   {label:<22}" + "".join(f"{row[k]:>11.2e}" for k in KINDS)
              + f"{mult:>12.1f}x")


if __name__ == "__main__":
    main()
