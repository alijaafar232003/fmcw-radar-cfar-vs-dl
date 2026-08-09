"""Step 2b verification — does the new clutter do what the config asks?

Three claims to check, none of which should be taken on trust:

  1. CNR calibration is accurate: asking for 15 dB gives 15 dB in the RD map.
  2. The distribution is actually heavy-tailed: a Weibull fit to the clutter-cell amplitudes
     recovers the requested shape, and shape < 2 really is heavier than Rayleigh.
  3. The ridge is where it should be: energy concentrated near zero Doppler, not spread.

Claim 2 is the one that matters most. The data-quality audit measured the OLD clutter at Weibull
shape 2.03 -- i.e. Gaussian-like, the thing this rewrite exists to fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.calibration import (                                     # noqa: E402
    clutter_scale_for_cnr,
    noise_variance_for_reference_snr,
    processing_gains,
)
from radar.config import RadarConfig                                # noqa: E402
from radar.rd_processing import compute_rd_map                      # noqa: E402
from radar.simulator import ClutterConfig, add_awgn_var, simulate_cpi  # noqa: E402

CFG = RadarConfig()
N_FRAMES = 12


def ridge_cells(mag: np.ndarray, width_bins: int) -> np.ndarray:
    centre = mag.shape[1] // 2
    lo, hi = max(0, centre - width_bins), min(mag.shape[1], centre + width_bins + 1)
    return mag[:, lo:hi]


def run(cnr_db: float, distribution: str, shape: float, width_bins: int = 3) -> dict:
    noise_var = noise_variance_for_reference_snr(15.0, CFG)
    _, g_noise = processing_gains(CFG)
    noise_cell_power = noise_var * g_noise

    clutter = ClutterConfig(
        enabled=True, cnr_db=cnr_db, width_bins=width_bins,
        distribution=distribution, shape=shape,
    )

    ridge_pow, amps, centre_amps, off_pow = [], [], [], []

    for k in range(N_FRAMES):
        cpi = simulate_cpi([], CFG, noise_var=noise_var, seed=1000 + k, clutter=clutter)
        _, mag, _, _, _ = compute_rd_map(cpi, CFG)

        r = ridge_cells(mag, width_bins)
        ridge_pow.append(np.mean(r ** 2))
        amps.append(r.ravel())

        # A single Doppler column isolates the range-domain amplitude model. Pooling the whole
        # ridge mixes columns with different mean levels (the ridge is tapered in Doppler), and
        # a mixture of differing means fits as a spuriously heavy tail -- which is exactly what
        # made "rayleigh" come back at c = 1.65 on the first attempt.
        centre_amps.append(mag[:, mag.shape[1] // 2])

        # A Doppler band far from the ridge: should be thermal noise only.
        off_pow.append(np.mean(mag[:, :max(1, mag.shape[1] // 2 - 4 * width_bins)] ** 2))

    amps = np.concatenate(amps)
    centre_amps = np.concatenate(centre_amps)

    measured_cnr = 10 * np.log10(np.mean(ridge_pow) / noise_cell_power)
    off_cnr = 10 * np.log10(np.mean(off_pow) / noise_cell_power)

    # Weibull shape of the clutter-cell amplitudes (floc=0 -> two-parameter fit).
    fit_shape, _, _ = stats.weibull_min.fit(amps, floc=0)
    fit_centre, _, _ = stats.weibull_min.fit(centre_amps, floc=0)

    return {
        "requested": cnr_db,
        "measured": measured_cnr,
        "off_ridge": off_cnr,
        "fit_shape": fit_shape,
        "fit_centre": fit_centre,
        "ratio_99_50": float(np.percentile(amps, 99) / np.percentile(amps, 50)),
    }


def main() -> None:
    print("1. CNR CALIBRATION  (weibull shape 1.5)")
    print(f"{'requested':>10}{'measured':>10}{'error dB':>10}{'off-ridge':>11}")
    for cnr in (5.0, 10.0, 15.0, 20.0, 25.0):
        r = run(cnr, "weibull", 1.5)
        print(f"{r['requested']:>10.1f}{r['measured']:>10.2f}"
              f"{r['measured'] - r['requested']:>+10.2f}{r['off_ridge']:>11.2f}")

    print("\n2. DISTRIBUTION SHAPE  (CNR 20 dB)")
    print(f"{'requested':<22}{'c (1 Doppler col)':>19}{'c (whole ridge)':>17}{'p99/p50':>10}")
    for dist, shp, label in (
        ("rayleigh", 2.0, "rayleigh"),
        ("weibull", 2.0, "weibull c=2.0"),
        ("weibull", 1.5, "weibull c=1.5"),
        ("weibull", 1.2, "weibull c=1.2"),
        ("k", 4.0, "K nu=4.0"),
        ("k", 1.5, "K nu=1.5"),
    ):
        r = run(20.0, dist, shp)
        print(f"{label:<22}{r['fit_centre']:>19.2f}{r['fit_shape']:>17.2f}"
              f"{r['ratio_99_50']:>10.2f}")

    print("\nFitted c near 2.0 = Rayleigh-like (the OLD behaviour, measured at 2.03).")
    print("Lower c = heavier tail. p99/p50 rising confirms more extreme spikes.")
    print("Off-ridge should sit near 0 dB: clutter confined to the ridge, not leaking.")


if __name__ == "__main__":
    main()
