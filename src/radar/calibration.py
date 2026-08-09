"""SNR calibration for the RD processing chain.

Why this exists
---------------
Setting SNR on the raw CPI samples is misleading: the range and Doppler FFTs
coherently integrate the target while the noise averages out, so the target peak
in the RD map ends up ~44 dB stronger than the nominal SNR suggests. A sweep of
"-10..+20 dB" nominal SNR is therefore entirely inside the always-detectable
region and produces a flat Pd curve.

Instead we define SNR **at the detector**:

    cell SNR = (target peak-cell power) / (mean noise power per RD cell)

and calibrate the injected noise to hit a requested value. This is the quantity
CFAR detection theory is written in terms of.

The gains below are exact for the windowed 2-D FFT chain in `rd_processing`:
for input complex noise of variance `s2` per CPI sample, the mean noise power per
RD cell is `s2 * g_noise`; for an on-bin tone of linear amplitude `A`, the peak
cell power is `A**2 * g_sig`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal

from radar.config import RadarConfig


@lru_cache(maxsize=None)
def processing_gains(cfg: RadarConfig) -> tuple[float, float]:
    """Return (g_sig, g_noise) for the windowed range+Doppler FFT chain.

    g_sig   : coherent peak-power gain for an on-bin tone  = (sum w_r)^2 (sum w_d)^2
    g_noise : per-cell noise power gain                    = (sum w_r^2)(sum w_d^2)
    """
    w_r = signal.windows.blackman(cfg.N)
    w_d = signal.windows.blackman(cfg.M)

    g_sig = float(w_r.sum() ** 2 * w_d.sum() ** 2)
    g_noise = float((w_r ** 2).sum() * (w_d ** 2).sum())

    return g_sig, g_noise


def snr_processing_gain_db(cfg: RadarConfig) -> float:
    """How many dB the RD processing adds to a target's SNR (~44 dB here)."""
    g_sig, g_noise = processing_gains(cfg)
    return float(10.0 * np.log10(g_sig / g_noise))


def noise_variance_for_reference_snr(reference_cell_snr_db: float, cfg: RadarConfig) -> float:
    """Input noise variance giving a 0 dB-amplitude on-bin target this cell SNR.

    A real target of amplitude `a_db` then lands at roughly
    `reference_cell_snr_db + a_db` (its exact value is measured per frame).
    """
    g_sig, g_noise = processing_gains(cfg)
    s_linear = 10.0 ** (reference_cell_snr_db / 10.0)
    return float(g_sig / (s_linear * g_noise))


def measure_peak_cell_power(
    rd_mag_clean: np.ndarray,
    range_bin: int,
    velocity_bin: int,
    search: int = 3,
) -> float:
    """Peak |RD|^2 within +/- `search` bins of a labelled cell, on the CLEAN map.

    Searching a small window (rather than reading the exact labelled cell) captures
    windowing/scalloping loss, so the reported SNR matches what a detector sees.
    """
    n_r, n_v = rd_mag_clean.shape

    r0, r1 = max(0, range_bin - search), min(n_r, range_bin + search + 1)
    v0, v1 = max(0, velocity_bin - search), min(n_v, velocity_bin + search + 1)

    return float(rd_mag_clean[r0:r1, v0:v1].max() ** 2)


@lru_cache(maxsize=None)
def unit_clutter_cell_power(
    cfg: RadarConfig,
    width_bins: int,
    distribution: str,
    shape: float,
    n_frames: int = 8,
    seed: int = 0,
) -> float:
    """Mean RD cell power inside the clutter ridge for UNIT-scale clutter.

    Measured through the real windowed FFT chain rather than derived analytically. The
    band-limiting, the window gains and the amplitude distribution all affect the result, and
    measuring is both simpler and less error-prone than composing three closed forms -- the same
    reasoning used for the cell-SNR calibration.

    Cached: `RadarConfig` is frozen and the rest are primitives, so a dataset run pays this once
    rather than once per frame.
    """
    from radar.rd_processing import compute_rd_map
    from radar.simulator import add_near_zero_doppler_clutter

    half = width_bins
    total = 0.0

    for k in range(n_frames):
        cpi = np.zeros((cfg.M, cfg.N), dtype=complex)
        cpi = add_near_zero_doppler_clutter(
            cpi,
            scale=1.0,
            width_bins=width_bins,
            distribution=distribution,
            shape=shape,
            seed=seed + k,
        )
        _, mag, _, _, _ = compute_rd_map(cpi, cfg)

        centre = mag.shape[1] // 2
        lo, hi = max(0, centre - half), min(mag.shape[1], centre + half + 1)
        total += float(np.mean(mag[:, lo:hi] ** 2))

    return total / n_frames


def clutter_scale_for_cnr(
    cnr_db: float,
    noise_var: float,
    cfg: RadarConfig,
    width_bins: int = 3,
    distribution: str = "weibull",
    shape: float = 1.5,
) -> float:
    """Amplitude scale placing the mean clutter cell `cnr_db` above the noise floor.

    Cell power scales with the square of the amplitude, so once the unit-scale power is known
    the scale follows in closed form.
    """
    _, g_noise = processing_gains(cfg)
    noise_cell_power = noise_var * g_noise

    target = noise_cell_power * (10.0 ** (cnr_db / 10.0))
    unit = unit_clutter_cell_power(cfg, width_bins, distribution, shape)

    if unit <= 0.0:
        return 0.0

    return float(np.sqrt(target / unit))


def cell_snr_db(peak_cell_power: float, noise_var: float, cfg: RadarConfig) -> float:
    """Convert a measured peak-cell power + injected noise variance into cell SNR (dB)."""
    _, g_noise = processing_gains(cfg)
    noise_cell_power = noise_var * g_noise

    if peak_cell_power <= 0.0 or noise_cell_power <= 0.0:
        return float("nan")

    return float(10.0 * np.log10(peak_cell_power / noise_cell_power))
