"""Integrated-CFAR statistics — an ISOLATED attribution experiment.

    How much of TinyCNN's advantage over CFAR is explained by the CNN integrating neighbouring
    mainlobe energy that single-cell CFAR discards?

Standard CFAR tests ONE cell against a noise estimate. A CNN with a 21x9 receptive field can
combine the target's whole mainlobe. This module builds the intermediate detector: the same CFAR
noise estimate, but a CUT statistic integrated over the measured mainlobe.

**Nothing in `src/` is modified.** The noise estimators are IMPORTED from `src/cfar/cfar.py` and
called unchanged, so the denominator here is computed by exactly the shipped code. Only the
numerator differs. `verify_denominator_matches_shipped()` asserts that.

Two statistics, deliberately:

  (a) UNIFORM      — plain sum over the integration region.
  (b) PSF-WEIGHTED — weights proportional to the measured normalised PSF, i.e. the matched filter
                     for a known signal shape in white noise.

⚠️ CALIBRATION IS NOT OPTIONAL FOR THESE. Summing K correlated cells changes the null distribution:
it is no longer exponential, so the closed form `alpha = N(Pfa^(-1/N) - 1)` does not apply and the
project's cached alphas (`configs/cfar_alpha_table.json`) are meaningless for this statistic. Use
`calibrate_alpha()`, which mirrors `scripts/cfar_calibrate_table.py`: Monte Carlo on pure noise,
verified on a held-out seed.

Note that alpha does NOT affect any result scored at matched MEASURED Pfa — the threshold is swept
there — exactly as established for the standard variants on 21 Aug. It fixes the DESIGN operating
point, and it is reported because an uncalibrated design point would be a silent error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import correlate

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import (CfarConfig, cfar_ratio_map,          # noqa: E402
                       _ring_mean, _half_ring_means, _os_ring_statistic)
from cfar.calibrate import make_noise_power_frames          # noqa: E402


# ----------------------------------------------------------------------
# The noise estimate — the shipped estimators, called unchanged
# ----------------------------------------------------------------------

def noise_estimate(power: np.ndarray, cfg: CfarConfig) -> np.ndarray:
    """The local noise estimate `cfar_ratio_map` would divide by, for any supported kind.

    This dispatch mirrors `cfar_ratio_map`'s, but every estimator it calls is the imported,
    unmodified function from `src/cfar/cfar.py`. `verify_denominator_matches_shipped()` checks the
    result reproduces the shipped ratio map exactly.
    """
    hw_r, hw_d = cfg.half_width_r, cfg.half_width_d
    g_r, g_d = cfg.guard_r, cfg.guard_d

    if cfg.kind == "CA":
        return _ring_mean(power, hw_r, hw_d, g_r, g_d)
    if cfg.kind == "GO":
        lead, lag = _half_ring_means(power, hw_r, hw_d, g_r, g_d, axis=cfg.go_axis)
        return np.maximum(lead, lag)
    if cfg.kind == "SO":
        lead, lag = _half_ring_means(power, hw_r, hw_d, g_r, g_d, axis=cfg.go_axis)
        return np.minimum(lead, lag)
    if cfg.kind == "OS":
        rank = int(round(cfg.os_rank_frac * cfg.num_ref_cells))
        return _os_ring_statistic(power, hw_r, hw_d, g_r, g_d, rank)
    raise ValueError(f"Unknown CFAR kind: {cfg.kind!r}")


def verify_denominator_matches_shipped(power: np.ndarray, cfg: CfarConfig) -> float:
    """Max |shipped ratio - power/our noise| over valid cells. Must be ~0."""
    ratio, valid, _ = cfar_ratio_map(power, cfg)
    ours = power / (noise_estimate(power, cfg) + 1e-30)
    return float(np.abs(np.where(valid, ours, 0.0) - ratio).max())


# ----------------------------------------------------------------------
# Integration kernels
# ----------------------------------------------------------------------

def uniform_kernel(half_r: int, half_d: int) -> np.ndarray:
    """Plain sum over the region, normalised to sum 1."""
    k = np.ones((2 * half_r + 1, 2 * half_d + 1), dtype=float)
    return k / k.sum()


def psf_kernel(psf_patch: np.ndarray, half_r: int, half_d: int,
               symmetrise: bool = True) -> np.ndarray:
    """Matched-filter weights from a measured normalised PSF power patch, normalised to sum 1.

    `psf_patch` is the averaged target-centred POWER patch, its centre at the middle. Only the
    central (2*half_r+1) x (2*half_d+1) is used.

    `symmetrise` averages the patch with its two flips and its 180-degree rotation. Targets are
    placed off-grid by design, so an average over finitely many of them carries a sampling
    asymmetry that the true response does not have; the expected shape is symmetric.
    """
    cr, cd = psf_patch.shape[0] // 2, psf_patch.shape[1] // 2
    k = psf_patch[cr - half_r:cr + half_r + 1, cd - half_d:cd + half_d + 1].astype(float).copy()

    if symmetrise:
        k = (k + k[::-1, :] + k[:, ::-1] + k[::-1, ::-1]) / 4.0

    k = np.clip(k, 0.0, None)
    return k / k.sum()


def kernel_asymmetry(psf_patch: np.ndarray, half_r: int, half_d: int) -> float:
    """Relative asymmetry of the raw patch, before symmetrising — reported, not hidden."""
    a = psf_kernel(psf_patch, half_r, half_d, symmetrise=False)
    b = psf_kernel(psf_patch, half_r, half_d, symmetrise=True)
    return float(np.abs(a - b).sum() / b.sum())


# ----------------------------------------------------------------------
# The integrated ratio map
# ----------------------------------------------------------------------

def integrated_ratio_map(power: np.ndarray, cfg: CfarConfig, kernel: np.ndarray):
    """(ratio, valid) with an INTEGRATED CUT and the shipped CFAR noise estimate.

    ratio[r, v] = sum_ij kernel[i, j] * power[r+i, v+j]  /  noise_estimate[r, v]

    The kernel must fit inside the guard region, or the numerator would draw on cells the
    denominator also uses and the two would be correlated. `check_kernel_fits_guard` enforces it.
    """
    check_kernel_fits_guard(kernel, cfg)

    cut = correlate(power, kernel, mode="nearest")
    noise = noise_estimate(power, cfg)
    ratio = cut / (noise + 1e-30)

    hw_r, hw_d = cfg.half_width_r, cfg.half_width_d
    valid = np.zeros(power.shape, dtype=bool)
    valid[hw_r:-hw_r, hw_d:-hw_d] = True

    return np.where(valid, ratio, 0.0), valid


def check_kernel_fits_guard(kernel: np.ndarray, cfg: CfarConfig) -> None:
    half_r, half_d = kernel.shape[0] // 2, kernel.shape[1] // 2
    if half_r > cfg.guard_r or half_d > cfg.guard_d:
        raise ValueError(
            f"integration region +/-({half_r},{half_d}) exceeds the guard "
            f"+/-({cfg.guard_r},{cfg.guard_d}); the numerator would overlap the reference ring")


# ----------------------------------------------------------------------
# Calibration — mandatory for these statistics
# ----------------------------------------------------------------------

def calibrate_alpha(radar_cfg, cfg: CfarConfig, kernel: np.ndarray, target_pfa: float = 1e-4,
                    n_frames: int = 300, seed: int = 0) -> float:
    """Monte-Carlo alpha giving `target_pfa` per valid cell on pure noise.

    Same protocol as `scripts/cfar_calibrate_table.py`: pool the statistic over pure-noise frames
    and take the (1 - target_pfa) quantile. No closed form is used, because none applies to a sum
    of correlated cells.
    """
    pool = []
    for power in make_noise_power_frames(radar_cfg, n_frames, seed=seed):
        ratio, valid = integrated_ratio_map(power, cfg, kernel)
        pool.append(ratio[valid])
    return float(np.quantile(np.concatenate(pool), 1.0 - target_pfa))


def achieved_pfa(radar_cfg, cfg: CfarConfig, kernel: np.ndarray, alpha: float,
                 n_frames: int = 200, seed: int = 9000) -> float:
    """Per-cell false-alarm rate at `alpha` on a HELD-OUT pure-noise seed."""
    fa = n = 0
    for power in make_noise_power_frames(radar_cfg, n_frames, seed=seed):
        ratio, valid = integrated_ratio_map(power, cfg, kernel)
        fa += int((ratio[valid] > alpha).sum())
        n += int(valid.sum())
    return fa / n
