"""R0 background-selection protocol — the single definition used everywhere in this audit.

Why this module exists
----------------------
Two preliminary passes produced non-authoritative numbers because the background population
was chosen inconsistently:

  * one trimmed the top decile, which biases the coefficient of variation downward;
  * one left the zero-Doppler ridge in, which inflates it enormously.

Neither is repeated here. There is exactly one background definition, it is applied
identically to CARRADA and to the simulator, and **no percentile trimming is ever applied.**

The eligible background population
----------------------------------
A cell is background if it survives all three exclusions:

  1. **Object support, dilated.** Every annotated/labelled object cell, dilated by
     ``OBJECT_DILATION`` bins in both axes, is removed. The dilation covers the target
     mainlobe (measured at ~2 bins, -30 dB half-width) so that target skirts do not leak
     into the "background".

  2. **Zero-Doppler ridge.** Doppler bins within ``ridge_half`` of the zero-Doppler bin are
     removed for the *ordinary* background statistics. The ridge is real structure, not
     speckle, and mixing it into a speckle estimate is what corrupted the earlier pass.
     Ridge strength is reported separately by ``ridge_profile``.

  3. **Detector border.** The outer ``BORDER_R`` range bins and ``BORDER_D`` Doppler bins are
     removed, matching the region a 21x9 CFAR window can actually test
     (guard 2 + train 8 = 10 in range, guard 2 + train 2 = 4 in Doppler). This keeps the
     audited population equal to the population a detector would see.

Known, documented mismatch
--------------------------
"Object support" is not the same object in the two domains. In the simulator it is the 3x3
label mask around a point target; in CARRADA it is a human-drawn segmentation blob whose
median car footprint is 14x9 bins. The dilation is identical, but the thing being dilated is
not. This asymmetry cannot be removed and is stated wherever these statistics are reported.
"""
from __future__ import annotations

import numpy as np

# Detector valid region: 21x9 CFAR window, guard 2/2 and train 8/2 (src/cfar/cfar.py).
BORDER_R = 10
BORDER_D = 4

# Dilation applied to object support before exclusion, in bins, both axes.
OBJECT_DILATION = 3

# Zero-Doppler bin. Simulator: velocity_axis has its minimum |v| at index 32.
# CARRADA: 64 Doppler bins over +-13.44 m/s, same convention. Asserted, not assumed,
# by r0_audit.py against the measured ridge peak.
ZERO_DOPPLER_BIN = 32

RD_SHAPE = (256, 64)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation by `radius` bins, by shifted ORs. Does NOT wrap at the array edges."""
    if radius <= 0:
        return mask.copy()
    rows, cols = mask.shape
    out = np.zeros_like(mask)
    for dr in range(-radius, radius + 1):
        dst_r0, dst_r1 = max(0, dr), min(rows, rows + dr)
        src_r0, src_r1 = max(0, -dr), min(rows, rows - dr)
        for dd in range(-radius, radius + 1):
            dst_d0, dst_d1 = max(0, dd), min(cols, cols + dd)
            src_d0, src_d1 = max(0, -dd), min(cols, cols - dd)
            out[dst_r0:dst_r1, dst_d0:dst_d1] |= mask[src_r0:src_r1, src_d0:src_d1]
    return out


def object_mask_from_cells(cells, shape=RD_SHAPE) -> np.ndarray:
    """Boolean object-support map from an iterable of (range_bin, doppler_bin)."""
    m = np.zeros(shape, dtype=bool)
    for r, d in cells:
        if 0 <= r < shape[0] and 0 <= d < shape[1]:
            m[r, d] = True
    return m


def border_mask(shape=RD_SHAPE) -> np.ndarray:
    """True where a 21x9 CFAR window fits entirely inside the map."""
    m = np.zeros(shape, dtype=bool)
    m[BORDER_R:shape[0] - BORDER_R, BORDER_D:shape[1] - BORDER_D] = True
    return m


def ridge_mask(ridge_half: int, shape=RD_SHAPE) -> np.ndarray:
    """True on Doppler bins within `ridge_half` of zero Doppler (i.e. the cells to EXCLUDE)."""
    m = np.zeros(shape, dtype=bool)
    lo = max(0, ZERO_DOPPLER_BIN - ridge_half)
    hi = min(shape[1], ZERO_DOPPLER_BIN + ridge_half + 1)
    m[:, lo:hi] = True
    return m


def background_mask(object_cells, ridge_half: int, *, exclude_ridge: bool = True,
                    shape=RD_SHAPE) -> np.ndarray:
    """The eligible background population for one frame.

    Parameters
    ----------
    object_cells : iterable of (range_bin, doppler_bin) for every annotated/labelled object cell
    ridge_half   : half-width in Doppler bins of the excluded zero-Doppler band
    exclude_ridge: set False only for the explicitly-labelled ridge-included side analysis

    Returns
    -------
    bool array, True where the cell is eligible background.
    """
    keep = border_mask(shape)
    obj = object_mask_from_cells(object_cells, shape)
    if obj.any():
        keep &= ~dilate(obj, OBJECT_DILATION)
    if exclude_ridge:
        keep &= ~ridge_mask(ridge_half, shape)
    return keep


def frame_stats(power: np.ndarray, keep: np.ndarray) -> dict:
    """Background statistics for one frame. NO trimming of any kind.

    `power` is linear power; `keep` is the eligible-background mask.
    Lag-1 correlations are computed on the rectangular eligible block so that neighbouring
    pairs are genuinely adjacent; pairs where either member is excluded are dropped.
    """
    p = power[keep]
    n = p.size
    if n < 100:
        return {}
    mean, std = float(p.mean()), float(p.std())
    db = 10.0 * np.log10(np.maximum(power, 1e-300))
    out = {
        "n_cells": n,
        "cv": std / mean,
        "gamma_k_moment": (mean / std) ** 2,
        "db_std": float(db[keep].std()),
        "mean_power": mean,
        "median_power": float(np.median(p)),
    }
    for axis, name in ((0, "lag1_range"), (1, "lag1_doppler")):
        a = np.take(power, range(0, power.shape[axis] - 1), axis=axis)
        b = np.take(power, range(1, power.shape[axis]), axis=axis)
        ka = np.take(keep, range(0, keep.shape[axis] - 1), axis=axis)
        kb = np.take(keep, range(1, keep.shape[axis]), axis=axis)
        both = ka & kb
        if both.sum() > 100:
            x, y = a[both], b[both]
            out[name] = float(np.corrcoef(x, y)[0, 1])
        else:
            out[name] = np.nan
    return out


def gamma_k_logmoment(p: np.ndarray) -> float:
    """Gamma shape by the log-moment estimator: solve ln k - digamma(k) = ln(mean) - mean(ln x).

    Less sensitive than the method of moments to the upper tail, and reported beside it so the
    reader can see whether the two agree rather than trusting one.
    """
    from scipy.special import digamma
    p = p[p > 0]
    s = float(np.log(p.mean()) - np.log(p).mean())
    if s <= 0:
        return np.nan
    lo, hi = 1e-4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (np.log(mid) - digamma(mid)) > s:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ridge_profile(power: np.ndarray, object_cells, shape=RD_SHAPE) -> np.ndarray:
    """Mean power per Doppler bin over object-free, border-valid range cells.

    The ridge itself is deliberately NOT excluded here -- this is the measurement that
    characterises it.
    """
    keep = border_mask(shape)
    obj = object_mask_from_cells(object_cells, shape)
    if obj.any():
        keep &= ~dilate(obj, OBJECT_DILATION)
    prof = np.full(shape[1], np.nan)
    for d in range(shape[1]):
        col = keep[:, d]
        if col.sum() > 10:
            prof[d] = power[col, d].mean()
    return prof
