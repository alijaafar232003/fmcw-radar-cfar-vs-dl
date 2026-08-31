"""Common detector-eligible region, object eligibility, and the pre-registered strata.

Experiment-local. The valid region is IMPORTED from the frozen evaluator rather than
reimplemented, so the endpoint convention cannot silently drift (PREREGISTRATION §5.1).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(_HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
_SRC = os.path.join(ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from dl.infer import cfar_valid_mask  # frozen, imported not copied  # noqa: E402

RD_SHAPE = (256, 64)

# Pre-registered strata, identical to R0's (PREREGISTRATION §9.4 S2/S3).
RANGE_BLOCKS = [(10, 60), (60, 110), (110, 160), (160, 210), (210, 246)]
ZERO_DOPPLER_BAND = (28, 37)          # bins 28..36 inclusive == the R0 common +-4 band


def valid_region(shape=RD_SHAPE) -> np.ndarray:
    """The region a 21x9 CFAR window can test. Frozen convention, imported."""
    return cfar_valid_mask(shape)


def valid_support(dense: np.ndarray, shape=RD_SHAPE) -> np.ndarray:
    """valid_support = dense_mask ∩ valid_region, as a (K, 2) array of cells."""
    if len(dense) == 0:
        return np.zeros((0, 2), dtype=int)
    v = valid_region(shape)
    keep = v[dense[:, 0], dense[:, 1]]
    return dense[keep]


def is_eligible(vs: np.ndarray) -> bool:
    """Primary eligibility: at least one support cell inside the valid region."""
    return len(vs) >= 1


def is_strictly_inside(dense: np.ndarray, shape=RD_SHAPE) -> bool:
    """Secondary S4: the whole mask lies inside the valid region."""
    if len(dense) == 0:
        return False
    return len(valid_support(dense, shape)) == len(dense)


def range_stratum(vs: np.ndarray):
    """Range block of an object, by the MEDIAN range index of its valid support.

    Annotation-only by construction: the detector's prediction never decides an object's stratum
    (PREREGISTRATION §9.4 S2, final ruling 4).
    """
    if len(vs) == 0:
        return None
    med = float(np.median(vs[:, 0]))
    for lo, hi in RANGE_BLOCKS:
        if lo <= med < hi:
            return "{}-{}".format(lo, hi)
    return None


def doppler_stratum(vs: np.ndarray) -> str:
    """near_zero if the valid support meets the R0 +-4 band, else away_zero."""
    if len(vs) == 0:
        return "none"
    lo, hi = ZERO_DOPPLER_BAND
    return "near_zero" if np.any((vs[:, 1] >= lo) & (vs[:, 1] < hi)) else "away_zero"


def dilate_cells(cells: np.ndarray, radius: int, shape=RD_SHAPE) -> np.ndarray:
    """Geometric dilation of a cell set by `radius` bins, clipped to the map. No wraparound.

    Applied identically to both detector arms (PREREGISTRATION §9.2).
    """
    if len(cells) == 0:
        return cells
    if radius <= 0:
        return cells
    offs = [(dr, dd) for dr in range(-radius, radius + 1)
            for dd in range(-radius, radius + 1)]
    out = set()
    for r, d in cells:
        for dr, dd in offs:
            rr, dd2 = r + dr, d + dd
            if 0 <= rr < shape[0] and 0 <= dd2 < shape[1]:
                out.add((rr, dd2))
    return np.array(sorted(out), dtype=int)


def describe_object(obj, shape=RD_SHAPE) -> dict:
    """Annotation-only descriptors. Computes NO detector quantity."""
    dense = obj["dense"]
    vs = valid_support(dense, shape)
    return {
        "sequence": obj["sequence"], "frame": obj["frame"], "object_id": obj["object_id"],
        "class": obj["class"], "label": obj["label"],
        "dense_cells": int(len(dense)),
        "valid_support_cells": int(len(vs)),
        "eligible": bool(is_eligible(vs)),
        "strictly_inside": bool(is_strictly_inside(dense, shape)),
        "range_stratum": range_stratum(vs),
        "doppler_stratum": doppler_stratum(vs),
    }
