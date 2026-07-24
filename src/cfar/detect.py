"""Turn a CFAR ratio map into a detection point list.

A CFAR pass flags many cells around each target (its mainlobe spreads over a few
bins). Reporting every flagged cell would massively over-count. So we:

1. threshold the ratio map (ratio > alpha),
2. keep only cells that are BOTH above threshold AND a local maximum of the ratio
   (non-maximum suppression over a small window ~ the mainlobe),

collapsing each blob to a single detection located at its peak. Each detection
carries its ratio value as a **score**, so the evaluator can sweep a threshold to
trace the ROC.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def extract_detections(
    ratio: np.ndarray,
    valid: np.ndarray,
    alpha: float,
    nms_size: int = 5,
) -> np.ndarray:
    """Ratio map -> (K, 3) array of (range_bin, velocity_bin, score).

    ratio    : CUT / noise estimate (from cfar_ratio_map)
    valid    : validity mask (border cells excluded)
    alpha    : detection threshold on the ratio (design-Pfa operating point)
    nms_size : local-maximum window per side, ~ the mainlobe (so one target -> one point)
    """
    peak = ratio == maximum_filter(ratio, size=2 * nms_size + 1, mode="nearest")
    detected = valid & peak & (ratio > alpha)

    r, v = np.nonzero(detected)
    score = ratio[r, v]

    return np.column_stack([r, v, score]).astype(float)


def all_peaks(ratio: np.ndarray, valid: np.ndarray, nms_size: int = 5) -> np.ndarray:
    """Every local-maximum cell with its ratio score, ignoring the threshold.

    Used to build the ROC: keep all candidate peaks and let the evaluator sweep the
    score threshold, rather than committing to one alpha here.
    """
    peak = ratio == maximum_filter(ratio, size=2 * nms_size + 1, mode="nearest")
    detected = valid & peak & (ratio > 0)

    r, v = np.nonzero(detected)
    score = ratio[r, v]

    return np.column_stack([r, v, score]).astype(float)
