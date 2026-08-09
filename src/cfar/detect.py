"""Turn a CFAR ratio map into a detection point list.

A CFAR pass flags many cells around each target (its mainlobe spreads over a few
bins). Reporting every flagged cell would massively over-count. So we:

1. threshold the ratio map (ratio > alpha),
2. keep only cells that are BOTH above threshold AND a local maximum of the ratio
   (non-maximum suppression over a small window ~ the mainlobe),

collapsing each blob to a single detection located at its peak. Each detection
carries its ratio value as a **score**, so the evaluator can sweep a threshold to
trace the ROC.

**nms_size is a real design parameter, and 5 was kept after measuring alternatives.**
The window is (2*nms_size+1) per side, so 5 gives 11x11 = 2.15 m in range and 4.6 m/s in
Doppler — far wider than the ~2-bin mainlobe. That does suppress resolvable neighbours: on
clean, noiseless maps only 38% of masking pairs keep both targets, and 5.9% for pairs under
1 m apart. It therefore caps the masking stratum at roughly Pd 0.69 (see PROGRESS "Step 6d").

Shrinking it is worse, though, and measurably so. NMS is not only merging targets, it is
also filtering noise: at nms=2 there are ~4x more candidate peaks per noise frame (427 vs
113), which at matched measured Pfa forces the threshold up and costs Pd everywhere —
including `isolated`, where nothing can be merged. Swept at the 21x9 geometry, Pd at matched
Pfa on isolated ran 0.291 (nms=2), 0.293 (3), 0.372 (4), 0.521 (5), 0.638 (7).

nms_size=1 looks good on that sweep but is disqualified: it returns 1.20 detections per
target, so a target gets several attempts and the evaluator forgives all but one inside its
guard region. That is the free-attempts failure the plateau fix removed. nms >= 2 gives at
most one detection per target (0.96 down to 0.80).

7 scores marginally above 5 but spans 2.9 m, which would merge two adjacent vehicles, so 5
is kept as the established value. The masking cap is documented as a limitation rather than
tuned away, and nms_size belongs in the attribution sweep because it interacts with the
evaluator's guard-region forgiveness.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label, maximum_filter


def _collapse_plateaus(peak_mask: np.ndarray) -> np.ndarray:
    """Keep exactly one cell per connected peak region.

    `score == maximum_filter(score)` marks EVERY cell of a plateau as a local maximum. Continuous
    statistics (the CFAR ratio) rarely tie, but a network's sigmoid output has flat regions, so it
    would emit several detections for a single target blob — inflating Pd and, because duplicates
    fall inside a truth's guard region, incurring no false-alarm penalty. Collapsing each connected
    region to one representative removes that asymmetry and applies identically to every detector.
    """
    lbl, n = label(peak_mask)

    if n == 0:
        return peak_mask

    flat = lbl.ravel()
    order = np.argsort(flat, kind="stable")
    first = order[np.searchsorted(flat[order], np.arange(1, n + 1))]

    keep = np.zeros(flat.size, dtype=bool)
    keep[first] = True
    return keep.reshape(peak_mask.shape)


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
    detected = _collapse_plateaus(valid & peak & (ratio > alpha))

    r, v = np.nonzero(detected)
    score = ratio[r, v]

    return np.column_stack([r, v, score]).astype(float)


def all_peaks(ratio: np.ndarray, valid: np.ndarray, nms_size: int = 5) -> np.ndarray:
    """Every local-maximum cell with its ratio score, ignoring the threshold.

    Used to build the ROC: keep all candidate peaks and let the evaluator sweep the
    score threshold, rather than committing to one alpha here.
    """
    peak = ratio == maximum_filter(ratio, size=2 * nms_size + 1, mode="nearest")
    detected = _collapse_plateaus(valid & peak & (ratio > 0))

    r, v = np.nonzero(detected)
    score = ratio[r, v]

    return np.column_stack([r, v, score]).astype(float)
