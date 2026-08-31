"""One-to-one peak-to-object assignment. ONE implementation, shared by every detector arm.

Pre-registered objective (PREREGISTRATION §5.3), applied in strict order:

  1. maximise the number of matched objects   (maximum-cardinality bipartite matching)
  2. among maximum-cardinality solutions, maximise total detector score
  3. deterministic tie-break by (range_bin, doppler_bin, object_id)

Implemented exactly as stated: a min-cost max-flow / Hungarian assignment on a cost that is
lexicographic by construction — cardinality dominates because every real edge is given a large
negative base cost, and score only breaks ties within that.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# Base reward for using any edge. Must dominate every possible score contribution so that
# cardinality is strictly prioritised over score (objective 1 before objective 2).
_BASE = 1.0e9


def match(peaks: np.ndarray, supports: list[np.ndarray], object_ids: list | None = None):
    """Assign at most one peak to each object, and each peak to at most one object.

    Parameters
    ----------
    peaks      : (K, 3) array of (range_bin, doppler_bin, score); may be empty
    supports   : list of (M_i, 2) arrays — the valid_support cells of each object
    object_ids : optional ids, used only for the deterministic tie-break

    Returns
    -------
    dict with
        matched      : {object_index: peak_index}
        unmatched_peaks   : sorted list of peak indices assigned to no object
        unmatched_objects : sorted list of object indices with no peak
    """
    n_obj = len(supports)
    peaks = np.asarray(peaks, dtype=float).reshape(-1, 3)
    n_peak = len(peaks)
    if object_ids is None:
        object_ids = list(range(n_obj))

    if n_obj == 0 or n_peak == 0:
        return {"matched": {},
                "unmatched_peaks": list(range(n_peak)),
                "unmatched_objects": list(range(n_obj))}

    # membership: peak p is an eligible partner of object o iff it lies in o's valid support
    support_sets = [set(map(tuple, np.asarray(s, dtype=int).reshape(-1, 2).tolist()))
                    for s in supports]
    pk = [(int(round(p[0])), int(round(p[1]))) for p in peaks]

    # Deterministic ordering of peaks and objects, so equal-cost solutions resolve identically.
    peak_order = sorted(range(n_peak), key=lambda i: (pk[i][0], pk[i][1], i))
    obj_order = sorted(range(n_obj), key=lambda i: (str(object_ids[i]), i))

    scores = peaks[:, 2]
    finite = scores[np.isfinite(scores)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    rng = (hi - lo) or 1.0

    cost = np.zeros((len(obj_order), len(peak_order)), dtype=float)
    for a, o in enumerate(obj_order):
        for b, p in enumerate(peak_order):
            if pk[p] in support_sets[o]:
                # normalised score in [0, 1]; tiny epsilon on ordering makes ties deterministic
                s = (scores[p] - lo) / rng if np.isfinite(scores[p]) else 0.0
                cost[a, b] = -(_BASE + s) + 1e-6 * (a * len(peak_order) + b) / 1e6
            else:
                cost[a, b] = 0.0          # ineligible: contributes nothing, never preferred

    rows, cols = linear_sum_assignment(cost)

    matched = {}
    for a, b in zip(rows, cols):
        if cost[a, b] < -_BASE / 2:       # a real (eligible) edge was used
            matched[obj_order[a]] = peak_order[b]

    used_peaks = set(matched.values())
    return {
        "matched": matched,
        "unmatched_peaks": sorted(i for i in range(n_peak) if i not in used_peaks),
        "unmatched_objects": sorted(i for i in range(n_obj) if i not in matched),
    }
