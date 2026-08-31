"""Pre-registered R1 metrics. Only the metrics in PREREGISTRATION §9 exist here.

Deliberately ABSENT, and not to be added later (PREREGISTRATION §9.3):
  centroid error, brightest-cell error, box-centre error, sparse-centroid error, exact-bin
  localization, or any "distance to nearest mask cell" reported as a localization metric. For a
  successful detection that distance is identically zero, so it cannot discriminate among
  successes; and a nearest-of-many rule would reward whichever detector emits more peaks.
"""
from __future__ import annotations

import numpy as np

from eligibility import dilate_cells, valid_support
from matching import match

TOLERANCES = (0, 1, 3)          # exact mask (headline), then +-1 and +-3 dilation


def score_frame(peaks: np.ndarray, objects: list[dict], tolerance: int = 0,
                shape=(256, 64)) -> dict:
    """Match one frame's peaks against its objects at one tolerance level.

    Returns per-object outcomes plus the frame's unmatched-detection burden.

    NOTE on the burden (PREREGISTRATION §7.3): CARRADA objects are extended and the frozen NMS
    window is 11x11, so a single object can yield several surviving peaks. Under one-to-one
    matching exactly one is matched and the rest count as UNMATCHED. That is expected and is a
    metric-transfer artefact, not evidence of detector noise.
    """
    supports, ids = [], []
    for o in objects:
        vs = valid_support(o["dense"], shape)
        if tolerance > 0 and len(vs) > 0:
            vs = valid_support(dilate_cells(o["dense"], tolerance, shape), shape)
        supports.append(vs)
        ids.append(o["object_id"])

    eligible = [i for i, s in enumerate(supports) if len(s) >= 1]
    res = match(peaks, supports, ids)

    per_object = []
    for i, o in enumerate(objects):
        mp = res["matched"].get(i)
        per_object.append({
            "sequence": o["sequence"], "frame": o["frame"], "object_id": o["object_id"],
            "class": o["class"], "eligible": i in eligible, "tolerance": tolerance,
            "matched": mp is not None,
            "peak_range": int(peaks[mp][0]) if mp is not None else None,
            "peak_doppler": int(peaks[mp][1]) if mp is not None else None,
            "peak_score": float(peaks[mp][2]) if mp is not None else None,
        })

    n_pk = len(np.asarray(peaks).reshape(-1, 3))
    return {
        "per_object": per_object,
        "n_peaks": n_pk,
        "n_matched_peaks": len(res["matched"]),
        "n_unmatched_peaks": n_pk - len(res["matched"]),
        "n_eligible_objects": len(eligible),
        "n_matched_eligible": sum(1 for i in eligible if i in res["matched"]),
    }


def object_recall(per_object_rows) -> float:
    """matched eligible objects / total eligible objects. NaN when the denominator is zero."""
    el = [r for r in per_object_rows if r["eligible"]]
    if not el:
        return float("nan")
    return sum(1 for r in el if r["matched"]) / len(el)


def burden_summary(per_frame_rows) -> dict:
    """Unmatched-detection burden.

    NEVER call this Pfa, false-positive rate, false alarm rate, or FPR. Only the 77 tracked
    objects are annotated, so a peak outside every mask is not a known false alarm
    (PREREGISTRATION §6).
    """
    u = np.array([r["n_unmatched_peaks"] for r in per_frame_rows], dtype=float)
    if u.size == 0:
        return {"n_frames": 0}
    return {
        "n_frames": int(u.size),
        "unmatched_per_frame_mean": float(u.mean()),
        "unmatched_per_frame_median": float(np.median(u)),
        "unmatched_per_frame_p90": float(np.percentile(u, 90)),
        "unmatched_per_frame_max": float(u.max()),
    }
