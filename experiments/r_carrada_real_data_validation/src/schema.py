"""R1 output schemas, fixed before scoring (PREREGISTRATION §12 / Phase 4 §X).

Definitions only — no TEST result file is created here. Phase 5 populates these.
"""
from __future__ import annotations

import pandas as pd

PER_OBJECT_COLUMNS = [
    "split", "sequence", "frame", "object_id", "class",
    "eligible", "range_stratum", "doppler_stratum",
    "detector", "cfar_kind", "seed", "tolerance",
    "matched", "peak_range", "peak_doppler", "peak_score",
]

PER_FRAME_COLUMNS = [
    "split", "sequence", "frame", "detector", "cfar_kind", "seed",
    "n_peaks", "n_matched_peaks", "n_unmatched_peaks",
    "n_eligible_objects", "has_annotation",
]

PER_SEQUENCE_COLUMNS = [
    "split", "sequence", "detector", "cfar_kind", "seed",
    "eligible_objects", "recall_exact", "recall_tol1", "recall_tol3",
    "unmatched_per_frame", "n_frames",
]


def empty(kind: str) -> pd.DataFrame:
    cols = {"per_object": PER_OBJECT_COLUMNS,
            "per_frame": PER_FRAME_COLUMNS,
            "per_sequence": PER_SEQUENCE_COLUMNS}[kind]
    return pd.DataFrame(columns=cols)


def example(kind: str) -> pd.DataFrame:
    """One illustrative row per schema, so the shape is reviewable before Phase 5 runs.

    Values are placeholders, not measurements.
    """
    rows = {
        "per_object": [{
            "split": "Test", "sequence": "<seq-id>", "frame": "000123", "object_id": "000652",
            "class": "car", "eligible": True, "range_stratum": "110-160",
            "doppler_stratum": "away_zero", "detector": "tinycnn", "cfar_kind": None, "seed": 0,
            "tolerance": 0, "matched": None, "peak_range": None, "peak_doppler": None,
            "peak_score": None}],
        "per_frame": [{
            "split": "Test", "sequence": "<seq-id>", "frame": "000123", "detector": "cfar",
            "cfar_kind": "CA", "seed": None, "n_peaks": None, "n_matched_peaks": None,
            "n_unmatched_peaks": None, "n_eligible_objects": None, "has_annotation": True}],
        "per_sequence": [{
            "split": "Test", "sequence": "<seq-id>", "detector": "tinycnn", "cfar_kind": None,
            "seed": 0, "eligible_objects": None, "recall_exact": None, "recall_tol1": None,
            "recall_tol3": None, "unmatched_per_frame": None, "n_frames": None}],
    }[kind]
    return pd.DataFrame(rows)[
        {"per_object": PER_OBJECT_COLUMNS, "per_frame": PER_FRAME_COLUMNS,
         "per_sequence": PER_SEQUENCE_COLUMNS}[kind]]
