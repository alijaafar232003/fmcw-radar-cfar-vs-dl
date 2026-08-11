"""Shared detection-scoring package (used identically by CFAR and deep learning)."""

from eval.detection import (
    MatchConfig,
    FrameEval,
    evaluate_frame,
    aggregate,
    roc_curve,
    roc_thresholds,
    pd_vs_snr,
)

__all__ = [
    "MatchConfig",
    "FrameEval",
    "evaluate_frame",
    "aggregate",
    "roc_curve",
    "roc_thresholds",
    "pd_vs_snr",
]
