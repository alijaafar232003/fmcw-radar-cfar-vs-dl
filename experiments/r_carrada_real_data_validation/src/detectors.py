"""Detector arms. Wrappers ADAPT INPUT/OUTPUT FORMAT ONLY — no algorithm behaviour changes.

Everything computational is imported from the frozen project:
    src.dl.models.TinyCNN, src.dl.infer.predict_points   (which calls the frozen normalize_rd)
    src.cfar.cfar.cfar_ratio_map, src.cfar.detect.all_peaks
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(_HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
_SRC = os.path.join(ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map        # noqa: E402
from cfar.detect import all_peaks                        # noqa: E402
from dl.infer import cfar_valid_mask, predict_points     # noqa: E402
from dl.models import TinyCNN                            # noqa: E402

import carrada_input as ci                                   # noqa: E402

NMS_SIZE = 5                     # FROZEN. Never retuned for CARRADA (PREREGISTRATION §7.3).
CFAR_KINDS = ("CA", "GO", "SO", "OS")
PRIMARY_CFAR = "CA"              # final ruling, PREREGISTRATION §7.2
SEEDS = tuple(range(8))


def checkpoint_path(seed: int) -> str:
    return os.path.join(ROOT, "results", "dl", "tinycnn_frozen_s{}_ep021.pt".format(seed))


def load_frozen_tinycnn(seed: int, device="cpu"):
    """Load one frozen checkpoint in eval mode. No adaptation of any kind."""
    torch.set_num_threads(2)                      # frozen convention; load-bearing for weights
    model = TinyCNN()
    state = torch.load(checkpoint_path(seed), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tinycnn_peaks(model, db: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """CARRADA dB frame -> (K, 3) peaks, entirely through the FROZEN path.

    `to_tinycnn_magnitude` returns 10**(D/20); the frozen `normalize_rd` then computes
    10*log10(mag**2 / median(mag**2)) == D - median(D), which IS the registered adapter. No
    preprocessing is reimplemented here.
    """
    mag = ci.to_tinycnn_magnitude(db)
    if valid is None:
        valid = cfar_valid_mask(db.shape)
    return predict_points(model, mag, valid=valid, nms_size=NMS_SIZE)


def cfar_peaks(db: np.ndarray, kind: str = PRIMARY_CFAR,
               valid: np.ndarray | None = None, cfg: CfarConfig | None = None) -> np.ndarray:
    """CARRADA dB frame -> (K, 3) CFAR peaks, frozen geometry/statistic/alpha.

    Input power uses the frame median as the cancelling constant. CFAR is a ratio test, so this
    constant provably drops out; `validate_adapters.py` unit-tests that invariance.
    """
    cfg = cfg or CfarConfig(kind=kind)
    power = ci.to_cfar_power(db)
    ratio, _cfar_valid, _alpha = cfar_ratio_map(power, cfg)   # (ratio_map, valid_mask, alpha)
    if valid is None:
        valid = cfar_valid_mask(db.shape)
    return all_peaks(ratio, valid, nms_size=NMS_SIZE)


def cfar_alpha(kind: str = PRIMARY_CFAR, cfg: CfarConfig | None = None) -> float:
    """The FROZEN calibrated alpha for this CFAR variant — its operating point.

    `all_peaks` deliberately keeps every local maximum with ratio > 0 so a ROC can be swept, so it
    does NOT apply alpha. The primary strict-zero-shot arm (PREREGISTRATION §8.2) is
    `ratio >= alpha`, and forgetting to apply it would leave CFAR unthresholded while TinyCNN is
    thresholded — not a like-for-like comparison. Caught during the Phase 4 visual sanity check.
    """
    cfg = cfg or CfarConfig(kind=kind)
    return float(cfg.resolved_alpha())


def cfar_peaks_at_alpha(db: np.ndarray, kind: str = PRIMARY_CFAR,
                        valid: np.ndarray | None = None,
                        cfg: CfarConfig | None = None) -> np.ndarray:
    """CFAR peaks at the FROZEN operating point: ratio >= calibrated alpha."""
    cfg = cfg or CfarConfig(kind=kind)
    pk = cfar_peaks(db, kind=kind, valid=valid, cfg=cfg)
    return apply_threshold(pk, cfar_alpha(kind, cfg))


def apply_threshold(peaks: np.ndarray, threshold: float) -> np.ndarray:
    """Keep peaks with score >= threshold. Shared by both arms."""
    p = np.asarray(peaks, dtype=float).reshape(-1, 3)
    if len(p) == 0:
        return p
    return p[p[:, 2] >= threshold]
