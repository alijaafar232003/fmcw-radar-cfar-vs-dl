"""CUT-centred patch extraction and guard masking for E4. Reads project code; edits nothing.

WHY PATCHES AT ALL. Guard masking is CUT-RELATIVE: for CUT (r,v) the 24 cells of its 5x5 guard
must be hidden, but each of those cells is itself a CUT elsewhere and must then be visible. A dense
fully-convolutional pass computes every output from ONE input map and cannot satisfy both. Proven
in `scripts/verify_control.py`.

WHY THIS IS SOUND FOR INFERENCE. The centre output of a 21x9 patch draws on exactly those 21x9
inputs and never on padding, and eval-mode BatchNorm is a fixed affine map. Measured: the frozen
dense model's logit equals its 21x9 patch-centre logit to 1.9e-06.

WHAT IT CHANGES FOR TRAINING, stated rather than hidden:
  * BatchNorm statistics -- dense normalises over 256x64 per channel, a patch over 21x9;
  * loss population -- dense supervises 16,384 cells per frame, a patch supervises its centre.
Both differences apply EQUALLY to the masked and unmasked patch arms, which is why E4's primary
contrast is masked-vs-unmasked PATCH, with the frozen dense model as a reference point rather than
as the arm being differenced against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig                     # noqa: E402

CFG = CfarConfig()
HW_R, HW_D = CFG.half_width_r, CFG.half_width_d      # 10, 4  -> 21x9 patch
G_R, G_D = CFG.guard_r, CFG.guard_d                  # 2, 2   -> 5x5 guard
NEUTRAL = 0.0                                        # normalize_rd at the frame median


def guard_mask_2d() -> np.ndarray:
    """(21, 9) bool: True where a cell must be HIDDEN. The CUT itself stays visible."""
    m = np.zeros((2 * HW_R + 1, 2 * HW_D + 1), dtype=bool)
    m[HW_R - G_R:HW_R + G_R + 1, HW_D - G_D:HW_D + G_D + 1] = True
    m[HW_R, HW_D] = False                            # CUT VISIBLE -- CFAR tests it
    return m


MASK = guard_mask_2d()


def verify_mask() -> dict:
    """Assertions on the mask, from the authoritative CFAR geometry."""
    out = {
        "patch_shape": [int(2 * HW_R + 1), int(2 * HW_D + 1)],
        "guard_shape": [int(2 * G_R + 1), int(2 * G_D + 1)],
        "cells_masked": int(MASK.sum()),
        "cut_visible": bool(not MASK[HW_R, HW_D]),
        "reference_cells_visible": int((~MASK).sum() - 1),
    }
    assert out["cells_masked"] == (2 * G_R + 1) * (2 * G_D + 1) - 1 == 24
    assert out["cut_visible"], "the CUT must never be masked"
    assert MASK.shape == (2 * HW_R + 1, 2 * HW_D + 1)
    # every masked cell is inside the guard; every guard cell except the CUT is masked
    for i in range(MASK.shape[0]):
        for j in range(MASK.shape[1]):
            in_guard = abs(i - HW_R) <= G_R and abs(j - HW_D) <= G_D
            is_cut = (i, j) == (HW_R, HW_D)
            assert MASK[i, j] == (in_guard and not is_cut), f"mask wrong at {(i, j)}"
    out["reference_cells_expected"] = int(CFG.num_ref_cells)
    assert out["reference_cells_visible"] == CFG.num_ref_cells
    return out


def sample_cuts(shape, n_per_frame: int, rng) -> np.ndarray:
    """Uniform CUT positions inside the valid interior.

    UNIFORM, deliberately: it preserves the dense run's positive/negative class distribution in
    expectation, so `pos_weight` keeps its meaning and is not retuned. A stratified scheme would
    have raised the positive rate by ~100x and made pos_weight=100 a different experiment.
    The cost is fewer positives per epoch, which applies equally to both arms.
    """
    r = rng.integers(HW_R, shape[0] - HW_R, n_per_frame)
    v = rng.integers(HW_D, shape[1] - HW_D, n_per_frame)
    return np.stack([r, v], axis=1)


def extract(x: np.ndarray, cuts: np.ndarray, masked: bool) -> np.ndarray:
    """(N, 21, 9) patches centred on `cuts`, guard-masked if requested."""
    out = np.empty((len(cuts), 2 * HW_R + 1, 2 * HW_D + 1), dtype=np.float32)
    for k, (r, v) in enumerate(cuts):
        out[k] = x[r - HW_R:r + HW_R + 1, v - HW_D:v + HW_D + 1]
    if masked:
        out[:, MASK] = NEUTRAL
    return out


def centre_logits(model, patches: torch.Tensor) -> torch.Tensor:
    """The centre output of each patch -- the CUT's logit."""
    return model(patches[:, None])[:, 0, HW_R, HW_D]
