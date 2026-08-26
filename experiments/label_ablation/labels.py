"""The three label representations, and the pos_weight rule that keeps them comparable.

ISOLATED. Nothing in `src/`, `scripts/` or the project's `data/`/`results/` is modified. The flat
3x3 arm calls the project's own `make_label_mask` so the control is provably identical to the
frozen run's supervision.

    (a) point     1.0 at the quantised truth bin only
    (b) flat3x3   1.0 on a 3x3 block  -- the incumbent, `radar.labels.make_label_mask(1, 1)`
    (c) gaussian  a 3x3 footprint weighted by the MEASURED PSF

All three share the same support convention at the frame border (clipped, not wrapped) because
they are built through the same indexing.

SIGMA IS DERIVED, NOT CHOSEN. `sigma_from_psf()` reads `results/psf_profile.csv` -- the output of
`scripts/measure_psf.py` -- finds the -3 dB width of the measured response on each axis, and
converts it with the Gaussian half-power relation

    FWHM = 2 * sqrt(2 ln 2) * sigma  ->  sigma = FWHM / 2.3548

The OFF-BIN (worst case) profile is used, for two reasons: it is the case the project already
quotes (1.62 / 1.65 bins), and targets in this dataset are placed off-grid by design, so a target
sitting exactly on a bin centre is the exception rather than the rule.

MULTI-TARGET COMBINATION. The Gaussian arm takes the ELEMENTWISE MAXIMUM across targets, not the
sum. CFARNet (Liang 2025) uses `min(1, sum_k ...)`, but a sum lets two nearby targets saturate into
a flat plateau -- exactly the flat-supervision failure this experiment exists to test. Maximum keeps
each target's peak unique at 1.0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from radar.labels import make_label_mask          # noqa: E402  -- the project's own, unmodified

ARMS = ("point", "flat3x3", "gaussian")
REFERENCE_ARM = "flat3x3"
REFERENCE_POS_WEIGHT = 100.0                      # configs/frozen_dl.yaml, won its 3-way ablation
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548


def sigma_from_psf(case: str = "off-bin (worst case)") -> tuple[float, float]:
    """(sigma_range, sigma_doppler) in bins, from the measured PSF profile."""
    P = pd.read_csv(ROOT / "results" / "psf_profile.csv")
    out = []
    for axis in ("range", "doppler"):
        d = P[(P.case == case) & (P.axis == axis)].sort_values("offset_bins")
        x, y = d.offset_bins.to_numpy(float), d.rel_db.to_numpy(float)
        pk = int(np.argmax(y))

        def crossing(step):
            idx = range(pk, len(x) - 1) if step > 0 else range(pk, 0, -1)
            for i in idx:
                j = i + step
                if y[i] >= -3.0 >= y[j]:
                    t = (-3.0 - y[i]) / (y[j] - y[i])
                    return x[i] + t * (x[j] - x[i])
            raise ValueError(f"no -3 dB crossing on {axis}")

        out.append((crossing(+1) - crossing(-1)) / FWHM_TO_SIGMA)
    return float(out[0]), float(out[1])


def psf_weights(sigma_r: float, sigma_d: float, half: int = 1) -> np.ndarray:
    """The 3x3 (by default) Gaussian footprint, peak normalised to 1.0."""
    r = np.arange(-half, half + 1)[:, None]
    d = np.arange(-half, half + 1)[None, :]
    w = np.exp(-(r ** 2) / (2 * sigma_r ** 2) - (d ** 2) / (2 * sigma_d ** 2))
    return w / w.max()


def build_label(arm: str, labels_df: pd.DataFrame, shape: tuple[int, int],
                weights: np.ndarray | None = None) -> np.ndarray:
    """One frame's target map for `arm`, as float32 in [0, 1]."""
    if arm == "flat3x3":
        recs = [{"range_bin": int(r), "velocity_bin": int(v)}
                for r, v in labels_df[["range_bin", "velocity_bin"]].to_numpy()]
        return make_label_mask(recs, shape, radius_r=1, radius_d=1).astype(np.float32)

    y = np.zeros(shape, dtype=np.float32)

    if arm == "point":
        for r, v in labels_df[["range_bin", "velocity_bin"]].to_numpy():
            y[int(r), int(v)] = 1.0
        return y

    if arm == "gaussian":
        if weights is None:
            raise ValueError("gaussian arm needs `weights`")
        half = weights.shape[0] // 2
        for r, v in labels_df[["range_bin", "velocity_bin"]].to_numpy():
            r, v = int(r), int(v)
            r0, r1 = max(0, r - half), min(shape[0], r + half + 1)
            d0, d1 = max(0, v - half), min(shape[1], v + half + 1)
            w = weights[half - (r - r0): half + (r1 - r), half - (v - d0): half + (d1 - v)]
            y[r0:r1, d0:d1] = np.maximum(y[r0:r1, d0:d1], w)      # max, not sum -- see module doc
        return y

    raise ValueError(f"unknown arm {arm!r}")


def pos_weight_for(arm_mass: float, reference_mass: float,
                   reference_pos_weight: float = REFERENCE_POS_WEIGHT) -> float:
    """Scale pos_weight so TOTAL POSITIVE GRADIENT MASS is comparable across arms.

    `BCEWithLogitsLoss(pos_weight=w)` weights the positive term of every cell by `w * y`, so the
    loss's total emphasis on "this is a target" is proportional to `w * sum(y)`. Holding that
    product fixed removes label shape as a confound with class weighting:

        w_arm = w_ref * ( sum(y_ref) / sum(y_arm) )

    ANCHORED TO THE INCUMBENT, DELIBERATELY. The obvious alternative -- re-deriving `pos_weight`
    per arm from the class frequency (w = N_neg / N_pos) -- is exact class balancing, and this
    project has already MEASURED that to be worse: pos_weight 974 lost the three-way ablation on 6
    of 7 conditions and manufactured a `clutter` deficit that had to be retracted (Correction C3).
    Anchoring to the empirically chosen 100 and scaling by positive mass preserves the operating
    point that was actually shown to work, while still controlling the confound.
    """
    return float(reference_pos_weight * reference_mass / arm_mass)
