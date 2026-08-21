"""CFAR detectors on Range-Doppler power maps (CA / GO / OS).

Design notes (see docs and PROGRESS.md):

* Operates on LINEAR power |RD|^2. The alpha<->Pfa relation assumes exponentially
  distributed cell power, so applying CFAR to log/dB data is invalid.
* Each detector returns a **ratio map** R = CUT / noise_estimate plus a **validity
  mask**. The noise estimate does not depend on the scaling factor alpha, so a full
  ROC is obtained by thresholding R once (thresholding R at alpha == detecting at the
  corresponding design Pfa) — no need to re-run CFAR per Pfa.
* Border cells lack a full window and are marked invalid (not padded — padding would
  invent data and bias the estimate). The evaluator counts only valid cells.
* Vectorised with scipy.ndimage so reported runtime reflects the algorithm, not Python
  loop overhead.

Window geometry (per side, in bins):
    guard    : cells around the CUT excluded from the estimate (covers the mainlobe;
               measured ~+/-2 bins on this radar, see scripts/measure_psf.py).
    train    : training ring beyond the guard, used to estimate local noise.
    window half-width = guard + train.

The window may be RECTANGULAR. `guard`/`train` set both axes; `guard_v`/`train_v` override the
Doppler axis alone. That matters on this sensor: the Doppler axis is only 64 bins, so a window
sized for the range axis spans a third of it and drags clutter-ridge energy across a wide band of
thresholds, while also costing testable map to border exclusion. Range has 256 bins and no
comparable structure, so the two axes do not want the same window.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter, rank_filter, correlate


# ----------------------------------------------------------------------
# Calibrated alpha table
# ----------------------------------------------------------------------
#
# The closed form `alpha = N(Pfa^(-1/N) - 1)` is derived for the arithmetic mean of N INDEPENDENT
# exponential cells. Neither assumption holds here: the Blackman window correlates adjacent
# reference cells, and three of the four variants are not a mean at all (OS is an order statistic,
# GO a maximum of two half-ring means, SO a minimum). Measured on pure noise, the closed form landed
# between 0.13x and 6.55x of the target rate depending on the variant.
#
# So alpha is taken from a Monte-Carlo table when one exists for this exact geometry, and falls back
# to the closed form when it does not. Regenerate with `scripts/cfar_calibrate_table.py`.
#
# NOTE: alpha does NOT affect `ratio_map`. It is the detection threshold applied to that map, so
# switching to calibrated values cannot change any result scored at matched MEASURED Pfa -- which is
# every result in this project. It changes the DESIGN operating point only.

_ALPHA_TABLE_PATH = Path(__file__).resolve().parents[2] / "configs" / "cfar_alpha_table.json"
_ALPHA_TABLE: dict | None = None


def _alpha_table() -> dict:
    global _ALPHA_TABLE
    if _ALPHA_TABLE is None:
        try:
            _ALPHA_TABLE = json.loads(_ALPHA_TABLE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _ALPHA_TABLE = {"entries": {}}
    return _ALPHA_TABLE


def alpha_key(cfg: "CfarConfig") -> str:
    """Canonical key for the calibrated-alpha table. Geometry-specific by construction."""
    return (f"{cfg.kind}|g{cfg.guard_r}-{cfg.guard_d}|t{cfg.train_r}-{cfg.train_d}"
            f"|pfa{cfg.pfa:g}|os{cfg.os_rank_frac:g}|ax{cfg.go_axis}")


@dataclass(frozen=True)
class CfarConfig:
    # Defaults are the 21x9 window selected in step 6 on the dedicated tuning split
    # (bench2_tune, 400 frames, seed disjoint from every evaluation split). Chosen by Pd at
    # matched MEASURED Pfa, per condition. See PROGRESS.md "Step 6".
    kind: str = "CA"            # "CA", "GO", "OS"
    guard: int = 2             # guard cells per side, RANGE axis (and Doppler unless overridden)
    train: int = 8             # training cells per side, RANGE axis (ditto)
    pfa: float = 1e-4          # design false-alarm rate -> alpha
    os_rank_frac: float = 0.75  # OS: rank as a fraction of the training-cell count
    go_axis: int = 1           # GO leading/lagging split axis (1 = Doppler/velocity)
    guard_v: int | None = 2     # Doppler-axis guard; None -> same as `guard`
    train_v: int | None = 2     # Doppler-axis training cells; None -> same as `train`

    # Defaulting the Doppler axis to the range axis keeps every square configuration -- and every
    # existing call site -- behaving exactly as before.

    @property
    def guard_r(self) -> int:
        return self.guard

    @property
    def train_r(self) -> int:
        return self.train

    @property
    def guard_d(self) -> int:
        return self.guard if self.guard_v is None else self.guard_v

    @property
    def train_d(self) -> int:
        return self.train if self.train_v is None else self.train_v

    @property
    def half_width_r(self) -> int:
        return self.guard_r + self.train_r

    @property
    def half_width_d(self) -> int:
        return self.guard_d + self.train_d

    @property
    def half_width(self) -> int:
        """Kept for square configurations; raises if the window is rectangular."""
        if self.half_width_r != self.half_width_d:
            raise ValueError(
                "half_width is ambiguous for a rectangular window; "
                "use half_width_r / half_width_d."
            )
        return self.half_width_r

    @property
    def num_ref_cells(self) -> int:
        win = (2 * self.half_width_r + 1) * (2 * self.half_width_d + 1)
        guard_win = (2 * self.guard_r + 1) * (2 * self.guard_d + 1)
        return win - guard_win

    @property
    def num_half_ring_cells(self) -> int:
        """Cells in ONE half-ring, for GO/SO — counted, not halved.

        The split excludes the centre line along `go_axis`, so a half-ring is NOT
        `num_ref_cells // 2`. On the 21x9 window each half holds 74 cells, not 82: the 16 cells
        of the centre Doppler column belong to neither half. Halving overstates the sample size
        and so understates the threshold GO needs for its design Pfa.
        """
        hw_r, hw_d = self.half_width_r, self.half_width_d
        ring = np.ones((2 * hw_r + 1, 2 * hw_d + 1), dtype=bool)
        ring[hw_r - self.guard_r: hw_r + self.guard_r + 1,
             hw_d - self.guard_d: hw_d + self.guard_d + 1] = False

        if self.go_axis == 0:
            side = np.arange(2 * hw_r + 1)[:, None] - hw_r > 0
        else:
            side = np.arange(2 * hw_d + 1)[None, :] - hw_d > 0

        return int((ring & side).sum())

    def alpha(self) -> float:
        """CA/OS threshold multiplier for the design Pfa (mean-of-N-exponentials)."""
        n = self.num_ref_cells
        return n * (self.pfa ** (-1.0 / n) - 1.0)

    def alpha_go(self) -> float:
        """GO threshold multiplier — same form, but over one half-ring's cells."""
        n = self.num_half_ring_cells
        return n * (self.pfa ** (-1.0 / n) - 1.0)

    def closed_form_alpha(self) -> float:
        """The analytic multiplier this kind would use with no calibration available.

        GO and SO are half-ring statistics, so both take the half-ring cell count; CA and OS take
        the full ring. For OS this is only a first approximation -- an order statistic has a
        different distribution entirely, which is precisely why the table exists.
        """
        return self.alpha_go() if self.kind in ("GO", "SO") else self.alpha()

    def resolved_alpha(self) -> float:
        """The multiplier actually used: calibrated if this geometry is in the table, else closed.

        `calibrated_alpha_available()` reports which one was returned, so callers that need to
        disclose it (the notebooks do) are not left guessing.
        """
        hit = _alpha_table().get("entries", {}).get(alpha_key(self))
        return float(hit) if hit is not None else self.closed_form_alpha()

    def calibrated_alpha_available(self) -> bool:
        return alpha_key(self) in _alpha_table().get("entries", {})


# ----------------------------------------------------------------------
# Windowed mean over the training ring (outer window minus guard region)
# ----------------------------------------------------------------------

def _ring_footprint(hw_r: int, hw_d: int, g_r: int, g_d: int) -> np.ndarray:
    """Boolean training ring: outer rectangle minus the guard rectangle."""
    ring = np.ones((2 * hw_r + 1, 2 * hw_d + 1), dtype=bool)
    ring[hw_r - g_r: hw_r + g_r + 1, hw_d - g_d: hw_d + g_d + 1] = False
    return ring


def _ring_mean(power: np.ndarray, hw_r: int, hw_d: int, g_r: int, g_d: int) -> np.ndarray:
    """Mean of the training ring = (outer-window sum - guard-window sum) / count.

    uniform_filter gives a box MEAN in O(1) per cell regardless of window size; convert to sums
    via the cell counts, subtract, and divide by the ring count. Rectangular windows are handled
    by passing a per-axis size tuple.
    """
    outer_r, outer_d = 2 * hw_r + 1, 2 * hw_d + 1
    inner_r, inner_d = 2 * g_r + 1, 2 * g_d + 1

    outer_sum = uniform_filter(power, size=(outer_r, outer_d), mode="nearest") * (outer_r * outer_d)
    guard_sum = uniform_filter(power, size=(inner_r, inner_d), mode="nearest") * (inner_r * inner_d)

    ring_count = outer_r * outer_d - inner_r * inner_d
    return (outer_sum - guard_sum) / ring_count


def _half_ring_means(power: np.ndarray, hw_r: int, hw_d: int, g_r: int, g_d: int, axis: int):
    """Leading / lagging half-ring MEANS, split along `axis` (for GO/SO).

    The training ring is split into the two halves on either side of the CUT along `axis`; the
    centre line is excluded. Each half's mean is computed with an explicit normalised kernel
    (exact — no shift tricks). Dividing a full-ring sum by half the cell count is what the
    original half-ring bug did, so the kernels are built and normalised individually.
    """
    ring = _ring_footprint(hw_r, hw_d, g_r, g_d)

    idx_r = np.arange(2 * hw_r + 1) - hw_r   # signed offset from centre, range axis
    idx_d = np.arange(2 * hw_d + 1) - hw_d   # ... Doppler axis

    if axis == 0:
        lead_fp = ring & (idx_r[:, None] > 0)
        lag_fp = ring & (idx_r[:, None] < 0)
    else:
        lead_fp = ring & (idx_d[None, :] > 0)
        lag_fp = ring & (idx_d[None, :] < 0)

    lead_k = lead_fp / lead_fp.sum()
    lag_k = lag_fp / lag_fp.sum()

    lead = correlate(power, lead_k, mode="nearest")
    lag = correlate(power, lag_k, mode="nearest")
    return lead, lag


def _os_ring_statistic(power: np.ndarray, hw_r: int, hw_d: int,
                       g_r: int, g_d: int, rank: int) -> np.ndarray:
    """k-th smallest training-ring value via a ring-footprint rank filter."""
    footprint = _ring_footprint(hw_r, hw_d, g_r, g_d)
    rank = int(np.clip(rank, 0, int(footprint.sum()) - 1))
    return rank_filter(power, rank=rank, footprint=footprint, mode="nearest")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def cfar_ratio_map(power: np.ndarray, cfg: CfarConfig):
    """Return (ratio_map, valid_mask, alpha).

    ratio_map  : CUT / local noise estimate for every cell.
    valid_mask : True where a full window fits (border cells excluded).
    alpha      : detect at design Pfa via  ratio_map > alpha.
    """
    hw_r, hw_d = cfg.half_width_r, cfg.half_width_d
    g_r, g_d = cfg.guard_r, cfg.guard_d

    if cfg.kind == "CA":
        noise = _ring_mean(power, hw_r, hw_d, g_r, g_d)
        alpha = cfg.resolved_alpha()

    elif cfg.kind == "GO":
        lead, lag = _half_ring_means(power, hw_r, hw_d, g_r, g_d, axis=cfg.go_axis)
        noise = np.maximum(lead, lag)
        # alpha over the cells one half-ring actually holds. GO's exact Pfa is still slightly
        # conservative (taking a maximum biases the estimate up) — documented, and irrelevant to
        # any comparison made at matched MEASURED Pfa.
        alpha = cfg.resolved_alpha()

    elif cfg.kind == "SO":
        # Smallest-of: the multiple-target specialist, and the counterpart to GO. Where GO takes
        # the larger half-ring to survive a clutter edge, SO takes the SMALLER one so that an
        # interfering target sitting in one half cannot inflate the noise estimate and mask its
        # neighbour. It is the natural classical baseline for the `masking` stratum, and its
        # absence left that comparison against OS alone.
        lead, lag = _half_ring_means(power, hw_r, hw_d, g_r, g_d, axis=cfg.go_axis)
        noise = np.minimum(lead, lag)
        # Same half-ring cell count as GO; SO's true Pfa is optimistic rather than conservative
        # (taking a minimum biases the estimate down), which matched-measured-Pfa scoring removes.
        alpha = cfg.resolved_alpha()

    elif cfg.kind == "OS":
        rank = int(round(cfg.os_rank_frac * cfg.num_ref_cells))
        noise = _os_ring_statistic(power, hw_r, hw_d, g_r, g_d, rank)
        # OS threshold multiplier is genuinely empirical -- an order statistic has no
        # closed form here -- so the calibrated table matters most for this kind.
        alpha = cfg.resolved_alpha()

    else:
        raise ValueError(f"Unknown CFAR kind: {cfg.kind!r}")

    ratio = power / (noise + 1e-30)

    valid = np.zeros(power.shape, dtype=bool)
    valid[hw_r:-hw_r, hw_d:-hw_d] = True

    ratio = np.where(valid, ratio, 0.0)
    return ratio, valid, float(alpha)
