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
               measured ~+/-2-3 bins on this radar).
    train    : training ring beyond the guard, used to estimate local noise.
    window half-width = guard + train ; full window = (2*hw+1) square.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter, rank_filter, correlate


@dataclass(frozen=True)
class CfarConfig:
    kind: str = "CA"            # "CA", "GO", "OS"
    guard: int = 2             # guard cells per side (mainlobe cover)
    train: int = 8             # training cells per side
    pfa: float = 1e-4          # design false-alarm rate -> alpha
    os_rank_frac: float = 0.75  # OS: rank as a fraction of the training-cell count
    go_axis: int = 1           # GO leading/lagging split axis (1 = Doppler/velocity)

    @property
    def half_width(self) -> int:
        return self.guard + self.train

    @property
    def num_ref_cells(self) -> int:
        win = (2 * self.half_width + 1) ** 2
        guard_win = (2 * self.guard + 1) ** 2
        return win - guard_win

    def alpha(self) -> float:
        """CA/GO threshold multiplier for the design Pfa (mean-of-N-exponentials)."""
        n = self.num_ref_cells
        return n * (self.pfa ** (-1.0 / n) - 1.0)


# ----------------------------------------------------------------------
# Windowed mean over the training ring (outer window minus guard region)
# ----------------------------------------------------------------------

def _ring_mean(power: np.ndarray, half_width: int, guard: int) -> np.ndarray:
    """Mean of the training ring = (outer-window sum - guard-window sum) / count.

    uniform_filter gives a box MEAN in O(1) per cell; convert to sums via the cell
    counts, subtract, and divide by the ring count.
    """
    outer = 2 * half_width + 1
    inner = 2 * guard + 1

    outer_sum = uniform_filter(power, size=outer, mode="nearest") * (outer ** 2)
    guard_sum = uniform_filter(power, size=inner, mode="nearest") * (inner ** 2)

    ring_count = outer ** 2 - inner ** 2
    return (outer_sum - guard_sum) / ring_count


def _half_ring_means(power: np.ndarray, half_width: int, guard: int, axis: int):
    """Leading / lagging half-ring MEANS, split along `axis` (for GO/SO).

    The training ring (outer square minus guard square) is split into the two halves on
    either side of the CUT along `axis`; the centre line is excluded. Each half's mean is
    computed with an explicit normalised kernel (exact — no shift tricks).
    """
    size = 2 * half_width + 1
    ring = np.ones((size, size), dtype=bool)
    g0, g1 = half_width - guard, half_width + guard + 1
    ring[g0:g1, g0:g1] = False  # blank the guard square

    idx = np.arange(size) - half_width  # signed offset from centre along each axis
    if axis == 0:
        lead_fp = ring & (idx[:, None] > 0)
        lag_fp = ring & (idx[:, None] < 0)
    else:
        lead_fp = ring & (idx[None, :] > 0)
        lag_fp = ring & (idx[None, :] < 0)

    lead_k = lead_fp / lead_fp.sum()
    lag_k = lag_fp / lag_fp.sum()

    lead = correlate(power, lead_k, mode="nearest")
    lag = correlate(power, lag_k, mode="nearest")
    return lead, lag


def _os_ring_statistic(power: np.ndarray, half_width: int, guard: int, rank: int) -> np.ndarray:
    """k-th smallest training-ring value via a ring-footprint rank filter.

    Footprint = outer (2*hw+1) square with the (2*guard+1) guard square blanked out.
    """
    outer = 2 * half_width + 1
    footprint = np.ones((outer, outer), dtype=bool)
    g0, g1 = half_width - guard, half_width + guard + 1
    footprint[g0:g1, g0:g1] = False

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
    hw = cfg.half_width

    if cfg.kind == "CA":
        noise = _ring_mean(power, hw, cfg.guard)
        alpha = cfg.alpha()

    elif cfg.kind == "GO":
        lead, lag = _half_ring_means(power, hw, cfg.guard, axis=cfg.go_axis)
        noise = np.maximum(lead, lag)
        # alpha for a half-ring estimate (GO/SO use N/2 cells; GO's exact Pfa is
        # slightly conservative — documented, refined during the sensitivity sweep).
        n_half = cfg.num_ref_cells // 2
        alpha = n_half * (cfg.pfa ** (-1.0 / n_half) - 1.0)

    elif cfg.kind == "OS":
        rank = int(round(cfg.os_rank_frac * cfg.num_ref_cells))
        noise = _os_ring_statistic(power, hw, cfg.guard, rank)
        # OS threshold multiplier is set empirically to the design Pfa during the
        # sensitivity sweep; start from the CA value as a first approximation.
        alpha = cfg.alpha()

    else:
        raise ValueError(f"Unknown CFAR kind: {cfg.kind!r}")

    ratio = power / (noise + 1e-30)

    valid = np.zeros(power.shape, dtype=bool)
    valid[hw:-hw, hw:-hw] = True

    ratio = np.where(valid, ratio, 0.0)
    return ratio, valid, float(alpha)
