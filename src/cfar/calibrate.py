"""Empirical (Monte-Carlo) calibration of CFAR thresholds.

Why this is needed
------------------
CA-CFAR has a closed-form threshold multiplier

    alpha_CA = N * (Pfa^(-1/N) - 1)          (N = number of training cells)

because its noise estimate is the arithmetic mean of N exponential (square-law) cells,
whose ratio to the cell-under-test has a known distribution. **This formula is specific
to the arithmetic mean.** OS-CFAR uses an order statistic (the k-th ranked training cell)
and GO-CFAR uses the maximum of two half-window means — different statistics with
different distributions. Reusing alpha_CA for them does not yield the intended Pfa: with
guard=2, train=8 it left OS and GO far too conservative (≈0 false alarms on pure noise).

Rather than derive a separate closed form per variant, we calibrate **empirically**, which
works for any CFAR statistic:

1. Process many pure-noise CPIs through the identical RD chain -> noise power maps.
2. Run the CFAR ratio map (CUT / local noise estimate) on each.
3. Pool the ratio values over all validly-tested cells.
4. The threshold for a target per-cell Pfa is the **(1 - Pfa) empirical quantile** of that
   pooled ratio distribution — i.e. the level exceeded by a fraction Pfa of noise cells.

This is a standard, defensible approach ("threshold set by Monte-Carlo to achieve the
design false-alarm rate"). We validate it by confirming the empirical CA threshold matches
the analytic formula (they agree to ~1%).
"""

from __future__ import annotations

import numpy as np

from radar.config import RadarConfig
from radar.rd_processing import compute_rd_map
from cfar.cfar import CfarConfig, cfar_ratio_map


def make_noise_power_frames(radar_cfg: RadarConfig, n_frames: int, seed: int = 0):
    """Yield `n_frames` pure-noise RD power maps from the real processing chain.

    Unit-variance complex white noise -> the same windowed 2-D FFT used for real data,
    so the cell statistics (exponential marginal + the window's mild inter-cell
    correlation) match the actual noise-only frames.
    """
    rng = np.random.default_rng(seed)
    m, n = radar_cfg.M, radar_cfg.N

    for _ in range(n_frames):
        cpi = (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2)
        _, mag, _, _, _ = compute_rd_map(cpi, radar_cfg)
        yield mag ** 2


def calibrate_alpha(
    cfar_cfg: CfarConfig,
    radar_cfg: RadarConfig,
    target_pfa: float,
    n_frames: int = 60,
    seed: int = 0,
) -> tuple[float, int]:
    """Monte-Carlo threshold multiplier for a target per-cell Pfa.

    Returns (alpha, n_cells_used). Detect with:  ratio_map > alpha.
    """
    pooled = []

    for power in make_noise_power_frames(radar_cfg, n_frames, seed):
        ratio, valid, _ = cfar_ratio_map(power, cfar_cfg)
        pooled.append(ratio[valid])

    ratios = np.concatenate(pooled)
    alpha = float(np.quantile(ratios, 1.0 - target_pfa))
    return alpha, int(ratios.size)


# SO was missing from this tuple until 19 Aug 2026, which is why it was never calibrated: it was
# added to `cfar.py` on 15 Aug and nothing here knew about it. Keep all four.
ALL_KINDS = ("CA", "GO", "OS", "SO")


def calibrate_all(
    radar_cfg: RadarConfig,
    target_pfa: float,
    guard: int = 2,
    train: int = 8,
    guard_v: int | None = 2,
    train_v: int | None = 2,
    kinds=ALL_KINDS,
    n_frames: int = 60,
    seed: int = 0,
) -> dict[str, float]:
    """Calibrated alpha for each CFAR kind at a common geometry and target Pfa.

    `guard_v` / `train_v` are exposed because the selected window is RECTANGULAR (21x9). Calibrating
    at the square default would produce alphas for a geometry the project does not use.
    """
    out = {}
    for k in kinds:
        cfg = CfarConfig(kind=k, guard=guard, train=train, guard_v=guard_v, train_v=train_v,
                         pfa=target_pfa)
        alpha, _ = calibrate_alpha(cfg, radar_cfg, target_pfa, n_frames=n_frames, seed=seed)
        out[k] = alpha
    return out
