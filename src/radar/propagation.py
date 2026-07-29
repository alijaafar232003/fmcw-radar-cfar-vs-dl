"""Range-dependent target amplitude (the radar equation).

Why this exists
---------------
The data-quality audit (`docs/data_quality_criteria.md`) flagged that target amplitude and
range were **independent** in the generator: a target at 100 m could be just as strong as one
at 10 m. That is physically impossible — the radar equation gives received power

    P_r  ∝  σ / R⁴

so a target's echo falls by 40 dB per decade of range. Real recordings show exactly this
(far targets are systematically weaker), so making amplitude range-dependent closes one of
the external-validity gaps identified in the audit.

Convention
----------
`Target.amplitude_db` keeps its meaning as the target's intrinsic reflectivity (a proxy for
RCS, σ). The **propagation loss** is applied on top, referenced to `ref_range_m` so the scale
stays interpretable:

    amplitude_db_effective = amplitude_db - 40 * log10(R / ref_range_m)

At R = ref_range_m the loss is 0 dB, so a target there behaves exactly as before. Beyond it
the echo weakens, and closer in it strengthens — with the correct 1/R⁴ power law
(equivalently 1/R² in amplitude, hence the 40·log10 in dB of power).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PropagationConfig:
    """Range-dependent amplitude settings.

    enabled      : apply the radar-equation range law.
    ref_range_m  : range at which the loss is 0 dB (targets here keep their nominal amplitude).
    exponent_db  : dB of power loss per decade of range; 40 = the 1/R⁴ radar equation.
    """

    enabled: bool = False
    ref_range_m: float = 50.0
    exponent_db: float = 40.0


def range_loss_db(range_m, cfg: PropagationConfig):
    """Two-way propagation loss in dB (positive = weaker), referenced to `ref_range_m`."""
    if not cfg.enabled:
        return np.zeros_like(np.asarray(range_m, dtype=float))

    r = np.maximum(np.asarray(range_m, dtype=float), 1e-3)  # guard against R -> 0
    return cfg.exponent_db * np.log10(r / cfg.ref_range_m)


def apply_range_law(targets, cfg: PropagationConfig):
    """Return targets with the radar-equation range loss folded into their amplitude."""
    if not cfg.enabled:
        return list(targets)

    out = []
    for t in targets:
        loss = float(range_loss_db(t.range_m, cfg))
        out.append(type(t)(
            range_m=t.range_m,
            velocity_mps=t.velocity_mps,
            amplitude_db=t.amplitude_db - loss,
        ))
    return out
