"""Sequence-aware statistics (PREREGISTRATION §10).

The CARRADA test split has SIX sequences. That is the effective sample size, and nothing here
pretends otherwise.

PROHIBITED, and deliberately not implemented: bootstrapping RD cells; treating object-frames as
independent; any interval whose n is a frame or object count.
"""
from __future__ import annotations

import numpy as np

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260831


def paired_sequence_bootstrap(delta_by_sequence: dict, n_boot: int = N_BOOTSTRAP,
                              seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict:
    """Paired bootstrap over SEQUENCES.

    `delta_by_sequence` maps sequence id -> paired difference (TinyCNN - CFAR) for that sequence.
    Sequences are resampled with replacement; the SAME resampled indices apply to both arms
    because the difference is already paired per sequence.
    """
    seqs = sorted(delta_by_sequence)
    d = np.array([delta_by_sequence[s] for s in seqs], dtype=float)
    n = len(d)
    if n == 0:
        return {"n_sequences": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = d[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n_sequences": n,
        "sequences": seqs,
        "per_sequence_delta": {s: float(v) for s, v in zip(seqs, d)},
        "mean_delta": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_level": 1 - alpha,
        "n_bootstrap": n_boot,
        "bootstrap_seed": seed,
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }


def seed_variability(per_seed_by_sequence: dict) -> dict:
    """TinyCNN between-seed spread, reported SEPARATELY from sequence uncertainty.

    `per_seed_by_sequence` maps sequence -> {seed: metric}. Never merged with the sequence
    bootstrap into a single standard error: the two answer different questions, and pooling them
    would misstate both (PREREGISTRATION §10).
    """
    out = {}
    for s, per_seed in per_seed_by_sequence.items():
        v = np.array([per_seed[k] for k in sorted(per_seed)], dtype=float)
        out[s] = {"seed_mean": float(v.mean()), "seed_sd": float(v.std(ddof=1)) if v.size > 1
                  else float("nan"), "seed_min": float(v.min()), "seed_max": float(v.max()),
                  "n_seeds": int(v.size)}
    allv = np.array([x for p in per_seed_by_sequence.values() for x in p.values()], dtype=float)
    out["_pooled"] = {"mean": float(allv.mean()),
                      "sd_across_all_seed_sequence_pairs": float(allv.std(ddof=1))
                      if allv.size > 1 else float("nan"),
                      "note": "reported separately from sequence-level uncertainty; never merged"}
    return out
