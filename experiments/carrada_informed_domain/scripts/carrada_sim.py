"""The CARRADA-informed synthetic background. Phase 4.

WHAT THIS CHANGES, AND WHAT IT DOES NOT
---------------------------------------
Three background/clutter properties are changed, all of them measured from CARRADA TRAIN only
(`domain_audit/train_derived_targets.json`):

  1. MULTI-LOOK SMOOTHNESS.  The original chain is single-look: its background spread is 5.569 dB
     against a closed-form single-look exponential value of 5.570 dB. CARRADA's is ~7.4 looks.
     Each frame is therefore formed as the incoherent (power) average of L independent background
     realisations sharing ONE deterministic target field, with L sampled per frame.

  2. RANGE DEPENDENCE.  The measured train range profile, unit-mean in power, is applied to the
     background only -- never to the targets, whose amplitudes come from the frozen radar equation.

  3. RIDGE STRUCTURE AND PREVALENCE.  A narrow, strong zero-Doppler ridge in essentially every
     frame, replacing the original's wide flat-topped plateau present in only 39% of frames.

Nothing else moves. `generate_dechirped_cpi`, the Blackman range/Doppler windows and
`compute_rd_map` are the frozen originals, called unmodified, so the target model, the point-spread
function, the 21x9 receptive-field equality and the guard geometry are untouched. Labels, target
placement, target SNR sampling and the dataset schema are untouched.

DECLARED RESIDUAL: lag-1 Doppler correlation is NOT matched. It is set by the slow-time window,
which also defines the target PSF, and the PSF is frozen. See the audit.

WHY THE LOOK COUNT IS UNIFORM ACROSS RANGE
------------------------------------------
The audit first read the far block (210-246) as a second, spikier regime at ~2.1 looks. It is not.
Detrending each cell by its own range bin's mean background power brings that block from L=2.11 to
L=7.51, in line with every other block. The apparent regime change -- and the far block's inflated
lag-1 correlations in BOTH axes -- is the steep mean roll-off inside it acting as a common-mode
component. So the range dependence lives entirely in the MEAN, and one look count plus one mean
profile reproduces it. A range-dependent look count would double-count the same effect.

HOW IT HOOKS IN
---------------
`install()` replaces two names in `data.generate_single_frame`, and only those:

    simulate_cpi   -> returns the deterministic target CPI and records it in a context
    compute_rd_map -> if handed exactly that array object, builds the multi-look background and
                      returns the averaged map; otherwise delegates to the frozen original

The identity test (`cpi is ctx.target_cpi`) is what keeps the two call sites apart: the same
`compute_rd_map` name is also used for the noise-free clean map from which cell SNR is measured,
and that call must keep its original single-look, background-free behaviour or every recorded
`cell_snr_db` would change meaning.

SNR CALIBRATION.  Incoherent averaging leaves mean target power and mean background power
unchanged and divides only the background VARIANCE by L, so cell SNR is preserved in expectation.
The range profile is unit-mean in power for the same reason. Both are asserted by
`selfcheck()` rather than taken on trust.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))

from radar.calibration import clutter_scale_for_cnr                      # noqa: E402
from radar.config import RadarConfig                                     # noqa: E402
from radar.simulator import (ClutterConfig, add_near_zero_doppler_clutter,  # noqa: E402
                             generate_dechirped_cpi)

import data.generate_single_frame as gsf                                 # noqa: E402
from radar.rd_processing import compute_rd_map as _compute_rd_map_orig   # noqa: E402

TARGETS_JSON = os.path.join(EXP, "domain_audit", "train_derived_targets.json")


# ---------------------------------------------------------------------------------- configuration
class DomainConfig:
    """Every knob the intervention exposes. Values come from the train-derived JSON."""

    def __init__(self, looks=(7, 8), look_weights=None, ridge_cnr_db=(19.4, 26.4),
                 ridge_width_bins=0, ridge_distribution="weibull", ridge_shape=(1.1, 2.0),
                 ridge_prevalence=0.99, range_profile=None, apply_range_profile=True):
        self.looks = tuple(int(v) for v in looks)
        w = np.ones(len(self.looks)) if look_weights is None else np.asarray(look_weights, float)
        self.look_weights = w / w.sum()
        self.ridge_cnr_db = tuple(float(v) for v in ridge_cnr_db)
        self.ridge_width_bins = int(ridge_width_bins)
        self.ridge_distribution = str(ridge_distribution)
        self.ridge_shape = tuple(float(v) for v in ridge_shape)
        self.ridge_prevalence = float(ridge_prevalence)
        self.apply_range_profile = bool(apply_range_profile)
        self.range_profile = (None if range_profile is None
                              else np.asarray(range_profile, dtype=np.float64))

    @classmethod
    def from_train_derived(cls, path=TARGETS_JSON, **over):
        d = json.load(open(path, encoding="utf-8"))
        kw = dict(
            ridge_cnr_db=(d["ridge"]["peak_over_floor_db_p05"],
                          d["ridge"]["peak_over_floor_db_p95"]),
            ridge_prevalence=d["ridge"]["fraction_frames_over_10db"],
            range_profile=d["range_profile"]["unit_mean_power"],
        )
        kw.update(over)
        return cls(**kw)

    def describe(self):
        return {
            "looks": list(self.looks),
            "look_weights": [float(v) for v in self.look_weights],
            "looks_mean": float(np.dot(self.looks, self.look_weights)),
            "ridge_cnr_db": list(self.ridge_cnr_db),
            "ridge_width_bins": self.ridge_width_bins,
            "ridge_distribution": self.ridge_distribution,
            "ridge_shape": list(self.ridge_shape),
            "ridge_prevalence": self.ridge_prevalence,
            "apply_range_profile": self.apply_range_profile,
            "range_profile_len": (0 if self.range_profile is None
                                  else int(len(self.range_profile))),
        }


# ---------------------------------------------------------------------------------- range shaping
def _amp_full(profile_power, n_fft):
    """Per-FFT-bin background AMPLITUDE from a unit-mean POWER profile over kept range bins.

    `compute_rd_map` keeps FFT bins 0..N/2-1 as range bins 0..255, so the profile indexes the
    first half directly; the second half is mirrored so the two halves carry the same mean and the
    per-sample variance normalisation below is exact.
    """
    half = n_fft // 2
    p = np.asarray(profile_power, dtype=np.float64)
    if len(p) < half:
        p = np.pad(p, (0, half - len(p)), mode="edge")
    p = p[:half]
    full = np.concatenate([p, p[::-1]])
    amp = np.sqrt(full)
    return amp / np.sqrt(np.mean(amp ** 2))          # mean power exactly 1 -> SNR untouched


def _shaped_noise(shape_mn, noise_var, amp_full, rng):
    """Complex noise of per-sample variance `noise_var`, with a range-dependent spectrum.

    Synthesised in the RANGE-FREQUENCY domain and inverse-transformed to fast time. Building it
    there rather than shaping flat fast-time noise is the same trick the frozen clutter routine
    uses, and for the same reason: a fast-time sample is not a range cell.
    """
    m, n = shape_mn
    spec = (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2.0)
    spec *= amp_full[None, :]
    # ifft * sqrt(n) undoes numpy's 1/N so that mean|y|^2 == mean_r(|spec|^2) == noise_var.
    return np.fft.ifft(spec, axis=1) * np.sqrt(n) * np.sqrt(noise_var)


def _shape_range(cpi, amp_full):
    """Apply the range-dependent amplitude profile to an existing fast-time field."""
    return np.fft.ifft(np.fft.fft(cpi, axis=1) * amp_full[None, :], axis=1)


# ---------------------------------------------------------------------------------- the hook
class _Ctx:
    target_cpi = None
    noise_var = None
    seed = None
    clutter = None
    params = None


_CTX = _Ctx()
_CFG = None
_RECORD = []


def _simulate_cpi_ci(targets, cfg, noise_var, seed=None, clutter=ClutterConfig(),
                     multipath=None, **kw):
    """Deterministic target field only. The background is built per look in compute_rd_map."""
    from radar.simulator import MultipathConfig
    mp = MultipathConfig() if multipath is None else multipath
    tcpi = generate_dechirped_cpi(targets=targets, cfg=cfg, multipath=mp)
    _CTX.target_cpi = tcpi
    _CTX.noise_var = float(noise_var)
    _CTX.seed = seed
    _CTX.clutter = clutter
    return tcpi


def _compute_rd_map_ci(cpi, cfg):
    """Multi-look averaged RD map for the frame CPI; the frozen original for anything else."""
    if _CTX.target_cpi is None or cpi is not _CTX.target_cpi:
        return _compute_rd_map_orig(cpi=cpi, cfg=cfg)

    dc = _CFG
    rng = np.random.default_rng(_CTX.seed)
    m, n = cpi.shape

    L = int(rng.choice(dc.looks, p=dc.look_weights))
    amp = (_amp_full(dc.range_profile, n) if (dc.apply_range_profile
                                              and dc.range_profile is not None)
           else np.ones(n))

    ridge_on = bool(rng.random() < dc.ridge_prevalence)
    cnr_db = float(rng.uniform(*dc.ridge_cnr_db)) if ridge_on else float("nan")
    shape = float(rng.uniform(*dc.ridge_shape))

    scale = 0.0
    if ridge_on:
        scale = clutter_scale_for_cnr(
            cnr_db=cnr_db, noise_var=_CTX.noise_var, cfg=cfg,
            width_bins=dc.ridge_width_bins, distribution=dc.ridge_distribution, shape=shape)

    power = np.zeros((n // 2, m), dtype=np.float64)
    last = None
    for ell in range(L):
        bg = _shaped_noise((m, n), _CTX.noise_var, amp, rng)
        if ridge_on:
            cl = add_near_zero_doppler_clutter(
                np.zeros_like(cpi), scale=scale, width_bins=dc.ridge_width_bins,
                distribution=dc.ridge_distribution, shape=shape,
                seed=int(rng.integers(0, 2 ** 31 - 1)))
            bg = bg + _shape_range(cl, amp)
        _, rd_mag, rd_c, range_axis, velocity_axis = _compute_rd_map_orig(
            cpi=cpi + bg, cfg=cfg)
        power += rd_mag.astype(np.float64) ** 2
        last = rd_c
    power /= L

    rd_mag = np.sqrt(power)
    rd_db = 20.0 * np.log10(rd_mag / np.max(rd_mag) + 1e-12)

    _RECORD.append({"looks": L, "ridge": ridge_on, "ridge_cnr_db": cnr_db,
                    "ridge_shape": shape, "noise_var": _CTX.noise_var})
    _CTX.target_cpi = None            # one use per frame; a second call must not reuse it
    return rd_db, rd_mag, last, range_axis, velocity_axis


def install(domain_cfg):
    """Patch the two names in `data.generate_single_frame`. Returns the previous pair."""
    global _CFG
    _CFG = domain_cfg
    prev = (gsf.simulate_cpi, gsf.compute_rd_map)
    gsf.simulate_cpi = _simulate_cpi_ci
    gsf.compute_rd_map = _compute_rd_map_ci
    return prev


def uninstall(prev):
    gsf.simulate_cpi, gsf.compute_rd_map = prev
    global _CFG
    _CFG = None


def drain_record():
    global _RECORD
    r, _RECORD = _RECORD, []
    return r


# ---------------------------------------------------------------------------------- self-checks
def selfcheck(verbose=True):
    """Assert the two invariants the frozen protocol depends on. Cheap; run before generating."""
    cfg = RadarConfig()
    n = cfg.N
    dc = DomainConfig.from_train_derived()

    # 1. the range profile is unit MEAN POWER, so the average background level is unchanged
    amp = _amp_full(dc.range_profile, n)
    mp = float(np.mean(amp ** 2))
    assert abs(mp - 1.0) < 1e-12, "range profile is not unit mean power: %r" % mp

    # 2. shaped noise has the requested per-sample variance, so the SNR calibration still holds
    rng = np.random.default_rng(0)
    nv = 3.7
    y = _shaped_noise((cfg.M, n), nv, amp, rng)
    got = float(np.mean(np.abs(y) ** 2))
    assert abs(got / nv - 1.0) < 0.02, "shaped-noise variance off by %.3f%%" % (
        100 * (got / nv - 1))

    # 3. flat-profile shaped noise reproduces the frozen add_awgn_var variance
    y2 = _shaped_noise((cfg.M, n), nv, np.ones(n), rng)
    got2 = float(np.mean(np.abs(y2) ** 2))
    assert abs(got2 / nv - 1.0) < 0.02, "flat shaped-noise variance off"

    if verbose:
        print("selfcheck OK")
        print("  range profile mean power   %.12f" % mp)
        print("  shaped-noise variance      %.4f  (requested %.4f, %+.2f%%)"
              % (got, nv, 100 * (got / nv - 1)))
        print("  flat-profile variance      %.4f  (%+.2f%%)" % (got2, 100 * (got2 / nv - 1)))
        print("  looks %s  weights %s  mean %.2f"
              % (dc.looks, np.round(dc.look_weights, 3), np.dot(dc.looks, dc.look_weights)))
    return True


if __name__ == "__main__":
    selfcheck()
