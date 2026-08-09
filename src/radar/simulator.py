from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy.constants import c, pi
from scipy.special import gamma as gamma_fn

from radar.config import RadarConfig, derive_radar_params


@dataclass(frozen=True)
class Target:
    range_m: float
    velocity_mps: float
    amplitude_db: float


@dataclass(frozen=True)
class ClutterConfig:
    """Near-zero-Doppler ground clutter.

    `cnr_db` is a **clutter-to-noise ratio**: how many dB the mean clutter cell sits above the
    thermal-noise floor in the RD map. It replaces the previous absolute `strength_db`, which
    had no fixed meaning -- it had to be raised to ~45 before it cleared a noise floor whose
    value depended on the requested SNR. The scale that realises `cnr_db` is computed by
    `radar.calibration.clutter_scale_for_cnr`, in the same spirit as the cell-SNR calibration.

    `distribution` sets the amplitude statistics ACROSS RANGE BINS, which is what a CFAR
    training window actually sees:
      "rayleigh" - Gaussian in-phase/quadrature; the easy, previously-used case.
      "weibull"  - shape < 2 is heavier-tailed than Rayleigh (2.0 IS Rayleigh). Land clutter.
      "k"        - compound Gamma texture x Rayleigh speckle; small `shape` = spiky. Sea/land.
    """

    enabled: bool = False
    cnr_db: float = 15.0
    width_bins: int = 3
    distribution: str = "weibull"
    shape: float = 1.5


@dataclass(frozen=True)
class MultipathConfig:
    enabled: bool = False
    delay_m: float = 2.0
    attenuation_db: float = -12.0


def db_to_amp(x_db: float | np.ndarray) -> float | np.ndarray:
    return 10 ** (np.asarray(x_db) / 20)


@lru_cache(maxsize=None)
def make_axes(cfg: RadarConfig):
    """Fast-time, absolute-time, range and velocity axes.

    Cached, because these depend only on the (frozen, hashable) radar config and
    were previously recomputed several times per frame.
    """
    p = derive_radar_params(cfg)

    t_fast = np.arange(cfg.N) / p.fs
    t_abs = np.arange(cfg.M)[:, None] * cfg.Tc + t_fast[None, :]

    freq_axis = np.fft.fftfreq(cfg.N, d=1 / p.fs)[: cfg.N // 2]
    range_axis = freq_axis * c / (2 * p.slope)

    doppler_axis = np.fft.fftshift(np.fft.fftfreq(cfg.M, d=cfg.Tc))
    velocity_axis = doppler_axis * p.lambda_m / 2

    # These arrays are cached and shared between callers, so make them read-only:
    # a caller mutating one in place would otherwise corrupt every later frame.
    for arr in (t_fast, t_abs, range_axis, velocity_axis):
        arr.flags.writeable = False

    return t_fast, t_abs, range_axis, velocity_axis


def validate_targets(targets: List[Target], cfg: RadarConfig) -> None:
    p = derive_radar_params(cfg)

    for i, target in enumerate(targets, start=1):
        fb = 2 * target.range_m * p.slope / c

        if fb >= p.fs / 2:
            raise ValueError(
                f"Target {i}: range {target.range_m} m exceeds Nyquist beat limit."
            )

        if abs(target.velocity_mps) >= p.vel_max:
            raise ValueError(
                f"Target {i}: velocity {target.velocity_mps:+.2f} m/s exceeds ±{p.vel_max:.2f} m/s."
            )


def generate_tx(cfg: RadarConfig) -> np.ndarray:
    p = derive_radar_params(cfg)
    t_fast, t_abs, _, _ = make_axes(cfg)
    t_chirp = t_fast[None, :]

    tx = np.exp(
        1j
        * 2
        * pi
        * (
            cfg.fc * t_abs
            + 0.5 * p.slope * t_chirp**2
        )
    )

    return tx


def generate_dechirped_cpi(
    targets: List[Target],
    cfg: RadarConfig,
    multipath: MultipathConfig = MultipathConfig(),
) -> np.ndarray:
    validate_targets(targets, cfg)

    p = derive_radar_params(cfg)
    t_fast, t_abs, _, _ = make_axes(cfg)
    t_chirp = t_fast[None, :]

    tx = generate_tx(cfg)
    rx = np.zeros((cfg.M, cfg.N), dtype=np.complex128)

    for target in targets:
        amp = db_to_amp(target.amplitude_db)
        r_t = target.range_m + target.velocity_mps * t_abs
        tau = 2 * r_t / c

        rx += amp * np.exp(
            1j
            * 2
            * pi
            * (
                cfg.fc * (t_abs - tau)
                + 0.5 * p.slope * (t_chirp - tau) ** 2
            )
        )

        if multipath.enabled:
            r_mp = target.range_m + multipath.delay_m + target.velocity_mps * t_abs
            tau_mp = 2 * r_mp / c

            rx += amp * db_to_amp(multipath.attenuation_db) * np.exp(
                1j
                * 2
                * pi
                * (
                    cfg.fc * (t_abs - tau_mp)
                    + 0.5 * p.slope * (t_chirp - tau_mp) ** 2
                )
            )

    dechirped = tx * np.conj(rx)

    return dechirped


def add_awgn_var(x: np.ndarray, noise_var: float, seed: int | None = None) -> np.ndarray:
    """Add complex AWGN of a given per-sample variance.

    The variance is supplied by the caller (see radar.calibration) rather than derived
    from the signal power, so the resulting *cell* SNR in the RD map is controlled.
    """
    rng = np.random.default_rng(seed)

    noise = np.sqrt(noise_var / 2) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )

    return x + noise


def clutter_amplitudes(n: int, distribution: str, shape: float, rng) -> np.ndarray:
    """Per-range-bin clutter TEXTURE amplitude, normalised to unit mean power (E[a^2] = 1).

    This is the *texture* of the compound-Gaussian clutter model, not the observed amplitude.
    Speckle is supplied for free by the processing chain: each RD cell is a coherent sum of
    random-phase contributions, so it is Rayleigh-distributed even for a perfectly uniform
    scene. The cell a detector sees is therefore

        observed amplitude  =  texture  x  speckle(Rayleigh)

    which is the standard model -- constant texture gives Rayleigh clutter, Gamma texture gives
    the K-distribution, and a Weibull texture gives heavier-than-Rayleigh land clutter.

    Drawing a Rayleigh *texture* as well (the first attempt) double-counts the speckle: it made
    the "rayleigh" setting come out at a fitted shape of 1.62 rather than 2.0, i.e. heavier-tailed
    than the very baseline it was supposed to define.

    Unit mean power keeps the choice orthogonal to the CNR calibration: changing the tail shape
    does not move the mean, so a requested CNR means the same thing in every case.
    """
    dist = distribution.lower()

    if dist == "rayleigh":
        # No texture. The processing chain's speckle alone gives Rayleigh cells.
        return np.ones(n)

    if dist == "weibull":
        if not 0.1 <= shape <= 10.0:
            raise ValueError(f"Weibull shape out of range: {shape}")
        a = (-np.log(rng.random(n))) ** (1.0 / shape)
        # E[a^2] = Gamma(1 + 2/c) for this parameterisation.
        return a / np.sqrt(gamma_fn(1.0 + 2.0 / shape))

    if dist == "k":
        if shape <= 0:
            raise ValueError(f"K-distribution shape must be > 0: {shape}")
        # Gamma texture of unit mean; combined with the chain's speckle this is the
        # K-distribution, with small `shape` the spiky case.
        return np.sqrt(rng.gamma(shape, 1.0 / shape, size=n))

    raise ValueError(f"Unknown clutter distribution: {distribution!r}")


def add_near_zero_doppler_clutter(
    cpi: np.ndarray,
    scale: float,
    width_bins: int = 3,
    distribution: str = "weibull",
    shape: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """Add ground clutter: heavy-tailed across range, concentrated near zero Doppler.

    The heavy tail is applied **per range BIN**, in the range-frequency domain, and then
    inverse-transformed into fast time. This matters: a fast-time sample is not a range cell.
    Each range bin is a tone spanning all N samples, so drawing heavy-tailed amplitudes per
    *sample* and letting the range FFT run over them averages 512 draws per bin and Gaussianises
    the result -- measured, the fitted Weibull shape came back at 1.97 for every requested
    distribution. Synthesising the range profile directly avoids that.

    Physically this is also the right picture: one range cell may contain a bright scatterer
    while its neighbour does not, and CFAR's training window spans range cells.
    """
    rng = np.random.default_rng(seed)
    n_chirps, n_samples = cpi.shape

    # Slow-time fluctuation, independently per range bin, band-limited to the clutter ridge.
    temporal = (
        rng.standard_normal(cpi.shape) + 1j * rng.standard_normal(cpi.shape)
    ) / np.sqrt(2)

    spec = np.fft.fftshift(np.fft.fft(temporal, axis=0), axes=0)

    centre = n_chirps // 2
    keep = np.zeros(n_chirps, dtype=bool)
    keep[max(0, centre - width_bins): min(n_chirps, centre + width_bins + 1)] = True
    spec[~keep, :] = 0.0

    temporal = np.fft.ifft(np.fft.ifftshift(spec, axes=0), axis=0)

    # Renormalise PER RANGE BIN, not globally. Band-limiting leaves only 2*width+1 of M Doppler
    # bins, so each column's surviving power is itself a high-variance random draw. Left in, that
    # acts as a second texture multiplying the amplitude model, and the compound comes out
    # heavy-tailed no matter what `distribution` asked for -- measured, "rayleigh" fitted c = 1.62
    # instead of 2.0. Per-column normalisation makes `distribution` the sole control on the tail.
    rms = np.sqrt(np.mean(np.abs(temporal) ** 2, axis=0, keepdims=True))
    temporal = np.divide(temporal, rms, out=np.zeros_like(temporal), where=rms > 0)

    # Heavy-tailed reflectivity per range bin, each with an independent phase.
    amps = clutter_amplitudes(n_samples, distribution, shape, rng)
    phase = np.exp(2j * np.pi * rng.random(n_samples))

    # Build the field in the RANGE-BIN domain, then transform to fast time. sqrt(n_samples)
    # undoes numpy's 1/N ifft normalisation so amplitudes stay O(1); any residual constant is
    # absorbed by the empirical CNR calibration.
    field = temporal * (amps * phase)[None, :]
    clutter = np.fft.ifft(field, axis=1) * np.sqrt(n_samples)

    return cpi + scale * clutter


def simulate_cpi(
    targets: List[Target],
    cfg: RadarConfig,
    noise_var: float,
    seed: int | None = None,
    clutter: ClutterConfig = ClutterConfig(),
    multipath: MultipathConfig = MultipathConfig(),
) -> np.ndarray:
    """Targets (+ multipath) -> clutter -> thermal noise of variance `noise_var`."""
    cpi = generate_dechirped_cpi(
        targets=targets,
        cfg=cfg,
        multipath=multipath,
    )

    if clutter.enabled:
        # Local import: radar.calibration imports radar.rd_processing, which imports this
        # module. Importing it lazily keeps that cycle from forming at module load.
        from radar.calibration import clutter_scale_for_cnr

        # Distinct seed: clutter and thermal noise are physically independent processes.
        # Sharing one seed made them draw the identical random field.
        clutter_seed = None if seed is None else (seed + 1) % (2 ** 31 - 1)

        scale = clutter_scale_for_cnr(
            cnr_db=clutter.cnr_db,
            noise_var=noise_var,
            cfg=cfg,
            width_bins=clutter.width_bins,
            distribution=clutter.distribution,
            shape=clutter.shape,
        )

        cpi = add_near_zero_doppler_clutter(
            cpi,
            scale=scale,
            width_bins=clutter.width_bins,
            distribution=clutter.distribution,
            shape=clutter.shape,
            seed=clutter_seed,
        )

    cpi = add_awgn_var(cpi, noise_var=noise_var, seed=seed)

    return cpi