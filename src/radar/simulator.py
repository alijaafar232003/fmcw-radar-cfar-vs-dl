from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy.constants import c, pi

from radar.config import RadarConfig, derive_radar_params


@dataclass(frozen=True)
class Target:
    range_m: float
    velocity_mps: float
    amplitude_db: float


@dataclass(frozen=True)
class ClutterConfig:
    enabled: bool = False
    strength_db: float = -18.0
    width_bins: int = 3


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


def add_near_zero_doppler_clutter(
    cpi: np.ndarray,
    strength_db: float = -18.0,
    width_bins: int = 3,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    amp = db_to_amp(strength_db)

    clutter = amp * (
        rng.standard_normal(cpi.shape) + 1j * rng.standard_normal(cpi.shape)
    ) / np.sqrt(2)

    clutter_fft = np.fft.fftshift(np.fft.fft(clutter, axis=0), axes=0)

    center = clutter_fft.shape[0] // 2
    mask = np.zeros(clutter_fft.shape[0], dtype=bool)

    start = max(0, center - width_bins)
    end = min(clutter_fft.shape[0], center + width_bins + 1)

    mask[start:end] = True
    clutter_fft[~mask, :] = 0

    clutter_filtered = np.fft.ifft(
        np.fft.ifftshift(clutter_fft, axes=0),
        axis=0,
    )

    return cpi + clutter_filtered


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
        # Distinct seed: clutter and thermal noise are physically independent processes.
        # Sharing one seed made them draw the identical random field.
        clutter_seed = None if seed is None else (seed + 1) % (2 ** 31 - 1)

        cpi = add_near_zero_doppler_clutter(
            cpi,
            strength_db=clutter.strength_db,
            width_bins=clutter.width_bins,
            seed=clutter_seed,
        )

    cpi = add_awgn_var(cpi, noise_var=noise_var, seed=seed)

    return cpi