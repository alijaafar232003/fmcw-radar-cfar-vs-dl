from dataclasses import dataclass
from functools import lru_cache

from scipy.constants import c


@dataclass(frozen=True)
class RadarConfig:
    """Sensor configuration, matched to the CARRADA dataset's range-Doppler geometry.

    These values reproduce CARRADA's RD map bin-for-bin (256 range x 64 Doppler,
    0.195 m and 0.42 m/s per bin, 50 m and +/-13.4 m/s span) so that simulated and
    real maps are directly comparable and a simulation-trained detector can be
    tested on real measurements without a resolution mismatch confounding it.

    B is derived from CARRADA's *bin geometry*, not from the 4 GHz sweep bandwidth
    quoted in their Table I -- that figure is inconsistent with their own stated
    range resolution and maximum range. Full derivation and evidence in
    docs/radar_config_carrada.md.
    """

    fc: float = 77e9       # carrier frequency [Hz]
    B: float = 768e6       # bandwidth [Hz]      -> 0.195 m range bins over 50 m
    Tc: float = 72.4e-6    # chirp duration [s]  -> +/-13.44 m/s unambiguous
    N: int = 512           # fast-time samples per chirp -> 256 range bins
    M: int = 64            # number of chirps            -> 64 Doppler bins


@dataclass(frozen=True)
class RadarDerived:
    lambda_m: float
    slope: float
    fs: float
    Ts: float
    range_res: float
    range_max: float
    vel_res: float
    vel_max: float


@lru_cache(maxsize=None)
def derive_radar_params(cfg: RadarConfig) -> RadarDerived:
    """Derived radar quantities. Cached: RadarConfig is frozen, so it's hashable."""
    lambda_m = c / cfg.fc
    slope = cfg.B / cfg.Tc
    fs = cfg.N / cfg.Tc
    Ts = 1 / fs

    range_res = c / (2 * cfg.B)
    range_max = c * fs / (4 * slope)

    vel_res = lambda_m / (2 * cfg.M * cfg.Tc)
    vel_max = lambda_m / (4 * cfg.Tc)

    return RadarDerived(
        lambda_m=lambda_m,
        slope=slope,
        fs=fs,
        Ts=Ts,
        range_res=range_res,
        range_max=range_max,
        vel_res=vel_res,
        vel_max=vel_max,
    )