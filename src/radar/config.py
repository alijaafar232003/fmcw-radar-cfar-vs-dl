from dataclasses import dataclass
from functools import lru_cache

from scipy.constants import c


@dataclass(frozen=True)
class RadarConfig:
    fc: float = 77e9       # carrier frequency [Hz]
    B: float = 300e6       # bandwidth [Hz]
    Tc: float = 32e-6      # chirp duration [s]
    N: int = 512           # fast-time samples per chirp
    M: int = 128           # number of chirps


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