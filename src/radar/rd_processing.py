import numpy as np
from scipy import signal

from radar.config import RadarConfig
from radar.simulator import make_axes


def compute_rd_map(cpi: np.ndarray, cfg: RadarConfig):
    range_window = signal.windows.blackman(cfg.N)
    doppler_window = signal.windows.blackman(cfg.M)

    cpi_windowed = cpi * doppler_window[:, None] * range_window[None, :]

    range_fft = np.fft.fft(cpi_windowed, axis=1)
    range_fft = range_fft[:, : cfg.N // 2]

    rd_complex = np.fft.fftshift(
        np.fft.fft(range_fft, axis=0),
        axes=0,
    )

    rd_mag = np.abs(rd_complex).T

    rd_db = 20 * np.log10(rd_mag / np.max(rd_mag) + 1e-12)

    _, _, range_axis, velocity_axis = make_axes(cfg)

    return rd_db, rd_mag, rd_complex, range_axis, velocity_axis