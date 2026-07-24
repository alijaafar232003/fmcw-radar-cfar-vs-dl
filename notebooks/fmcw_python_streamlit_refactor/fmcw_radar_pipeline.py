"""
FMCW Radar Range-Doppler Pipeline

Refactored from the notebook:
fmcw_rd_map_v3_VALIDATION_REFACTORED_EXECUTED.ipynb

Core simulation path intentionally preserved:
TX chirp -> RX delayed chirp(s) -> dechirp/mix -> Range FFT -> Doppler FFT -> RD map.

Run:
    python fmcw_radar_pipeline.py

This will generate:
    outputs/demo_rd_map.png
    outputs/example_frame/rd_map.npy
    outputs/example_frame/rd_map_preview.png
    outputs/example_frame/labels.json
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import c, pi


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class RadarConfig:
    """Radar parameters matching the notebook defaults."""

    carrier_hz: float = 24e9        # f
    chirp_duration_s: float = 40e-6 # Tc
    bandwidth_hz: float = 500e6     # bw
    num_chirps: int = 128           # M, slow-time samples
    samples_per_chirp: int = 2048   # N, fast-time samples


@dataclass(frozen=True)
class Target:
    """Point target definition.

    Positive velocity means the target is moving away from the radar.
    amplitude_db is relative amplitude in dB.
    """

    range_m: float
    velocity_mps: float
    amplitude_db: float = 0.0


@dataclass(frozen=True)
class RadarDerived:
    wavelength_m: float
    slope_hz_per_s: float
    fs_hz: float
    range_resolution_m: float
    max_range_m: float
    max_velocity_mps: float
    velocity_resolution_mps: float


# -----------------------------
# Parameter helpers
# -----------------------------

def db_to_amp(x_db: float | np.ndarray) -> float | np.ndarray:
    """Convert amplitude dB to linear amplitude."""
    return 10 ** (np.asarray(x_db) / 20)


def compute_derived(config: RadarConfig) -> RadarDerived:
    """Compute derived radar quantities."""
    wavelength = c / config.carrier_hz
    slope = config.bandwidth_hz / config.chirp_duration_s
    fs = config.samples_per_chirp / config.chirp_duration_s

    return RadarDerived(
        wavelength_m=wavelength,
        slope_hz_per_s=slope,
        fs_hz=fs,
        range_resolution_m=c / (2 * config.bandwidth_hz),
        max_range_m=c * fs / (4 * slope),
        max_velocity_mps=wavelength / (4 * config.chirp_duration_s),
        velocity_resolution_mps=wavelength / (2 * config.num_chirps * config.chirp_duration_s),
    )


def compute_f_beat(range_m: float, config: RadarConfig) -> float:
    """Beat frequency [Hz] for a target at range_m."""
    d = compute_derived(config)
    return (2 * range_m * d.slope_hz_per_s) / c


def compute_phase_diff(velocity_mps: float, config: RadarConfig) -> float:
    """Inter-chirp Doppler phase shift [rad] for radial velocity."""
    d = compute_derived(config)
    return 4 * pi * velocity_mps * config.chirp_duration_s / d.wavelength_m


def make_axes(config: RadarConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create fast-time, absolute-time, range, Doppler, and velocity axes."""
    d = compute_derived(config)

    n = config.samples_per_chirp
    m = config.num_chirps

    t_fast = np.arange(n) / d.fs_hz
    t_abs = np.arange(m)[:, None] * config.chirp_duration_s + t_fast[None, :]

    freq_axis = np.fft.fftfreq(n, d=1 / d.fs_hz)[: n // 2]
    range_axis = freq_axis * c / (2 * d.slope_hz_per_s)

    doppler_axis = np.fft.fftshift(np.fft.fftfreq(m, d=config.chirp_duration_s))
    velocity_axis = doppler_axis * d.wavelength_m / 2

    return t_fast, t_abs, range_axis, doppler_axis, velocity_axis


# -----------------------------
# Validation helpers
# -----------------------------

def normalize_targets(targets: Sequence[Target | tuple[float, float, float]]) -> list[Target]:
    """Accept either Target objects or notebook-style tuples."""
    normalized: list[Target] = []

    for item in targets:
        if isinstance(item, Target):
            normalized.append(item)
        else:
            r, v, amp_db = item
            normalized.append(Target(float(r), float(v), float(amp_db)))

    return normalized


def check_targets(targets: Sequence[Target | tuple[float, float, float]], config: RadarConfig) -> None:
    """Raise an error if a target violates basic range/velocity limits."""
    d = compute_derived(config)
    normalized = normalize_targets(targets)

    for i, target in enumerate(normalized, start=1):
        fb = compute_f_beat(target.range_m, config)

        if fb >= d.fs_hz / 2:
            raise ValueError(
                f"Target {i}: beat frequency {fb / 1e6:.2f} MHz exceeds "
                f"Nyquist fs/2 = {d.fs_hz / 2 / 1e6:.2f} MHz. "
                "Reduce range, increase N/fs, or adjust radar parameters."
            )

        if abs(target.velocity_mps) >= d.max_velocity_mps:
            raise ValueError(
                f"Target {i}: velocity {target.velocity_mps:+.2f} m/s exceeds "
                f"max unambiguous velocity ±{d.max_velocity_mps:.2f} m/s. "
                "Reduce |v| or reduce chirp duration Tc."
            )


# -----------------------------
# Signal generation
# -----------------------------

def generate_tx(config: RadarConfig) -> np.ndarray:
    """Generate the complex analytic FMCW TX chirps."""
    d = compute_derived(config)
    t_fast, t_abs, *_ = make_axes(config)
    t_chirp = t_fast[None, :]

    tx = np.exp(
        1j
        * 2
        * pi
        * (
            config.carrier_hz * t_abs
            + 0.5 * d.slope_hz_per_s * t_chirp**2
        )
    )

    return tx


def generate_cpi(
    targets: Sequence[Target | tuple[float, float, float]],
    config: RadarConfig,
    return_rx: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate a clean dechirped CPI for a list of point targets.

    This preserves the notebook's canonical simulation path:
    1. Generate TX chirp.
    2. Generate delayed RX signal for each target.
    3. Mix/dechirp using tx * conj(rx).

    Shape:
        CPI = [num_chirps, samples_per_chirp]
    """
    check_targets(targets, config)

    d = compute_derived(config)
    t_fast, t_abs, *_ = make_axes(config)
    t_chirp = t_fast[None, :]

    tx = generate_tx(config)
    rx = np.zeros((config.num_chirps, config.samples_per_chirp), dtype=np.complex128)

    for target in normalize_targets(targets):
        amp = db_to_amp(target.amplitude_db)

        # Positive velocity = moving away from radar.
        range_t = target.range_m + target.velocity_mps * t_abs
        tau = 2 * range_t / c

        rx += amp * np.exp(
            1j
            * 2
            * pi
            * (
                config.carrier_hz * (t_abs - tau)
                + 0.5 * d.slope_hz_per_s * (t_chirp - tau) ** 2
            )
        )

    # Convention chosen so positive range gives positive beat frequency.
    cpi = tx * np.conj(rx)

    if return_rx:
        return cpi, rx

    return cpi


def add_awgn(x: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add complex AWGN at the requested SNR."""
    rng = np.random.default_rng(seed)

    signal_power = np.mean(np.abs(x) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )

    return x + noise


# -----------------------------
# Range-Doppler processing
# -----------------------------

def compute_rd_map(
    cpi: np.ndarray,
    config: RadarConfig,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Range-Doppler map.

    Processing preserved from the notebook:
    - Blackman range window across fast time.
    - Blackman Doppler window across slow time.
    - Range FFT across axis=1.
    - Keep positive range half only.
    - Doppler FFT across axis=0.
    - fftshift only Doppler axis.
    - Output RD dB with rows=range bins, columns=velocity bins.

    Returns:
        rd_db      : [range_bins, velocity_bins]
        rd_complex : complex FFT result before transpose, shape [velocity_bins, range_bins]
        rd_mag     : [range_bins, velocity_bins]
    """
    m = config.num_chirps
    n = config.samples_per_chirp

    range_window = signal.windows.blackman(n)
    doppler_window = signal.windows.blackman(m)
    window_2d = doppler_window[:, None] * range_window[None, :]

    cpi_windowed = cpi * window_2d

    range_fft = np.fft.fft(cpi_windowed, axis=1)
    range_fft = range_fft[:, : n // 2]

    rd_complex = np.fft.fftshift(
        np.fft.fft(range_fft, axis=0),
        axes=0,
    )

    rd_mag = np.abs(rd_complex).T
    rd_db = 20 * np.log10(rd_mag / np.max(rd_mag) + eps)

    return rd_db, rd_complex, rd_mag


def simulate_rd_map(
    targets: Sequence[Target | tuple[float, float, float]],
    config: RadarConfig,
    snr_db: float = 10,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-call helper: targets -> noisy CPI -> RD map."""
    _, _, range_axis, _, velocity_axis = make_axes(config)

    cpi = generate_cpi(targets, config)
    cpi_noisy = add_awgn(cpi, snr_db, seed=seed)
    rd_db, rd_complex, rd_mag = compute_rd_map(cpi_noisy, config)

    return rd_db, rd_complex, rd_mag, range_axis, velocity_axis


# -----------------------------
# Labels and validation
# -----------------------------

def expected_label_for_targets(
    targets: Sequence[Target | tuple[float, float, float]],
    config: RadarConfig,
) -> list[dict]:
    """Create label dictionaries with nearest range/velocity bin locations."""
    _, _, range_axis, _, velocity_axis = make_axes(config)

    labels: list[dict] = []

    for target in normalize_targets(targets):
        r_bin = int(np.argmin(np.abs(range_axis - target.range_m)))
        v_bin = int(np.argmin(np.abs(velocity_axis - target.velocity_mps)))

        labels.append(
            {
                "range_m": float(target.range_m),
                "velocity_mps": float(target.velocity_mps),
                "amplitude_db": float(target.amplitude_db),
                "range_bin": r_bin,
                "velocity_bin": v_bin,
                "nearest_range_axis_m": float(range_axis[r_bin]),
                "nearest_velocity_axis_mps": float(velocity_axis[v_bin]),
            }
        )

    return labels


def validate_peaks_records(
    rd_mag: np.ndarray,
    targets: Sequence[Target | tuple[float, float, float]],
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    search_r: int = 8,
    search_v: int = 4,
) -> list[dict]:
    """Return local peak validation records near expected target coordinates.

    This is validation, not detection. It searches near known truth.
    """
    records: list[dict] = []
    noise_floor = np.median(rd_mag) + 1e-12

    global_idx = np.unravel_index(np.argmax(rd_mag), rd_mag.shape)
    global_r = float(range_axis[global_idx[0]])
    global_v = float(velocity_axis[global_idx[1]])

    for i, target in enumerate(normalize_targets(targets), start=1):
        ri = int(np.argmin(np.abs(range_axis - target.range_m)))
        vi = int(np.argmin(np.abs(velocity_axis - target.velocity_mps)))

        r0 = max(0, ri - search_r)
        r1 = min(rd_mag.shape[0] - 1, ri + search_r)
        v0 = max(0, vi - search_v)
        v1 = min(rd_mag.shape[1] - 1, vi + search_v)

        patch = rd_mag[r0 : r1 + 1, v0 : v1 + 1]
        pr, pv = np.unravel_index(patch.argmax(), patch.shape)

        r_est = float(range_axis[r0 + pr])
        v_est = float(velocity_axis[v0 + pv])
        local_peak = float(patch.max())

        records.append(
            {
                "target": i,
                "r_exp": float(target.range_m),
                "r_est": r_est,
                "dr": r_est - float(target.range_m),
                "v_exp": float(target.velocity_mps),
                "v_est": v_est,
                "dv": v_est - float(target.velocity_mps),
                "rel_db": 20 * np.log10(local_peak / (rd_mag.max() + 1e-12) + 1e-12),
                "peak_to_floor_db": 20 * np.log10((local_peak + 1e-12) / noise_floor),
                "global_r": global_r,
                "global_v": global_v,
            }
        )

    return records


def format_validation_records(records: Sequence[dict]) -> str:
    """Create compact monospaced validation text."""
    lines = [
        "#   Rexp   Rest    ΔR |   Vexp   Vest    ΔV | Pk-floor",
        "-" * 62,
    ]

    for rec in records:
        lines.append(
            f"{rec['target']:<1} "
            f"{rec['r_exp']:>6.1f} {rec['r_est']:>6.1f} {rec['dr']:>+6.1f} | "
            f"{rec['v_exp']:>6.1f} {rec['v_est']:>6.1f} {rec['dv']:>+6.1f} | "
            f"{rec['peak_to_floor_db']:>7.1f} dB"
        )

    if records:
        lines += [
            "",
            f"Global max ≈ R={records[0]['global_r']:.1f} m, "
            f"V={records[0]['global_v']:+.2f} m/s",
        ]

    return "\n".join(lines)


# -----------------------------
# Plotting and saving
# -----------------------------

def plot_rd_map(
    rd_db: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    targets: Sequence[Target | tuple[float, float, float]] | None = None,
    title: str = "Range-Doppler Map",
    max_range_m: float | None = None,
    vmin_db: float = -60,
    vmax_db: float = 0,
    cmap: str = "inferno",
    output_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one RD map."""
    extent = [
        velocity_axis[0],
        velocity_axis[-1],
        range_axis[0],
        range_axis[-1],
    ]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    im = ax.imshow(
        rd_db,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=vmin_db,
        vmax=vmax_db,
    )

    if targets:
        for target in normalize_targets(targets):
            ax.plot(
                target.velocity_mps,
                target.range_m,
                marker="x",
                color="cyan",
                linestyle="None",
                ms=11,
                mew=1.4,
            )

    if max_range_m is not None:
        ax.set_ylim(0, max_range_m)

    ax.set_title(title)
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Range [m]")
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Relative magnitude [dB]")

    if output_path:
        fig.savefig(output_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def save_example_frame(
    rd_db: np.ndarray,
    targets: Sequence[Target | tuple[float, float, float]],
    config: RadarConfig,
    snr_db: float,
    output_dir: str | Path = "outputs/example_frame",
    cmap: str = "inferno",
    vmin_db: float = -60,
    vmax_db: float = 0,
) -> dict:
    """Save RD map array, preview image, and labels JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, range_axis, _, velocity_axis = make_axes(config)
    d = compute_derived(config)

    np.save(output_dir / "rd_map.npy", rd_db.astype(np.float32))
    plt.imsave(output_dir / "rd_map_preview.png", rd_db, cmap=cmap, vmin=vmin_db, vmax=vmax_db)

    labels = {
        "snr_db": float(snr_db),
        "rd_map_file": "rd_map.npy",
        "rd_map_preview_file": "rd_map_preview.png",
        "rd_map_shape": list(rd_db.shape),
        "radar_config": asdict(config),
        "range_resolution_m": float(d.range_resolution_m),
        "velocity_resolution_mps": float(d.velocity_resolution_mps),
        "range_axis_min_m": float(range_axis[0]),
        "range_axis_max_m": float(range_axis[-1]),
        "velocity_axis_min_mps": float(velocity_axis[0]),
        "velocity_axis_max_mps": float(velocity_axis[-1]),
        "targets": expected_label_for_targets(targets, config),
    }

    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    return labels


# -----------------------------
# Demo
# -----------------------------

def print_radar_summary(config: RadarConfig) -> None:
    """Print radar parameters and derived performance."""
    d = compute_derived(config)

    print("─── Radar Parameters ──────────────────────────────")
    print(f"  Carrier        : {config.carrier_hz / 1e9:.0f} GHz")
    print(f"  Bandwidth      : {config.bandwidth_hz / 1e6:.0f} MHz")
    print(f"  Chirp duration : {config.chirp_duration_s * 1e6:.0f} µs")
    print(f"  Chirps (M)     : {config.num_chirps}")
    print(f"  Samples (N)    : {config.samples_per_chirp}")
    print(f"  ADC fs         : {d.fs_hz / 1e6:.2f} MHz")
    print("─── Performance ───────────────────────────────────")
    print(f"  Range res      : {d.range_resolution_m:.3f} m")
    print(f"  Max range      : {d.max_range_m:.1f} m")
    print(f"  Max velocity   : ±{d.max_velocity_mps:.2f} m/s")
    print(f"  Velocity res   : {d.velocity_resolution_mps:.3f} m/s")
    print(f"  Nyquist beat   : {d.fs_hz / 2 / 1e6:.2f} MHz")


def demo() -> None:
    """Small runnable demo that outputs a full RD map."""
    config = RadarConfig()
    snr_db = 10

    targets = [
        Target(30, 10.0, 20),
        Target(55, -6.0, 10),
        Target(80, 3.0, -8),
    ]

    print_radar_summary(config)

    print("\n─── Targets ───────────────────────────────────────")
    for i, target in enumerate(targets, start=1):
        print(
            f"  [{i}] range={target.range_m:.0f} m   "
            f"vel={target.velocity_mps:+.1f} m/s   "
            f"amp={target.amplitude_db:+.0f} dB   "
            f"f_beat={compute_f_beat(target.range_m, config) / 1e3:.1f} kHz"
        )

    rd_db, _, rd_mag, range_axis, velocity_axis = simulate_rd_map(
        targets,
        config,
        snr_db=snr_db,
        seed=42,
    )

    records = validate_peaks_records(rd_mag, targets, range_axis, velocity_axis)

    print("\n─── Local Peak Validation ─────────────────────────")
    print(format_validation_records(records))

    output_root = Path("outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    r_plot_max = max(t.range_m for t in targets) * 1.3
    plot_rd_map(
        rd_db,
        range_axis,
        velocity_axis,
        targets=targets,
        title=f"Noisy Range-Doppler Map — SNR = {snr_db} dB",
        max_range_m=r_plot_max,
        vmin_db=-60,
        vmax_db=0,
        cmap="inferno",
        output_path=output_root / "demo_rd_map.png",
        show=True,
    )

    labels = save_example_frame(
        rd_db,
        targets,
        config,
        snr_db=snr_db,
        output_dir=output_root / "example_frame",
    )

    print("\nSaved files:")
    print(f"  Demo plot            : {output_root / 'demo_rd_map.png'}")
    print(f"  RD map numeric array : {output_root / 'example_frame' / 'rd_map.npy'}")
    print(f"  RD map PNG preview   : {output_root / 'example_frame' / 'rd_map_preview.png'}")
    print(f"  Labels               : {output_root / 'example_frame' / 'labels.json'}")


if __name__ == "__main__":
    demo()
