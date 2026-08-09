"""Step 2a — measure the point-spread function (mainlobe) at the new radar settings.

Two questions this answers, both of which later steps depend on:

  1. How many bins does a single target smear over?  -> sets the CFAR GUARD cells, which exist to
     keep a target's own mainlobe out of its noise estimate.
  2. What separation counts as "resolvable"?         -> sets the masking / unresolved scenarios.

It also tests a claim made while planning the switch: that mainlobe width *in bins* is set by the
FFT window (Blackman) rather than by the radar's resolution, and therefore barely changes when the
sensor changes. Old and new configurations are measured side by side so that claim is checked
rather than assumed.

Scalloping matters here: a target exactly on a bin centre gives the narrowest, tallest response; a
target halfway between bins is the worst case. Both are measured.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.config import RadarConfig, derive_radar_params          # noqa: E402
from radar.rd_processing import compute_rd_map                      # noqa: E402
from radar.simulator import Target, make_axes, simulate_cpi         # noqa: E402

OLD = RadarConfig(fc=77e9, B=300e6, Tc=32e-6, N=512, M=128)
NEW = RadarConfig()  # current default = CARRADA-matched


def _spread_bins(profile_db: np.ndarray, peak: int, floor_db: float) -> int:
    """Half-width in bins beyond which the response never again rises above `floor_db`.

    Robust where a null search is not: an on-bin target has exact nulls, an off-bin target has
    none (they fill in), so "first local minimum" walks to the map edge. This instead asks the
    question the guard cells actually care about — how far out does the target still contribute
    more than `floor_db` below its own peak.
    """
    thresh = profile_db[peak] + floor_db  # floor_db is negative
    above = np.nonzero(profile_db > thresh)[0]
    return int(np.abs(above - peak).max()) if above.size else 0


def _width_at(profile_db: np.ndarray, peak: int, drop_db: float) -> float:
    """Width in bins where the profile crosses (peak - drop_db), linearly interpolated."""
    target = profile_db[peak] - drop_db

    def cross(step: int) -> float:
        i = peak
        n = profile_db.size
        while 0 < i + step < n - 1 and profile_db[i + step] > target:
            i += step
        j = i + step
        if not (0 <= j < n):
            return float(abs(i - peak))
        hi, lo = profile_db[i], profile_db[j]
        frac = 0.0 if hi == lo else (hi - target) / (hi - lo)
        return abs(i - peak) + frac

    return cross(-1) + cross(+1)


FLOORS = (-20.0, -30.0, -40.0)


def measure(cfg: RadarConfig, offset_bins: float, label: str) -> dict:
    d = derive_radar_params(cfg)
    _, _, range_axis, vel_axis = make_axes(cfg)

    dr = range_axis[1] - range_axis[0]
    dv = vel_axis[1] - vel_axis[0]

    # Put the target mid-map, displaced by `offset_bins` of a bin on both axes.
    r0 = range_axis[len(range_axis) // 2] + offset_bins * dr
    v0 = vel_axis[len(vel_axis) // 2] + offset_bins * dv

    cpi = simulate_cpi([Target(r0, v0, 0.0)], cfg, noise_var=0.0, seed=0)
    _, mag, _, _, _ = compute_rd_map(cpi, cfg)

    power_db = 20 * np.log10(mag + 1e-30)
    pr, pv = np.unravel_index(np.argmax(power_db), power_db.shape)

    out = {"label": label, "peak_bin": (int(pr), int(pv))}

    for name, prof, peak, step in (
        ("range", power_db[:, pv], pr, dr),
        ("doppler", power_db[pr, :], pv, dv),
    ):
        out[name] = {
            "w3db_bins": _width_at(prof, peak, 3.0),
            "spread": {f: _spread_bins(prof, peak, f) for f in FLOORS},
            "step": step,
        }
    return out


def resolvable(cfg: RadarConfig, sep_bins: float, weak_db: float, axis: str) -> bool:
    """Do two targets `sep_bins` apart produce two distinct peaks?

    This is the measurement that defines the boundary between the `masking` scenario (two peaks
    exist; a detector may still miss the weak one) and `unresolved` (only one peak exists, so no
    detector can find both). Guessing it from the -3 dB width would be an approximation; this
    tests it directly.
    """
    _, _, range_axis, vel_axis = make_axes(cfg)
    dr = range_axis[1] - range_axis[0]
    dv = vel_axis[1] - vel_axis[0]
    r0 = range_axis[len(range_axis) // 2]
    v0 = vel_axis[len(vel_axis) // 2]

    if axis == "range":
        pair = [Target(r0, v0, 0.0), Target(r0 + sep_bins * dr, v0, weak_db)]
    else:
        pair = [Target(r0, v0, 0.0), Target(r0, v0 + sep_bins * dv, weak_db)]

    cpi = simulate_cpi(pair, cfg, noise_var=0.0, seed=0)
    _, mag, _, _, _ = compute_rd_map(cpi, cfg)

    pr, pv = np.unravel_index(np.argmax(mag), mag.shape)
    prof = 20 * np.log10(mag[:, pv] if axis == "range" else mag[pr, :]) + 1e-30

    # Rayleigh-style criterion: the saddle between the two responses must sit at least
    # `dip_db` below the weaker of the two peaks. Counting bare local maxima is not enough --
    # on a noiseless map, numerical ripple in the sidelobes creates spurious ones.
    dip_db = 3.0
    centre = len(prof) // 2
    i1, i2 = centre, int(round(centre + sep_bins))

    # Need at least one bin strictly between the two responses, otherwise there is no saddle to
    # measure and the test degenerates into comparing two adjacent samples.
    if i2 - i1 < 2 or i2 >= len(prof) - 1:
        return False

    p1 = prof[max(i1 - 1, 0):i1 + 2].max()
    p2 = prof[i2 - 1:i2 + 2].max()

    # The weak response must actually be a local maximum, not just a point on the strong
    # target's skirt.
    if not (prof[i2] >= prof[i2 - 1] and prof[i2] >= prof[i2 + 1]):
        return False

    saddle = prof[i1 + 1:i2].min()
    return (min(p1, p2) - saddle) >= dip_db


def report_resolution(cfg: RadarConfig) -> None:
    _, _, range_axis, vel_axis = make_axes(cfg)
    dr = range_axis[1] - range_axis[0]
    dv = vel_axis[1] - vel_axis[0]

    print(f"\n{'axis':<10}{'weak target':<14}{'min sep (bins)':>16}{'in metres / (m/s)':>20}")
    for axis, step, unit in (("range", dr, "m"), ("doppler", dv, "m/s")):
        for weak_db in (0.0, -6.0, -12.0):
            found = None
            for sep in np.arange(1.0, 10.25, 0.25):
                if resolvable(cfg, float(sep), weak_db, axis):
                    found = float(sep)
                    break
            tag = "equal" if weak_db == 0 else f"{weak_db:+.0f} dB"
            if found is None:
                print(f"{axis:<10}{tag:<14}{'>10':>16}{'—':>20}")
            else:
                print(f"{axis:<10}{tag:<14}{found:>16.2f}{found * step:>17.2f} {unit}")


def main() -> None:
    for cfg, name in ((OLD, "OLD (300 MHz, M=128)"), (NEW, "NEW (768 MHz, M=64)")):
        d = derive_radar_params(cfg)
        print(f"\n{'=' * 74}\n{name}   map {cfg.N // 2} x {cfg.M}   "
              f"{d.range_res:.3f} m/bin, {d.vel_res:.3f} (m/s)/bin\n{'=' * 74}")
        hdr = "".join(f"{'>-' + str(int(-f)) + 'dB':>10}" for f in FLOORS)
        print(f"{'position':<14}{'axis':<10}{'-3dB width':>12}{hdr}")
        for off, pos in ((0.0, "on-bin"), (0.5, "off-bin 1/2")):
            m = measure(cfg, off, pos)
            for axis in ("range", "doppler"):
                a = m[axis]
                cells = "".join(f"{a['spread'][f]:>10d}" for f in FLOORS)
                print(f"{pos:<14}{axis:<10}{a['w3db_bins']:>12.2f}{cells}")

    print("\nColumns after -3dB: half-width in bins beyond which the target never again")
    print("rises above that level relative to its own peak. The CFAR GUARD should cover")
    print("the -30 dB column so a target's own energy stays out of its noise estimate.")
    print("Read the OFF-BIN rows -- worst case.")

    print(f"\n{'=' * 74}\nRESOLUTION LIMIT (new radar) -- minimum separation giving TWO peaks")
    print(f"{'=' * 74}")
    report_resolution(NEW)
    print("\nBelow the minimum: one peak only -> `unresolved` (no detector can win).")
    print("Above it: two peaks exist -> `masking` (a detector CAN find both; CA often does not).")


if __name__ == "__main__":
    main()
