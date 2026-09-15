"""One deliberately difficult range-Doppler frame, for the defence deck.

NOT a new simulator. Every physical step is the frozen project code:

    radar.config.RadarConfig              77 GHz, 768 MHz, Tc 72.4 us, N 512, M 64
    radar.simulator.Target                target record
    radar.simulator.generate_dechirped_cpi  signal model + the multipath model
    radar.simulator.add_near_zero_doppler_clutter  the zero-Doppler ridge
    radar.simulator.add_awgn_var          thermal noise
    radar.calibration.clutter_scale_for_cnr        ridge level from a CNR
    radar.calibration.noise_variance_for_reference_snr   cell-SNR calibration
    radar.calibration.measure_peak_cell_power / cell_snr_db
    radar.propagation.apply_range_law     the radar-equation range law
    radar.rd_processing.compute_rd_map    Blackman windows + the two FFTs
    radar.labels.make_target_labels / make_label_mask

The scene is specified in PHYSICAL units (metres, m/s, dB). The simulator maps them to
bins; no bin index is hard-coded anywhere.

Four intended phenomena:

  1. a narrow zero-Doppler clutter ridge          (ClutterConfig, width_bins below the
                                                   benchmark's 3)
  2. two targets inside one mainlobe              (separation below the measured 0.39 m
                                                   equal-strength resolution limit)
  3. a multipath ghost                            (MultipathConfig, applied to one target
                                                   only -- see `_build_cpi`)
  4. a true target buried in the ridge skirt      (low cell SNR *relative to its local
                                                   background*, not absurdly weak)

Reads no experiment artefact and writes nothing outside presentation_assets/.

    venv/Scripts/python.exe presentation_assets/generate_challenging_rd_map.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from radar.calibration import (cell_snr_db, clutter_scale_for_cnr,          # noqa: E402
                               measure_peak_cell_power,
                               noise_variance_for_reference_snr)
from radar.config import RadarConfig, derive_radar_params                   # noqa: E402
from radar.labels import make_label_mask, make_target_labels                # noqa: E402
from radar.propagation import PropagationConfig, apply_range_law            # noqa: E402
from radar.rd_processing import compute_rd_map                             # noqa: E402
from radar.simulator import (ClutterConfig, MultipathConfig, Target,        # noqa: E402
                             add_awgn_var, add_near_zero_doppler_clutter,
                             generate_dechirped_cpi, make_axes, simulate_cpi)

OUT_FIG = os.path.join(HERE, "figures")
OUT_DATA = os.path.join(HERE, "data")

# ----------------------------------------------------------------------- scene
SEED = 20260915

# The frozen sensor. RadarConfig's defaults ARE the thesis configuration.
CFG = RadarConfig()

# The benchmark's own scene settings (configs/bench2_test.yaml).
PROPAGATION = PropagationConfig(enabled=True, ref_range_m=25.0, exponent_db=40.0)

# The ridge. `width_bins` 1 rather than the benchmark's 3: the brief asks for a ridge
# narrower than the broad synthetic band, and this is the model's own width control.
CLUTTER = ClutterConfig(enabled=True, cnr_db=16.0, width_bins=1,
                        distribution="weibull", shape=1.6)

# The benchmark's multipath model and delay. Attenuation is -8 dB rather than the
# benchmark's -12 dB, so the ghost is plainly visible on a projected slide; both are
# ordinary values for the same model.
MULTIPATH = MultipathConfig(enabled=True, delay_m=2.0, attenuation_db=-8.0)

# Reference cell SNR for a 0 dB-amplitude target; inside the benchmark's -4..14 sweep.
REFERENCE_SNR_DB = 6.0

# Measured by scripts/measure_psf.py and quoted in configs/bench2_test.yaml.
RES_RANGE_EQUAL_M = 0.39      # equal-strength range resolution
RES_RANGE_WEAK_M = 0.59       # with a weak partner
RES_VELOCITY_MPS = 1.16       # Doppler resolution
MAINLOBE_BINS = 2

# Physical scene. amplitude_db is NOMINAL, i.e. before the radar-equation range law.
SCENE = [
    # id           range_m  vel_mps  amp_db  role
    # Upper-left: two targets inside one mainlobe. 0.28 m apart in range, below the
    # measured 0.39 m equal-strength limit; well clear of the ridge, so the ambiguity
    # is resolution, not clutter.
    ("pair_a",      34.00,   -5.60,   20.0,  "unresolved_pair"),
    ("pair_b",      34.28,   -5.42,   18.0,  "unresolved_pair"),
    # Lower-right: the brightest target in the frame. Its ghost lands 2 m further out
    # at the same velocity, 8 dB down (MULTIPATH).
    ("multipath",   13.00,    6.40,    7.5,  "multipath_primary"),
    # Inside the zero-Doppler ridge at short range: a real target whose cell SNR is no
    # higher than the clutter it sits in, so the local background hides it.
    ("weak",        10.50,   -0.45,   -3.5,  "weak_masked"),
]
MULTIPATH_ID = "multipath"


# ------------------------------------------------------------------ simulation
def _build_cpi(targets, ids, multipath: MultipathConfig):
    """Dechirped CPI with multipath applied to ONE target only.

    `generate_dechirped_cpi` applies its MultipathConfig to every target it is given, so a
    single call cannot give one target a ghost and leave the others clean. The dechirped
    signal is linear in the received field --- tx * conj(rx_A + rx_B) = tx * conj(rx_A) +
    tx * conj(rx_B) --- so two calls summed are exactly one call over the union. That
    identity is asserted in `validate()`.
    """
    plain = [t for t, i in zip(targets, ids) if i != MULTIPATH_ID]
    ghosted = [t for t, i in zip(targets, ids) if i == MULTIPATH_ID]

    cpi = generate_dechirped_cpi(targets=plain, cfg=CFG,
                                 multipath=MultipathConfig(enabled=False))
    cpi = cpi + generate_dechirped_cpi(targets=ghosted, cfg=CFG, multipath=multipath)
    return cpi


def _add_scene(cpi, noise_var, seed, clutter: ClutterConfig):
    """Clutter then thermal noise.

    A line-for-line mirror of the tail of `radar.simulator.simulate_cpi`, including its
    seed derivation, because that function builds its own CPI and cannot take a prebuilt
    one. `validate()` asserts this path reproduces `simulate_cpi` bit-for-bit.
    """
    if clutter.enabled:
        clutter_seed = None if seed is None else (seed + 1) % (2 ** 31 - 1)
        scale = clutter_scale_for_cnr(
            cnr_db=clutter.cnr_db, noise_var=noise_var, cfg=CFG,
            width_bins=clutter.width_bins, distribution=clutter.distribution,
            shape=clutter.shape,
        )
        cpi = add_near_zero_doppler_clutter(
            cpi, scale=scale, width_bins=clutter.width_bins,
            distribution=clutter.distribution, shape=clutter.shape, seed=clutter_seed,
        )
    return add_awgn_var(cpi, noise_var=noise_var, seed=seed)


def simulate(scene=SCENE, clutter=CLUTTER, multipath=MULTIPATH,
             reference_snr_db=REFERENCE_SNR_DB, seed=SEED):
    """Run the frozen chain once. Returns everything the outputs need."""
    ids = [s[0] for s in scene]
    roles = {s[0]: s[4] for s in scene}
    nominal = [Target(range_m=s[1], velocity_mps=s[2], amplitude_db=s[3]) for s in scene]

    # Radar equation folded into amplitude before the clean measurement, as in
    # data/generate_single_frame.py.
    targets = apply_range_law(nominal, PROPAGATION)
    noise_var = noise_variance_for_reference_snr(reference_snr_db, CFG)

    # Clean map (targets + ghost, no clutter, no noise) -> per-target cell SNR.
    clean_cpi = _build_cpi(targets, ids, multipath)
    _, rd_mag_clean, _, range_axis, velocity_axis = compute_rd_map(clean_cpi, CFG)

    prelim = make_target_labels(targets, range_axis, velocity_axis)
    cell_snrs = [
        cell_snr_db(measure_peak_cell_power(rd_mag_clean, p["range_bin"],
                                            p["velocity_bin"]), noise_var, CFG)
        for p in prelim
    ]

    # The frame itself.
    cpi = _add_scene(_build_cpi(targets, ids, multipath), noise_var, seed, clutter)
    rd_db, rd_mag, _, range_axis, velocity_axis = compute_rd_map(cpi, CFG)

    labels = make_target_labels(targets, range_axis, velocity_axis, cell_snr_db=cell_snrs)
    for lab, i, nom in zip(labels, ids, nominal):
        lab["id"] = i
        lab["role"] = roles[i]
        lab["nominal_amplitude_db"] = float(nom.amplitude_db)
        lab["propagated_amplitude_db"] = lab.pop("amplitude_db")

    # The ghost: same velocity, range + delay_m, amplitude + attenuation_db. Recorded as a
    # non-target return so it is never counted as ground truth.
    mp = next(t for t, i in zip(targets, ids) if i == MULTIPATH_ID)
    ghost_r = mp.range_m + multipath.delay_m
    g_rbin = int(np.argmin(np.abs(range_axis - ghost_r)))
    g_vbin = int(np.argmin(np.abs(velocity_axis - mp.velocity_mps)))
    ghost = {
        "id": "multipath_ghost", "role": "multipath_ghost_NOT_a_target",
        "range_m": float(ghost_r), "velocity_mps": float(mp.velocity_mps),
        "range_bin": g_rbin, "velocity_bin": g_vbin,
        "nearest_range_m": float(range_axis[g_rbin]),
        "nearest_velocity_mps": float(velocity_axis[g_vbin]),
        "propagated_amplitude_db": float(mp.amplitude_db + multipath.attenuation_db),
        "cell_snr_db": float(cell_snr_db(
            measure_peak_cell_power(rd_mag_clean, g_rbin, g_vbin), noise_var, CFG)),
    }

    mask = make_label_mask(labels, rd_db.shape, radius_r=1, radius_d=1)

    return dict(rd_db=rd_db, rd_db_above_noise=db_above_noise(rd_mag),
                rd_mag=rd_mag, mask=mask, range_axis=range_axis,
                velocity_axis=velocity_axis, labels=labels, ghost=ghost,
                noise_var=noise_var, targets=targets, ids=ids, seed=seed,
                rd_mag_clean=rd_mag_clean)


# -------------------------------------------------------------------- plotting
def db_above_noise(rd_mag):
    """RD power in dB above the frame's own median (noise-floor) level.

    The thesis RD-map convention, copied from `scripts/make_ch3_figures.py:rd_db`, which
    is what every RD map printed in the report uses. `compute_rd_map` also returns a
    dB map normalised to the frame MAXIMUM; that is the dev-preview convention in
    `data/generate_single_frame.py` and is kept in the .npz but not plotted, because
    normalising to the brightest target buries everything else in the colour ramp.
    """
    power = rd_mag ** 2
    return 10.0 * np.log10(power / np.median(power) + 1e-12)


def _base(ax, r, extent):
    """The thesis RD-map convention: viridis, 0-30 dB above the frame noise floor."""
    im = ax.imshow(r["rd_db_above_noise"], aspect="auto", origin="lower", extent=extent,
                   cmap="viridis", vmin=0, vmax=30, interpolation="nearest")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Range [m]")
    return im


def render(r):
    extent = [float(r["velocity_axis"][0]), float(r["velocity_axis"][-1]),
              float(r["range_axis"][0]), float(r["range_axis"][-1])]
    os.makedirs(OUT_FIG, exist_ok=True)
    written = []

    def _save(fig, name):
        for ext in ("png", "svg"):
            p = os.path.join(OUT_FIG, "%s.%s" % (name, ext))
            fig.savefig(p, dpi=200)
            written.append(p)
        plt.close(fig)

    # (a) clean presentation version -- axes and colourbar only
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    im = _base(ax, r, extent)
    fig.colorbar(im, ax=ax).set_label("dB above noise floor")
    _save(fig, "s01_challenging_rd_map")

    # (b) annotated debug version
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    im = _base(ax, r, extent)
    fig.colorbar(im, ax=ax).set_label("dB above noise floor")
    style = {"unresolved_pair": ("#ff3b3b", "o"), "multipath_primary": ("#ff8c1a", "s"),
             "weak_masked": ("#ffffff", "D")}
    # Per-target label offsets in points: the pair sits inside one mainlobe and the ghost
    # sits two metres from its primary, so the annotations must be pushed apart by hand
    # or they overprint each other.
    off = {"pair_a": (16, 14), "pair_b": (16, -30), "multipath": (18, -28),
           "weak": (16, 12), "multipath_ghost": (-16, 18)}
    for lab in r["labels"]:
        col, mk = style[lab["role"]]
        ax.scatter(lab["velocity_mps"], lab["range_m"], marker=mk, s=150,
                   facecolors="none", edgecolors=col, linewidths=2.0, zorder=5)
        ax.annotate("%s  %.1f dB" % (lab["id"], lab["cell_snr_db"]),
                    (lab["velocity_mps"], lab["range_m"]),
                    textcoords="offset points", xytext=off[lab["id"]], color=col,
                    fontsize=10, fontweight="bold")
    g = r["ghost"]
    ax.scatter(g["velocity_mps"], g["range_m"], marker="x", s=150, color="#ff3bd1",
               linewidths=2.2, zorder=5)
    ax.annotate("multipath ghost  %.1f dB  (NOT a target)" % g["cell_snr_db"],
                (g["velocity_mps"], g["range_m"]), textcoords="offset points",
                xytext=off["multipath_ghost"], color="#ff3bd1", fontsize=10,
                fontweight="bold", ha="right")
    ax.set_title("Ground truth: 4 targets (2 unresolved) + 1 multipath ghost")
    _save(fig, "s02_challenging_rd_map_annotated")

    # (c) ground-truth mask overlay
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    im = _base(ax, r, extent)
    fig.colorbar(im, ax=ax).set_label("dB above noise floor")
    ax.contour(r["mask"].astype(float), levels=[0.5], colors="#ff3b3b", linewidths=1.8,
               extent=extent, origin="lower")
    ax.set_title("Ground-truth label mask (3 $\\times$ 3 per target)")
    _save(fig, "s03_challenging_rd_map_truth_mask")

    return written


# ------------------------------------------------------------------ validation
def validate(r):
    """Assert each intended phenomenon is produced by the simulator, not by drawing."""
    out = {}
    p = derive_radar_params(CFG)

    # 1 -- the two-call multipath split equals one call over the union.
    union = generate_dechirped_cpi(targets=r["targets"], cfg=CFG,
                                   multipath=MultipathConfig(enabled=False))
    split = (generate_dechirped_cpi(
                 targets=[t for t, i in zip(r["targets"], r["ids"])
                          if i != MULTIPATH_ID],
                 cfg=CFG, multipath=MultipathConfig(enabled=False))
             + generate_dechirped_cpi(
                 targets=[t for t, i in zip(r["targets"], r["ids"])
                          if i == MULTIPATH_ID],
                 cfg=CFG, multipath=MultipathConfig(enabled=False)))
    out["cpi_superposition_max_abs_err"] = float(np.max(np.abs(union - split)))
    assert out["cpi_superposition_max_abs_err"] < 1e-9

    # 2 -- `_add_scene` reproduces `simulate_cpi` exactly.
    ref = simulate_cpi(targets=r["targets"], cfg=CFG, noise_var=r["noise_var"],
                       seed=r["seed"], clutter=CLUTTER,
                       multipath=MultipathConfig(enabled=False))
    mine = _add_scene(generate_dechirped_cpi(targets=r["targets"], cfg=CFG,
                                             multipath=MultipathConfig(enabled=False)),
                      r["noise_var"], r["seed"], CLUTTER)
    out["simulate_cpi_max_abs_err"] = float(np.max(np.abs(ref - mine)))
    assert out["simulate_cpi_max_abs_err"] == 0.0

    # 3 -- the ghost comes from the multipath model: switch it off, the energy goes.
    g = r["ghost"]
    no_mp = _add_scene(_build_cpi(r["targets"], r["ids"],
                                  MultipathConfig(enabled=False)),
                       r["noise_var"], r["seed"], CLUTTER)
    rd_no_mp, _, _, _, _ = compute_rd_map(no_mp, CFG)
    out["ghost_db_with_multipath"] = float(r["rd_db"][g["range_bin"], g["velocity_bin"]])
    out["ghost_db_without_multipath"] = float(rd_no_mp[g["range_bin"], g["velocity_bin"]])
    out["ghost_db_drop"] = (out["ghost_db_with_multipath"]
                            - out["ghost_db_without_multipath"])
    assert out["ghost_db_drop"] > 5.0

    # 4 -- the ridge comes from the clutter model: switch it off, the ridge goes.
    zero_bin = int(np.argmin(np.abs(r["velocity_axis"])))
    no_cl = _add_scene(_build_cpi(r["targets"], r["ids"], MULTIPATH),
                       r["noise_var"], r["seed"], ClutterConfig(enabled=False))
    rd_no_cl, _, _, _, _ = compute_rd_map(no_cl, CFG)
    out["ridge_zero_doppler_bin"] = zero_bin
    out["ridge_median_db_with_clutter"] = float(np.median(r["rd_db"][:, zero_bin]))
    out["ridge_median_db_without_clutter"] = float(np.median(rd_no_cl[:, zero_bin]))
    out["ridge_rise_db"] = (out["ridge_median_db_with_clutter"]
                            - out["ridge_median_db_without_clutter"])
    assert out["ridge_rise_db"] > 6.0

    # ridge width: Doppler bins whose column median is within 6 dB of the peak column
    col = np.median(r["rd_db"], axis=0)
    out["ridge_width_bins_6db"] = int(np.sum(col >= col.max() - 6.0))
    out["ridge_width_mps_6db"] = float(out["ridge_width_bins_6db"] * p.vel_res)

    # 5 -- the close pair is two distinct truths inside one mainlobe.
    a = next(x for x in r["labels"] if x["id"] == "pair_a")
    b = next(x for x in r["labels"] if x["id"] == "pair_b")
    dr = abs(a["range_m"] - b["range_m"])
    dv = abs(a["velocity_mps"] - b["velocity_mps"])
    out["pair_separation_m"] = float(dr)
    out["pair_separation_mps"] = float(dv)
    out["pair_separation_range_bins"] = float(dr / p.range_res)
    out["pair_separation_velocity_bins"] = float(dv / p.vel_res)
    out["pair_below_equal_strength_limit"] = bool(dr < RES_RANGE_EQUAL_M)
    out["pair_within_mainlobe"] = bool(dr / p.range_res <= MAINLOBE_BINS)
    out["pair_distinct_truth_objects"] = bool(a is not b and
                                              (a["range_m"], a["velocity_mps"])
                                              != (b["range_m"], b["velocity_mps"]))
    assert out["pair_below_equal_strength_limit"] and out["pair_distinct_truth_objects"]

    # how merged they look: is there a dip between the two peaks?
    r0, r1 = sorted((a["range_bin"], b["range_bin"]))
    vcol = a["velocity_bin"]
    cut = r["rd_db"][max(0, r0 - 4):min(CFG.N // 2, r1 + 5), vcol]
    out["pair_profile_db"] = [round(float(x), 2) for x in cut]
    out["pair_local_maxima"] = int(np.sum((cut[1:-1] > cut[:-2])
                                          & (cut[1:-1] > cut[2:])))
    # Both members resolve to the SAME peak cell, so `measure_peak_cell_power` returns one
    # value for both -- itself direct evidence that they are not separable.
    out["pair_shares_one_peak"] = bool(abs(a["cell_snr_db"] - b["cell_snr_db"]) < 1e-9)

    # 6 -- the weak target is in the truth list and sits below its local background.
    w = next(x for x in r["labels"] if x["id"] == "weak")
    out["weak_in_truth"] = True
    out["weak_cell_snr_db"] = round(w["cell_snr_db"], 2)
    out["weak_cell_db"] = float(r["rd_db"][w["range_bin"], w["velocity_bin"]])
    ring = r["rd_db"][max(0, w["range_bin"] - 12):w["range_bin"] + 13,
                      w["velocity_bin"]]
    out["weak_local_background_median_db"] = float(np.median(ring))
    out["weak_excess_over_local_db"] = (out["weak_cell_db"]
                                        - out["weak_local_background_median_db"])
    assert r["mask"][w["range_bin"], w["velocity_bin"]] == 1

    return out


# ----------------------------------------------------------------------- output
def save(r, checks, figs):
    os.makedirs(OUT_DATA, exist_ok=True)
    npz = os.path.join(OUT_DATA, "s01_challenging_rd_map.npz")
    np.savez_compressed(npz,
                        rd_db_above_noise=r["rd_db_above_noise"],   # the plotted map
                        rd_db_rel_max=r["rd_db"],                   # compute_rd_map output
                        rd_mag=r["rd_mag"], label_mask=r["mask"],
                        range_axis=r["range_axis"],
                        velocity_axis=r["velocity_axis"])

    p = derive_radar_params(CFG)
    meta = {
        "what": "One deliberately difficult RD frame for the defence deck. Produced by the "
                "frozen project simulator; not an illustration.",
        "seed": r["seed"],
        "radar_config": asdict(CFG),
        "radar_derived": {"range_res_m": p.range_res, "range_max_m": p.range_max,
                          "vel_res_mps": p.vel_res, "vel_max_mps": p.vel_max,
                          "map_shape_range_doppler": [CFG.N // 2, CFG.M]},
        "reference_cell_snr_db": REFERENCE_SNR_DB,
        "noise_variance": r["noise_var"],
        "clutter": asdict(CLUTTER),
        "multipath": asdict(MULTIPATH),
        "propagation": asdict(PROPAGATION),
        "measured_resolution_limits": {
            "range_equal_strength_m": RES_RANGE_EQUAL_M,
            "range_weak_partner_m": RES_RANGE_WEAK_M,
            "velocity_mps": RES_VELOCITY_MPS,
            "mainlobe_bins": MAINLOBE_BINS,
            "source": "scripts/measure_psf.py, quoted in configs/bench2_test.yaml",
        },
        "targets": r["labels"],
        "non_target_returns": [r["ghost"]],
        "validation": checks,
        "source_files_used": [
            "src/radar/config.py", "src/radar/simulator.py",
            "src/radar/rd_processing.py", "src/radar/calibration.py",
            "src/radar/propagation.py", "src/radar/labels.py",
            "configs/bench2_test.yaml (scene settings)",
            "src/data/generate_single_frame.py (frame recipe and plot convention)",
        ],
        "outputs": [os.path.relpath(p_, ROOT) for p_ in figs + [npz]],
    }
    js = os.path.join(OUT_DATA, "s01_challenging_rd_map.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return npz, js


def main():
    r = simulate()
    checks = validate(r)
    figs = render(r)
    npz, js = save(r, checks, figs)

    p = derive_radar_params(CFG)
    print("radar: %.0f GHz, B %.0f MHz, %d x %d, %.4f m/bin, %.4f m/s per bin"
          % (CFG.fc / 1e9, CFG.B / 1e6, CFG.N // 2, CFG.M, p.range_res, p.vel_res))
    print("seed %d | reference cell SNR %.1f dB | noise_var %.4g"
          % (r["seed"], REFERENCE_SNR_DB, r["noise_var"]))
    print("\n%-12s %-18s %8s %8s %7s %7s %9s" %
          ("id", "role", "range_m", "vel_mps", "r_bin", "v_bin", "cellSNR"))
    for lab in r["labels"]:
        print("%-12s %-18s %8.2f %8.2f %7d %7d %9.2f"
              % (lab["id"], lab["role"], lab["range_m"], lab["velocity_mps"],
                 lab["range_bin"], lab["velocity_bin"], lab["cell_snr_db"]))
    g = r["ghost"]
    print("%-12s %-18s %8.2f %8.2f %7d %7d %9.2f"
          % (g["id"], g["role"], g["range_m"], g["velocity_mps"],
             g["range_bin"], g["velocity_bin"], g["cell_snr_db"]))

    print("\nvalidation")
    for k, v in checks.items():
        if k == "pair_profile_db":
            continue
        print("   %-38s %s" % (k, round(v, 4) if isinstance(v, float) else v))
    print("   pair range profile [dB]:", checks["pair_profile_db"])

    print("\nwrote")
    for f in figs + [npz, js]:
        print("   ", os.path.relpath(f, ROOT))


if __name__ == "__main__":
    main()
