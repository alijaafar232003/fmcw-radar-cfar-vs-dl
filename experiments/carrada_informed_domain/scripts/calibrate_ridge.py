"""Map the generator's `cnr_db` knob onto the MEASURED ridge peak-over-floor. Phase 4, step 2.

These are two different quantities and conflating them would silently mis-set the ridge.
`cnr_db` is the mean clutter cell power over the noise floor, calibrated by the frozen
`clutter_scale_for_cnr`. What the audit measures on CARRADA is the peak of the mean background
Doppler profile over its floor. With the clutter concentrated into one slow-time spectral bin
instead of spread over seven, a given `cnr_db` produces a much higher peak than it used to, so the
mapping has to be measured rather than assumed.

Background-only frames (no targets), scored with the audit's own `ridge_profile`, through the same
multi-look path the dataset generator will use. No CARRADA data is read: the only CARRADA-derived
inputs are the already-frozen train targets in train_derived_targets.json.

    python calibrate_ridge.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(R_EXP, "src"))
sys.path.insert(0, HERE)

import r0_background as bg                                               # noqa: E402
from radar.config import RadarConfig                                     # noqa: E402
from radar.calibration import noise_variance_for_reference_snr           # noqa: E402

import carrada_sim as cs                                                 # noqa: E402
from audit_domain import ridge_metrics                                   # noqa: E402
from derive_targets import looks_from_db                                 # noqa: E402

OUT = os.path.join(EXP, "domain_audit")


def one_frame(dc, cfg, noise_var, cnr_db, width_bins, shape, looks, seed):
    """Background-only multi-look frame, built by the same code path as the generator."""
    rng = np.random.default_rng(seed)
    m, n = cfg.M, cfg.N
    amp = (cs._amp_full(dc.range_profile, n)
           if (dc.apply_range_profile and dc.range_profile is not None) else np.ones(n))
    zero = np.zeros((m, n), dtype=complex)
    scale = 0.0
    if np.isfinite(cnr_db):
        from radar.calibration import clutter_scale_for_cnr
        scale = clutter_scale_for_cnr(cnr_db=cnr_db, noise_var=noise_var, cfg=cfg,
                                      width_bins=width_bins,
                                      distribution=dc.ridge_distribution, shape=shape)
    power = None
    for _ in range(looks):
        b = cs._shaped_noise((m, n), noise_var, amp, rng)
        if scale:
            cl = cs.add_near_zero_doppler_clutter(
                zero, scale=scale, width_bins=width_bins,
                distribution=dc.ridge_distribution, shape=shape,
                seed=int(rng.integers(0, 2 ** 31 - 1)))
            b = b + cs._shape_range(cl, amp)
        _, rd_mag, _, _, _ = cs._compute_rd_map_orig(cpi=b, cfg=cfg)
        p = rd_mag.astype(np.float64) ** 2
        power = p if power is None else power + p
    return power / looks


def measure(power_maps):
    """Ridge metrics and background spread, using the audit's own definitions."""
    rm, dbs = [], []
    keep_bg = bg.background_mask([], 4)
    for p in power_maps:
        prof = bg.ridge_profile(p, [])
        m = ridge_metrics(prof)
        if m:
            rm.append(m)
        st = bg.frame_stats(p, keep_bg)
        if st:
            dbs.append(st["db_std"])
    out = {}
    if rm:
        for k in rm[0]:
            out[k] = float(np.median([r[k] for r in rm]))
    if dbs:
        out["db_std"] = float(np.median(dbs))
        out["looks_implied"] = looks_from_db(out["db_std"])
    return out


def main(n_frames=12):
    cfg = RadarConfig()
    tgt = json.load(open(os.path.join(OUT, "train_derived_targets.json"), encoding="utf-8"))
    want_peak = tgt["ridge"]["peak_over_floor_db_median"]
    want_p05 = tgt["ridge"]["peak_over_floor_db_p05"]
    want_p95 = tgt["ridge"]["peak_over_floor_db_p95"]
    want_half = tgt["ridge"]["halfpower_bins_median"]
    want_above3 = tgt["ridge"]["bins_above_3db_median"]
    dc = cs.DomainConfig.from_train_derived()

    # Mid-scale noise variance: the CNR calibration is a ratio, so the level only has to be
    # representative. -4..14 dB is the frozen snr_db sweep; 5 dB sits in the middle.
    noise_var = noise_variance_for_reference_snr(5.0, cfg)

    print("CARRADA train target:  peak %.1f dB (p5-p95 %.1f-%.1f), "
          "half-power %.2f bins, >3dB %.2f bins"
          % (want_peak, want_p05, want_p95, want_half, want_above3))
    print("")
    print("width  cnr_db |  peak_over_floor  halfpower  >3dB   bg dB spread  implied L")
    print("-" * 84)

    rows = []
    for width in (0, 1):
        for cnr in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
            maps = [one_frame(dc, cfg, noise_var, cnr, width, 1.5, 8, 5000 + 97 * i)
                    for i in range(n_frames)]
            r = measure(maps)
            rows.append(dict(width_bins=width, cnr_db=cnr, **r))
            print("  %d      %5.1f |    %7.2f dB      %5.2f     %4.2f      %5.3f      %5.2f"
                  % (width, cnr, r.get("ridge_peak_over_floor_db", float("nan")),
                     r.get("ridge_halfpower_bins", float("nan")),
                     r.get("ridge_bins_above_3db", float("nan")),
                     r.get("db_std", float("nan")), r.get("looks_implied", float("nan"))))

    # Invert: which cnr_db lands on the wanted peak, for each width?
    print("")
    best = {}
    for width in (0, 1):
        sub = [r for r in rows if r["width_bins"] == width
               and np.isfinite(r.get("ridge_peak_over_floor_db", np.nan))
               and r["cnr_db"] > 0]
        if len(sub) < 2:
            continue
        x = np.array([r["cnr_db"] for r in sub])
        y = np.array([r["ridge_peak_over_floor_db"] for r in sub])
        o = np.argsort(y)
        inv = lambda p: float(np.interp(p, y[o], x[o]))
        best[str(width)] = {"cnr_for_median": inv(want_peak),
                            "cnr_for_p05": inv(want_p05),
                            "cnr_for_p95": inv(want_p95),
                            "peak_at_cnr_grid": {float(a): float(b) for a, b in zip(x, y)},
                            "halfpower_at_grid": {
                                float(r["cnr_db"]): float(r["ridge_halfpower_bins"])
                                for r in sub}}
        print("width %d: cnr_db %.2f -> peak %.1f dB;  p5 %.2f -> %.1f;  p95 %.2f -> %.1f"
              % (width, inv(want_peak), want_peak, inv(want_p05), want_p05,
                 inv(want_p95), want_p95))

    with open(os.path.join(OUT, "ridge_calibration.json"), "w", encoding="utf-8") as fh:
        json.dump({"target": tgt["ridge"], "grid": rows, "inverted": best,
                   "n_frames_per_point": n_frames,
                   "note": "cnr_db is mean clutter power over the noise floor; peak_over_floor is "
                           "the measured Doppler-profile peak. This file maps one to the other."},
                  fh, indent=1, default=cs.json_default if hasattr(cs, "json_default") else str)
    print("\nwrote ridge_calibration.json")


if __name__ == "__main__":
    main()
