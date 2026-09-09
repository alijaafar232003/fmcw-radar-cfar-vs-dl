"""Run the Integrated-CFAR attribution experiment on bench2_val. Writes to ./results/.

    How much of TinyCNN's advantage over CFAR is explained by the CNN integrating neighbouring
    mainlobe energy that single-cell CFAR discards?

ISOLATED BY CONSTRUCTION. Reads `src/`, `data/bench2_val/`, and the cached baselines in
`results/cfar_val/` and `results/dl/`. Writes ONLY inside this folder. Modifies nothing.

⚠️ `bench2_test` is never touched. This is an attribution experiment on the validation split.

    python experiments/integrated_cfar/run_integrated_cfar.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# ------------------------------------------------- E6 PATCH (auto-generated)
# Isolated copy at NMS radius 2. The original is not modified and keeps its radius-5
# results. Checkpoints and source are read from the original experiment; every write
# goes into the e6 folder.
from pathlib import Path as _P
HERE = _P("C:/Users/USER/Desktop/Radar-fyp/My work/Simulations/experiments/integrated_cfar")
ROOT = _P("C:/Users/USER/Desktop/Radar-fyp/My work/Simulations")
E6_RES = _P("C:/Users/USER/Desktop/Radar-fyp/My work/Simulations/experiments/e6_nms_audit") / "results" / "p10_attribution" / "e2"
E6_RES.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from cfar.cfar import CfarConfig                                        # noqa: E402
from cfar.detect import all_peaks                                       # noqa: E402
from dl.infer import cfar_valid_mask                                    # noqa: E402
from dl.stratified_eval import _at_measured_pfa                         # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,               # noqa: E402
                  roc_curve, roc_thresholds)
from radar.config import RadarConfig                                    # noqa: E402

from integrated_cfar import (achieved_pfa, calibrate_alpha, integrated_ratio_map,   # noqa: E402
                             kernel_asymmetry, noise_estimate, psf_kernel,
                             uniform_kernel, verify_denominator_matches_shipped)

SPLIT = "bench2_val"
SHAPE = (256, 64)
NMS = 2          # E6 PATCH
TARGET_PFA = 1e-4
N_BOOT = 2000
KINDS = ("CA", "GO", "OS", "SO")
STATS = ("uniform", "psf")
RES = E6_RES
RES.mkdir(exist_ok=True)

RADAR = RadarConfig()
CFG = CfarConfig()
VALID = cfar_valid_mask(SHAPE, CFG)
RANGE_RES, VEL_RES = 0.19518, 0.42013


# ---------------------------------------------------------------- step 1: the PSF
def measure_psf_patch(half: int = 4, snr_quantile: float = 0.75):
    """Averaged, noise-floor-subtracted, normalised target-centred POWER patch.

    Measured on `isolated` targets only -- one target per frame, so no neighbour contaminates the
    patch -- and on the top-SNR quartile, so the average reflects the response and not the noise.
    """
    frames = sorted((ROOT / "data" / SPLIT / "frames").iterdir())
    iso = [fd for fd in frames
           if json.loads((fd / "metadata.json").read_text())["condition"] == "isolated"]

    patches, snrs, aligned = [], [], 0
    for fd in iso:
        lab = pd.read_csv(fd / "labels.csv")
        if len(lab) != 1:
            continue
        power = np.load(fd / "rd_map_mag.npy").astype(np.float64) ** 2
        r, v = int(lab.range_bin.iloc[0]), int(lab.velocity_bin.iloc[0])
        if not (half <= r < SHAPE[0] - half and half <= v < SHAPE[1] - half):
            continue
        sub = power[r - half:r + half + 1, v - half:v + half + 1]
        pr, pv = np.unravel_index(np.argmax(sub), sub.shape)
        aligned += (pr == half and pv == half)
        patches.append(sub)
        snrs.append(float(lab.cell_snr_db.iloc[0]))

    patches, snrs = np.array(patches), np.array(snrs)
    hi = snrs >= np.quantile(snrs, snr_quantile)

    avg = patches[hi].mean(axis=0)
    avg = np.clip(avg - np.median(avg[[0, -1], :]), 0.0, None)     # subtract the noise floor
    avg /= avg.max()

    meta = dict(n_targets=int(len(patches)), n_used=int(hi.sum()),
                snr_min=float(snrs[hi].min()), snr_max=float(snrs[hi].max()),
                label_is_local_peak_pct=float(100 * aligned / len(patches)),
                label_is_local_peak_pct_hiSNR=float(
                    100 * np.mean([np.unravel_index(np.argmax(p), p.shape) == (half, half)
                                   for p in patches[hi]])))
    return avg, meta


def energy_capture(patch: np.ndarray) -> pd.DataFrame:
    c = patch.shape[0] // 2
    tot = patch.sum()
    rows = []
    for hr in range(0, c + 1):
        for hd in range(0, c + 1):
            reg = patch[c - hr:c + hr + 1, c - hd:c + hd + 1]
            rows.append(dict(half_r=hr, half_d=hd,
                             region=f"{2 * hr + 1}x{2 * hd + 1}", cells=int(reg.size),
                             energy_fraction=float(reg.sum() / tot)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring helpers
def per_frame_hits(frames, thr, mc):
    h, n = [], []
    for pk, tru, shape, valid in frames:
        det = pk[pk[:, 2] >= thr][:, :2] if (len(pk) and thr == thr) else np.zeros((0, 2))
        ev = evaluate_frame(det, tru, shape, mc, valid_mask=valid)
        h.append(ev.tp)
        n.append(len(tru))
    return np.array(h, float), np.array(n, float)


def matched_threshold(frames, mc):
    return _at_measured_pfa(roc_curve(frames, roc_thresholds(frames, mc), mc), TARGET_PFA)[1]


def load_peaks(p: Path) -> dict:
    a = np.load(p)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def save_peaks(pk: dict, path: Path) -> None:
    rows = [np.column_stack([np.full(len(p), i, float), p]) for i, p in sorted(pk.items()) if len(p)]
    np.savez_compressed(path, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))


def main() -> None:
    print("=" * 78)
    print("INTEGRATED CFAR - attribution experiment on", SPLIT)
    print("=" * 78)

    # ---------------- step 1
    patch, pmeta = measure_psf_patch()
    np.save(RES / "psf_patch.npy", patch)
    cap = energy_capture(patch)
    cap.to_csv(RES / "psf_energy_capture.csv", index=False)
    print(f"\nPSF from {pmeta['n_used']} isolated targets "
          f"(cell SNR {pmeta['snr_min']:.1f}-{pmeta['snr_max']:.1f} dB)")
    for _, r in cap[(cap.half_r == cap.half_d) & (cap.half_r <= 2)].iterrows():
        print(f"  {r.region:>5}: {r.cells:2d} cells, {100 * r.energy_fraction:5.1f}% of energy")

    HALF_R = HALF_D = 1                      # 3x3 -- see the energy table; fits inside guard 2/2
    kernels = {"uniform": uniform_kernel(HALF_R, HALF_D),
               "psf": psf_kernel(patch, HALF_R, HALF_D)}
    np.savez(RES / "kernels.npz", **kernels)
    print(f"\nintegration region 3x3 (half {HALF_R},{HALF_D}); guard is "
          f"{CFG.guard_r},{CFG.guard_d} so the numerator stays inside it")
    print(f"raw-patch asymmetry before symmetrising: "
          f"{100 * kernel_asymmetry(patch, HALF_R, HALF_D):.1f}%")
    print("psf kernel (normalised to sum 1):")
    print(pd.DataFrame(kernels["psf"], index=[-1, 0, 1], columns=[-1, 0, 1]).round(4).to_string())

    # ---------------- denominator identity check
    fd0 = sorted((ROOT / "data" / SPLIT / "frames").iterdir())[0]
    p0 = np.load(fd0 / "rd_map_mag.npy").astype(np.float64) ** 2
    print("\ndenominator identical to shipped cfar_ratio_map (max abs diff):")
    for k in KINDS:
        d = verify_denominator_matches_shipped(p0, CfarConfig(kind=k))
        print(f"  {k}: {d:.3e}")
        assert d < 1e-9, f"denominator differs from the shipped one for {k}"

    # ---------------- step 3: calibration
    print("\ncalibrating alpha by Monte Carlo (no closed form applies to a sum of correlated "
          "cells)")
    rows = []
    for k in KINDS:
        for s in STATS:
            cfg = CfarConfig(kind=k)
            a = calibrate_alpha(RADAR, cfg, kernels[s], TARGET_PFA, n_frames=300, seed=0)
            got = achieved_pfa(RADAR, cfg, kernels[s], a, n_frames=200, seed=9000)
            rows.append(dict(kind=k, statistic=s, alpha=a, achieved_pfa_heldout=got,
                             ratio_to_target=got / TARGET_PFA))
            print(f"  {k:<3} {s:<8} alpha {a:8.4f}   held-out Pfa {got:.3e} "
                  f"({got / TARGET_PFA:.2f}x target)")
    CAL = pd.DataFrame(rows)
    CAL.to_csv(RES / "calibration.csv", index=False)

    # ---------------- step 4: detections over the split
    fm = pd.read_csv(ROOT / "results" / "cfar_val" / "frames.csv")
    truths = pd.read_csv(ROOT / "results" / "cfar_val" / "truths.csv")
    TB = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}
    ids = fm.frame.to_numpy()

    print(f"\nextracting integrated detections over {len(ids)} frames x {len(KINDS)} kinds "
          f"x {len(STATS)} statistics")
    peaks = {(k, s): {} for k in KINDS for s in STATS}
    for n, i in enumerate(ids):
        power = np.load(ROOT / "data" / SPLIT / "frames" / f"frame_{int(i):06d}" /
                        "rd_map_mag.npy").astype(np.float64) ** 2
        for k in KINDS:
            cfg = CfarConfig(kind=k)
            noise = noise_estimate(power, cfg)                     # once per kind per frame
            for s in STATS:
                from scipy.ndimage import correlate
                cut = correlate(power, kernels[s], mode="nearest")
                ratio = np.where(VALID, cut / (noise + 1e-30), 0.0)
                peaks[(k, s)][int(i)] = all_peaks(ratio, VALID, nms_size=NMS)
        if (n + 1) % 200 == 0:
            print(f"    {n + 1}/{len(ids)}")

    for (k, s), pk in peaks.items():
        save_peaks(pk, RES / f"peaks_int_{k}_{s}.npz")

    # ---------------- cached baselines (never recomputed, never overwritten)
    CV = ROOT / "results" / "cfar_val"
    R2PK = _P("C:/Users/USER/Desktop/Radar-fyp/My work/Simulations/experiments/e6_nms_audit") / "results" / "final" / "peaks"          # E6 PATCH
    base = {k: load_peaks(R2PK / ("bench2_val_r2_%s.npz" % k)) for k in KINDS}
    dl = [load_peaks(p) for p in sorted(R2PK.glob("bench2_val_r2_DL_s*.npz"))]
    print(f"\ncached baselines: {len(base)} CFAR variants, {len(dl)} DL seeds")

    conds = sorted(c for c in fm.condition.unique() if c != "noise")
    rng = np.random.default_rng(0)
    summary, boots, locs = [], [], []

    for tol in (1, 3):
        mc = MatchConfig(tol, tol)
        for cond in conds:
            cid = fm.loc[fm.condition == cond, "frame"].to_numpy()
            fr_of = (lambda pk: [(pk.get(int(i), np.zeros((0, 3))), TB.get(i, np.zeros((0, 2))),
                                  SHAPE, VALID) for i in cid])

            arms = {}
            for k in KINDS:
                arms[f"CFAR-{k}"] = base[k]
                for s in STATS:
                    arms[f"INT-{s}-{k}"] = peaks[(k, s)]

            hits, n_t = {}, None
            for name, pk in arms.items():
                f = fr_of(pk)
                thr = matched_threshold(f, mc)
                h, n_t = per_frame_hits(f, thr, mc)
                hits[name] = h
                a = aggregate([evaluate_frame(
                    p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2)),
                    t, sh, mc, valid_mask=v) for p, t, sh, v in f],
                    range_res_m=RANGE_RES, vel_res_mps=VEL_RES)
                summary.append(dict(gate=tol, condition=cond, arm=name, threshold=thr,
                                    pd=h.sum() / n_t.sum(), measured_pfa=a["pfa_per_cell"],
                                    fa_per_frame=a["fa_per_frame"]))
                locs.append(dict(gate=tol, condition=cond, arm=name,
                                 dr_bins=a["mean_abs_dr_bins"], dv_bins=a["mean_abs_dv_bins"]))

            dl_h = np.mean([per_frame_hits(fr_of(pk), matched_threshold(fr_of(pk), mc), mc)[0]
                            for pk in dl], axis=0)
            hits["DL"] = dl_h
            summary.append(dict(gate=tol, condition=cond, arm="DL", threshold=np.nan,
                                pd=dl_h.sum() / n_t.sum(), measured_pfa=np.nan,
                                fa_per_frame=np.nan))

            # best-of per family, then the two deltas the experiment exists to estimate
            best_cfar = max(KINDS, key=lambda k: hits[f"CFAR-{k}"].sum())
            k_idx = rng.integers(0, len(cid), size=(N_BOOT, len(cid)))
            tot = n_t[k_idx].sum(axis=1)

            def boot(a, b):
                d = (a[k_idx].sum(axis=1) - b[k_idx].sum(axis=1)) / tot
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                return float(lo), float(hi), float(p)

            for s in STATS:
                best_int = max(KINDS, key=lambda k: hits[f"INT-{s}-{k}"].sum())
                hi_, hc, hd = hits[f"INT-{s}-{best_int}"], hits[f"CFAR-{best_cfar}"], hits["DL"]
                lo1, hi1, p1 = boot(hi_, hc)
                lo2, hi2, p2 = boot(hd, hi_)
                boots.append(dict(
                    gate=tol, condition=cond, statistic=s,
                    best_cfar=best_cfar, best_int=best_int,
                    pd_cfar=hc.sum() / n_t.sum(), pd_int=hi_.sum() / n_t.sum(),
                    pd_dl=hd.sum() / n_t.sum(),
                    delta_integration=(hi_.sum() - hc.sum()) / n_t.sum(),
                    di_lo=lo1, di_hi=hi1, di_p=p1,
                    delta_remaining=(hd.sum() - hi_.sum()) / n_t.sum(),
                    dr_lo=lo2, dr_hi=hi2, dr_p=p2))
            print(f"  gate +/-{tol}  {cond:<14} done")

    pd.DataFrame(summary).to_csv(RES / "summary_pd.csv", index=False)
    pd.DataFrame(locs).to_csv(RES / "localization.csv", index=False)
    B = pd.DataFrame(boots)
    B["alpha_bonf"] = 0.05 / len(conds)
    B["sig_integration"] = B.di_p < B.alpha_bonf
    B.to_csv(RES / "deltas.csv", index=False)

    print("\n" + "=" * 78)
    for s in STATS:
        d = B[(B.gate == 1) & (B.statistic == s)]
        print(f"gate +/-1, {s:<8} mean delta_integration {d.delta_integration.mean():+.4f}, "
              f"significant on {int(d.sig_integration.sum())}/{len(d)} conditions")
    print(f"\nwrote {RES}")


if __name__ == "__main__":
    main()
