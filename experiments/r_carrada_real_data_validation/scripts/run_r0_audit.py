"""R0 — synthetic vs CARRADA domain audit.

READ-ONLY with respect to CARRADA, the simulator, the frozen models and every previous
result. Writes only into experiments/r_carrada_real_data_validation/results/.

Runs NO detector: no TinyCNN inference, no CFAR pass, no threshold, no Pd, no Pfa.

Stages
  1  representation audit          _numpy vs _raw
  2  geometry / coordinate audit
  3  zero-Doppler ridge band selection (deterministic subsample), then ridge strength
  4  background statistics, CARRADA   (sequence-aware aggregation)
  5  background statistics, simulator (condition-aware aggregation, bench2_val only)
  6  frame-median normalization audit
  7  object-support geometry
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(EXP, "src"))
import r0_background as bg  # noqa: E402

CARRADA = r"C:/Users/USER/Desktop/CARRADA/Carrada"
META = os.path.join(ROOT, "data", "carrada_meta")
SIM_SPLIT = "bench2_val"          # development split; bench2_test is NOT touched
RESULTS = os.path.join(EXP, "results")
SEED = 20260831

CLASS_NAMES = {1: "pedestrian", 2: "cyclist", 3: "car"}


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------------------
def load_meta():
    light = json.load(open(os.path.join(META, "light_dataset_frame_oriented.json")))
    fo = json.load(open(os.path.join(META, "annotations_frame_oriented.json")))
    seq_ref = json.load(open(os.path.join(META, "data_seq_ref.json")))
    return light, fo, seq_ref


def carrada_power(seq, frame):
    """Linear power (up to a constant scale) from range_doppler_numpy, which is power in dB."""
    db = np.load("{}/{}/range_doppler_numpy/{}.npy".format(CARRADA, seq, frame))
    return 10.0 ** (db / 10.0), db


def carrada_object_cells(fo, seq, frame):
    cells = []
    for v in fo[seq].get(frame, {}).values():
        cells.extend(tuple(c) for c in v["range_doppler"]["dense"])
    return cells


def sim_frames():
    d = os.path.join(ROOT, "data", SIM_SPLIT, "frames")
    md = pd.read_csv(os.path.join(ROOT, "data", SIM_SPLIT, "metadata.csv"))
    cond = dict(zip(md.frame_id, md.condition))
    return d, sorted(os.listdir(d)), cond


def sim_power(fdir, name):
    mag = np.load(os.path.join(fdir, name, "rd_map_mag.npy"))
    return np.asarray(mag, dtype=np.float64) ** 2


def sim_object_cells(fdir, name):
    lab = np.load(os.path.join(fdir, name, "label_mask.npy"))
    return [tuple(c) for c in np.argwhere(lab > 0)]


# --------------------------------------------------------------------------------------
# 1. representation audit
# --------------------------------------------------------------------------------------
def stage_representation(light, fo, n_per_seq=8):
    log("stage 1: representation audit (_numpy vs _raw)")
    rng = np.random.default_rng(SEED)
    rows = []
    for seq in sorted(light):
        for fr in rng.choice(light[seq], size=min(n_per_seq, len(light[seq])), replace=False):
            fr = str(fr)
            db = np.load("{}/{}/range_doppler_numpy/{}.npy".format(CARRADA, seq, fr))
            raw = np.load("{}/{}/range_doppler_raw/{}.npy".format(CARRADA, seq, fr))
            x = 10.0 * np.log10(raw)
            A = np.vstack([x.ravel(), np.ones(x.size)]).T
            coef, *_ = np.linalg.lstsq(A, db.ravel(), rcond=None)
            resid = db - (coef[0] * x + coef[1])
            # rank correlation between the two arrays (are they the same ordering?)
            ra = np.argsort(np.argsort(db.ravel()))
            rb = np.argsort(np.argsort(raw.ravel()))
            rows.append({
                "sequence": seq, "frame": fr,
                "slope_vs_10log10raw": coef[0], "intercept": coef[1],
                "resid_std_db": float(resid.std()), "resid_absmax_db": float(np.abs(resid).max()),
                "corr_resid_level": float(np.corrcoef(db.ravel(), resid.ravel())[0, 1]),
                "resid_range_profile_std": float(resid.mean(axis=1).std()),
                "resid_doppler_profile_std": float(resid.mean(axis=0).std()),
                "spearman_db_vs_raw": float(np.corrcoef(ra, rb)[0, 1]),
                # power statistics agreement, on the eligible background only
                "cv_from_numpy": np.nan, "cv_from_raw_sq": np.nan, "cv_from_raw": np.nan,
            })
            keep = bg.background_mask(carrada_object_cells(fo, seq, fr), ridge_half=4)
            pn = (10.0 ** (db / 10.0))[keep]
            pr2 = (raw ** 2)[keep]
            pr = raw[keep]
            rows[-1]["cv_from_numpy"] = float(pn.std() / pn.mean())
            rows[-1]["cv_from_raw_sq"] = float(pr2.std() / pr2.mean())
            rows[-1]["cv_from_raw"] = float(pr.std() / pr.mean())
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "r0_representation_frames.csv"), index=False)

    # peak alignment: does the array's max inside an annotated box land in the dense mask?
    rng = np.random.default_rng(SEED + 1)
    hit_n = hit_r = tot = agree = 0
    for seq in sorted(fo):
        ann = [f for f in fo[seq] if len(fo[seq][f]) > 0]
        for fr in rng.choice(ann, size=min(20, len(ann)), replace=False):
            fr = str(fr)
            db = np.load("{}/{}/range_doppler_numpy/{}.npy".format(CARRADA, seq, fr))
            raw = np.load("{}/{}/range_doppler_raw/{}.npy".format(CARRADA, seq, fr))
            for v in fo[seq][fr].values():
                cells = np.array(v["range_doppler"]["dense"])
                if len(cells) == 0:
                    continue
                r0, r1 = cells[:, 0].min(), cells[:, 0].max()
                d0, d1 = cells[:, 1].min(), cells[:, 1].max()
                sn, sr = db[r0:r1 + 1, d0:d1 + 1], raw[r0:r1 + 1, d0:d1 + 1]
                if sn.size == 0:
                    continue
                pn = np.unravel_index(np.argmax(sn), sn.shape)
                pr = np.unravel_index(np.argmax(sr), sr.shape)
                mask = set(map(tuple, cells.tolist()))
                hit_n += (r0 + pn[0], d0 + pn[1]) in mask
                hit_r += (r0 + pr[0], d0 + pr[1]) in mask
                agree += int(pn == pr)
                tot += 1

    summary = {
        "frames_analysed": int(len(df)),
        "sequences": int(df.sequence.nunique()),
        "slope_vs_10log10raw_mean": float(df.slope_vs_10log10raw.mean()),
        "slope_vs_10log10raw_sd": float(df.slope_vs_10log10raw.std()),
        "intercept_mean": float(df.intercept.mean()),
        "intercept_sd": float(df.intercept.std()),
        "resid_std_db_mean": float(df.resid_std_db.mean()),
        "resid_absmax_db_mean": float(df.resid_absmax_db.mean()),
        "corr_resid_level_mean": float(df.corr_resid_level.mean()),
        "resid_range_profile_std_mean": float(df.resid_range_profile_std.mean()),
        "resid_doppler_profile_std_mean": float(df.resid_doppler_profile_std.mean()),
        "spearman_db_vs_raw_mean": float(df.spearman_db_vs_raw.mean()),
        "cv_from_numpy_mean": float(df.cv_from_numpy.mean()),
        "cv_from_raw_sq_mean": float(df.cv_from_raw_sq.mean()),
        "cv_from_raw_mean": float(df.cv_from_raw.mean()),
        "peak_alignment_instances": int(tot),
        "peak_in_mask_numpy": hit_n / tot if tot else np.nan,
        "peak_in_mask_raw": hit_r / tot if tot else np.nan,
        "peak_cell_agreement": agree / tot if tot else np.nan,
    }
    s = summary
    s["cv_pct_gap_numpy_vs_rawsq"] = abs(s["cv_from_numpy_mean"] - s["cv_from_raw_sq_mean"]) \
        / s["cv_from_raw_sq_mean"] * 100
    s["cv_pct_gap_numpy_vs_raw"] = abs(s["cv_from_numpy_mean"] - s["cv_from_raw_mean"]) \
        / s["cv_from_raw_mean"] * 100
    return summary


# --------------------------------------------------------------------------------------
# 2. geometry audit
# --------------------------------------------------------------------------------------
def stage_geometry(light, fo):
    log("stage 2: geometry / coordinate audit")
    ax = np.load(os.path.join(ROOT, "data", SIM_SPLIT, "axes.npz"))
    rng_axis, vel_axis = ax["range_axis"], ax["velocity_axis"]
    zero_bin = int(np.argmin(np.abs(vel_axis)))

    shapes = set()
    for seq in sorted(light):
        for fr in light[seq][:3]:
            shapes.add(np.load("{}/{}/range_doppler_numpy/{}.npy".format(
                CARRADA, seq, fr)).shape)

    # annotation coordinate ranges: do they index (range, Doppler) of a 256x64 array?
    rmin = dmin = 10 ** 9
    rmax = dmax = -10 ** 9
    n_cells = 0
    for seq in sorted(fo):
        for fr in fo[seq]:
            for v in fo[seq][fr].values():
                c = np.array(v["range_doppler"]["dense"])
                if len(c) == 0:
                    continue
                rmin, rmax = min(rmin, c[:, 0].min()), max(rmax, c[:, 0].max())
                dmin, dmax = min(dmin, c[:, 1].min()), max(dmax, c[:, 1].max())
                n_cells += len(c)

    return {
        "carrada_rd_shapes_seen": sorted(str(s) for s in shapes),
        "simulator_rd_shape": [int(len(rng_axis)), int(len(vel_axis))],
        "simulator_range_axis_m": [float(rng_axis[0]), float(rng_axis[-1])],
        "simulator_range_bin_m": float(rng_axis[1] - rng_axis[0]),
        "simulator_velocity_axis_mps": [float(vel_axis[0]), float(vel_axis[-1])],
        "simulator_velocity_bin_mps": float(vel_axis[1] - vel_axis[0]),
        "simulator_zero_doppler_bin": zero_bin,
        "assumed_zero_doppler_bin": bg.ZERO_DOPPLER_BIN,
        "annotation_range_bin_min": int(rmin), "annotation_range_bin_max": int(rmax),
        "annotation_doppler_bin_min": int(dmin), "annotation_doppler_bin_max": int(dmax),
        "annotation_cells_total": int(n_cells),
        "annotation_within_bounds": bool(rmin >= 0 and rmax < 256 and dmin >= 0 and dmax < 64),
        "resampling_required": False,
    }


# --------------------------------------------------------------------------------------
# 3. ridge band selection + ridge strength
# --------------------------------------------------------------------------------------
def _profile_to_band(prof, excess_db=3.0):
    """Doppler half-width of the CONTIGUOUS run around zero Doppler that exceeds the
    off-ridge floor by more than `excess_db`.

    Grows outward one bin at a time and stops at the first offset where NEITHER edge bin is
    still above threshold. An earlier version asked whether *any* bin in the window was above
    threshold, which the centre bin always satisfies, so the band grew to the loop bound every
    time. That version is superseded and its results are not retained.
    """
    z = bg.ZERO_DOPPLER_BIN
    off = np.concatenate([prof[:max(0, z - 8)], prof[min(len(prof), z + 9):]])
    floor = np.nanmedian(off)
    with np.errstate(invalid="ignore", divide="ignore"):
        over = 10 * np.log10(prof / floor) > excess_db
    half = 0
    for h in range(1, 16):
        lo, hi = z - h, z + h
        lo_over = bool(over[lo]) if 0 <= lo < len(prof) else False
        hi_over = bool(over[hi]) if 0 <= hi < len(prof) else False
        if lo_over or hi_over:
            half = h
        else:
            break
    return half, float(floor)


def _peak_and_plateau(prof, floor, tol_db=0.5):
    """Peak bin plus every bin within tol_db of the peak, so a flat top is visible rather than
    being reported as a spurious offset from zero Doppler."""
    d = 10 * np.log10(prof / floor)
    pk = int(np.nanargmax(d))
    plateau = [int(i) for i in np.where(np.isfinite(d) & (d >= np.nanmax(d) - tol_db))[0]]
    return pk, plateau, float(np.nanmax(d))


def stage_ridge(light, fo, n_per_seq=20, n_sim=400):
    log("stage 3: zero-Doppler ridge band selection")
    rng = np.random.default_rng(SEED + 2)
    profs_c = []
    for seq in sorted(light):
        for fr in rng.choice(light[seq], size=min(n_per_seq, len(light[seq])), replace=False):
            fr = str(fr)
            p, _ = carrada_power(seq, fr)
            profs_c.append(bg.ridge_profile(p, carrada_object_cells(fo, seq, fr)))
    prof_c = np.nanmean(np.array(profs_c), axis=0)

    fdir, names, _ = sim_frames()
    sel = rng.choice(names, size=min(n_sim, len(names)), replace=False)
    profs_s = [bg.ridge_profile(sim_power(fdir, n), sim_object_cells(fdir, n)) for n in sel]
    prof_s = np.nanmean(np.array(profs_s), axis=0)

    half_c, floor_c = _profile_to_band(prof_c)
    half_s, floor_s = _profile_to_band(prof_s)
    common = int(max(half_c, half_s))

    pk_c, plat_c, top_c = _peak_and_plateau(prof_c, floor_c)
    pk_s, plat_s, top_s = _peak_and_plateau(prof_s, floor_s)
    out = {
        "band_selection_frames_carrada": len(profs_c),
        "band_selection_frames_simulator": len(profs_s),
        "excess_threshold_db": 3.0,
        "ridge_half_width_carrada": int(half_c),
        "ridge_half_width_simulator": int(half_s),
        "ridge_half_width_common_applied": common,
        "carrada_ridge_peak_bin": pk_c,
        "carrada_ridge_plateau_bins": plat_c,
        "simulator_ridge_peak_bin": pk_s,
        "simulator_ridge_plateau_bins": plat_s,
        "plateau_note": ("peak_bin is argmax and can land anywhere on a flat top; "
                         "plateau_bins lists every bin within 0.5 dB of the peak, which is "
                         "what shows the ridge is actually centred on the zero-Doppler bin"),
        "carrada_ridge_peak_over_floor_db": top_c,
        "simulator_ridge_peak_over_floor_db": top_s,
    }
    np.savez(os.path.join(RESULTS, "r0_ridge_profiles.npz"),
             carrada=prof_c, simulator=prof_s,
             carrada_floor=floor_c, simulator_floor=floor_s)
    return out, common


# --------------------------------------------------------------------------------------
# 4/5. background statistics
# --------------------------------------------------------------------------------------
def stage_background_carrada(light, fo, ridge_half, seq_ref):
    log("stage 4: CARRADA background statistics (all {} frames)".format(
        sum(len(v) for v in light.values())))
    rows = []
    for seq in sorted(light):
        for fr in light[seq]:
            p, db = carrada_power(seq, fr)
            cells = carrada_object_cells(fo, seq, fr)
            keep = bg.background_mask(cells, ridge_half)
            st = bg.frame_stats(p, keep)
            if not st:
                continue
            st.update(sequence=seq, frame=fr, split=seq_ref[seq]["split"],
                      n_objects=len(fo[seq].get(fr, {})),
                      frame_median_db=float(np.median(db)),
                      frame_median_db_bg=float(np.median(db[keep])))
            keep_r = bg.background_mask(cells, ridge_half, exclude_ridge=False)
            pr = p[keep_r]
            st["cv_ridge_included"] = float(pr.std() / pr.mean())
            rows.append(st)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "r0_background_carrada_frames.csv"), index=False)
    return df


def stage_background_simulator(ridge_half):
    fdir, names, cond = sim_frames()
    log("stage 5: simulator background statistics ({} frames, {})".format(len(names), SIM_SPLIT))
    rows = []
    for n in names:
        p = sim_power(fdir, n)
        cells = sim_object_cells(fdir, n)
        keep = bg.background_mask(cells, ridge_half)
        st = bg.frame_stats(p, keep)
        if not st:
            continue
        db = 10 * np.log10(np.maximum(p, 1e-300))
        st.update(frame=n, condition=cond.get(n, "unknown"),
                  frame_median_db=float(np.median(db)),
                  frame_median_db_bg=float(np.median(db[keep])))
        keep_r = bg.background_mask(cells, ridge_half, exclude_ridge=False)
        pr = p[keep_r]
        st["cv_ridge_included"] = float(pr.std() / pr.mean())
        rows.append(st)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "r0_background_simulator_frames.csv"), index=False)
    return df


def aggregate(df, group_col, metrics):
    """Sequence-aware (or condition-aware) two-level summary.

    Frames within a sequence are correlated, so uncertainty is taken ACROSS groups, never
    across frames. n is the number of groups.

    Within a group the frames are combined by MEDIAN, not mean. Reason, measured rather than
    assumed: the per-frame CV distribution is heavy-tailed (CARRADA median 0.49, 99th pct 6.8,
    max 41), because a minority of frames contain very bright UNANNOTATED reflectors -- only
    the 77 tracked objects carry annotations, so the object-exclusion mask cannot remove them.
    A mean over that distribution is dominated by the tail, which breaks the exact per-frame
    identity k = 1/CV^2 at the summary level (mean CV said 1.00 while mean k said 3.97; the
    medians agree exactly, 0.4948 <-> 4.08). The mean-based summary is retained beside it as
    `*_meanagg` so the difference is visible rather than hidden.
    """
    out = {}
    per_group = df.groupby(group_col)[metrics].median()
    per_group_mean = df.groupby(group_col)[metrics].mean()
    for m in metrics:
        v = per_group[m].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        w = per_group_mean[m].to_numpy(dtype=float)
        w = w[np.isfinite(w)]
        out[m] = {"mean": float(v.mean()),
                  "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
                  "n_groups": int(len(v)),
                  "group_min": float(v.min()), "group_max": float(v.max()),
                  "meanagg_mean": float(w.mean()),
                  "meanagg_se": float(w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 1
                  else float("nan")}
    out["_n_frames"] = int(len(df))
    out["_within_group_combiner"] = "median"
    # heavy-tail diagnostic: how often is a "background" frame contaminated?
    if "cv" in df:
        out["_cv_tail"] = {"frac_cv_gt_2": float((df.cv > 2).mean()),
                           "frac_cv_gt_5": float((df.cv > 5).mean()),
                           "frac_cv_gt_10": float((df.cv > 10).mean()),
                           "cv_median": float(df.cv.median()),
                           "cv_p99": float(df.cv.quantile(0.99)),
                           "cv_max": float(df.cv.max())}
    return out, per_group


# --------------------------------------------------------------------------------------
# 6. Gamma shape by log-moment, on a pooled deterministic subsample
# --------------------------------------------------------------------------------------
def stage_logmoment(light, fo, ridge_half, n_per_seq=20, n_sim=400):
    log("stage 6: Gamma shape by log-moment estimator")
    rng = np.random.default_rng(SEED + 3)
    kc = []
    for seq in sorted(light):
        vals = []
        for fr in rng.choice(light[seq], size=min(n_per_seq, len(light[seq])), replace=False):
            fr = str(fr)
            p, _ = carrada_power(seq, fr)
            keep = bg.background_mask(carrada_object_cells(fo, seq, fr), ridge_half)
            vals.append(bg.gamma_k_logmoment(p[keep]))
        kc.append(np.nanmean(vals))
    fdir, names, _ = sim_frames()
    sel = rng.choice(names, size=min(n_sim, len(names)), replace=False)
    ks = [bg.gamma_k_logmoment(sim_power(fdir, n)[bg.background_mask(
        sim_object_cells(fdir, n), ridge_half)]) for n in sel]
    kc, ks = np.array(kc, float), np.array(ks, float)
    return {
        "carrada_k_logmoment_mean": float(np.nanmean(kc)),
        "carrada_k_logmoment_se": float(np.nanstd(kc, ddof=1) / np.sqrt(np.isfinite(kc).sum())),
        "carrada_k_logmoment_n_sequences": int(np.isfinite(kc).sum()),
        "simulator_k_logmoment_mean": float(np.nanmean(ks)),
        "simulator_k_logmoment_se": float(np.nanstd(ks, ddof=1) / np.sqrt(np.isfinite(ks).sum())),
        "simulator_k_logmoment_n_frames": int(np.isfinite(ks).sum()),
    }


# --------------------------------------------------------------------------------------
# 6b. range-block decomposition
# --------------------------------------------------------------------------------------
RANGE_BLOCKS = [(10, 60), (60, 110), (110, 160), (160, 210), (210, 246), (200, 246)]


def stage_range_blocks(light, fo, ridge_half, n_per_seq=10, n_sim=300):
    """Background statistics as a function of range block.

    The preliminary (superseded) pass measured only far range, bins 200-255. This stage exists
    so the difference between that pass and the full-map R0 numbers is explained by measurement
    rather than asserted. The last block (200, 246) is the closest reproducible analogue of the
    preliminary region, clipped to the detector border.
    """
    log("stage 6b: range-block decomposition")
    rng = np.random.default_rng(SEED + 4)
    rows = []
    for seq in sorted(light):
        for fr in rng.choice(light[seq], size=min(n_per_seq, len(light[seq])), replace=False):
            fr = str(fr)
            p, _ = carrada_power(seq, fr)
            keep0 = bg.background_mask(carrada_object_cells(fo, seq, fr), ridge_half)
            for r0, r1 in RANGE_BLOCKS:
                k = np.zeros_like(keep0); k[r0:r1] = keep0[r0:r1]
                st = bg.frame_stats(p, k)
                if st:
                    st.update(domain="CARRADA", group=seq, block="{}-{}".format(r0, r1))
                    rows.append(st)
    fdir, names, cond = sim_frames()
    for n in rng.choice(names, size=min(n_sim, len(names)), replace=False):
        p = sim_power(fdir, n)
        keep0 = bg.background_mask(sim_object_cells(fdir, n), ridge_half)
        for r0, r1 in RANGE_BLOCKS:
            k = np.zeros_like(keep0); k[r0:r1] = keep0[r0:r1]
            st = bg.frame_stats(p, k)
            if st:
                st.update(domain="simulator", group=cond.get(n, "?"),
                          block="{}-{}".format(r0, r1))
                rows.append(st)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "r0_range_blocks.csv"), index=False)
    summ = {}
    for (dom, blk), g_ in df.groupby(["domain", "block"]):
        per = g_.groupby("group")[["cv", "gamma_k_moment", "db_std", "lag1_range"]].mean()
        summ.setdefault(dom, {})[blk] = {
            c: {"mean": float(per[c].mean()),
                "se": float(per[c].std(ddof=1) / np.sqrt(len(per)))}
            for c in per.columns}
    return summ


# --------------------------------------------------------------------------------------
# 7. object support geometry
# --------------------------------------------------------------------------------------
def stage_footprint(fo):
    log("stage 7: object-support geometry")
    rows = []
    for seq in sorted(fo):
        for fr in fo[seq]:
            for ik, v in fo[seq][fr].items():
                rd = v["range_doppler"]
                dense = np.array(rd["dense"])
                sparse = np.array(rd["sparse"])
                if len(dense) == 0:
                    continue
                box = rd["box"]
                rows.append({
                    "sequence": seq, "frame": fr, "instance": ik,
                    "label": rd["label"], "class": CLASS_NAMES.get(rd["label"], "?"),
                    "range_extent": int(dense[:, 0].max() - dense[:, 0].min() + 1),
                    "doppler_extent": int(dense[:, 1].max() - dense[:, 1].min() + 1),
                    "box_range_extent": int(box[1][0] - box[0][0] + 1),
                    "box_doppler_extent": int(box[1][1] - box[0][1] + 1),
                    "dense_cells": int(len(dense)),
                    "sparse_cells": int(len(sparse)),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "r0_object_footprint.csv"), index=False)
    summ = {}
    for cls, g in df.groupby("class"):
        summ[cls] = {"object_frames": int(len(g))}
        for col in ("range_extent", "doppler_extent", "box_range_extent",
                    "box_doppler_extent", "dense_cells", "sparse_cells"):
            summ[cls][col] = {"median": float(g[col].median()),
                              "p90": float(g[col].quantile(0.90)),
                              "max": int(g[col].max())}
    summ["ALL"] = {"object_frames": int(len(df))}
    return summ, df


# --------------------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    light, fo, seq_ref = load_meta()
    report = {"seed": SEED, "simulator_split": SIM_SPLIT,
              "background_protocol": {
                  "border_range_bins": bg.BORDER_R, "border_doppler_bins": bg.BORDER_D,
                  "object_dilation_bins": bg.OBJECT_DILATION,
                  "percentile_trimming": "NONE - never applied",
                  "zero_doppler_bin": bg.ZERO_DOPPLER_BIN}}

    report["representation"] = stage_representation(light, fo)
    report["geometry"] = stage_geometry(light, fo)
    ridge, ridge_half = stage_ridge(light, fo)
    report["ridge"] = ridge
    report["background_protocol"]["ridge_half_width_applied"] = ridge_half

    dfc = stage_background_carrada(light, fo, ridge_half, seq_ref)
    dfs = stage_background_simulator(ridge_half)

    metrics = ["cv", "gamma_k_moment", "db_std", "lag1_range", "lag1_doppler",
               "frame_median_db", "cv_ridge_included"]
    agg_c, per_seq = aggregate(dfc, "sequence", metrics)
    agg_s, per_cond = aggregate(dfs, "condition", metrics)
    per_seq.to_csv(os.path.join(RESULTS, "r0_background_carrada_per_sequence.csv"))
    per_cond.to_csv(os.path.join(RESULTS, "r0_background_simulator_per_condition.csv"))
    report["background_carrada"] = agg_c
    report["background_simulator"] = agg_s
    report["gamma_logmoment"] = stage_logmoment(light, fo, ridge_half)
    report["range_blocks"] = stage_range_blocks(light, fo, ridge_half)

    # frame-median normalization audit: spread of the per-frame median, in dB
    report["frame_median_audit"] = {
        "carrada_frame_median_db_sd_within_sequence": float(
            dfc.groupby("sequence").frame_median_db.std().mean()),
        "carrada_frame_median_db_sd_across_all": float(dfc.frame_median_db.std()),
        "carrada_frame_median_db_range": [float(dfc.frame_median_db.min()),
                                          float(dfc.frame_median_db.max())],
        "simulator_frame_median_db_sd_across_all": float(dfs.frame_median_db.std()),
        "simulator_frame_median_db_range": [float(dfs.frame_median_db.min()),
                                            float(dfs.frame_median_db.max())],
        "note": ("TinyCNN input is 10*log10(P/median(P)) = dB - median(dB); because dB is a "
                 "monotone transform of power, median(P) = 10**(median(dB)/10) EXACTLY, so the "
                 "adapter is exact in the dB domain and the unknown CARRADA scale constant "
                 "cancels identically."),
    }

    fp, dffp = stage_footprint(fo)
    report["object_footprint"] = fp
    report["elapsed_s"] = round(time.time() - t0, 1)

    with open(os.path.join(RESULTS, "r0_audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    log("wrote r0_audit.json  ({:.1f} s total)".format(report["elapsed_s"]))

    # console summary
    print("\n" + "=" * 78)
    print("R0 HEADLINE COMPARISON  (ridge excluded, NO trimming, sequence-aware SE)")
    print("=" * 78)
    print("{:26s} {:>20s} {:>20s}".format("statistic", "CARRADA", "simulator"))
    label = {"cv": "CV(power)", "gamma_k_moment": "Gamma shape k (moment)",
             "db_std": "background std [dB]", "lag1_range": "lag-1 range corr",
             "lag1_doppler": "lag-1 Doppler corr"}
    for m in ["cv", "gamma_k_moment", "db_std", "lag1_range", "lag1_doppler"]:
        a, b = agg_c[m], agg_s[m]
        print("{:26s} {:11.4f} +-{:.4f} {:11.4f} +-{:.4f}".format(
            label[m], a["mean"], a["se"], b["mean"], b["se"]))
    print("{:26s} {:11d}       {:11d}".format(
        "aggregation groups (n)", agg_c["cv"]["n_groups"], agg_s["cv"]["n_groups"]))
    print("{:26s} {:11d}       {:11d}".format(
        "frames used", agg_c["_n_frames"], agg_s["_n_frames"]))
    print("{:26s} {:11.1f}       {:11.1f}".format(
        "zero-Doppler ridge [dB]", ridge["carrada_ridge_peak_over_floor_db"],
        ridge["simulator_ridge_peak_over_floor_db"]))
    print("\nconsistency check  k == 1/CV^2 at the summary level:")
    for tag, a in (("CARRADA", agg_c), ("simulator", agg_s)):
        print("   {:10s} k = {:.3f}   1/CV^2 = {:.3f}   (agree to {:.1%})".format(
            tag, a["gamma_k_moment"]["mean"], 1.0 / a["cv"]["mean"] ** 2,
            abs(a["gamma_k_moment"]["mean"] - 1.0 / a["cv"]["mean"] ** 2)
            / a["gamma_k_moment"]["mean"]))
    print("\nheavy-tail diagnostic (frames whose background holds a bright unannotated return):")
    for tag, a in (("CARRADA", agg_c), ("simulator", agg_s)):
        t = a["_cv_tail"]
        print("   {:10s} CV>2 {:6.2%}   CV>5 {:6.2%}   CV>10 {:6.2%}   median {:.3f}  max {:.1f}"
              .format(tag, t["frac_cv_gt_2"], t["frac_cv_gt_5"], t["frac_cv_gt_10"],
                      t["cv_median"], t["cv_max"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
