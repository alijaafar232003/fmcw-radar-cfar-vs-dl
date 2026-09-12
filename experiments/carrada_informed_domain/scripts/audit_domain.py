"""Domain audit -- ONE implementation, run identically on every domain we compare.

Phases 2, 3 and 5 of the CARRADA-informed domain intervention all need the same measurement:
given a source of range--Doppler frames, characterise its background/clutter statistics. Running
three different scripts on three domains is how domain "gaps" become artefacts of the measuring
code, so there is exactly one script and the domain is an argument.

    python audit_domain.py --domain carrada_train
    python audit_domain.py --domain sim --split bench2_train
    python audit_domain.py --domain sim --split bench2_val
    python audit_domain.py --domain newsim --root <dir>        (Phase 5)

The background-selection protocol is NOT reimplemented. It is imported from
`experiments/r_carrada_real_data_validation/src/r0_background.py`, the module Chapter 7's R0 audit
used, so these numbers are commensurable with the published domain audit:

  * annotated/labelled object support, dilated by 3 bins, excluded;
  * zero-Doppler band of half-width `--ridge-half` (default 4, R0's common band) excluded for the
    ordinary background statistics, and measured separately by `ridge_profile`;
  * outer 10 range / 4 Doppler bins excluded (the 21x9 CFAR-testable interior);
  * NO percentile trimming, ever.

DATA DISCIPLINE. With `--domain carrada_train` this script may open ONLY the 18 official CARRADA
Train sequences. A path guard enforces it: every CARRADA read goes through `_guard()`, which raises
on any sequence not in the Train list, and every sequence actually opened is recorded in the output
JSON so the claim is checkable rather than asserted. CARRADA Validation and Test are never listed,
never globbed and never read.

Runs NO detector: no TinyCNN, no CFAR, no threshold, no Pd, no recall.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
sys.path.insert(0, os.path.join(R_EXP, "src"))

import r0_background as bg                                              # noqa: E402

CARRADA = r"C:/Users/USER/Desktop/CARRADA/Carrada"
META = os.path.join(ROOT, "data", "carrada_meta")
OUT = os.path.join(EXP, "domain_audit")

# Frozen Chapter 7 strata, imported by value so this audit and the eventual evaluation agree.
RANGE_BLOCKS = [(10, 60), (60, 110), (110, 160), (160, 210), (210, 246)]
RIDGE_HALF_DEFAULT = 4            # R0's "common applied" band
RD_SHAPE = (256, 64)


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# ------------------------------------------------------------------ CARRADA access, guarded
class Guard:
    """Allows reads only from an explicit sequence allow-list, and records what was opened."""

    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.opened = set()

    def __call__(self, seq):
        if seq not in self.allowed:
            raise PermissionError(
                "BLOCKED: sequence %r is not in the allowed split. This audit may read only the "
                "%d allowed sequences; CARRADA Validation and Test are out of bounds."
                % (seq, len(self.allowed)))
        self.opened.add(seq)
        return seq


def carrada_power(guard, seq, frame):
    """Linear power (up to a constant scale) from range_doppler_numpy, which is power in dB."""
    guard(seq)
    db = np.load("%s/%s/range_doppler_numpy/%s.npy" % (CARRADA, seq, frame))
    return 10.0 ** (db / 10.0), db


def carrada_object_cells(fo, seq, frame):
    cells = []
    for v in fo[seq].get(frame, {}).values():
        cells.extend(tuple(c) for c in v["range_doppler"]["dense"])
    return cells


# ------------------------------------------------------------------ simulator access
def sim_power(fdir, name):
    mag = np.load(os.path.join(fdir, name, "rd_map_mag.npy"))
    return np.asarray(mag, dtype=np.float64) ** 2


def sim_object_cells(fdir, name):
    p = os.path.join(fdir, name, "label_mask.npy")
    if not os.path.exists(p):
        return []
    return [tuple(c) for c in np.argwhere(np.load(p) > 0)]


# ------------------------------------------------------------------ extra measurements
def ridge_metrics(prof, ridge_half_probe=10):
    """Strength and width of the zero-Doppler ridge from one mean-power Doppler profile.

    `prof` is mean background power per Doppler bin (ridge NOT excluded). The floor is the median
    of the bins outside a generous probe band, so a wide ridge cannot raise its own reference.
    """
    z = bg.ZERO_DOPPLER_BIN
    n = len(prof)
    outside = np.ones(n, dtype=bool)
    lo, hi = max(0, z - ridge_half_probe), min(n, z + ridge_half_probe + 1)
    outside[lo:hi] = False
    ok = outside & np.isfinite(prof)
    if ok.sum() < 8 or not np.isfinite(prof[z]):
        return {}
    floor = float(np.median(prof[ok]))
    if floor <= 0:
        return {}
    peak = float(np.nanmax(prof[lo:hi]))
    excess_db = 10.0 * np.log10(prof / floor)

    # Width: number of Doppler bins whose excess exceeds 3 dB, and the half-power (-3 dB from
    # peak) width. Both are reported because a strong narrow ridge and a weak wide one can share
    # either measure alone.
    above3 = int(np.nansum(excess_db > 3.0))
    peak_db = 10.0 * np.log10(peak / floor)
    half = int(np.nansum(excess_db > (peak_db - 3.0)))
    return {
        "ridge_peak_over_floor_db": peak_db,
        "ridge_bins_above_3db": above3,
        "ridge_halfpower_bins": half,
        "ridge_floor_power": floor,
    }


def range_block_stats(power, keep):
    """Background statistics inside each frozen range block."""
    out = {}
    for r0, r1 in RANGE_BLOCKS:
        k = np.zeros_like(keep)
        k[r0:r1, :] = keep[r0:r1, :]
        st = bg.frame_stats(power, k)
        if st:
            out["%d-%d" % (r0, r1)] = st
    return out


# ------------------------------------------------------------------ aggregation
def agg(df, cols, groupcol):
    """Group-level then across-group summary, so sequences are the unit rather than frames."""
    res = {}
    for c in cols:
        if c not in df:
            continue
        g = df.groupby(groupcol)[c].mean().dropna()
        if not len(g):
            continue
        res[c] = {
            "mean": float(g.mean()),
            "sd_across_groups": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
            "se": float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0,
            "group_min": float(g.min()),
            "group_max": float(g.max()),
            "p05_frames": float(df[c].quantile(0.05)),
            "p95_frames": float(df[c].quantile(0.95)),
            "n_groups": int(len(g)),
            "n_frames": int(df[c].notna().sum()),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=("carrada_train", "sim", "newsim"))
    ap.add_argument("--split", default="bench2_train", help="simulator split name (--domain sim)")
    ap.add_argument("--root", default=None, help="frames root (--domain newsim)")
    ap.add_argument("--tag", default=None, help="output basename; defaults from the domain")
    ap.add_argument("--ridge-half", type=int, default=RIDGE_HALF_DEFAULT)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=20260912)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    tag = a.tag or (a.domain if a.domain != "sim" else "sim_" + a.split)
    t0 = time.perf_counter()
    rows, ridge_rows, block_rows = [], [], []
    guard = None

    if a.domain == "carrada_train":
        splits = json.load(open(os.path.join(
            R_EXP, "configs", "official_splits.json")))["splits"]
        allowed = splits["Train"]
        assert len(allowed) == 18, "expected 18 official Train sequences, got %d" % len(allowed)
        # Never even name the other splits below this point.
        forbidden = set(splits["Validation"]) | set(splits["Test"])
        assert not (set(allowed) & forbidden), "split overlap"
        guard = Guard(allowed)

        light = json.load(open(os.path.join(META, "light_dataset_frame_oriented.json")))
        fo = json.load(open(os.path.join(META, "annotations_frame_oriented.json")))
        log("CARRADA TRAIN: %d sequences, %d annotated frames"
            % (len(allowed), sum(len(light.get(s, {})) for s in allowed)))

        for seq in sorted(allowed):
            frames = sorted(light.get(seq, {}))
            if a.max_frames:
                frames = frames[:a.max_frames]
            for fr in frames:
                p, db = carrada_power(guard, seq, fr)
                cells = carrada_object_cells(fo, seq, fr)
                keep = bg.background_mask(cells, a.ridge_half)
                st = bg.frame_stats(p, keep)
                if not st:
                    continue
                st.update(group=seq, frame=fr,
                          frame_median_db=float(np.median(db)),
                          frame_median_db_bg=float(np.median(db[keep])))
                rows.append(st)

                prof = bg.ridge_profile(p, cells)
                rm = ridge_metrics(prof)
                if rm:
                    rm.update(group=seq, frame=fr)
                    ridge_rows.append(rm)

                for blk, bst in range_block_stats(p, keep).items():
                    bst.update(group=seq, frame=fr, block=blk)
                    block_rows.append(bst)
            log("  %s done (%d frames so far)" % (seq, len(rows)))
    else:
        if a.domain == "sim":
            fdir = os.path.join(ROOT, "data", a.split, "frames")
            md = pd.read_csv(os.path.join(ROOT, "data", a.split, "metadata.csv"))
            cond = dict(zip(md.frame_id, md.condition))
        else:
            fdir = os.path.join(a.root, "frames")
            mdp = os.path.join(a.root, "metadata.csv")
            cond = {}
            if os.path.exists(mdp):
                md = pd.read_csv(mdp)
                cond = dict(zip(md.frame_id, md.condition))
        names = sorted(os.listdir(fdir))
        if a.max_frames:
            rng = np.random.default_rng(a.seed)
            names = [names[i] for i in sorted(
                rng.choice(len(names), size=min(a.max_frames, len(names)), replace=False))]
        log("%s: %d frames from %s" % (a.domain, len(names), fdir))

        for i, nm in enumerate(names):
            p = sim_power(fdir, nm)
            cells = sim_object_cells(fdir, nm)
            keep = bg.background_mask(cells, a.ridge_half)
            st = bg.frame_stats(p, keep)
            if not st:
                continue
            db = 10.0 * np.log10(np.maximum(p, 1e-300))
            st.update(group=cond.get(nm, "all"), frame=nm,
                      frame_median_db=float(np.median(db)),
                      frame_median_db_bg=float(np.median(db[keep])))
            rows.append(st)

            prof = bg.ridge_profile(p, cells)
            rm = ridge_metrics(prof)
            if rm:
                rm.update(group=cond.get(nm, "all"), frame=nm)
                ridge_rows.append(rm)

            for blk, bst in range_block_stats(p, keep).items():
                bst.update(group=cond.get(nm, "all"), frame=nm, block=blk)
                block_rows.append(bst)
            if (i + 1) % 400 == 0:
                log("  %d/%d" % (i + 1, len(names)))

    df = pd.DataFrame(rows)
    rdf = pd.DataFrame(ridge_rows)
    bdf = pd.DataFrame(block_rows)
    df.to_csv(os.path.join(OUT, "%s_frames.csv" % tag), index=False)
    rdf.to_csv(os.path.join(OUT, "%s_ridge.csv" % tag), index=False)
    bdf.to_csv(os.path.join(OUT, "%s_blocks.csv" % tag), index=False)

    # Pooled log-moment Gamma shape: a second estimator on a pooled background sample, so the
    # moment estimate is not trusted alone. Sampled deterministically across groups.
    CORE = ["cv", "gamma_k_moment", "db_std", "lag1_range", "lag1_doppler",
            "mean_power", "median_power", "frame_median_db", "frame_median_db_bg"]
    RIDGE = ["ridge_peak_over_floor_db", "ridge_bins_above_3db", "ridge_halfpower_bins"]

    summary = {
        "domain": a.domain,
        "tag": tag,
        "split": a.split if a.domain == "sim" else None,
        "root": a.root if a.domain == "newsim" else None,
        "ridge_half_excluded": a.ridge_half,
        "background_protocol": {
            "module": "experiments/r_carrada_real_data_validation/src/r0_background.py",
            "border_range_bins": bg.BORDER_R,
            "border_doppler_bins": bg.BORDER_D,
            "object_dilation_bins": bg.OBJECT_DILATION,
            "zero_doppler_bin": bg.ZERO_DOPPLER_BIN,
            "percentile_trimming": "NONE - never applied",
        },
        "n_frames": int(len(df)),
        "n_groups": int(df.group.nunique()) if len(df) else 0,
        "groups_opened": sorted(guard.opened) if guard else sorted(df.group.unique().tolist()),
        "background": agg(df, CORE, "group"),
        "ridge": agg(rdf, RIDGE, "group") if len(rdf) else {},
        "range_blocks": {},
        "per_group": {},
        "runtime_s": None,
    }

    for blk in ["%d-%d" % b for b in RANGE_BLOCKS]:
        sub = bdf[bdf.block == blk] if len(bdf) else bdf
        if len(sub):
            summary["range_blocks"][blk] = agg(
                sub, ["cv", "gamma_k_moment", "db_std", "mean_power",
                      "lag1_range", "lag1_doppler"], "group")

    if len(df):
        for g, sub in df.groupby("group"):
            summary["per_group"][str(g)] = {
                c: float(sub[c].mean()) for c in CORE if c in sub}
            summary["per_group"][str(g)]["n_frames"] = int(len(sub))

    # Range non-stationarity, as one number per statistic: spread of the block means.
    ns = {}
    for c in ("cv", "gamma_k_moment", "db_std", "mean_power"):
        vals = [summary["range_blocks"][b][c]["mean"]
                for b in summary["range_blocks"] if c in summary["range_blocks"][b]]
        if len(vals) > 1:
            ns[c] = {"block_min": min(vals), "block_max": max(vals),
                     "block_sd": float(np.std(vals, ddof=1)),
                     "max_over_min": (max(vals) / min(vals)) if min(vals) > 0 else None}
    if ns and "mean_power" in ns:
        ns["mean_power_db_span"] = 10.0 * np.log10(
            ns["mean_power"]["block_max"] / ns["mean_power"]["block_min"])
    summary["range_nonstationarity"] = ns

    summary["runtime_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(OUT, "%s_stats.json" % tag), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    log("wrote %s_stats.json  (%d frames, %d groups, %.1f min)"
        % (tag, summary["n_frames"], summary["n_groups"], summary["runtime_s"] / 60))
    for c in CORE[:5]:
        if c in summary["background"]:
            v = summary["background"][c]
            log("  %-16s %.4f  (group sd %.4f, range %.4f..%.4f)"
                % (c, v["mean"], v["sd_across_groups"], v["group_min"], v["group_max"]))
    for c in RIDGE:
        if c in summary.get("ridge", {}):
            v = summary["ridge"][c]
            log("  %-16s %.3f  (group sd %.3f)" % (c, v["mean"], v["sd_across_groups"]))


if __name__ == "__main__":
    main()
