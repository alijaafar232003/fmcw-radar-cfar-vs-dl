"""Task 2.1 — the CFAR baseline on the FULL `bench2_test` split. Run once.

This is the pre-registered classical baseline: CFAR has no learnable parameters, so touching the
test split with it cannot contaminate anything, and doing it before the network exists means the
numbers cannot be unconsciously tuned against.

Geometry is whatever `CfarConfig` defaults to — the 21x9 window (guard 2/2, train 8/2) selected in
step 6 on `bench2_tune`. Nothing is re-tuned here.

**Detections are cached, not just summarised.** Every local maximum with its score is written to
`results/cfar_test/peaks_<METHOD>.npz`, so the notebook can re-derive any operating point, ROC,
or per-SNR breakdown in seconds without re-running CFAR. Re-running this script is only necessary
if the data or the detector changes.

Output (results/cfar_test/):
    peaks_CA.npz / peaks_GO.npz / peaks_OS.npz   cached detections + runtimes
    frames.csv                                    per-frame condition + truth index
    truths.csv                                    every target: bins, cell SNR, condition
    summary.csv                                   per-condition Pd at matched measured Pfa
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map          # noqa: E402
from cfar.detect import all_peaks                          # noqa: E402
from dl.stratified_eval import NMS, _at_measured_pfa       # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,          # noqa: E402
                  roc_curve, roc_thresholds)

import argparse

METHODS = ("CA", "GO", "OS", "SO")
TARGET_PFA = 1e-4

# Set by main(); `bench2_test` is the reported baseline, `bench2_val` is what every comparison
# made while choices are still open must use instead.
SPLIT = "bench2_test"
OUT = ROOT / "results" / "cfar_test"

# Dense in the upper tail: the operating point sits in the top ~1% of the score pool, and a
# uniform grid steps straight over it (see PROGRESS.md Milestone 20b).
QUANTILES = np.clip(np.concatenate([
    np.linspace(0.0, 0.99, 40),
    1.0 - np.logspace(np.log10(0.01), np.log10(2e-5), 40),
]), 0.0, 1.0)


def collect() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One pass over the split: every method's detections, cached."""
    frame_dirs = sorted((ROOT / "data" / SPLIT / "frames").iterdir())
    print(f"{len(frame_dirs)} frames in {SPLIT}")

    frames_meta, truth_rows = [], []
    peaks = {m: [] for m in METHODS}
    elapsed = {m: 0.0 for m in METHODS}

    for i, fd in enumerate(frame_dirs):
        md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
        mag = np.load(fd / "rd_map_mag.npy")
        power = mag.astype(np.float64) ** 2

        lab = (pd.read_csv(fd / "labels.csv")
               if (fd / "labels.csv").stat().st_size > 5 else pd.DataFrame())

        for t, L in lab.iterrows():
            truth_rows.append(dict(
                frame=i, target=int(t),
                range_bin=int(L["range_bin"]), velocity_bin=int(L["velocity_bin"]),
                cell_snr_db=float(L["cell_snr_db"]), amplitude_db=float(L["amplitude_db"]),
                range_m=float(L["range_m"]), velocity_mps=float(L["velocity_mps"]),
                condition=md.get("condition"),
            ))

        cl = md.get("clutter", {})
        frames_meta.append(dict(
            frame=i, condition=md.get("condition"), num_targets=len(lab),
            cnr_db=cl.get("cnr_db") if cl.get("enabled") else np.nan,
            shape=cl.get("shape") if cl.get("enabled") else np.nan,
        ))

        for m in METHODS:
            t0 = time.perf_counter()
            ratio, valid, _ = cfar_ratio_map(power, CfarConfig(kind=m))
            pk = all_peaks(ratio, valid, nms_size=NMS)
            elapsed[m] += time.perf_counter() - t0

            if len(pk):
                peaks[m].append(np.column_stack([np.full(len(pk), i, dtype=float), pk]))

        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(frame_dirs)}")

    OUT.mkdir(parents=True, exist_ok=True)
    n = len(frame_dirs)

    for m in METHODS:
        arr = np.concatenate(peaks[m]) if peaks[m] else np.zeros((0, 4))
        np.savez_compressed(OUT / f"peaks_{m}.npz", peaks=arr,
                            ms_per_frame=1000.0 * elapsed[m] / n)
        print(f"  {m}: {len(arr):,} detections, {1000 * elapsed[m] / n:.1f} ms/frame")

    return pd.DataFrame(frames_meta), pd.DataFrame(truth_rows), elapsed


def score(frames_meta: pd.DataFrame, truths: pd.DataFrame) -> pd.DataFrame:
    """Per-condition Pd at matched MEASURED Pfa, from the cached detections."""
    cfg = CfarConfig()
    shape = (256, 64)
    valid = np.zeros(shape, dtype=bool)
    valid[cfg.half_width_r:-cfg.half_width_r, cfg.half_width_d:-cfg.half_width_d] = True

    mc = MatchConfig(1, 1)
    tru_by_frame = {f: g[["range_bin", "velocity_bin"]].to_numpy(float)
                    for f, g in truths.groupby("frame")}

    rows = []
    for m in METHODS:
        d = np.load(OUT / f"peaks_{m}.npz")
        arr, ms = d["peaks"], float(d["ms_per_frame"])
        by_frame = {f: arr[arr[:, 0] == f][:, 1:] for f in np.unique(arr[:, 0])}

        for cond, g in frames_meta.groupby("condition"):
            fr = [(by_frame.get(float(i), np.zeros((0, 3))),
                   tru_by_frame.get(i, np.zeros((0, 2))), shape, valid)
                  for i in g.frame]

            roc = roc_curve(fr, roc_thresholds(fr, mc), mc)
            pd_interp, thr = _at_measured_pfa(roc, TARGET_PFA)

            # Interpolation's job is to FIND the threshold; Pd is then measured AT it. Reporting
            # the interpolated Pd instead carries the interpolation error into the headline (up to
            # 0.017, and biased optimistic on the clutter strata, where the ROC is steepest).
            evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr)
                                  else np.zeros((0, 2)), t, s, mc, valid_mask=v)
                   for p, t, s, v in fr]
            a = aggregate(evs)

            # Fixed-threshold view: the operating point CFAR theory prescribes. GO's whole design
            # goal -- surviving a clutter edge -- shows up here as fewer false alarms, and is
            # invisible at matched Pfa, where equal false-alarm rates are imposed by construction.
            alpha = CfarConfig(kind=m).alpha_go() if m == "GO" else CfarConfig(kind=m).alpha()
            evs_d = [evaluate_frame(p[p[:, 2] >= alpha][:, :2] if len(p) else np.zeros((0, 2)),
                                    t, s, mc, valid_mask=v)
                     for p, t, s, v in fr]
            ad = aggregate(evs_d)

            rows.append(dict(condition=cond, method=m, frames=len(g),
                             pd=a["pd"], pd_interp=pd_interp, threshold=thr,
                             measured_pfa=a["pfa_per_cell"], fa_per_frame=a["fa_per_frame"],
                             tp=a["tp"], fp=a["fp"], fn=a["fn"], ms_per_frame=ms,
                             design_alpha=alpha, design_pd=ad["pd"],
                             design_pfa=ad["pfa_per_cell"], design_fa_per_frame=ad["fa_per_frame"]))
        print(f"  scored {m}")

    return pd.DataFrame(rows)


def main() -> None:
    global SPLIT, OUT, METHODS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="bench2_test")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset, e.g. 'SO'. Existing peaks_*.npz for methods NOT "
                         "listed are left untouched and their summary rows are merged back in, so "
                         "a late-added method can be filled in without re-reading the split for "
                         "the ones already measured.")
    ap.add_argument("--score-only", action="store_true",
                    help="skip detection and re-derive the summary from the cached peaks_*.npz. "
                         "This is what caching the detections was for.")
    args = ap.parse_args()
    SPLIT = args.split
    OUT = ROOT / "results" / f"cfar_{SPLIT.replace('bench2_', '')}"

    if args.methods:
        METHODS = tuple(m.strip().upper() for m in args.methods.split(","))
        print(f"running {METHODS} only; other methods' cached results are preserved")

    if args.score_only:
        frames_meta = pd.read_csv(OUT / "frames.csv")
        truths = pd.read_csv(OUT / "truths.csv")
        print(f"  --score-only: re-deriving the summary from cached peaks "
              f"({len(frames_meta)} frames)")
    else:
        frames_meta, truths, _ = collect()

        # Do not clobber the earlier run's frame/truth indices -- they are the join keys for every
        # cached peaks file, so a reordering would silently corrupt the methods NOT re-run.
        #
        # Compared with a tolerance, not exactly: the existing CSVs were written by an earlier run
        # and float64 does not survive a decimal round-trip bit-for-bit. The observed disagreement
        # is 3.6e-15 (one ulp) on cnr_db, which is serialisation, not a different dataset. NaN
        # patterns and every integer/string column must still match EXACTLY.
        for name, df in (("frames.csv", frames_meta), ("truths.csv", truths)):
            p = OUT / name
            if not p.exists():
                df.to_csv(p, index=False)
                continue
            old = pd.read_csv(p)
            new = df[old.columns.tolist()].reset_index(drop=True)
            if len(old) != len(new):
                raise SystemExit(f"{name}: {len(new)} rows against the existing {len(old)} -- "
                                 f"refusing to overwrite; cached peaks are keyed on it.")
            for c in old.columns:
                a, b = old[c], new[c]
                if a.dtype.kind == "f" or b.dtype.kind == "f":
                    af, bf = a.to_numpy(float), b.to_numpy(float)
                    ok = (np.array_equal(np.isnan(af), np.isnan(bf))
                          and np.allclose(af, bf, rtol=1e-9, atol=1e-12, equal_nan=True))
                else:
                    ok = a.equals(b)
                if not ok:
                    raise SystemExit(f"{name}: column {c!r} differs from the existing run -- "
                                     f"refusing to overwrite; cached peaks are keyed on it.")
            print(f"  {name} matches the existing run, left as is")

    summary = score(frames_meta, truths)

    # Merge rather than replace: rows for methods not re-run are carried over verbatim.
    sp = OUT / "summary.csv"
    if sp.exists():
        old = pd.read_csv(sp)
        kept = old[~old.method.isin(METHODS)]
        summary = pd.concat([kept, summary], ignore_index=True).sort_values(["condition", "method"])
        print(f"  merged: kept {kept.method.nunique()} existing method(s), "
              f"added/updated {len(METHODS)}")
    summary.to_csv(sp, index=False)

    piv = summary.pivot_table(index="condition", columns="method", values="pd")
    print(f"\nPd at matched MEASURED Pfa {TARGET_PFA:.0e}, {SPLIT}, 21x9 window\n")
    print(piv.round(3).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
