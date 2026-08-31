"""Phase 4 check C8 — non-test pipeline dry run, plus the extended-object NMS characteristic.

SOFTWARE SANITY ONLY — NOT PERFORMANCE ANALYSIS.

Uses CARRADA **Train** sequences only. Never Validation, never Test. Nothing measured here may
influence any methodological choice: thresholds, NMS, CFAR variant, mask definitions, matching and
preprocessing are all already frozen by the pre-registration.

Aggregate recall is deliberately NOT printed. What is printed is: does it run, are the shapes and
coordinates right, are there NaNs, and how often does one extended object contain several
post-NMS peaks (PREREGISTRATION §7.3, the registered metric-transfer limitation).
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
sys.path.insert(0, os.path.join(ROOT, "src"))

import carrada_input as ci      # noqa: E402
import detectors as det         # noqa: E402
import eligibility as el        # noqa: E402
import metrics as mt            # noqa: E402

RESULTS = os.path.join(EXP, "results")
SPLIT = "Train"                 # HARD-CODED. The only CARRADA split this script may read.
SEED = 20260831
N_SEQ, N_FRAMES = 6, 8


def frames_for_dry_run():
    rng = np.random.default_rng(SEED)
    seqs = ci.sequences(SPLIT)[:N_SEQ]
    out = []
    for s in seqs:
        ann = [f for f in ci.frames(s) if ci.objects(s, f)]
        for f in rng.choice(ann, size=min(N_FRAMES, len(ann)), replace=False):
            out.append((s, str(f)))
    return out


def load_thresholds():
    p = os.path.join(EXP, "configs", "tinycnn_zero_shot_thresholds.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    return {r["seed"]: r["threshold"] for r in d["thresholds"]}


def main():
    assert SPLIT == "Train", "dry run is restricted to CARRADA Train"
    print("=" * 74)
    print("C8  NON-TEST PIPELINE DRY RUN   —   SOFTWARE SANITY ONLY, NOT PERFORMANCE ANALYSIS")
    print("=" * 74)
    print("CARRADA split used: {} (never Validation, never Test)".format(SPLIT))

    pairs = frames_for_dry_run()
    thr = load_thresholds()
    print("frames: {} across {} sequences".format(len(pairs), N_SEQ))
    print("TinyCNN thresholds loaded: {}".format("yes" if thr else "NO — using unthresholded peaks"))

    model = det.load_frozen_tinycnn(0)
    t0 = time.time()

    problems, multi = [], []
    n_nan = 0
    peak_counts = {"tinycnn": [], "CA": []}

    for seq, fr in pairs:
        D = ci.load_rd_db(seq, fr)
        objs = ci.objects(seq, fr)
        if not np.isfinite(D).all():
            problems.append("non-finite RD in {}/{}".format(seq, fr))

        pk_dl = det.tinycnn_peaks(model, D)
        pk_ca = det.cfar_peaks_at_alpha(D, kind="CA")   # frozen operating point: ratio >= alpha
        for name, pk in (("tinycnn", pk_dl), ("CA", pk_ca)):
            if pk.size and not np.isfinite(pk).all():
                n_nan += 1
            if pk.size and (pk[:, 0].min() < 0 or pk[:, 0].max() > 255
                            or pk[:, 1].min() < 0 or pk[:, 1].max() > 63):
                problems.append("peak outside map in {}/{} ({})".format(seq, fr, name))
            peak_counts[name].append(len(pk))

        if thr:
            pk_dl = det.apply_threshold(pk_dl, thr[0])

        # matching + metrics must simply RUN and return finite structures
        for tol in mt.TOLERANCES:
            res = mt.score_frame(pk_dl, objs, tolerance=tol)
            if res["n_matched_peaks"] > res["n_peaks"]:
                problems.append("matched > total peaks in {}/{}".format(seq, fr))

        # --- V: extended-object multi-peak characteristic (metric behaviour, not performance) ---
        for o in objs:
            vs = el.valid_support(o["dense"])
            if len(vs) == 0:
                continue
            sup = set(map(tuple, vs.tolist()))
            for name, pk in (("tinycnn", pk_dl), ("CA", pk_ca)):
                inside = sum(1 for p in pk if (int(p[0]), int(p[1])) in sup)
                multi.append({"sequence": seq, "frame": fr, "object_id": o["object_id"],
                              "class": o["class"], "detector": name,
                              "support_cells": len(vs), "peaks_inside_support": inside})

    dt = time.time() - t0
    dfm = pd.DataFrame(multi)
    dfm.to_csv(os.path.join(RESULTS, "phase4_multipeak_characteristic.csv"), index=False)

    print("\n--- software sanity ---")
    print("pipeline completed          : {} frames in {:.1f}s ({:.2f}s/frame both arms)".format(
        len(pairs), dt, dt / max(1, len(pairs))))
    print("non-finite peak arrays      : {}".format(n_nan))
    print("structural problems         : {}".format(len(problems)))
    for p in problems[:5]:
        print("   ", p)
    print("peaks/frame  TinyCNN (pre-threshold): median {:.0f}".format(
        np.median(peak_counts["tinycnn"])))
    print("peaks/frame  CA-CFAR (pre-threshold): median {:.0f}".format(
        np.median(peak_counts["CA"])))

    print("\n--- V: extended-object / frozen-NMS characteristic ---")
    print("This is a METRIC TRANSFER CHARACTERISTIC, not a performance result.")
    print("NMS is NOT altered in response to it.\n")
    rows = []
    for name, g in dfm.groupby("detector"):
        frac_multi = float((g.peaks_inside_support >= 2).mean())
        rows.append({"detector": name,
                     "objects": int(len(g)),
                     "median_peaks_inside": float(g.peaks_inside_support.median()),
                     "max_peaks_inside": int(g.peaks_inside_support.max()),
                     "frac_objects_with_>=2_peaks": frac_multi})
        print("  {:8s}  objects {:4d}   median peaks inside {:.0f}   max {:2d}   "
              ">=2 peaks in {:.1%} of objects".format(
                  name, len(g), g.peaks_inside_support.median(),
                  int(g.peaks_inside_support.max()), frac_multi))
    if len(dfm):
        big = dfm[dfm["class"] == "car"]
        if len(big):
            print("  cars only, any detector: >=2 peaks inside support in {:.1%}".format(
                float((big.peaks_inside_support >= 2).mean())))

    out = {"note": "SOFTWARE SANITY ONLY - NOT PERFORMANCE ANALYSIS",
           "carrada_split_used": SPLIT,
           "frames": len(pairs), "seconds": round(dt, 1),
           "non_finite_peak_arrays": n_nan, "structural_problems": problems,
           "multipeak_by_detector": rows,
           "interpretation": ("A CARRADA car spans ~18x13 bins against an 11x11 suppression "
                              "window, so one object can hold several post-NMS peaks. Under "
                              "one-to-one matching exactly one is matched and the surplus counts "
                              "as UNMATCHED, which inflates unmatched-detection burden for BOTH "
                              "arms. Registered in PREREGISTRATION 7.3; NMS is not changed.")}
    with open(os.path.join(RESULTS, "phase4_dry_run.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    ok = not problems and n_nan == 0
    print("\nC8 PASS:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
