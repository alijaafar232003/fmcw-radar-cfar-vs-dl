"""R1 analysis. READS the preserved Test outputs; never reruns a detector.

Order of interpretation follows PREREGISTRATION and the Phase 5 brief: primary recall first, then
the pre-registered strata, then burden, then seed stability, then secondary CFAR variants.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import eligibility as el   # noqa: E402
import stats as st         # noqa: E402

OUT = os.path.join(EXP, "results", "r1_test")


def recall(df):
    e = df[df["eligible"]]
    return float(e["matched"].mean()) if len(e) else float("nan")


def arm_label(row):
    return "CFAR-" + row["cfar_kind"] if row["detector"] == "cfar" else "TinyCNN"


def main():
    dfo = pd.read_csv(os.path.join(OUT, "per_object.csv"))
    dff = pd.read_csv(os.path.join(OUT, "per_frame.csv"))
    res = {}

    ex = dfo[dfo["tolerance"] == 0]
    elig_total = int(ex[(ex.detector == "cfar") & (ex.cfar_kind == "CA")]["eligible"].sum())
    res["eligible_test_objects"] = elig_total

    # ---------------- primary: exact-mask object recall ----------------
    cfar = {k: recall(ex[(ex.detector == "cfar") & (ex.cfar_kind == k)])
            for k in ("CA", "GO", "SO", "OS")}
    per_seed = {int(s): recall(ex[(ex.detector == "tinycnn") & (ex.seed == s)])
                for s in sorted(ex[ex.detector == "tinycnn"]["seed"].dropna().unique())}
    tiny = float(np.mean(list(per_seed.values())))
    res["primary"] = {"tinycnn_seed_mean": tiny, "tinycnn_per_seed": per_seed,
                      "tinycnn_seed_sd": float(np.std(list(per_seed.values()), ddof=1)),
                      "cfar": cfar, "primary_cfar": "CA"}

    # ---------------- per-sequence, paired ----------------
    seqs = sorted(ex["sequence"].unique())
    rows, delta = [], {}
    for s in seqs:
        sub = ex[ex.sequence == s]
        t_by_seed = {int(sd): recall(sub[(sub.detector == "tinycnn") & (sub.seed == sd)])
                     for sd in sorted(sub[sub.detector == "tinycnn"]["seed"].dropna().unique())}
        t = float(np.mean(list(t_by_seed.values())))
        c = recall(sub[(sub.detector == "cfar") & (sub.cfar_kind == "CA")])
        n_el = int(sub[(sub.detector == "cfar") & (sub.cfar_kind == "CA")]["eligible"].sum())
        delta[s] = t - c
        rows.append({"sequence": s, "eligible_objects": n_el, "tinycnn": t, "cfar_CA": c,
                     "delta": t - c,
                     **{"cfar_" + k: recall(sub[(sub.detector == "cfar") & (sub.cfar_kind == k)])
                        for k in ("GO", "SO", "OS")}})
    res["per_sequence"] = rows
    res["paired_bootstrap"] = st.paired_sequence_bootstrap(delta)

    # ---------------- tolerance ladder ----------------
    tol = {}
    for t_ in (0, 1, 3):
        d = dfo[dfo["tolerance"] == t_]
        tol["tol{}".format(t_)] = {
            "tinycnn": float(np.mean([recall(d[(d.detector == "tinycnn") & (d.seed == s)])
                                      for s in sorted(d[d.detector == "tinycnn"]["seed"]
                                                      .dropna().unique())])),
            **{"cfar_" + k: recall(d[(d.detector == "cfar") & (d.cfar_kind == k)])
               for k in ("CA", "GO", "SO", "OS")}}
    res["tolerance"] = tol

    # ---------------- strata ----------------
    def strat(col):
        out = {}
        for v in sorted(x for x in ex[col].dropna().unique()):
            sub = ex[ex[col] == v]
            n_el = int(sub[(sub.detector == "cfar") & (sub.cfar_kind == "CA")]["eligible"].sum())
            nseq = int(sub[sub["eligible"]]["sequence"].nunique())
            out[str(v)] = {
                "eligible_objects": n_el, "sequences": nseq,
                "tinycnn": float(np.mean([recall(sub[(sub.detector == "tinycnn") & (sub.seed == s)])
                                          for s in sorted(sub[sub.detector == "tinycnn"]["seed"]
                                                          .dropna().unique())])),
                "cfar_CA": recall(sub[(sub.detector == "cfar") & (sub.cfar_kind == "CA")]),
                "cfar_OS": recall(sub[(sub.detector == "cfar") & (sub.cfar_kind == "OS")])}
        return out

    res["by_class"] = strat("class")
    res["by_doppler"] = strat("doppler_stratum")
    res["by_range"] = strat("range_stratum")

    # tolerance ladder per class too
    for cls, d in res["by_class"].items():
        sub1 = dfo[(dfo["class"] == cls) & (dfo.tolerance == 1)]
        sub3 = dfo[(dfo["class"] == cls) & (dfo.tolerance == 3)]
        d["tinycnn_tol1"] = float(np.mean([recall(sub1[(sub1.detector == "tinycnn") &
                                                       (sub1.seed == s)])
                                           for s in sorted(sub1[sub1.detector == "tinycnn"]["seed"]
                                                           .dropna().unique())]))
        d["tinycnn_tol3"] = float(np.mean([recall(sub3[(sub3.detector == "tinycnn") &
                                                       (sub3.seed == s)])
                                           for s in sorted(sub3[sub3.detector == "tinycnn"]["seed"]
                                                           .dropna().unique())]))
        d["cfar_CA_tol1"] = recall(sub1[(sub1.detector == "cfar") & (sub1.cfar_kind == "CA")])
        d["cfar_CA_tol3"] = recall(sub3[(sub3.detector == "cfar") & (sub3.cfar_kind == "CA")])

    # ---------------- unmatched-detection burden (NOT Pfa) ----------------
    burden = {}
    for lab, sub in (("TinyCNN", dff[dff.detector == "tinycnn"]),
                     *[("CFAR-" + k, dff[(dff.detector == "cfar") & (dff.cfar_kind == k)])
                       for k in ("CA", "GO", "SO", "OS")]):
        u = sub["n_unmatched_peaks"].to_numpy(float)
        ua = sub[sub.has_annotation]["n_unmatched_peaks"].to_numpy(float)
        un = sub[~sub.has_annotation]["n_unmatched_peaks"].to_numpy(float)
        burden[lab] = {
            "mean": float(u.mean()), "median": float(np.median(u)),
            "p90": float(np.percentile(u, 90)), "max": float(u.max()),
            "annotated_frames_mean": float(ua.mean()) if ua.size else float("nan"),
            "unannotated_frames_mean": float(un.mean()) if un.size else float("nan"),
            "per_sequence_mean": {s: float(sub[sub.sequence == s]["n_unmatched_peaks"].mean())
                                  for s in seqs}}
    res["unmatched_burden"] = burden

    with open(os.path.join(OUT, "r1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    # ---------------- console ----------------
    p = res["primary"]
    print("=" * 78)
    print("R1 PRIMARY RESULT — exact dense-mask object recall, CARRADA TEST")
    print("=" * 78)
    print("eligible test objects: {:,}".format(elig_total))
    print()
    print("  frozen TinyCNN (8-seed mean) : {:.4f}   (sd across seeds {:.4f})".format(
        p["tinycnn_seed_mean"], p["tinycnn_seed_sd"]))
    print("  frozen CA-CFAR   (primary)   : {:.4f}".format(cfar["CA"]))
    b = res["paired_bootstrap"]
    print("  paired difference TinyCNN - CA: {:+.4f}   95% CI [{:+.4f}, {:+.4f}]  (n = {} sequences)"
          .format(b["mean_delta"], b["ci_lo"], b["ci_hi"], b["n_sequences"]))
    print("  CI crosses zero: {}".format(b["crosses_zero"]))
    print()
    print("secondary CFAR variants:", {k: round(v, 4) for k, v in cfar.items()})
    print()
    print("per-sequence (TinyCNN seed-mean vs CA):")
    for r in rows:
        print("   {}  n={:4d}   TinyCNN {:.4f}   CA {:.4f}   delta {:+.4f}".format(
            r["sequence"], r["eligible_objects"], r["tinycnn"], r["cfar_CA"], r["delta"]))
    print()
    print("tolerance ladder (exact / ±1 / ±3):")
    for arm in ("tinycnn", "cfar_CA"):
        print("   {:9s} {:.4f}  {:.4f}  {:.4f}".format(
            arm, tol["tol0"][arm], tol["tol1"][arm], tol["tol3"][arm]))
    print()
    print("zero-Doppler stratum:")
    for k, v in res["by_doppler"].items():
        print("   {:10s} n={:5d}  TinyCNN {:.4f}  CA {:.4f}".format(
            k, v["eligible_objects"], v["tinycnn"], v["cfar_CA"]))
    print()
    print("by class:")
    for k, v in res["by_class"].items():
        print("   {:11s} n={:5d} seqs={}  TinyCNN {:.4f}  CA {:.4f}".format(
            k, v["eligible_objects"], v["sequences"], v["tinycnn"], v["cfar_CA"]))
    print()
    print("by range block:")
    for k, v in res["by_range"].items():
        print("   {:10s} n={:5d}  TinyCNN {:.4f}  CA {:.4f}".format(
            k, v["eligible_objects"], v["tinycnn"], v["cfar_CA"]))
    print()
    print("unmatched detections per frame (NOT Pfa):")
    for k, v in burden.items():
        print("   {:9s} mean {:6.2f}  median {:5.1f}  p90 {:6.1f}   "
              "annotated {:6.2f}  unannotated {:6.2f}".format(
                  k, v["mean"], v["median"], v["p90"], v["annotated_frames_mean"],
                  v["unannotated_frames_mean"]))
    print()
    print("TinyCNN per-seed exact recall:", {k: round(v, 4) for k, v in per_seed.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
