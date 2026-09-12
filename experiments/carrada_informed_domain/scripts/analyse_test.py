"""Phase 10 analysis: arm B against arm A (primary) and arm C (secondary).

Arms
    A  original TinyCNN, 8 frozen seeds       from e6_nms_audit/results/p4_carrada/r1_test
    B  CARRADA-informed TinyCNN, 8 seeds      from results/test_armB_{r5,r2}
    C  CA-CFAR                                from e6_nms_audit/results/p4_carrada/r1_test

A and C are read, never recomputed. The statistics come from the original experiment's own
`stats.paired_sequence_bootstrap`, imported unmodified, with its own N_BOOTSTRAP (10,000) and its
own BOOTSTRAP_SEED (20260831). Reusing that seed is deliberate: the same sequence resamples are
drawn as in Chapter 7, so the new interval is directly comparable with the published one.

Recall is computed exactly as Chapter 7 computes it: over ELIGIBLE objects only, per sequence,
seed-averaged within an arm before differencing, so that the resampling unit stays the sequence.

    python analyse_test.py --thresholds r5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
E6 = os.path.join(ROOT, "experiments", "e6_nms_audit")
sys.path.insert(0, os.path.join(R_EXP, "src"))

import stats as st                                                       # noqa: E402

REF = os.path.join(E6, "results", "p4_carrada", "r1_test")
PRIMARY_CFAR = "CA"


def recall_by_sequence(df, tol=0):
    """Eligible-object recall per sequence per seed. Returns {sequence: {seed: recall}}."""
    d = df[(df.tolerance == tol) & (df.eligible)]
    out = {}
    for (seq, seed), g in d.groupby(["sequence", "seed"], dropna=False):
        out.setdefault(seq, {})[seed] = float(g.matched.mean())
    return out


def pooled_recall(df, tol=0):
    """Pooled eligible-object recall, seed-averaged.

    dropna=False matters: the CFAR rows carry seed = NaN (it has no seeds), and a plain groupby
    would silently drop every one of them and return NaN for the classical arm.
    """
    d = df[(df.tolerance == tol) & (df.eligible)]
    per_seed = d.groupby("seed", dropna=False).matched.mean()
    return float(per_seed.mean()), {("cfar" if pd.isna(k) else k): float(v)
                                    for k, v in per_seed.items()}


def seq_mean(rbs):
    """Seed-average within each sequence."""
    return {s: float(np.mean(list(v.values()))) for s, v in rbs.items()}


def burden(dff):
    """Unmatched detections per frame. NOT false alarms, NOT Pfa."""
    per_seed = dff.groupby("seed", dropna=False).n_unmatched_peaks.mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0


def strat_recall(df, col, tol=0):
    d = df[(df.tolerance == tol) & (df.eligible)]
    out = {}
    for k, g in d.groupby(col, dropna=False):
        if k is None or (isinstance(k, float) and np.isnan(k)):
            continue
        per_seed = g.groupby("seed", dropna=False).matched.mean()
        out[str(k)] = {"recall": float(per_seed.mean()),
                       "n_objects": int(len(g) / max(1, g.seed.nunique()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="r5", choices=("r5", "r2"))
    a = ap.parse_args()

    B_dir = os.path.join(EXP, "results", "test_armB_%s" % a.thresholds)
    Bo = pd.read_csv(os.path.join(B_dir, "per_object.csv"))
    Bf = pd.read_csv(os.path.join(B_dir, "per_frame.csv"))
    Ro = pd.read_csv(os.path.join(REF, "per_object.csv"))
    Rf = pd.read_csv(os.path.join(REF, "per_frame.csv"))

    Ao = Ro[Ro.detector == "tinycnn"].copy()
    Af = Rf[Rf.detector == "tinycnn"].copy()
    Co = Ro[(Ro.detector == "cfar") & (Ro.cfar_kind == PRIMARY_CFAR)].copy()
    Cf = Rf[(Rf.detector == "cfar") & (Rf.cfar_kind == PRIMARY_CFAR)].copy()

    nA = int(Ao[(Ao.tolerance == 0) & (Ao.seed == 0)].eligible.sum())
    nB = int(Bo[(Bo.tolerance == 0) & (Bo.seed == 0)].eligible.sum())
    nC = int(Co[Co.tolerance == 0].eligible.sum())
    assert nA == nB == nC, "eligible-object counts differ: A=%d B=%d C=%d" % (nA, nB, nC)

    res = {"threshold_set": a.thresholds, "eligible_objects": nA,
           "arms": {"A": "original TinyCNN", "B": "CARRADA-informed TinyCNN",
                    "C": "CA-CFAR"}}

    # ---------------- primary: exact recall, B - A, paired over sequences
    rA, rB, rC = (recall_by_sequence(x) for x in (Ao, Bo, Co))
    mA, mB, mC = seq_mean(rA), seq_mean(rB), seq_mean(rC)
    pA, sA = pooled_recall(Ao)
    pB, sB = pooled_recall(Bo)
    pC, _ = pooled_recall(Co)

    dBA = {s: mB[s] - mA[s] for s in mA}
    dBC = {s: mB[s] - mC[s] for s in mA}
    dAC = {s: mA[s] - mC[s] for s in mA}

    res["pooled_exact_recall"] = {"A": pA, "B": pB, "C": pC}
    res["per_sequence_exact_recall"] = {"A": mA, "B": mB, "C": mC}
    res["primary_B_minus_A"] = st.paired_sequence_bootstrap(dBA)
    res["secondary_B_minus_C"] = st.paired_sequence_bootstrap(dBC)
    res["reference_A_minus_C"] = st.paired_sequence_bootstrap(dAC)

    # ---------------- tolerance ladder
    res["tolerance"] = {}
    for tol in (0, 1, 3):
        tA, _ = pooled_recall(Ao, tol)
        tB, _ = pooled_recall(Bo, tol)
        tC, _ = pooled_recall(Co, tol)
        dd = {s: v for s, v in zip(sorted(mA), [0] * len(mA))}
        rAt, rBt = seq_mean(recall_by_sequence(Ao, tol)), seq_mean(recall_by_sequence(Bo, tol))
        dd = {s: rBt[s] - rAt[s] for s in rAt}
        res["tolerance"]["tol%d" % tol] = {
            "A": tA, "B": tB, "C": tC,
            "B_minus_A": st.paired_sequence_bootstrap(dd)}

    # ---------------- output burden
    bA, bAsd = burden(Af)
    bB, bBsd = burden(Bf)
    bC = float(Cf.n_unmatched_peaks.mean())
    res["unmatched_per_frame"] = {
        "A": {"mean": bA, "sd_across_seeds": bAsd},
        "B": {"mean": bB, "sd_across_seeds": bBsd},
        "C": {"mean": bC},
        "note": "OUTPUT BURDEN. Never false alarms, never Pfa: CARRADA cannot support a "
                "false-alarm measurement."}
    for tag, df in (("A", Af), ("B", Bf), ("C", Cf)):
        ann = df[df.has_annotation]
        emp = df[~df.has_annotation]
        res["unmatched_per_frame"][tag]["annotated_frames_mean"] = float(
            ann.n_unmatched_peaks.mean())
        res["unmatched_per_frame"][tag]["unannotated_frames_mean"] = float(
            emp.n_unmatched_peaks.mean())

    # ---------------- strata
    for col, key in (("range_stratum", "by_range"), ("doppler_stratum", "by_doppler"),
                     ("class", "by_class")):
        res[key] = {"A": strat_recall(Ao, col), "B": strat_recall(Bo, col),
                    "C": strat_recall(Co, col)}

    # ---------------- seed variability, reported separately
    res["seed_variability"] = {
        "A": {"per_seed_pooled": {str(k): float(v) for k, v in sA.items()},
              "mean": float(np.mean(list(sA.values()))),
              "sd": float(np.std(list(sA.values()), ddof=1)),
              "min": float(min(sA.values())), "max": float(max(sA.values()))},
        "B": {"per_seed_pooled": {str(k): float(v) for k, v in sB.items()},
              "mean": float(np.mean(list(sB.values()))),
              "sd": float(np.std(list(sB.values()), ddof=1)),
              "min": float(min(sB.values())), "max": float(max(sB.values()))},
        "note": "NEVER merged with the sequence bootstrap; the two answer different questions."}
    res["seeds_B_above_CFAR"] = int(sum(1 for v in sB.values() if v > pC))
    res["seeds_A_above_CFAR"] = int(sum(1 for v in sA.values() if v > pC))

    p = os.path.join(EXP, "results", "test_analysis_%s.json" % a.thresholds)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    # ---------------- report
    P = res["primary_B_minus_A"]
    print("=" * 78)
    print("PRIMARY  exact dense-mask object recall, %d eligible objects, thresholds %s"
          % (nA, a.thresholds))
    print("=" * 78)
    print("  A original TinyCNN       %.4f" % pA)
    print("  B CARRADA-informed       %.4f" % pB)
    print("  C CA-CFAR                %.4f" % pC)
    print("")
    print("  B - A  sequence-balanced %+.4f   95%% CI [%+.4f, %+.4f]   crosses zero: %s"
          % (P["mean_delta"], P["ci_lo"], P["ci_hi"], P["crosses_zero"]))
    S = res["secondary_B_minus_C"]
    print("  B - C  sequence-balanced %+.4f   95%% CI [%+.4f, %+.4f]   crosses zero: %s"
          % (S["mean_delta"], S["ci_lo"], S["ci_hi"], S["crosses_zero"]))
    Rf_ = res["reference_A_minus_C"]
    print("  A - C  (Chapter 7 value) %+.4f   95%% CI [%+.4f, %+.4f]"
          % (Rf_["mean_delta"], Rf_["ci_lo"], Rf_["ci_hi"]))
    print("")
    print("  per sequence (A -> B, delta):")
    for s in sorted(mA):
        print("    %-22s %.4f -> %.4f   %+.4f" % (s, mA[s], mB[s], mB[s] - mA[s]))
    print("")
    print("  tolerance ladder        A        B        C     B-A")
    for tol in (0, 1, 3):
        t = res["tolerance"]["tol%d" % tol]
        print("    %-6s              %.4f   %.4f   %.4f   %+.4f [%+.4f, %+.4f]"
              % ("exact" if tol == 0 else "+/-%d" % tol, t["A"], t["B"], t["C"],
                 t["B_minus_A"]["mean_delta"], t["B_minus_A"]["ci_lo"],
                 t["B_minus_A"]["ci_hi"]))
    print("")
    u = res["unmatched_per_frame"]
    print("  unmatched detections/frame (OUTPUT BURDEN, not Pfa):")
    print("    A %.2f    B %.2f    C %.2f" % (u["A"]["mean"], u["B"]["mean"], u["C"]["mean"]))
    print("")
    print("  range strata (recall):        A        B        C")
    for k in sorted(res["by_range"]["A"]):
        print("    %-10s n=%4d       %.4f   %.4f   %.4f"
              % (k, res["by_range"]["A"][k]["n_objects"], res["by_range"]["A"][k]["recall"],
                 res["by_range"]["B"].get(k, {}).get("recall", float("nan")),
                 res["by_range"]["C"].get(k, {}).get("recall", float("nan"))))
    print("")
    print("  zero-Doppler strata:          A        B        C")
    for k in sorted(res["by_doppler"]["A"]):
        print("    %-10s n=%4d       %.4f   %.4f   %.4f"
              % (k, res["by_doppler"]["A"][k]["n_objects"], res["by_doppler"]["A"][k]["recall"],
                 res["by_doppler"]["B"].get(k, {}).get("recall", float("nan")),
                 res["by_doppler"]["C"].get(k, {}).get("recall", float("nan"))))
    print("")
    print("  seed spread   A %.4f +/- %.4f  [%.4f, %.4f]"
          % (res["seed_variability"]["A"]["mean"], res["seed_variability"]["A"]["sd"],
             res["seed_variability"]["A"]["min"], res["seed_variability"]["A"]["max"]))
    print("                B %.4f +/- %.4f  [%.4f, %.4f]"
          % (res["seed_variability"]["B"]["mean"], res["seed_variability"]["B"]["sd"],
             res["seed_variability"]["B"]["min"], res["seed_variability"]["B"]["max"]))
    print("\nwrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
