"""Phase 9 analysis + Phase 10 evaluation-only burden curve, on official CARRADA Validation.

Primary comparison        C - D   (fine-tuned vs CA-CFAR)
Adaptation comparison     C - B   (fine-tuned vs CARRADA-informed synthetic)
Context                   B - A

Statistics come from the original experiment's `stats.paired_sequence_bootstrap`, imported
unmodified: 10,000 replicates over the SIX Validation sequences, which are the resampling unit.
Seed variability is reported separately and never merged into that interval.

Phase 10's sweep is EVALUATION-ONLY. It re-thresholds the preserved candidate lists to trace recall
against unmatched detections per frame. It selects nothing: the deployed thresholds were frozen on
internal-development CARRADA TRAIN data before Validation was opened, and are not revisited.

    python analyse_validation.py
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
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
sys.path.insert(0, os.path.join(R_EXP, "src"))
sys.path.insert(0, HERE)

import guard                                                             # noqa: E402
guard.install(guard.ALLOW_VALIDATION)

import carrada_input as ci                                               # noqa: E402
import eligibility as el                                                 # noqa: E402
import metrics as mt                                                     # noqa: E402
import stats as st                                                       # noqa: E402

VAL = os.path.join(EXP, "results", "validation")
ARMS = ["A", "B", "C", "Cinh", "D"]
NAME = {"A": "original TinyCNN", "B": "CARRADA-informed TinyCNN",
        "C": "real-fine-tuned TinyCNN", "Cinh": "fine-tuned @ inherited threshold",
        "D": "CA-CFAR"}


def recall_seq(df, arm, tol=0):
    d = df[(df.arm == arm) & (df.tolerance == tol) & (df.eligible)]
    out = {}
    for (seq, seed), g in d.groupby(["sequence", "seed"], dropna=False):
        out.setdefault(seq, {})[seed] = float(g.matched.mean())
    return {s: float(np.mean(list(v.values()))) for s, v in out.items()}


def pooled(df, arm, tol=0):
    d = df[(df.arm == arm) & (df.tolerance == tol) & (df.eligible)]
    per_seed = d.groupby("seed", dropna=False).matched.mean()
    return float(per_seed.mean()), {str(k): float(v) for k, v in per_seed.items()}


def burden(dff, arm):
    d = dff[dff.arm == arm]
    per_seed = d.groupby("seed", dropna=False).n_unmatched_peaks.mean()
    return float(per_seed.mean()), (float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0)


def strat(df, arm, col, tol=0):
    d = df[(df.arm == arm) & (df.tolerance == tol) & (df.eligible)]
    out = {}
    for k, g in d.groupby(col, dropna=False):
        if k is None or (isinstance(k, float) and np.isnan(k)):
            continue
        out[str(k)] = {"recall": float(g.groupby("seed", dropna=False).matched.mean().mean()),
                       "n_objects": int(len(g) / max(1, g.seed.nunique(dropna=False)))}
    return out


# ------------------------------------------------------------------ Phase 10 sweep
def load_peaks(path):
    z = np.load(path, allow_pickle=True)
    pk, idx = z["peaks"], z["index"]
    return {(str(s), str(f)): pk[int(a):int(b)] for s, f, a, b in idx}


def sweep_arm(peaks_by_seed, objs, n_points=90, anchors=()):
    """Recall vs unmatched/frame over a threshold grid, seed-averaged. Selects nothing.

    The grid is quantile-based but deliberately DENSE in the upper tail, because burden falls
    steeply there and that is exactly where the CA-CFAR operating region sits; a uniform quantile
    grid resolves the low-burden end too coarsely to interpolate through. Each arm's own frozen
    threshold is also forced into the grid as an anchor, so the curve passes exactly through the
    deployed operating point instead of approximating it.
    """
    pts = []
    allsc = np.concatenate([p[:, 2] for pk in peaks_by_seed.values()
                            for p in pk.values() if len(p)])
    half = n_points // 2
    qs = np.concatenate([np.linspace(0.0, 0.90, half),
                         1.0 - np.geomspace(0.10, 1e-4, n_points - half)])
    grid = np.unique(np.concatenate([np.quantile(allsc, np.clip(qs, 0, 1)),
                                     np.asarray(anchors, dtype=float)]))
    for thr in grid:
        recs, burs = [], []
        for pk in peaks_by_seed.values():
            matched, unm, nfr = [], 0, 0
            for key, p in pk.items():
                keep = p[p[:, 2] >= thr] if len(p) else np.zeros((0, 3))
                r = mt.score_frame(keep, objs[key], tolerance=0)
                unm += r["n_unmatched_peaks"]
                nfr += 1
                for row in r["per_object"]:
                    if row["eligible"]:
                        matched.append(row["matched"])
            recs.append(float(np.mean(matched)) if matched else np.nan)
            burs.append(unm / max(1, nfr))
        pts.append({"threshold": float(thr), "recall": float(np.mean(recs)),
                    "burden": float(np.mean(burs))})
    return pts


def recall_at_burden(curve, target):
    """Interpolate recall at a target burden along the monotone burden axis."""
    c = sorted(curve, key=lambda x: x["burden"])
    b = np.array([x["burden"] for x in c])
    r = np.array([x["recall"] for x in c])
    ok = np.isfinite(r)
    if not ok.any() or target < b[ok].min() or target > b[ok].max():
        return None
    return float(np.interp(target, b[ok], r[ok]))


def main():
    dfo = pd.read_csv(os.path.join(VAL, "per_object.csv"))
    dff = pd.read_csv(os.path.join(VAL, "per_frame.csv"))
    res = {"split": "CARRADA Validation (official, 6 sequences)",
           "n_frames": int(dff[dff.arm == "D"].shape[0]),
           "eligible_objects": int(dfo[(dfo.arm == "D") & (dfo.tolerance == 0)].eligible.sum())}

    seqr = {a: recall_seq(dfo, a) for a in ARMS}
    pool = {}
    per_seed = {}
    for a in ARMS:
        pool[a], per_seed[a] = pooled(dfo, a)
    res["pooled_exact_recall"] = pool
    res["per_sequence_exact_recall"] = seqr
    res["per_seed_pooled"] = per_seed

    def paired(x, y):
        d = {s: seqr[x][s] - seqr[y][s] for s in seqr[x]}
        return st.paired_sequence_bootstrap(d)

    res["primary_C_minus_D"] = paired("C", "D")
    res["adaptation_C_minus_B"] = paired("C", "B")
    res["context_B_minus_A"] = paired("B", "A")
    res["context_A_minus_D"] = paired("A", "D")
    res["secondary_Cinh_minus_D"] = paired("Cinh", "D")

    res["tolerance"] = {}
    for tol in (0, 1, 3):
        row = {a: pooled(dfo, a, tol)[0] for a in ARMS}
        rs = {a: recall_seq(dfo, a, tol) for a in ARMS}
        row["C_minus_D"] = st.paired_sequence_bootstrap(
            {s: rs["C"][s] - rs["D"][s] for s in rs["C"]})
        res["tolerance"]["tol%d" % tol] = row

    res["unmatched_per_frame"] = {}
    for a in ARMS:
        m, sd = burden(dff, a)
        sub = dff[dff.arm == a]
        res["unmatched_per_frame"][a] = {
            "mean": m, "sd_across_seeds": sd,
            "annotated_frames_mean": float(sub[sub.has_annotation].n_unmatched_peaks.mean()),
            "unannotated_frames_mean": float(sub[~sub.has_annotation].n_unmatched_peaks.mean())}
    res["unmatched_per_frame"]["note"] = ("OUTPUT BURDEN. Never false alarms, never Pfa: "
                                          "CARRADA cannot support a false-alarm measurement.")

    for col, key in (("range_stratum", "by_range"), ("doppler_stratum", "by_doppler"),
                     ("class", "by_class")):
        res[key] = {a: strat(dfo, a, col) for a in ARMS}
    res["by_class_warning"] = ("class recall is confounded by mask size: larger objects present "
                               "more support cells and are easier to hit")

    res["seed_variability"] = {}
    for a in ARMS:
        v = np.array([x for x in per_seed[a].values()], dtype=float)
        res["seed_variability"][a] = {
            "n": int(v.size), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}
    res["seed_variability"]["note"] = "reported separately; never merged into the sequence CI"
    res["seeds_above_CA"] = {a: int(sum(1 for v in per_seed[a].values() if v > pool["D"]))
                             for a in ("A", "B", "C")}

    # ---------------- Phase 10: evaluation-only burden curve
    print("Phase 10: evaluation-only burden sweep (selects nothing)", flush=True)
    objs = {}
    for seq in ci.sequences("Validation"):
        for fr in ci.frames(seq):
            objs[(seq, fr)] = ci.objects(seq, fr)

    man = json.load(open(os.path.join(VAL, "run_manifest.json"), encoding="utf-8"))
    thr_of = man["thresholds"]
    curves = {}
    for a in ("A", "B", "C"):
        pk = {s: load_peaks(os.path.join(VAL, "%s_%d_peaks.npz" % (a, s))) for s in range(8)}
        anchors = [thr_of["%s_s%d" % (a, s)] for s in range(8)]
        curves[a] = sweep_arm(pk, objs, anchors=anchors)
        print("  %s done" % a, flush=True)
    pkD = {None: load_peaks(os.path.join(VAL, "D_None_peaks.npz"))}
    curves["D"] = sweep_arm(pkD, objs, anchors=[man["cfar_alpha"]])
    print("  D done", flush=True)

    # Sanity: the curve must pass through each arm's measured frozen operating point.
    res["curve_consistency"] = {}
    for a in ("A", "B", "C", "D"):
        got = recall_at_burden(curves[a], res["unmatched_per_frame"][a]["mean"])
        res["curve_consistency"][a] = {
            "frozen_recall": pool[a], "curve_recall_at_frozen_burden": got,
            "abs_diff": (None if got is None else abs(got - pool[a]))}
        print("    %s curve at frozen burden %.2f -> %.4f   (measured %.4f)"
              % (a, res["unmatched_per_frame"][a]["mean"],
                 got if got is not None else float("nan"), pool[a]), flush=True)
    res["burden_curve"] = curves

    bD = res["unmatched_per_frame"]["D"]["mean"]
    res["recall_at_matched_burden"] = {
        "CA_frozen_burden": bD,
        "at_CA_burden": {a: recall_at_burden(curves[a], bD) for a in ("A", "B", "C", "D")},
        "at_2x_CA_burden": {a: recall_at_burden(curves[a], 2 * bD) for a in ("A", "B", "C", "D")},
        "note": "evaluation-only; the deployed thresholds were frozen on CARRADA TRAIN "
                "development data before Validation was opened and are not revisited"}

    p = os.path.join(EXP, "results", "validation_analysis.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    # ---------------- report
    P, AD, CB = res["primary_C_minus_D"], res["context_A_minus_D"], res["adaptation_C_minus_B"]
    print("\n" + "=" * 78)
    print("OFFICIAL CARRADA VALIDATION -- %d frames, %d eligible objects"
          % (res["n_frames"], res["eligible_objects"]))
    print("=" * 78)
    for a in ARMS:
        print("  %-4s %-34s recall %.4f   unmatched/frame %7.2f"
              % (a, NAME[a], pool[a], res["unmatched_per_frame"][a]["mean"]))
    print("")
    print("  PRIMARY      C - D  %+.4f  95%% CI [%+.4f, %+.4f]  crosses zero: %s"
          % (P["mean_delta"], P["ci_lo"], P["ci_hi"], P["crosses_zero"]))
    print("  adaptation   C - B  %+.4f  95%% CI [%+.4f, %+.4f]  crosses zero: %s"
          % (CB["mean_delta"], CB["ci_lo"], CB["ci_hi"], CB["crosses_zero"]))
    print("  context      B - A  %+.4f  95%% CI [%+.4f, %+.4f]"
          % (res["context_B_minus_A"]["mean_delta"], res["context_B_minus_A"]["ci_lo"],
             res["context_B_minus_A"]["ci_hi"]))
    print("  context      A - D  %+.4f  95%% CI [%+.4f, %+.4f]"
          % (AD["mean_delta"], AD["ci_lo"], AD["ci_hi"]))
    print("")
    print("  tolerance     A       B       C       D      C-D")
    for tol in (0, 1, 3):
        t = res["tolerance"]["tol%d" % tol]
        print("    %-6s %.4f  %.4f  %.4f  %.4f   %+.4f [%+.4f, %+.4f]"
              % ("exact" if tol == 0 else "+/-%d" % tol, t["A"], t["B"], t["C"], t["D"],
                 t["C_minus_D"]["mean_delta"], t["C_minus_D"]["ci_lo"],
                 t["C_minus_D"]["ci_hi"]))
    print("")
    rb = res["recall_at_matched_burden"]
    print("  recall at matched burden (evaluation-only sweep):")
    print("    at CA's %.2f/frame :  %s" % (rb["CA_frozen_burden"],
          "  ".join("%s %s" % (k, ("%.4f" % v) if v is not None else "n/a")
                    for k, v in rb["at_CA_burden"].items())))
    print("    at 2x that         :  %s" %
          "  ".join("%s %s" % (k, ("%.4f" % v) if v is not None else "n/a")
                    for k, v in rb["at_2x_CA_burden"].items()))
    print("")
    print("  range strata        A       B       C       D")
    for k in sorted(res["by_range"]["D"]):
        print("    %-10s n=%4d  %.4f  %.4f  %.4f  %.4f"
              % (k, res["by_range"]["D"][k]["n_objects"],
                 *[res["by_range"][a].get(k, {}).get("recall", float("nan"))
                   for a in ("A", "B", "C", "D")]))
    print("")
    print("  zero-Doppler        A       B       C       D")
    for k in sorted(res["by_doppler"]["D"]):
        print("    %-10s n=%4d  %.4f  %.4f  %.4f  %.4f"
              % (k, res["by_doppler"]["D"][k]["n_objects"],
                 *[res["by_doppler"][a].get(k, {}).get("recall", float("nan"))
                   for a in ("A", "B", "C", "D")]))
    print("")
    print("  class               A       B       C       D")
    for k in sorted(res["by_class"]["D"]):
        print("    %-10s n=%4d  %.4f  %.4f  %.4f  %.4f"
              % (k, res["by_class"]["D"][k]["n_objects"],
                 *[res["by_class"][a].get(k, {}).get("recall", float("nan"))
                   for a in ("A", "B", "C", "D")]))
    print("")
    for a in ("A", "B", "C"):
        v = res["seed_variability"][a]
        print("  seeds %-4s mean %.4f  sd %.4f  [%.4f, %.4f]   above CA: %d/8"
              % (a, v["mean"], v["sd"], v["min"], v["max"], res["seeds_above_CA"][a]))
    print("\nwrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
