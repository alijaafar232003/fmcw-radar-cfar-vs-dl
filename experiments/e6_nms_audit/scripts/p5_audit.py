"""Phase 5 - conclusion-change audit: original radius 5 vs the audited radius.

Reads the frozen originals (never recomputed) and the e6 outputs, and classifies each conclusion.

Classification uses the confidence intervals ALREADY PUBLISHED in Chapter 5 as the yardstick,
because this audit deliberately skipped the bootstrap:

  NOT ESTABLISHED       |change| <= the old CI half-width for that condition. Per the frozen plan
                        this is never reported as "unchanged" -- the audit simply cannot resolve it.
  NUMERICAL, SAME       |change| within (1, 2] x CI half-width, sign of the difference preserved.
  MATERIAL              |change| > 2 x CI half-width, sign preserved.
  REVERSED              sign of the TinyCNN - CFAR difference flips.

Writes only inside the experiment folder.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))
CAR = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
OUT = os.path.join(EXP, "results", "p5_audit")


def classify(old_diff, new_diff, half):
    if not np.isfinite(old_diff) or not np.isfinite(new_diff):
        return "UNAVAILABLE"
    if np.sign(old_diff) != np.sign(new_diff) and abs(old_diff) > half and abs(new_diff) > half:
        return "REVERSED"
    d = abs(new_diff - old_diff)
    if d <= half:
        return "NOT ESTABLISHED"
    if d <= 2 * half:
        return "NUMERICAL, SAME"
    return "MATERIAL"


def synthetic_rows(radius):
    old = pd.read_csv(os.path.join(ROOT, "results", "dl",
                                   "bootstrap_frozen_ep021_test_tol1.csv")).set_index("condition")
    new = pd.read_csv(os.path.join(EXP, "results", "p3_synth", "holdout_scores.csv"))
    n1 = new[(new.gate == 1)]

    rows = []
    for cond in old.index:
        half = float((old.loc[cond, "ci_hi"] - old.loc[cond, "ci_lo"]) / 2.0)
        sub = n1[n1.condition == cond]
        if not len(sub):
            continue
        out = {}
        for r in (radius, 5):
            dl = sub[(sub.nms == r) & (sub.method == "DL")]
            cf = sub[(sub.nms == r) & (sub.method != "DL")]
            if not len(dl) or not len(cf):
                continue
            best = cf.loc[cf["pd"].idxmax()]
            out[r] = dict(dl=float(dl["pd"].iloc[0]), cfar=float(best["pd"]),
                          cfar_method=best["method"],
                          diff=float(dl["pd"].iloc[0]) - float(best["pd"]))
        if radius not in out or 5 not in out:
            continue
        rows.append(dict(
            domain="synthetic", item=cond,
            old_frames_r5_diff=float(old.loc[cond, "diff"]),
            new_frames_r5_diff=out[5]["diff"],
            new_frames_rN_diff=out[radius]["diff"],
            ci_half=half,
            # Protocol effect is measured on the SAME (fresh) frames, isolating radius from sampling.
            protocol_effect=out[radius]["diff"] - out[5]["diff"],
            sampling_effect=out[5]["diff"] - float(old.loc[cond, "diff"]),
            verdict=classify(out[5]["diff"], out[radius]["diff"], half),
        ))
    return rows


def carrada_rows(radius):
    op = os.path.join(CAR, "results", "r1_test", "r1_summary.json")
    np_ = os.path.join(EXP, "results", "p4_carrada", "r1_test", "r1_summary.json")
    if not (os.path.exists(op) and os.path.exists(np_)):
        return [], None, None
    o, n = json.load(open(op)), json.load(open(np_))

    # The published sequence-balanced CI is the yardstick here.
    pb = o.get("paired_bootstrap", {})
    half = float((pb.get("ci_hi", np.nan) - pb.get("ci_lo", np.nan)) / 2.0)

    rows = []

    def add(item, od, nd, h=half):
        rows.append(dict(domain="carrada", item=item, old=od, new=nd,
                         change=nd - od, ci_half=h, verdict=classify(od, nd, h)))

    od = o["primary"]["tinycnn_seed_mean"] - o["primary"]["cfar"]["CA"]
    nd = n["primary"]["tinycnn_seed_mean"] - n["primary"]["cfar"]["CA"]
    add("overall TinyCNN - CA recall", od, nd)

    o_lead = sum(1 for s in o["per_sequence"] if s["delta"] < 0)
    n_lead = sum(1 for s in n["per_sequence"] if s["delta"] < 0)
    rows.append(dict(domain="carrada", item="sequences where CA leads (of 6)",
                     old=o_lead, new=n_lead, change=n_lead - o_lead, ci_half=np.nan,
                     verdict="UNCHANGED" if o_lead == n_lead else "MATERIAL"))

    for t in ("tol0", "tol1", "tol3"):
        add("tolerance %s (TinyCNN - CA)" % t,
            o["tolerance"][t]["tinycnn"] - o["tolerance"][t]["cfar_CA"],
            n["tolerance"][t]["tinycnn"] - n["tolerance"][t]["cfar_CA"])

    for k in ("near_zero", "away_zero"):
        add("doppler %s (TinyCNN - CA)" % k,
            o["by_doppler"][k]["tinycnn"] - o["by_doppler"][k]["cfar_CA"],
            n["by_doppler"][k]["tinycnn"] - n["by_doppler"][k]["cfar_CA"])

    ob, nb = o.get("unmatched_burden", {}), n.get("unmatched_burden", {})
    for k in ("TinyCNN", "CFAR-CA"):
        if k in ob and k in nb:
            ov = ob[k] if not isinstance(ob[k], dict) else list(ob[k].values())[0]
            nv = nb[k] if not isinstance(nb[k], dict) else list(nb[k].values())[0]
            rows.append(dict(domain="carrada", item="unmatched/frame %s" % k,
                             old=float(ov), new=float(nv), change=float(nv) - float(ov),
                             ci_half=np.nan, verdict="see raw"))

    add("seed spread (sd)", o["primary"]["tinycnn_seed_sd"], n["primary"]["tinycnn_seed_sd"],
        h=0.02)
    return rows, o, n


def main():
    radius = int(json.load(open(os.path.join(EXP, "MANIFEST.json")))["selected_nms_radius"])
    os.makedirs(OUT, exist_ok=True)

    syn = synthetic_rows(radius)
    car, o, n = carrada_rows(radius)

    pd.DataFrame(syn).to_csv(os.path.join(OUT, "synthetic_audit.csv"), index=False)
    if car:
        pd.DataFrame(car).to_csv(os.path.join(OUT, "carrada_audit.csv"), index=False)

    print("=== SYNTHETIC: radius %d vs 5, on the SAME fresh holdout ===" % radius)
    for r in sorted(syn, key=lambda x: -abs(x["protocol_effect"])):
        print("  %-13s r5 %+0.4f -> r%d %+0.4f   protocol %+0.4f  sampling %+0.4f   %s"
              % (r["item"], r["new_frames_r5_diff"], radius, r["new_frames_rN_diff"],
                 r["protocol_effect"], r["sampling_effect"], r["verdict"]))
    if car:
        print("\n=== CARRADA: radius %d vs 5 ===" % radius)
        for r in car:
            print("  %-34s old %+0.4f  new %+0.4f  change %+0.4f   %s"
                  % (r["item"], r["old"], r["new"], r["change"], r["verdict"]))
    else:
        print("\n[CARRADA outputs not present yet]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
