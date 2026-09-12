"""POST-HOC diagnostic: recall at MATCHED output burden.

NOT part of the pre-analysis plan, and it does not change the primary result. It is run because
the plan's own interpretation rules pose the question directly:

    "If recall improves while unmatched output increases dramatically, the conclusion must reflect
     that trade-off. ... Is any recall gain still meaningful given the burden change?"

At its frozen operating point the CARRADA-informed arm emits far more detections per frame than
the original arm. A detector that simply fires more often will recover more objects, so recall at
two different output densities is not a like-for-like comparison. This re-scores every arm at
thresholds chosen to equalise unmatched detections per frame, and asks whether the advantage
survives.

Method. Each arm's raw candidate lists were preserved before thresholding, so no inference is
repeated. For each arm and seed, the threshold is solved for by bisection on the score
distribution so that the surviving candidates per frame hit a target, then the frame is scored
through the SAME `metrics.score_frame` at tolerance 0 and the realised burden and recall measured.

Arms A and C are read from the Chapter 7 artefacts; nothing there is recomputed or modified.

    python burden_matched.py
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
E6 = os.path.join(ROOT, "experiments", "e6_nms_audit")
sys.path.insert(0, os.path.join(R_EXP, "src"))

import carrada_input as ci                                               # noqa: E402
import eligibility as el                                                 # noqa: E402
import metrics as mt                                                     # noqa: E402
import stats as st                                                       # noqa: E402

REF = os.path.join(E6, "results", "p4_carrada", "r1_test")
ARMB = os.path.join(EXP, "results", "test_armB_r5")


def load_peaks(path):
    z = np.load(path, allow_pickle=True)
    pk, idx = z["peaks"], z["index"]
    out = {}
    for seq, fr, a, b in idx:
        out[(str(seq), str(fr))] = pk[int(a):int(b)]
    return out


def threshold_for_rate(peaks, target_per_frame):
    """Bisect on the pooled score distribution for a given surviving-candidates-per-frame rate."""
    allsc = np.concatenate([p[:, 2] for p in peaks.values() if len(p)]) if peaks else np.zeros(0)
    if allsc.size == 0:
        return np.inf
    n_frames = len(peaks)
    want = target_per_frame * n_frames
    if want >= allsc.size:
        return float(allsc.min()) - 1.0
    return float(np.partition(allsc, allsc.size - int(want))[allsc.size - int(want)])


def score_arm(peaks, objs_by_frame, thr):
    """Recall over eligible objects, and unmatched detections per frame, at one threshold."""
    matched, elig, unmatched, nfr = [], 0, 0, 0
    per_seq = {}
    for (seq, fr), pk in peaks.items():
        keep = pk[pk[:, 2] >= thr] if len(pk) else np.zeros((0, 3))
        objs = objs_by_frame[(seq, fr)]
        res = mt.score_frame(keep, objs, tolerance=0)
        unmatched += res["n_unmatched_peaks"]
        nfr += 1
        for row in res["per_object"]:
            if row["eligible"]:
                elig += 1
                matched.append(row["matched"])
                per_seq.setdefault(seq, []).append(row["matched"])
    return (float(np.mean(matched)) if matched else np.nan, unmatched / max(1, nfr),
            {s: float(np.mean(v)) for s, v in per_seq.items()})


def main():
    seqs = ci.sequences("Test")
    objs_by_frame = {}
    for seq in seqs:
        for fr in ci.frames(seq):
            objs_by_frame[(seq, fr)] = ci.objects(seq, fr)
    print("loaded annotations for %d frames" % len(objs_by_frame), flush=True)

    A = {s: load_peaks(os.path.join(REF, "tinycnn_seed%d_peaks.npz" % s)) for s in range(8)}
    B = {s: load_peaks(os.path.join(ARMB, "tinycnn_ci_seed%d_peaks.npz" % s)) for s in range(8)}
    C = load_peaks(os.path.join(REF, "cfar_CA_peaks.npz"))

    base = json.load(open(os.path.join(EXP, "results", "test_analysis_r5.json"), encoding="utf-8"))
    targets = {
        "CA-CFAR level": base["unmatched_per_frame"]["C"]["mean"],
        "original-TinyCNN level": base["unmatched_per_frame"]["A"]["mean"],
        "CARRADA-informed level": base["unmatched_per_frame"]["B"]["mean"],
    }

    rows = []
    for label, target in targets.items():
        print("\n--- matching burden to the %s: %.2f unmatched/frame" % (label, target),
              flush=True)
        for arm, peaks in (("A", A), ("B", B)):
            rec, bur, per_seq_all = [], [], {}
            for s in range(8):
                t = threshold_for_rate(peaks[s], target + 0.8)   # +matched, ~0.76/frame
                r, b, ps = score_arm(peaks[s], objs_by_frame, t)
                rec.append(r)
                bur.append(b)
                for k, v in ps.items():
                    per_seq_all.setdefault(k, []).append(v)
            m = {k: float(np.mean(v)) for k, v in per_seq_all.items()}
            rows.append(dict(target_label=label, target=target, arm=arm,
                             recall=float(np.mean(rec)), burden=float(np.mean(bur)),
                             per_sequence=m))
            print("    arm %s  recall %.4f   realised burden %.2f" % (arm, np.mean(rec),
                                                                     np.mean(bur)), flush=True)
        # CFAR has no seeds; score it once at each target too
        t = threshold_for_rate(C, target + 0.8)
        r, b, ps = score_arm(C, objs_by_frame, t)
        rows.append(dict(target_label=label, target=target, arm="C", recall=r, burden=b,
                         per_sequence=ps))
        print("    arm C  recall %.4f   realised burden %.2f" % (r, b), flush=True)

    out = {"what": "POST-HOC recall at matched output burden. Not pre-registered; the primary "
                   "result is unchanged.",
           "rows": rows}
    # B - A at each matched burden, same sequence-paired bootstrap
    for label in targets:
        a = [r for r in rows if r["target_label"] == label and r["arm"] == "A"][0]
        b = [r for r in rows if r["target_label"] == label and r["arm"] == "B"][0]
        d = {s: b["per_sequence"][s] - a["per_sequence"][s] for s in a["per_sequence"]}
        out.setdefault("B_minus_A_at_matched_burden", {})[label] = st.paired_sequence_bootstrap(d)

    p = os.path.join(EXP, "results", "burden_matched.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("\n" + "=" * 78)
    print("RECALL AT MATCHED OUTPUT BURDEN  (post-hoc)")
    print("=" * 78)
    print("  burden level              arm A     arm B     arm C     B-A [95% CI]")
    for label, target in targets.items():
        g = {r["arm"]: r for r in rows if r["target_label"] == label}
        bb = out["B_minus_A_at_matched_burden"][label]
        print("  %-24s %.4f    %.4f    %.4f    %+.4f [%+.4f, %+.4f]"
              % ("%s (%.1f/frame)" % (label.split()[0], target),
                 g["A"]["recall"], g["B"]["recall"], g["C"]["recall"],
                 bb["mean_delta"], bb["ci_lo"], bb["ci_hi"]))
    print("\nwrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
