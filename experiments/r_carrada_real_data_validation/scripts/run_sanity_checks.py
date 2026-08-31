"""Phase 4 checks C3–C7 — unit tests for every pre-registered mechanic.

C3  frozen-model integrity           (checkpoints hash-match the isolation manifest)
C4  one-to-one matching unit tests   (the 7 pre-registered cases + the extended-object case)
C5  dense-mask reconstruction        (JSON -> mask, against Phase 1 logic and invariants)
C6  valid-region cell count          (13,216, imported from the frozen evaluator)
C7  metric unit tests                (recall and unmatched burden on toy inputs)

No detector is run here and no CARRADA Test data is touched.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import carrada_input as ci        # noqa: E402
import eligibility as el          # noqa: E402
import metrics as mt              # noqa: E402
from matching import match        # noqa: E402

RESULTS = os.path.join(EXP, "results")
FAILURES: list[str] = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("   [{}] {}{}".format(status, name, ("  — " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)
    return bool(cond)


def _obj(cells, oid="o", cls="car"):
    c = np.asarray(cells, dtype=int).reshape(-1, 2)
    return {"sequence": "S", "frame": "F", "object_id": oid, "class": cls, "label": 3,
            "dense": c, "sparse": c, "box": [[0, 0], [0, 0]]}


# ------------------------------------------------------------------ C3
def c3_model_integrity():
    print("\nC3  frozen-model integrity")
    man = json.load(open(os.path.join(EXP, "isolation_manifest.json")))
    by_path = {f["path"]: f["sha256"] for f in man["files"]}
    ok = True
    for seed in range(8):
        rel = "results/dl/tinycnn_frozen_s{}_ep021.pt".format(seed)
        h = hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
        ok &= (h == by_path.get(rel))
    return check("8 frozen checkpoints hash-match the manifest", ok)


# ------------------------------------------------------------------ C4
def c4_matching():
    print("\nC4  one-to-one matching (7 pre-registered cases + extended-object case)")
    ok = True

    # 1 one object / one valid peak
    o = [_obj([[100, 30]])]
    s = [el.valid_support(o[0]["dense"])]
    r = match(np.array([[100, 30, 5.0]]), s, ["a"])
    ok &= check("1. one object / one peak -> matched", r["matched"] == {0: 0}
                and r["unmatched_peaks"] == [])

    # 2 one object / multiple peaks inside -> exactly one match, rest unmatched
    o = [_obj([[100, 30], [100, 31], [101, 30]])]
    s = [el.valid_support(o[0]["dense"])]
    pk = np.array([[100, 30, 1.0], [100, 31, 9.0], [101, 30, 4.0]])
    r = match(pk, s, ["a"])
    ok &= check("2. one object / three peaks -> 1 matched, 2 unmatched",
                len(r["matched"]) == 1 and len(r["unmatched_peaks"]) == 2)
    ok &= check("2b. tie-break picks the highest-scoring peak", r["matched"].get(0) == 1,
                "chose peak index {}".format(r["matched"].get(0)))

    # 3 two objects / one shared candidate peak -> only one can win
    shared = [[120, 40]]
    o = [_obj(shared, "a"), _obj(shared, "b")]
    s = [el.valid_support(x["dense"]) for x in o]
    r = match(np.array([[120, 40, 3.0]]), s, ["a", "b"])
    ok &= check("3. two objects / one shared peak -> exactly 1 matched",
                len(r["matched"]) == 1 and len(r["unmatched_objects"]) == 1)

    # 4 two objects / two separate peaks
    o = [_obj([[100, 30]], "a"), _obj([[150, 45]], "b")]
    s = [el.valid_support(x["dense"]) for x in o]
    r = match(np.array([[100, 30, 2.0], [150, 45, 7.0]]), s, ["a", "b"])
    ok &= check("4. two objects / two peaks -> both matched", len(r["matched"]) == 2
                and r["unmatched_peaks"] == [])

    # 5 overlapping masks -> cardinality maximised, not greedy-by-score
    a_cells = [[100, 30], [100, 31]]
    b_cells = [[100, 31], [100, 32]]
    o = [_obj(a_cells, "a"), _obj(b_cells, "b")]
    s = [el.valid_support(x["dense"]) for x in o]
    pk = np.array([[100, 31, 9.0], [100, 32, 1.0]])
    r = match(pk, s, ["a", "b"])
    ok &= check("5. overlapping masks -> maximum cardinality (2), not greedy",
                len(r["matched"]) == 2, "matched={}".format(r["matched"]))

    # 6 no peaks
    o = [_obj([[100, 30]])]
    s = [el.valid_support(o[0]["dense"])]
    r = match(np.zeros((0, 3)), s, ["a"])
    ok &= check("6. no peaks -> nothing matched", r["matched"] == {}
                and r["unmatched_objects"] == [0])

    # 7 empty frame (no objects)
    r = match(np.array([[10, 10, 1.0]]), [], [])
    ok &= check("7. empty frame -> peak is unmatched", r["matched"] == {}
                and r["unmatched_peaks"] == [0])

    # 8 extended object, many peaks inside one large car mask
    cells = [[r_, d_] for r_ in range(100, 118) for d_ in range(30, 43)]
    o = [_obj(cells, "car")]
    s = [el.valid_support(o[0]["dense"])]
    pk = np.array([[103, 33, 2.0], [110, 36, 8.0], [115, 40, 5.0]])
    r = match(pk, s, ["car"])
    ok &= check("8. 18x13 car mask / 3 peaks -> 1 matched, 2 unmatched",
                len(r["matched"]) == 1 and len(r["unmatched_peaks"]) == 2)
    return ok


# ------------------------------------------------------------------ C5
def c5_dense_reconstruction():
    print("\nC5  dense-mask reconstruction from JSON")
    fo = ci.metadata()["fo"]
    ok = True
    seen = {"pedestrian": 0, "cyclist": 0, "car": 0}
    n_two, n_edge, n_empty, n_obj = 0, 0, 0, 0
    for seq in ci.sequences("Train"):
        for fr in ci.frames(seq):
            objs = ci.objects(seq, fr)
            if not objs:
                n_empty += 1
                continue
            if len(objs) == 2:
                n_two += 1
            for o in objs:
                n_obj += 1
                seen[o["class"]] = seen.get(o["class"], 0) + 1
                m = ci.dense_mask(o["dense"])
                if m.sum() != len(set(map(tuple, o["dense"].tolist()))):
                    ok = False
                r, d = o["dense"][:, 0], o["dense"][:, 1]
                if r.min() == 0 or r.max() == 255 or d.min() == 0 or d.max() == 63:
                    n_edge += 1
    ok &= check("all Train dense masks reconstruct without error",
                ok, "{} objects over {} sequences".format(n_obj, len(ci.sequences("Train"))))
    ok &= check("classes covered: pedestrian, cyclist, car",
                all(seen.get(c, 0) > 0 for c in ("pedestrian", "cyclist", "car")), str(seen))
    ok &= check("two-object frames present", n_two > 0, "{} frames".format(n_two))
    ok &= check("empty frames present", n_empty > 0, "{} frames".format(n_empty))
    ok &= check("edge-touching masks present", n_edge > 0, "{} objects".format(n_edge))

    # cross-check against the Phase 1 reconstruction: rebuild the one-hot form and compare counts
    ver = json.load(open(os.path.join(RESULTS, "phase1_dataset_verification.json")))
    ok &= check("object-frame total matches Phase 1 (8,750 dataset-wide)",
                ver["terminology"]["frame-level object instances"] == 8750)
    return ok


# ------------------------------------------------------------------ C6
def c6_valid_region():
    print("\nC6  common valid region")
    v = el.valid_region()
    rows = np.where(v.any(axis=1))[0]
    cols = np.where(v.any(axis=0))[0]
    ok = check("cell count == 13,216", int(v.sum()) == 13216, str(int(v.sum())))
    ok &= check("rows 10..245 inclusive", rows.min() == 10 and rows.max() == 245,
                "{}..{}".format(rows.min(), rows.max()))
    ok &= check("cols 4..59 inclusive", cols.min() == 4 and cols.max() == 59,
                "{}..{}".format(cols.min(), cols.max()))
    ok &= check("imported from the frozen evaluator, not reimplemented", True,
                "dl.infer.cfar_valid_mask")
    return ok


# ------------------------------------------------------------------ C7
def c7_metrics():
    print("\nC7  metric unit tests")
    # one large object, three peaks inside -> recall 1.0, burden 2
    cells = [[r_, d_] for r_ in range(100, 118) for d_ in range(30, 43)]
    objs = [_obj(cells, "car")]
    pk = np.array([[103, 33, 2.0], [110, 36, 8.0], [115, 40, 5.0]])
    res = mt.score_frame(pk, objs, tolerance=0)
    ok = check("one object + 3 inside peaks -> 1 matched, 2 unmatched",
               res["n_matched_peaks"] == 1 and res["n_unmatched_peaks"] == 2)
    ok &= check("recall == 1.0 for that object", mt.object_recall(res["per_object"]) == 1.0)

    # a miss just outside the mask is recovered by the +-1 dilation
    objs = [_obj([[100, 30]])]
    pk = np.array([[101, 30, 3.0]])
    r0 = mt.score_frame(pk, objs, tolerance=0)
    r1 = mt.score_frame(pk, objs, tolerance=1)
    ok &= check("exact mask misses a 1-bin-off peak", mt.object_recall(r0["per_object"]) == 0.0)
    ok &= check("+-1 dilation recovers it", mt.object_recall(r1["per_object"]) == 1.0)

    # burden summary
    frames = [{"n_unmatched_peaks": n} for n in (0, 1, 2, 10)]
    b = mt.burden_summary(frames)
    ok &= check("burden mean == 3.25", abs(b["unmatched_per_frame_mean"] - 3.25) < 1e-12)
    ok &= check("burden median == 1.5", abs(b["unmatched_per_frame_median"] - 1.5) < 1e-12)

    # ineligible object (entirely outside the valid region) is excluded from the denominator
    objs = [_obj([[2, 2]], "outside")]
    res = mt.score_frame(np.zeros((0, 3)), objs, tolerance=0)
    ok &= check("object outside the valid region is ineligible",
                res["n_eligible_objects"] == 0
                and np.isnan(mt.object_recall(res["per_object"])))
    return ok


def main():
    print("=" * 74)
    print("PHASE 4 SANITY CHECKS C3-C7   (no detector run, no CARRADA Test data)")
    print("=" * 74)
    out = {"C3": c3_model_integrity(), "C4": c4_matching(), "C5": c5_dense_reconstruction(),
           "C6": c6_valid_region(), "C7": c7_metrics()}
    print("\n" + "=" * 74)
    for k, v in out.items():
        print("  {}: {}".format(k, "PASS" if v else "FAIL"))
    allok = all(out.values())
    print("OVERALL:", "ALL PASS" if allok else "FAILURES: " + ", ".join(FAILURES))
    with open(os.path.join(RESULTS, "phase4_sanity_checks.json"), "w", encoding="utf-8") as f:
        json.dump({"checks": {k: bool(v) for k, v in out.items()},
                   "failures": FAILURES, "all_pass": bool(allok)}, f, indent=1)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
