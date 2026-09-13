"""Phase 2 — the internal 15/3 split of CARRADA TRAIN, fixed before any result exists.

SEQUENCE LEVEL ONLY. Frames from one physical recording never straddle the two sides, because
consecutive frames of a sequence are near-duplicates and a frame-level split would leak the
fine-tuning set into the development set almost perfectly.

TWO RULES WERE PERMITTED: a fixed seeded draw, or a metadata-balanced selection using only
pre-existing counts and geometry. Both are recorded here, and the chronology is explicit.

RULE 1 (seeded draw), run first: one draw of 3 from 18 with `numpy.random.default_rng(20260913)`.
It returned 2019-09-16-12-58-42 / 2019-09-16-13-03-38 / 2020-02-28-13-15-36, whose combined
annotations contain 681 cars, 70 cyclists and **zero pedestrians**. Pedestrians are the majority
object class in real CARRADA, so that development set cannot represent the task: early stopping and
the burden-matched threshold rule would both have been fixed on a car-only sample.

RULE 2 (metadata-balanced), adopted: enumerate all C(18,3) = 816 sequence triples and take the one
minimising a balance objective computed ONLY from annotation counts --

    sum_c |p_dev(c) - p_train(c)|  +  |objects_dev / objects_train - 1/6|  +  1.0 if the triple
    does not span both recording campaigns

with ties broken lexicographically. Deterministic, exhaustive, and computed before any detector
runs: no loss, no recall, no threshold and no model output enters it. The switch was made because
of the class composition above, NOT because of any performance observation -- no detector had been
run on CARRADA train at that point, and none is run by this script.

The resulting balance is reported so the reader can judge it.

Writes configs/internal_split.json. CARRADA VALIDATION and TEST are never referenced except to
assert the chosen sequences are disjoint from both.

    python make_split.py
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
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
sys.path.insert(0, os.path.join(R_EXP, "src"))
sys.path.insert(0, HERE)

import guard                                                             # noqa: E402
guard.install(guard.ALLOW_TRAIN)

import carrada_input as ci                                               # noqa: E402
import eligibility as el                                                 # noqa: E402

SEED = 20260913
N_DEV = 3
OUT = os.path.join(EXP, "configs", "internal_split.json")
CLASS_NAMES = {1: "pedestrian", 2: "cyclist", 3: "car"}


def describe(seqs):
    """Counts only: frames, annotated frames, objects, classes, median object range bin."""
    valid = el.valid_region()
    out = {}
    for s in seqs:
        frames = ci.frames(s)
        n_obj, cls, rng_bins, n_ann = 0, {}, [], 0
        for fr in frames:
            objs = ci.objects(s, fr)
            if objs:
                n_ann += 1
            for o in objs:
                vs = el.valid_support(o["dense"])
                if len(vs) == 0:
                    continue
                n_obj += 1
                c = o.get("class") or CLASS_NAMES.get(o.get("class_id"), "?")
                cls[c] = cls.get(c, 0) + 1
                rng_bins.append(float(np.median(vs[:, 0])))
        out[s] = {"n_frames": len(frames), "n_annotated_frames": n_ann,
                  "n_eligible_objects": n_obj, "classes": cls,
                  "median_object_range_bin": float(np.median(rng_bins)) if rng_bins else None,
                  "date": s[:7]}
    return out


def balanced_triple(train, info):
    """Rule 2: the C(18,3) triple minimising the declared count-only balance objective."""
    import itertools
    classes = sorted({c for s in train for c in info[s]["classes"]})
    tot_cls = {c: sum(info[s]["classes"].get(c, 0) for s in train) for c in classes}
    tot_obj = sum(tot_cls.values())
    p_train = {c: tot_cls[c] / tot_obj for c in classes}

    best, best_score = None, None
    for tri in itertools.combinations(sorted(train), N_DEV):
        n_obj = sum(info[s]["n_eligible_objects"] for s in tri)
        if n_obj == 0:
            continue
        cls = {c: sum(info[s]["classes"].get(c, 0) for s in tri) for c in classes}
        p_dev = {c: cls[c] / n_obj for c in classes}
        score = sum(abs(p_dev[c] - p_train[c]) for c in classes)
        score += abs(n_obj / tot_obj - 1.0 / 6.0)
        if len({info[s]["date"][:4] for s in tri}) < 2:
            score += 1.0
        if best_score is None or score < best_score - 1e-12:
            best, best_score = tri, score
    return sorted(best), float(best_score)


def main():
    train = list(guard.TRAIN)
    assert len(train) == 18, "expected 18 official Train sequences"

    info = describe(train)

    # Rule 1, recorded for the chronology, then superseded -- see the module docstring.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train), size=N_DEV, replace=False)
    rule1 = sorted(train[int(i)] for i in idx)
    rule1_cls = {}
    for s in rule1:
        for c, n in info[s]["classes"].items():
            rule1_cls[c] = rule1_cls.get(c, 0) + n

    dev, bal_score = balanced_triple(train, info)
    fit = sorted(s for s in train if s not in dev)
    print("rule 1 (seeded draw) gave %s" % ", ".join(rule1))
    print("  its class counts: %s  -> rejected: no pedestrians" % rule1_cls)
    print("rule 2 (metadata-balanced, 816 triples enumerated) objective %.4f" % bal_score)

    assert len(fit) == 15 and len(dev) == 3
    assert not set(dev) & set(fit)
    assert not (set(dev) | set(fit)) & set(guard.VALIDATION), "leaked into Validation"
    assert not (set(dev) | set(fit)) & set(guard.TEST), "leaked into Test"

    print("  internal development:")
    for s in dev:
        print("    ", s)

    tot = lambda ss, k: sum(info[s][k] for s in ss)
    summary = {
        "fit": {"n_sequences": len(fit), "n_frames": tot(fit, "n_frames"),
                "n_annotated_frames": tot(fit, "n_annotated_frames"),
                "n_eligible_objects": tot(fit, "n_eligible_objects")},
        "dev": {"n_sequences": len(dev), "n_frames": tot(dev, "n_frames"),
                "n_annotated_frames": tot(dev, "n_annotated_frames"),
                "n_eligible_objects": tot(dev, "n_eligible_objects")},
    }
    for side, ss in (("fit", fit), ("dev", dev)):
        cls = {}
        for s in ss:
            for c, n in info[s]["classes"].items():
                cls[c] = cls.get(c, 0) + n
        summary[side]["classes"] = cls
        summary[side]["dates"] = sorted({info[s]["date"] for s in ss})

    payload = {
        "rule": "RULE 2 (adopted): exhaustive enumeration of all C(18,3)=816 sequence triples, "
                "taking the minimum of sum_c |p_dev(c)-p_train(c)| + |obj_dev/obj_train - 1/6| "
                "+ 1.0 if the triple does not span both recording campaigns; ties broken "
                "lexicographically. Counts only -- no detector output of any kind.",
        "rule_1_superseded": {
            "rule": "one draw of %d from 18, numpy.random.default_rng(%d)" % (N_DEV, SEED),
            "seed": SEED,
            "dev_sequences": rule1,
            "class_counts": rule1_cls,
            "why_superseded": "contained zero pedestrians, the majority object class in real "
                              "CARRADA, so early stopping and the burden-matched threshold rule "
                              "would have been fixed on a car-only development sample. The switch "
                              "was made on annotation counts alone; no detector had been run.",
        },
        "balance_objective": bal_score,
        "level": "sequence (never frame)",
        "fit_sequences": fit,
        "dev_sequences": dev,
        "summary": summary,
        "per_sequence": info,
        "carrada_validation_referenced": False,
        "carrada_test_referenced": False,
        "access_report": guard.access_report(),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "sha256"},
                   sort_keys=True).encode()).hexdigest()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print("\n           sequences  frames  annotated  eligible objects  dates")
    for side in ("fit", "dev"):
        s = summary[side]
        print("  %-8s %4d      %5d    %5d      %5d             %s"
              % (side, s["n_sequences"], s["n_frames"], s["n_annotated_frames"],
                 s["n_eligible_objects"], ",".join(s["dates"])))
    print("\n  classes  fit %s" % summary["fit"]["classes"])
    print("           dev %s" % summary["dev"]["classes"])
    print("\n  dev share of objects: %.1f%%"
          % (100 * summary["dev"]["n_eligible_objects"]
             / max(1, summary["fit"]["n_eligible_objects"]
                   + summary["dev"]["n_eligible_objects"])))
    a = guard.access_report()
    print("\n  access: %d train sequences opened, validation %s, test %s"
          % (len(a["opened_train"]), a["validation_opened"], a["test_opened"]))
    print("  split sha256:", payload["sha256"][:32])
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
