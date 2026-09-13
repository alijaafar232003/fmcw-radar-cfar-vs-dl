"""Phases 3-4 — real-data labels, cached, and the fine-tuning pos_weight.

LABEL DEFINITION (Phase 3). For each CARRADA train frame the positive target is the UNION of the
provided dense segmentation masks, restricted to the detector-valid interior. Nothing is invented:
no centroids, no brightest cells, no bounding-box centres, no synthetic point truths. Frames with
no annotation are kept as genuine all-background examples rather than discarded, because a detector
that never sees empty scenes has no way to learn when not to fire.

This is a REAL CHANGE OF SUPERVISION, and it is the point of the experiment. The synthetic arms
were trained on flat 3x3 blocks around point targets; here the network is asked to mark extended
object support. So this fine-tuning tests REAL-DOMAIN ADAPTATION, not a continuation of the
original point-target task, and the two label definitions are not interchangeable.

INPUT. The frozen adapter, unchanged: `carrada_input.to_tinycnn_magnitude` then the frozen
`dl.dataset.normalize_rd`, which together compute D - median(D), clipped to [-10, +60] dB, / 20.

POS_WEIGHT RULE (Phase 4), declared before the ratio was computed:

    pos_weight = round(N_negative / N_positive) over the 15 FIT sequences, counted inside the
    detector-valid interior only, and CAPPED AT 100.

The cap is declared in advance, with its reason: 100 is the value the frozen synthetic contract
uses, and the project's own 3-way ablation established that exact class balancing (pos_weight 974)
manufactured a clutter result that had to be retracted as Correction C3. Fine-tuning must not apply
a STRONGER positive weighting than the original training survived. If the measured ratio is below
100 the cap does not bind and the measured value is used as-is.

The 3 development sequences contribute nothing to this number.

    python build_cache.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
sys.path.insert(0, os.path.join(R_EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import guard                                                             # noqa: E402
guard.install(guard.ALLOW_TRAIN)

import carrada_input as ci                                               # noqa: E402
import eligibility as el                                                 # noqa: E402
from dl.dataset import normalize_rd                                      # noqa: E402

CACHE = os.path.join(EXP, "cache")
SPLIT = json.load(open(os.path.join(EXP, "configs", "internal_split.json"), encoding="utf-8"))
POS_WEIGHT_CAP = 100.0
SHAPE = (256, 64)


def frame_label(objs, valid):
    """Union of dense object masks, restricted to the detector-valid interior."""
    y = np.zeros(SHAPE, dtype=bool)
    for o in objs:
        d = np.asarray(o["dense"], dtype=int)
        if len(d) == 0:
            continue
        r, c = d[:, 0], d[:, 1]
        ok = (r >= 0) & (r < SHAPE[0]) & (c >= 0) & (c < SHAPE[1])
        y[r[ok], c[ok]] = True
    return y & valid


def build(side, seqs, valid):
    xs, ys, index = [], [], []
    t0 = time.perf_counter()
    for s in seqs:
        for fr in ci.frames(s):
            db = ci.load_rd_db(s, fr)
            x = normalize_rd(ci.to_tinycnn_magnitude(db)).astype(np.float32)
            y = frame_label(ci.objects(s, fr), valid)
            xs.append(x)
            ys.append(y)
            index.append((s, fr))
        print("    %s  %d frames" % (s, len(ci.frames(s))), flush=True)
    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.uint8)
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, "%s_x.npy" % side), X)
    np.save(os.path.join(CACHE, "%s_y.npy" % side), Y)
    with open(os.path.join(CACHE, "%s_index.json" % side), "w", encoding="utf-8") as fh:
        json.dump([list(t) for t in index], fh)
    print("  %s: %s frames, %.2f min" % (side, X.shape, (time.perf_counter() - t0) / 60))
    return X, Y


def main():
    valid = el.valid_region()
    n_valid = int(valid.sum())
    print("detector-valid interior: %d of %d cells" % (n_valid, SHAPE[0] * SHAPE[1]))

    print("\nbuilding the FIT cache (15 sequences)")
    Xf, Yf = build("fit", SPLIT["fit_sequences"], valid)
    print("\nbuilding the DEV cache (3 sequences)")
    Xd, Yd = build("dev", SPLIT["dev_sequences"], valid)

    # ---- pos_weight, from the FIT side only, inside the valid interior only
    pos = int(Yf.sum())
    tot = int(Yf.shape[0]) * n_valid
    neg = tot - pos
    ratio = neg / max(1, pos)
    pos_weight = float(min(round(ratio), POS_WEIGHT_CAP))
    capped = round(ratio) > POS_WEIGHT_CAP

    # the same numbers on the dev side, reported but NOT used to choose anything
    posd = int(Yd.sum())
    totd = int(Yd.shape[0]) * n_valid

    rec = {
        "label_definition": "union of the provided CARRADA dense segmentation masks, restricted "
                            "to the detector-valid interior; annotation-empty frames kept as "
                            "all-background examples; nothing invented",
        "input_adapter": "carrada_input.to_tinycnn_magnitude then the frozen dl.dataset."
                         "normalize_rd (= D - median(D), clipped [-10,+60] dB, /20)",
        "supervision_change": "extended real object masks, NOT the synthetic flat 3x3 point "
                              "labels. This is real-domain adaptation, not a continuation of the "
                              "original point-target task.",
        "pos_weight_rule": "round(N_neg / N_pos) over the 15 FIT sequences inside the valid "
                           "interior, capped at %g; cap and reason declared before computing"
                           % POS_WEIGHT_CAP,
        "fit": {"n_frames": int(Yf.shape[0]), "valid_cells_per_frame": n_valid,
                "positive_cells": pos, "negative_cells": neg, "total_cells": tot,
                "positive_rate": pos / tot, "neg_over_pos": ratio},
        "dev_reported_not_used": {"n_frames": int(Yd.shape[0]), "positive_cells": posd,
                                  "positive_rate": posd / totd},
        "pos_weight": pos_weight,
        "cap_applied": bool(capped),
        "cap_value": POS_WEIGHT_CAP,
        "synthetic_reference": {"frozen_pos_weight": 100.0,
                                "synthetic_measured_ratio_approx": 943,
                                "note": "the synthetic label was a 3x3 block around a point "
                                        "target; CARRADA masks are extended, so the positive "
                                        "density is far higher and the ratio far lower"},
        "access_report": guard.access_report(),
    }
    rec["sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in rec.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()
    p = os.path.join(EXP, "configs", "label_and_posweight.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)

    print("\n" + "=" * 72)
    print("REAL-LABEL CLASS BALANCE  (15 fit sequences, valid interior only)")
    print("=" * 72)
    print("  frames            %d" % Yf.shape[0])
    print("  valid cells/frame %d" % n_valid)
    print("  positive cells    %d" % pos)
    print("  negative cells    %d" % neg)
    print("  positive rate     %.5f   (1 in %.1f)" % (pos / tot, 1 / (pos / tot)))
    print("  neg / pos ratio   %.2f" % ratio)
    print("  -> pos_weight     %g   %s" % (pos_weight,
                                           "(CAPPED at %g)" % POS_WEIGHT_CAP if capped
                                           else "(measured; cap did not bind)"))
    print("  dev positive rate %.5f   (reported only; not used to choose anything)"
          % (posd / totd))
    a = guard.access_report()
    print("\n  access: %d train sequences, validation %s, test %s"
          % (len(a["opened_train"]), a["validation_opened"], a["test_opened"]))
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
