"""Rebuild the R1 per-object / per-frame tables from the PRESERVED raw Test output.

Why this exists: `run_r1_test.py` completed all detector inference and preserved the raw peaks for
all 12 arms, then failed on the very last step writing Parquet (neither pyarrow nor fastparquet is
installed). That is a SERIALIZATION failure after scoring, not a methodological one.

No detector is re-run. This script reads the immutable `*_peaks.npz`, applies the already-frozen
per-seed TinyCNN thresholds and frozen CFAR alphas, and runs the same frozen matcher — the identical
computation the crashed run performed in memory. Only the output format changes (CSV).
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

# ---------------------------------------------------------------- E6 PATCH (auto-generated)
# Isolated copy. The original lives in experiments/r_carrada_real_data_validation/scripts/ and is
# not modified. Two changes only: the NMS radius override below, and output paths redirected into
# the e6 experiment folder. Reads of frozen inputs still point at the original experiment.
E6_DIR = "C:/Users/USER/Desktop/Radar-fyp/My work/Simulations/experiments/e6_nms_audit"
E6_NMS_RADIUS = 2
# This copy lives one directory deeper than the original, so the script's own EXP/ROOT derivation
# from __file__ would point at the wrong places. Pin them to the originals: EXP is the CARRADA
# experiment (read-only source of src/ and its frozen inputs), ROOT is the project root.
EXP = "C:/Users/USER/Desktop/Radar-fyp/My work/Simulations/experiments/r_carrada_real_data_validation"
ROOT = "C:/Users/USER/Desktop/Radar-fyp/My work/Simulations"
# ----------------------------------------------------------------------------------------------

sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import carrada_input as ci   # noqa: E402
import detectors as det      # noqa: E402
det.NMS_SIZE = E6_NMS_RADIUS          # E6 PATCH
import eligibility as el     # noqa: E402
import metrics as mt         # noqa: E402

OUT = os.path.join(E6_DIR, "results", "p4_carrada", "r1_test")


def load_arm(tag):
    z = np.load(os.path.join(OUT, "{}_peaks.npz".format(tag)), allow_pickle=True)
    peaks, index = z["peaks"], z["index"]
    out = {}
    for seq, fr, a, b in index:
        out[(str(seq), str(fr))] = peaks[int(a):int(b)]
    return out


def main():
    thr = {int(r["seed"]): float(r["threshold"]) for r in json.load(
        open(os.path.join(E6_DIR, "configs", "tinycnn_zero_shot_thresholds.json")))["thresholds"]}
    alphas = {k: det.cfar_alpha(k) for k in det.CFAR_KINDS}

    arms = {}
    for s in det.SEEDS:
        arms[("tinycnn", None, s)] = (load_arm("tinycnn_seed{}".format(s)), thr[s])
    for k in det.CFAR_KINDS:
        arms[("cfar", k, None)] = (load_arm("cfar_{}".format(k)), alphas[k])

    per_object, per_frame, multipeak = [], [], []
    seqs = ci.sequences("Test")
    for seq in seqs:
        for fr in ci.frames(seq):
            objs = ci.objects(seq, fr)
            supports = {o["object_id"]: el.valid_support(o["dense"]) for o in objs}
            for (dname, kind, seed), (store, cut) in arms.items():
                pk_all = store.get((seq, fr), np.zeros((0, 3)))
                pk = det.apply_threshold(pk_all, cut)
                for tol in mt.TOLERANCES:
                    res = mt.score_frame(pk, objs, tolerance=tol)
                    for row in res["per_object"]:
                        vs = supports[row["object_id"]]
                        row.update(split="Test", detector=dname, cfar_kind=kind, seed=seed,
                                   range_stratum=el.range_stratum(vs),
                                   doppler_stratum=el.doppler_stratum(vs))
                        per_object.append(row)
                    if tol == 0:
                        per_frame.append({
                            "split": "Test", "sequence": seq, "frame": fr, "detector": dname,
                            "cfar_kind": kind, "seed": seed, "n_peaks": res["n_peaks"],
                            "n_matched_peaks": res["n_matched_peaks"],
                            "n_unmatched_peaks": res["n_unmatched_peaks"],
                            "n_eligible_objects": res["n_eligible_objects"],
                            "has_annotation": len(objs) > 0})
                        # multi-peak diagnostic on TEST (secondary)
                        for o in objs:
                            vs = supports[o["object_id"]]
                            if len(vs) == 0:
                                continue
                            sup = set(map(tuple, vs.tolist()))
                            n_in = sum(1 for p in pk if (int(p[0]), int(p[1])) in sup)
                            multipeak.append({
                                "sequence": seq, "frame": fr, "object_id": o["object_id"],
                                "class": o["class"], "detector": dname, "cfar_kind": kind,
                                "seed": seed, "peaks_inside_support": n_in})
        print("  rebuilt", seq, flush=True)

    pd.DataFrame(per_object).to_csv(os.path.join(OUT, "per_object.csv"), index=False)
    pd.DataFrame(per_frame).to_csv(os.path.join(OUT, "per_frame.csv"), index=False)
    pd.DataFrame(multipeak).to_csv(os.path.join(OUT, "multipeak_test.csv"), index=False)
    print("\nper_object rows {:,}  per_frame rows {:,}  multipeak rows {:,}".format(
        len(per_object), len(per_frame), len(multipeak)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
