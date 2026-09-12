"""PHASE 10 — the single frozen CARRADA TEST evaluation of the CARRADA-informed arm.

This is the one read of CARRADA test this experiment makes. Everything it uses was fixed
beforehand and hashed in results/freeze.json.

SCOPE. It scores ARM B ONLY (the CARRADA-informed 8-seed TinyCNN population). Arm A (original
TinyCNN) and arm C (CA-CFAR) are NOT recomputed: their radius-2 results already exist in
experiments/e6_nms_audit/results/p4_carrada/r1_test/, they are unchanged by this experiment, and
re-running them could only introduce a difference where there should be none. The analysis step
reads them from there.

PROTOCOL REUSE. Every scoring primitive is imported from the original CARRADA experiment's `src/`
and called unmodified:

    carrada_input.frames / load_rd_db / objects      the data path and the dB->power adapters
    detectors.tinycnn_peaks                          candidate extraction (uses det.NMS_SIZE)
    detectors.apply_threshold                        the frozen threshold application
    metrics.score_frame, metrics.TOLERANCES          one-to-one matching, exact / +-1 / +-3
    eligibility.valid_region / valid_support         the detector-valid interior
    eligibility.range_stratum / doppler_stratum      the frozen Chapter 7 strata

Two things are rebound before use, and only these two:

    det.NMS_SIZE        -> 2, the final protocol radius. `tinycnn_peaks` passes this through to
                           `predict_points` explicitly, so unlike the threshold-derivation path it
                           genuinely takes effect (see FROZEN_CONFIG.yaml).
    det.checkpoint_path -> this experiment's checkpoints.

Writes only under experiments/carrada_informed_domain/results/test_armB/.

    python run_test_armB.py --thresholds r5      # primary
    python run_test_armB.py --thresholds r2      # pre-registered sensitivity check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
sys.path.insert(0, os.path.join(R_EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import carrada_input as ci                                               # noqa: E402
import detectors as det                                                  # noqa: E402
import eligibility as el                                                 # noqa: E402
import metrics as mt                                                     # noqa: E402

NMS = 2
SPLIT = "Test"
CKPT = os.path.join(EXP, "checkpoints")

det.NMS_SIZE = NMS
det.checkpoint_path = lambda seed: os.path.join(CKPT, "tinycnn_ci_s%d_ep021.pt" % seed)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="r5", choices=("r5", "r2"))
    a = ap.parse_args()

    out = os.path.join(EXP, "results", "test_armB_%s" % a.thresholds)
    os.makedirs(out, exist_ok=True)

    tp = os.path.join(EXP, "configs", "thresholds_%s.json" % a.thresholds)
    T = json.load(open(tp, encoding="utf-8"))
    thr = {int(t["seed"]): float(t["threshold"]) for t in T["thresholds"]}
    assert T["carrada_data_used"] is False and T["source_split"] == "carrada_informed_val"

    seqs = ci.sequences(SPLIT)
    freeze = json.load(open(os.path.join(EXP, "results", "freeze.json"), encoding="utf-8"))

    print("=" * 78)
    print("CARRADA TEST ACCESS BEGINS HERE  (arm B, CARRADA-informed, thresholds %s)"
          % a.thresholds)
    print("=" * 78)
    print("timestamp        :", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("freeze rolling   :", freeze["rolling_sha256"][:16], "...")
    print("frozen config    :", freeze["frozen_config_sha256"][:16], "...")
    print("pre-analysis plan:", freeze["pre_analysis_plan_sha256"][:16], "...")
    print("NMS radius       :", det.NMS_SIZE)
    print("thresholds       :", {k: round(v, 4) for k, v in sorted(thr.items())})
    print("sequences        :", len(seqs))
    print()

    models = {s: det.load_frozen_tinycnn(s) for s in det.SEEDS}
    valid = el.valid_region()

    raw, per_object, per_frame = {}, [], []
    t0, n_frames = time.time(), 0

    for seq in seqs:
        frames = ci.frames(seq)
        tseq = time.time()
        for fr in frames:
            D = ci.load_rd_db(seq, fr)
            objs = ci.objects(seq, fr)
            n_frames += 1

            for s in det.SEEDS:
                pk_all = det.tinycnn_peaks(models[s], D, valid=valid)
                pk = det.apply_threshold(pk_all, thr[s])
                raw.setdefault("tinycnn_ci_seed%d" % s, []).append(
                    (seq, fr, pk_all if len(pk_all) else np.zeros((0, 3))))

                for tol in mt.TOLERANCES:
                    res = mt.score_frame(pk, objs, tolerance=tol)
                    for row in res["per_object"]:
                        vs = el.valid_support(
                            [o for o in objs if o["object_id"] == row["object_id"]][0]["dense"])
                        row.update(split=SPLIT, detector="tinycnn_ci", cfar_kind=None, seed=s,
                                   range_stratum=el.range_stratum(vs),
                                   doppler_stratum=el.doppler_stratum(vs))
                        per_object.append(row)
                    if tol == 0:
                        per_frame.append({
                            "split": SPLIT, "sequence": seq, "frame": fr,
                            "detector": "tinycnn_ci", "cfar_kind": None, "seed": s,
                            "n_peaks": res["n_peaks"],
                            "n_matched_peaks": res["n_matched_peaks"],
                            "n_unmatched_peaks": res["n_unmatched_peaks"],
                            "n_eligible_objects": res["n_eligible_objects"],
                            "has_annotation": len(objs) > 0})
        print("  %s  %d frames  (%.0fs)" % (seq, len(frames), time.time() - tseq), flush=True)

    for tag, rows in raw.items():
        idx, arrs, off = [], [], 0
        for seq, fr, arr in rows:
            idx.append((seq, fr, off, off + len(arr)))
            arrs.append(arr)
            off += len(arr)
        np.savez_compressed(os.path.join(out, "%s_peaks.npz" % tag),
                            peaks=np.vstack(arrs) if arrs else np.zeros((0, 3)),
                            index=np.array(idx, dtype=object))

    dfo = pd.DataFrame(per_object)
    dff = pd.DataFrame(per_frame)
    dfo.to_csv(os.path.join(out, "per_object.csv"), index=False)
    dff.to_csv(os.path.join(out, "per_frame.csv"), index=False)

    manifest = {
        "arm": "B (CARRADA-informed TinyCNN)",
        "threshold_set": a.thresholds,
        "thresholds_file_sha256": sha(tp),
        "thresholds": thr,
        "nms_size": NMS,
        "split": SPLIT,
        "sequences": seqs,
        "n_frames": n_frames,
        "checkpoints": {s: sha(det.checkpoint_path(s)) for s in det.SEEDS},
        "freeze_rolling_sha256": freeze["rolling_sha256"],
        "frozen_config_sha256": freeze["frozen_config_sha256"],
        "pre_analysis_plan_sha256": freeze["pre_analysis_plan_sha256"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": round(time.time() - t0, 1),
        "arms_A_and_C": "not recomputed; read from e6_nms_audit/results/p4_carrada/r1_test",
    }
    with open(os.path.join(out, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    elig = dfo[(dfo.tolerance == 0) & (dfo.seed == 0)].eligible.sum()
    print("\n%d frames, %d eligible objects per seed, %.1f min"
          % (n_frames, elig, (time.time() - t0) / 60))
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
