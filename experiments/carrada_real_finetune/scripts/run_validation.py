"""PHASE 9 — the one evaluation on the official CARRADA VALIDATION split.

Four arms, 6 sequences, 2,448 frames, all scored through the Chapter 7 protocol:

    A  original TinyCNN              results/dl/tinycnn_frozen_s{0..7}_ep021.pt
    B  CARRADA-informed TinyCNN      carrada_informed_domain/checkpoints/tinycnn_ci_s{0..7}_ep021.pt
    C  real-fine-tuned TinyCNN       checkpoints/tinycnn_ft_s{0..7}.pt          (this experiment)
    D  CA-CFAR                       frozen geometry and alpha

None of the four has ever been scored on Validation before, so all four are run here; nothing is
reused from a previous split.

Every threshold is read from a file hashed in results/freeze.json. Nothing is re-tuned, re-selected
or re-thresholded after this script runs: it reads frozen inputs and writes per-object and
per-frame outcomes.

Primitives are the originals, imported from the CARRADA experiment's src/ and called unmodified:
`carrada_input`, `detectors.tinycnn_peaks`/`cfar_peaks`/`apply_threshold`, `metrics.score_frame`,
`eligibility.valid_region`/`valid_support`/`range_stratum`/`doppler_stratum`. Only two things are
rebound, per arm: det.NMS_SIZE (to the final radius 2) and det.checkpoint_path.

The allow-list guard runs in VALIDATION mode: Validation and Train are readable, CARRADA TEST
raises. Test is forbidden for this experiment and is not opened here or anywhere else.

    python run_validation.py
"""
from __future__ import annotations

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
CI_EXP = os.path.join(ROOT, "experiments", "carrada_informed_domain")
sys.path.insert(0, os.path.join(R_EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import guard                                                             # noqa: E402
guard.install(guard.ALLOW_VALIDATION)

import torch                                                             # noqa: E402

import carrada_input as ci                                               # noqa: E402
import detectors as det                                                  # noqa: E402
import eligibility as el                                                 # noqa: E402
import metrics as mt                                                     # noqa: E402
from dl.infer import predict_points                                      # noqa: E402
from dl.models import TinyCNN                                            # noqa: E402

NMS = 2
det.NMS_SIZE = NMS
SPLIT = "Validation"
OUT = os.path.join(EXP, "results", "validation")


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck.get("state_dict", ck)
    m = TinyCNN(width=16)
    m.load_state_dict(st)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def arms():
    """(arm, seed) -> (model, threshold). Every threshold comes from a hashed frozen file."""
    out = {}

    thrA = {int(t["seed"]): float(t["threshold"]) for t in json.load(
        open(os.path.join(CI_EXP, "..", "e6_nms_audit", "configs",
                          "tinycnn_zero_shot_thresholds.json"), encoding="utf-8"))["thresholds"]}
    for s in range(8):
        out[("A", s)] = (os.path.join(ROOT, "results", "dl",
                                      "tinycnn_frozen_s%d_ep021.pt" % s), thrA[s])

    thrB = {int(t["seed"]): float(t["threshold"]) for t in json.load(
        open(os.path.join(CI_EXP, "configs", "thresholds_r5.json"),
             encoding="utf-8"))["thresholds"]}
    for s in range(8):
        out[("B", s)] = (os.path.join(CI_EXP, "checkpoints",
                                      "tinycnn_ci_s%d_ep021.pt" % s), thrB[s])

    T = json.load(open(os.path.join(EXP, "configs", "thresholds_ft.json"), encoding="utf-8"))
    thrC = {int(k): float(v) for k, v in T["primary_thresholds"].items()}
    thrCin = {int(k): float(v) for k, v in T["secondary_inherited_thresholds"].items()}
    for s in range(8):
        out[("C", s)] = (os.path.join(EXP, "checkpoints", "tinycnn_ft_s%d.pt" % s), thrC[s])
        out[("Cinh", s)] = (os.path.join(EXP, "checkpoints", "tinycnn_ft_s%d.pt" % s), thrCin[s])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(6)
    freeze = json.load(open(os.path.join(EXP, "results", "freeze.json"), encoding="utf-8"))
    seqs = ci.sequences(SPLIT)
    valid = el.valid_region()
    spec = arms()

    print("=" * 78)
    print("OFFICIAL CARRADA VALIDATION ACCESS BEGINS HERE")
    print("=" * 78)
    print("timestamp      :", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("freeze rolling :", freeze["rolling_sha256"][:16], "...")
    print("NMS radius     :", det.NMS_SIZE)
    print("sequences      :", len(seqs))
    print("arms           : A original, B CARRADA-informed, C fine-tuned, "
          "Cinh fine-tuned@inherited, D CA-CFAR")
    print()

    models = {}
    for key, (path, _) in spec.items():
        if key[0] == "Cinh":
            models[key] = models[("C", key[1])]
        else:
            models[key] = load_model(path)

    alpha = det.cfar_alpha("CA")
    per_object, per_frame, raw = [], [], {}
    t0, nfr = time.time(), 0

    for seq in seqs:
        frames = ci.frames(seq)
        tseq = time.time()
        for fr in frames:
            D = ci.load_rd_db(seq, fr)
            objs = ci.objects(seq, fr)
            nfr += 1

            todo = {}
            for (arm, s), (_, thr) in spec.items():
                mag = ci.to_tinycnn_magnitude(D)
                pk_all = predict_points(models[(arm, s)], mag, valid=valid, nms_size=NMS)
                todo[(arm, s)] = (pk_all, det.apply_threshold(pk_all, thr))
            pk_all = det.cfar_peaks(D, kind="CA", valid=valid)
            todo[("D", None)] = (pk_all, det.apply_threshold(pk_all, alpha))

            for (arm, s), (pk_all, pk) in todo.items():
                raw.setdefault("%s_%s" % (arm, s), []).append(
                    (seq, fr, pk_all if len(pk_all) else np.zeros((0, 3))))
                for tol in mt.TOLERANCES:
                    res = mt.score_frame(pk, objs, tolerance=tol)
                    for row in res["per_object"]:
                        vs = el.valid_support(
                            [o for o in objs if o["object_id"] == row["object_id"]][0]["dense"])
                        row.update(split=SPLIT, arm=arm, seed=s,
                                   range_stratum=el.range_stratum(vs),
                                   doppler_stratum=el.doppler_stratum(vs))
                        per_object.append(row)
                    if tol == 0:
                        per_frame.append({
                            "split": SPLIT, "sequence": seq, "frame": fr, "arm": arm, "seed": s,
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
        np.savez_compressed(os.path.join(OUT, "%s_peaks.npz" % tag),
                            peaks=np.vstack(arrs) if arrs else np.zeros((0, 3)),
                            index=np.array(idx, dtype=object))

    dfo = pd.DataFrame(per_object)
    dff = pd.DataFrame(per_frame)
    dfo.to_csv(os.path.join(OUT, "per_object.csv"), index=False)
    dff.to_csv(os.path.join(OUT, "per_frame.csv"), index=False)

    man = {
        "split": SPLIT, "sequences": seqs, "n_frames": nfr, "nms_size": NMS,
        "cfar_alpha": alpha,
        "thresholds": {"%s_s%s" % (a, s): t for (a, s), (_, t) in spec.items()},
        "checkpoints": {"%s_s%s" % (a, s): hashlib.sha256(open(p, "rb").read()).hexdigest()
                        for (a, s), (p, _) in spec.items()},
        "freeze_rolling_sha256": freeze["rolling_sha256"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": round(time.time() - t0, 1),
        "access_report": guard.access_report(),
    }
    with open(os.path.join(OUT, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)

    elig = int(dfo[(dfo.tolerance == 0) & (dfo.arm == "D")].eligible.sum())
    a = guard.access_report()
    print("\n%d frames, %d eligible objects, %.1f min" % (nfr, elig, (time.time() - t0) / 60))
    print("access: validation %d seqs, test opened %s" % (len(a["opened_validation"]),
                                                          a["test_opened"]))
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
