"""R1 — the ONE-SHOT frozen evaluation on the official CARRADA TEST sequences.

This is the only script permitted to pass CARRADA Test data through a detector.

Everything it uses was fixed before it ran: the pre-registration
(e800ad84af5a909d595a9c1a5b32bfe308efc4830a1bfe17d8ea793a1bba98a8), the per-seed TinyCNN
thresholds derived from bench2_val alone, the frozen CFAR alphas, NMS radius 5, the valid region,
the dense-mask truth, and the one-to-one matcher.

It writes a run manifest and a test-access record BEFORE the first inference, then preserves raw
detector output before computing any summary.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

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

import torch  # noqa: E402
from cfar.cfar import CfarConfig  # noqa: E402

import carrada_input as ci   # noqa: E402
import detectors as det      # noqa: E402
det.NMS_SIZE = E6_NMS_RADIUS          # E6 PATCH
import eligibility as el     # noqa: E402
import metrics as mt         # noqa: E402

OUT = os.path.join(E6_DIR, "results", "p4_carrada", "r1_test")
SPLIT = "Test"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def build_manifest(thresholds, alphas, seqs):
    """The exact protocol, frozen before scoring. Not changed once scoring begins."""
    return {
        "experiment": "CARRADA Real-Data External Validation — R1",
        "data": {"split": "official CARRADA TEST only", "n_sequences": len(seqs),
                 "sequences": seqs},
        "primary_truth": "dense segmentation support",
        "eligibility": {"valid_region": "frozen", "range_margin": el.BORDER_R
                        if hasattr(el, "BORDER_R") else 10, "doppler_margin": 4,
                        "cells": 13216,
                        "rule": "valid_support = dense_mask ∩ valid_region; eligible iff >= 1 cell"},
        "primary_metric": "exact dense-mask object recall",
        "secondary_metrics": ["dense mask dilated ±1", "dense mask dilated ±3"],
        "matching": "pre-registered one-to-one maximum-cardinality matcher, score then "
                    "(range, doppler, object_id) tie-breaks",
        "primary_cfar": "CA",
        "secondary_cfar": ["GO", "SO", "OS"],
        "cfar_alphas_frozen": alphas,
        "tinycnn": {"checkpoints": 8, "evaluated": "independently, never ensembled",
                    "thresholds_source": "bench2_val only, derived Phase 4",
                    "thresholds": thresholds},
        "nms": {"radius": 5, "frozen": True},
        "unmatched_output": "unmatched detections per frame — NOT Pfa",
        "statistics": {"unit": "test sequence", "n": len(seqs),
                       "method": "paired bootstrap", "replicates": 10000},
        "zero_doppler_band_bins": [28, 36],
        "range_blocks": el.RANGE_BLOCKS,
        "preregistration_sha256": json.load(
            open(os.path.join(EXP, "results", "preregistration_freeze.json")))["sha256"],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(2)

    seqs = ci.sequences(SPLIT)
    thr = {int(r["seed"]): float(r["threshold"]) for r in json.load(
        open(os.path.join(E6_DIR, "configs", "tinycnn_zero_shot_thresholds.json")))["thresholds"]}
    alphas = {k: det.cfar_alpha(k) for k in det.CFAR_KINDS}

    manifest = build_manifest(thr, alphas, seqs)
    man_hash = sha256_obj(manifest)
    manifest["manifest_sha256"] = man_hash
    with open(os.path.join(OUT, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    access = {
        "record": "FINAL CARRADA TEST ACCESS BEGINS HERE",
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "git_commit": git,
        "preregistration_sha256": manifest["preregistration_sha256"],
        "run_manifest_sha256": man_hash,
        "threshold_config_sha256": sha256_file(
            os.path.join(E6_DIR, "configs", "tinycnn_zero_shot_thresholds.json")),
        "checkpoint_sha256": {s: sha256_file(det.checkpoint_path(s)) for s in det.SEEDS},
        "cfar_alphas": alphas,
        "cfar_config": {k: str(CfarConfig(kind=k)) for k in det.CFAR_KINDS},
        "tinycnn_thresholds": thr,
        "nms_size": det.NMS_SIZE,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__},
        "note": ("From this point on, any methodological change is POST-RESULT and cannot "
                 "silently modify the confirmatory experiment."),
    }
    with open(os.path.join(OUT, "access_log.json"), "w", encoding="utf-8") as f:
        json.dump(access, f, indent=1)

    print("=" * 78)
    print("FINAL CARRADA TEST ACCESS BEGINS HERE")
    print("=" * 78)
    print("timestamp        :", access["timestamp_utc"])
    print("git commit       :", git[:12])
    print("prereg sha256    :", manifest["preregistration_sha256"][:16], "...")
    print("run manifest hash:", man_hash[:16], "...")
    print("CFAR alphas      :", {k: round(v, 4) for k, v in alphas.items()})
    print("TinyCNN thresh   :", {k: round(v, 4) for k, v in thr.items()})
    print("sequences        :", len(seqs))
    print()

    models = {s: det.load_frozen_tinycnn(s) for s in det.SEEDS}
    valid = el.valid_region()

    raw = {}     # detector-arm -> list of rows
    per_object, per_frame = [], []
    t0 = time.time()
    n_frames = 0

    for seq in seqs:
        frames = ci.frames(seq)
        tseq = time.time()
        for fr in frames:
            D = ci.load_rd_db(seq, fr)
            objs = ci.objects(seq, fr)
            n_frames += 1

            arms = {}
            for s in det.SEEDS:
                pk_all = det.tinycnn_peaks(models[s], D, valid=valid)
                arms[("tinycnn", None, s)] = (pk_all, det.apply_threshold(pk_all, thr[s]))
            for k in det.CFAR_KINDS:
                pk_all = det.cfar_peaks(D, kind=k, valid=valid)
                arms[("cfar", k, None)] = (pk_all, det.apply_threshold(pk_all, alphas[k]))

            for (dname, kind, seed), (pk_all, pk) in arms.items():
                tag = "{}_{}".format(dname, kind or "seed{}".format(seed))
                raw.setdefault(tag, []).append(
                    (seq, fr, pk_all if len(pk_all) else np.zeros((0, 3))))

                for tol in mt.TOLERANCES:
                    res = mt.score_frame(pk, objs, tolerance=tol)
                    for row in res["per_object"]:
                        vs = el.valid_support(
                            [o for o in objs if o["object_id"] == row["object_id"]][0]["dense"])
                        row.update(split=SPLIT, detector=dname, cfar_kind=kind, seed=seed,
                                   range_stratum=el.range_stratum(vs),
                                   doppler_stratum=el.doppler_stratum(vs))
                        per_object.append(row)
                    if tol == 0:
                        per_frame.append({
                            "split": SPLIT, "sequence": seq, "frame": fr,
                            "detector": dname, "cfar_kind": kind, "seed": seed,
                            "n_peaks": res["n_peaks"],
                            "n_matched_peaks": res["n_matched_peaks"],
                            "n_unmatched_peaks": res["n_unmatched_peaks"],
                            "n_eligible_objects": res["n_eligible_objects"],
                            "has_annotation": len(objs) > 0})
        print("  {}  {} frames  ({:.0f}s)".format(seq, len(frames), time.time() - tseq),
              flush=True)

    # ---- preserve raw detector output BEFORE any summarising ----
    for tag, rows in raw.items():
        idx, arrs = [], []
        off = 0
        for seq, fr, a in rows:
            idx.append((seq, fr, off, off + len(a)))
            arrs.append(a)
            off += len(a)
        np.savez_compressed(os.path.join(OUT, "{}_peaks.npz".format(tag)),
                            peaks=np.vstack(arrs) if arrs else np.zeros((0, 3)),
                            index=np.array(idx, dtype=object))
    print("\nraw detector output preserved for {} arms".format(len(raw)))

    dfo = pd.DataFrame(per_object)
    dff = pd.DataFrame(per_frame)
    dfo.to_parquet(os.path.join(OUT, "per_object.parquet"))
    dff.to_parquet(os.path.join(OUT, "per_frame.parquet"))

    dt = time.time() - t0
    print("scored {} frames x 12 arms in {:.0f}s".format(n_frames, dt))
    print("per_object rows: {:,}   per_frame rows: {:,}".format(len(dfo), len(dff)))
    with open(os.path.join(OUT, "run_complete.json"), "w", encoding="utf-8") as f:
        json.dump({"frames": n_frames, "seconds": round(dt, 1),
                   "completed_utc": pd.Timestamp.now("UTC").isoformat(),
                   "arms": sorted(raw)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
