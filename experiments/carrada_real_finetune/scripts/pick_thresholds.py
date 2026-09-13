"""Phase 8 — the burden-matched operating point, chosen on the 3 internal-development sequences.

WHY NOT A Pfa RULE. On synthetic data the operating point is set by a measured per-cell false-alarm
rate. CARRADA cannot support that: its annotations mark the objects that were labelled, not every
scatterer present, so an unmatched detection is not evidence of a false alarm. The operating point
must therefore be fixed by something CARRADA can actually measure, and what it can measure is
OUTPUT BURDEN -- how many detections a detector emits that no annotated object accounts for.

THE RULE, fixed before it was run:

    Run the frozen CA-CFAR on the 3 internal-development sequences and measure its unmatched
    detections per frame, B_CA. For each fine-tuned seed, sweep its own score threshold on those
    same sequences and select the threshold giving the HIGHEST exact dense-mask recall subject to
    unmatched detections per frame <= B_CA. If no threshold satisfies it exactly, take the closest
    one that does not exceed B_CA.

This produces a CA-burden-matched operating point without ever opening official CARRADA
Validation, which is what makes the eventual C-vs-CA comparison a comparison of detectors rather
than of output densities.

Official Validation cannot be read here: the allow-list guard runs in TRAIN mode, so any attempt
raises. The 15 fine-tuning sequences are also not used -- the models were fitted on them, so their
burden there would be optimistic.

Also records, as a SECONDARY operating point, the synthetic-derived threshold each seed inherits
from arm B, which answers "what if only the weights change and the threshold stays inherited?"

    python pick_thresholds.py
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

import torch                                                             # noqa: E402

import carrada_input as ci                                               # noqa: E402
import detectors as det                                                  # noqa: E402
import eligibility as el                                                 # noqa: E402
import metrics as mt                                                     # noqa: E402
from dl.infer import predict_points                                      # noqa: E402
from dl.models import TinyCNN                                            # noqa: E402

NMS = 2
det.NMS_SIZE = NMS
SPLIT = json.load(open(os.path.join(EXP, "configs", "internal_split.json"), encoding="utf-8"))
DEV = SPLIT["dev_sequences"]
CKPT = os.path.join(EXP, "checkpoints")
OUT = os.path.join(EXP, "configs", "thresholds_ft.json")


def dev_frames():
    out = []
    for s in DEV:
        for fr in ci.frames(s):
            out.append((s, fr))
    return out


def score_at(peaks_by_frame, objs_by_frame, thr):
    """Exact recall over eligible objects and unmatched detections per frame at one threshold."""
    matched, unmatched, nfr = [], 0, 0
    for key, pk in peaks_by_frame.items():
        keep = pk[pk[:, 2] >= thr] if len(pk) else np.zeros((0, 3))
        res = mt.score_frame(keep, objs_by_frame[key], tolerance=0)
        unmatched += res["n_unmatched_peaks"]
        nfr += 1
        for row in res["per_object"]:
            if row["eligible"]:
                matched.append(row["matched"])
    return (float(np.mean(matched)) if matched else np.nan, unmatched / max(1, nfr))


def main():
    torch.set_num_threads(6)
    valid = el.valid_region()
    frames = dev_frames()
    print("internal development: %d sequences, %d frames" % (len(DEV), len(frames)))

    objs = {}
    db_cache = {}
    for key in frames:
        objs[key] = ci.objects(*key)
        db_cache[key] = ci.load_rd_db(*key)

    # ---------------- CA-CFAR reference burden on the SAME sequences
    t0 = time.perf_counter()
    ca_peaks = {}
    alpha = det.cfar_alpha("CA")
    for key in frames:
        ca_peaks[key] = det.cfar_peaks(db_cache[key], kind="CA", valid=valid)
    ca_rec, ca_bur = score_at(ca_peaks, objs, alpha)
    print("CA-CFAR on internal dev:  recall %.4f   unmatched/frame %.3f   (%.0fs)"
          % (ca_rec, ca_bur, time.perf_counter() - t0))

    # ---------------- sweep each fine-tuned seed
    inherited = {int(t["seed"]): float(t["threshold"]) for t in json.load(
        open(os.path.join(ROOT, "experiments", "carrada_informed_domain", "configs",
                          "thresholds_r5.json"), encoding="utf-8"))["thresholds"]}

    rows = []
    for seed in range(8):
        ck = torch.load(os.path.join(CKPT, "tinycnn_ft_s%d.pt" % seed), weights_only=False)
        m = TinyCNN(width=16)
        m.load_state_dict(ck["state_dict"])
        m.eval()

        pk = {}
        for key in frames:
            mag = ci.to_tinycnn_magnitude(db_cache[key])
            pk[key] = predict_points(m, mag, valid=valid, nms_size=NMS)

        allsc = np.concatenate([p[:, 2] for p in pk.values() if len(p)])
        grid = np.unique(np.quantile(allsc, np.linspace(0.0, 1.0, 260)))
        grid = np.concatenate([grid, [allsc.max() + 1.0]])

        best = None
        curve = []
        for thr in grid:
            rec, bur = score_at(pk, objs, float(thr))
            curve.append({"threshold": float(thr), "recall": rec, "burden": bur})
            if bur <= ca_bur and np.isfinite(rec):
                if best is None or rec > best["recall"] + 1e-12:
                    best = {"threshold": float(thr), "recall": rec, "burden": bur}
        if best is None:                       # nothing reaches CA's burden: take the closest
            feas = min(curve, key=lambda c: c["burden"])
            best = {**feas, "note": "no threshold met the CA burden; closest taken"}

        inh_rec, inh_bur = score_at(pk, objs, inherited[seed])
        rows.append({
            "seed": seed,
            "threshold": best["threshold"],
            "dev_recall": best["recall"],
            "dev_burden": best["burden"],
            "ca_burden_reference": ca_bur,
            "meets_ca_burden": bool(best["burden"] <= ca_bur),
            "inherited_threshold": inherited[seed],
            "inherited_dev_recall": inh_rec,
            "inherited_dev_burden": inh_bur,
            "checkpoint_sha256": hashlib.sha256(
                open(os.path.join(CKPT, "tinycnn_ft_s%d.pt" % seed), "rb").read()).hexdigest(),
            "selected_epoch": int(ck["epoch"]),
            "curve": curve,
        })
        print("  seed %d  thr %.4f  dev recall %.4f  burden %.3f   |  inherited thr %.4f -> "
              "recall %.4f burden %.2f"
              % (seed, best["threshold"], best["recall"], best["burden"],
                 inherited[seed], inh_rec, inh_bur), flush=True)

    payload = {
        "rule": "highest exact dense-mask recall on the 3 internal-development CARRADA TRAIN "
                "sequences subject to unmatched detections/frame <= the frozen CA-CFAR burden on "
                "the same sequences; closest non-exceeding threshold if none qualifies",
        "nms_size": NMS,
        "dev_sequences": DEV,
        "n_dev_frames": len(frames),
        "ca_cfar": {"alpha": alpha, "dev_recall": ca_rec, "dev_burden": ca_bur},
        "primary_thresholds": {str(r["seed"]): r["threshold"] for r in rows},
        "secondary_inherited_thresholds": {str(k): v for k, v in inherited.items()},
        "per_seed": rows,
        "carrada_validation_opened": guard.access_report()["validation_opened"],
        "carrada_test_opened": guard.access_report()["test_opened"],
        "access_report": guard.access_report(),
    }
    payload["sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in payload.items() if k not in ("sha256", "per_seed")},
        sort_keys=True).encode()).hexdigest()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    r = np.array([x["dev_recall"] for x in rows])
    b = np.array([x["dev_burden"] for x in rows])
    print("\ninternal-dev summary (NOT a result -- development data):")
    print("  CA-CFAR        recall %.4f  burden %.3f" % (ca_rec, ca_bur))
    print("  fine-tuned     recall %.4f +/- %.4f  burden %.3f (all <= CA: %s)"
          % (r.mean(), r.std(ddof=1), b.mean(), all(x["meets_ca_burden"] for x in rows)))
    print("  inherited thr  recall %.4f  burden %.2f"
          % (np.mean([x["inherited_dev_recall"] for x in rows]),
             np.mean([x["inherited_dev_burden"] for x in rows])))
    print("\nwrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
