"""Restore `r_carrada_real_data_validation/results/tinycnn_val_roc_pooled.csv`.

WHY THIS EXISTS. The first run of the patched threshold script redirected its thresholds JSON but
not its SECOND write -- a pooled-ROC dump whose path is built inline from EXP, which the patched
copy points at the ORIGINAL CARRADA experiment so that detectors.py and src/ resolve. That write
therefore landed on a Chapter 7 artefact and overwrote it with the new arm's curves. The isolation
manifest caught it; `patch_scripts.py` now redirects that write too.

This regenerates the file's correct content by repeating exactly what the original script did:
the eight FROZEN checkpoints, `bench2_val`, and `dl.infer.collect_split` -- which extracts at
radius 5 because `predict_points` binds `nms_size` to `dl.infer.NMS_SIZE` at definition time. The
file is write-only in this repository: `grep` finds no reader other than the script that produces
it, so nothing downstream consumed the wrong content.

It writes ONE file and nothing else. It does not touch the thresholds JSON, which was never
modified (it lives under configs/, which the patched copy did redirect).

Verifies the restored file against the SHA-256 recorded in the isolation manifest and reports
whether the restoration is byte-identical.

    python restore_roc_dump.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
R_EXP = os.path.join(ROOT, "experiments", "r_carrada_real_data_validation")
sys.path.insert(0, os.path.join(R_EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch                                                             # noqa: E402

from dl.infer import collect_split                                       # noqa: E402
from eval.detection import MatchConfig, roc_curve, roc_thresholds        # noqa: E402

import detectors as det                                                  # noqa: E402

TARGET = os.path.join(R_EXP, "results", "tinycnn_val_roc_pooled.csv")
SPLIT = "bench2_val"
GATE = 1
MANIFEST = os.path.join(EXP, "results", "isolation_manifest.json")
REL = "experiments/r_carrada_real_data_validation/results/tinycnn_val_roc_pooled.csv"


def main():
    assert det.NMS_SIZE == 5, "detectors.NMS_SIZE must be its frozen 5 here"
    torch.set_num_threads(6)
    cfg = MatchConfig(GATE, GATE)

    curves = []
    for seed in det.SEEDS:
        model = det.load_frozen_tinycnn(seed)
        frames, _ = collect_split(model, SPLIT)
        thr = roc_thresholds(frames, cfg)
        roc = roc_curve(frames, thresholds=thr, cfg=cfg)
        roc = roc.sort_values("threshold").reset_index(drop=True)
        curves.append(roc.assign(seed=seed))
        print("  seed %d done" % seed, flush=True)

    pd.concat(curves).to_csv(TARGET, index=False)
    got = hashlib.sha256(open(TARGET, "rb").read()).hexdigest()

    want = None
    if os.path.exists(MANIFEST):
        want = json.load(open(MANIFEST, encoding="utf-8"))["files"].get(REL, {}).get("sha256")

    print("\nrestored %s" % os.path.relpath(TARGET, ROOT))
    print("  sha256 now      %s" % got)
    print("  manifest sha256 %s" % (want or "(not recorded)"))
    print("  %s" % ("BYTE-IDENTICAL to the pre-experiment state" if want == got
                    else "*** DIFFERS from the recorded hash -- investigate ***"))
    return 0 if (want is None or want == got) else 1


if __name__ == "__main__":
    sys.exit(main())
