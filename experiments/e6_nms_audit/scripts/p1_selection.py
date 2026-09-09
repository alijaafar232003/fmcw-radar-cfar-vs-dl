"""Phase 1 - NMS selection sweep on development data only.

Selection on bench2_tune; confirmation on bench2_val. Writes only inside the experiment folder.

Adapted from scripts/nms_sweep.py (which hard-codes results/ paths and cannot be run here).
The scoring primitives are IMPORTED from the project, not reimplemented: roc_curve,
_at_measured_pfa, evaluate_frame, all_peaks. Score maps are computed once per frame and each NMS
radius applied afterwards, so radius count costs almost nothing.

Adds the measurement the frozen SELECTION_RULE.md requires and nms_sweep.py did not provide:
ABSOLUTE free-attempt behaviour -- in-gate detections per matched truth, and the fraction of
matched truths receiving two or more in-gate detections.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map                              # noqa: E402
from cfar.detect import all_peaks                                            # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                        # noqa: E402
from dl.models import TinyCNN                                                # noqa: E402
from eval.detection import (MatchConfig, evaluate_frame, roc_curve,          # noqa: E402
                            roc_thresholds, aggregate)
from dl.stratified_eval import _at_measured_pfa                              # noqa: E402

SHAPE = (256, 64)
TARGET_PFA = 1e-4
MC = MatchConfig(1, 1)
NMS_RADII = (1, 2, 3, 5)
CFAR_METHODS = ("CA", "OS")
CONDITIONS = ("isolated", "masking", "unresolved", "noise")
OUT = os.path.join(EXP, "results", "p1_selection")


def load_models():
    ms = []
    for s in range(8):
        p = os.path.join(ROOT, "results", "dl", "tinycnn_frozen_s%d_ep021.pt" % s)
        ck = torch.load(p, weights_only=False)
        m = TinyCNN(width=ck["config"].get("width", 16))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        ms.append(m)
    return ms


def free_attempts(frames, thr):
    """ABSOLUTE duplicate behaviour, per SELECTION_RULE.md gate A.

    For every truth that the one-to-one assignment matched, count how many detections fall
    inside its gate. A detector emitting one detection per target scores 1.0; anything above
    that is a target receiving extra attempts the evaluator forgives.
    """
    in_gate, n_truth_matched, n_two_plus = 0, 0, 0
    for p, t, s, v in frames:
        det = p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2))
        if len(t) == 0:
            continue
        ev = evaluate_frame(det, t, s, MC, valid_mask=v)
        for k in ev.matched_truth_idx:
            tr = np.asarray(t, float)[k]
            n = int(((np.abs(det[:, 0] - tr[0]) <= MC.tol_r) &
                     (np.abs(det[:, 1] - tr[1]) <= MC.tol_v)).sum()) if len(det) else 0
            in_gate += n
            n_truth_matched += 1
            n_two_plus += int(n >= 2)
    if n_truth_matched == 0:
        return dict(in_gate_per_matched_truth=np.nan, frac_truths_two_plus=np.nan,
                    n_matched_truths=0)
    return dict(in_gate_per_matched_truth=in_gate / n_truth_matched,
                frac_truths_two_plus=n_two_plus / n_truth_matched,
                n_matched_truths=n_truth_matched)


def score(frames):
    """Pd, achieved Pfa, threshold, detections per target -- at the matched-Pfa threshold."""
    roc = roc_curve(frames, roc_thresholds(frames, MC), MC)
    _, thr = _at_measured_pfa(roc, TARGET_PFA)
    evs, ndet, ntgt = [], 0, 0
    for p, t, s, v in frames:
        det = p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2))
        evs.append(evaluate_frame(det, t, s, MC, valid_mask=v))
        ndet += len(det)
        ntgt += len(t)
    a = aggregate(evs)
    fa = free_attempts(frames, thr)
    return dict(pd=a["pd"], pfa=a["pfa_per_cell"], threshold=float(thr),
                det_per_target=ndet / max(ntgt, 1), **fa)


def build_index(split):
    """Frame ids per condition and truth points per frame, read from the dataset itself.

    Deliberately does NOT use results/cfar_*/frames.csv: no such cache exists for bench2_tune,
    and reading the dataset keeps this experiment independent of results/.
    """
    meta = pd.read_csv(os.path.join(ROOT, "data", split, "metadata.csv"))
    fdir = os.path.join(ROOT, "data", split, "frames")
    by_cond, tb = {}, {}
    for _, r in meta.iterrows():
        fid = int(str(r["frame_id"]).split("_")[-1])
        by_cond.setdefault(r["condition"], []).append(fid)
        lp = os.path.join(fdir, str(r["frame_id"]), "labels.csv")
        pts = np.zeros((0, 2))
        if os.path.exists(lp):
            try:
                lab = pd.read_csv(lp)
                if len(lab):
                    pts = lab[["range_bin", "velocity_bin"]].to_numpy(float)
            except Exception:
                pass
        tb[fid] = pts
    return {k: np.array(sorted(v)) for k, v in by_cond.items()}, tb


def run_split(split, models, valid):
    by_cond, tb = build_index(split)

    rows = []
    for cond in CONDITIONS:
        ids = by_cond.get(cond, np.array([], dtype=int))
        print("  %-12s %d frames" % (cond, len(ids)), flush=True)

        ratio_maps = {m: {} for m in CFAR_METHODS}
        logit_maps = [{} for _ in models]
        for i in ids:
            mag = np.load(os.path.join(ROOT, "data", split, "frames",
                                       "frame_%06d" % int(i), "rd_map_mag.npy"))
            power = mag.astype(np.float64) ** 2
            for m in CFAR_METHODS:
                ratio_maps[m][int(i)] = cfar_ratio_map(power, CfarConfig(kind=m))[0]
            for k, mdl in enumerate(models):
                logit_maps[k][int(i)] = predict_heatmap(mdl, mag)

        for nms in NMS_RADII:
            def fr_of(maps):
                return [(all_peaks(maps[int(i)], valid, nms_size=nms),
                         tb.get(i, np.zeros((0, 2))), SHAPE, valid) for i in ids]

            # Candidate density needs no threshold; measured on every condition.
            cand = float(np.mean([len(all_peaks(logit_maps[0][int(i)], valid, nms_size=nms))
                                  for i in ids]))
            cand_ca = float(np.mean([len(all_peaks(ratio_maps["CA"][int(i)], valid, nms_size=nms))
                                     for i in ids]))

            if cond == "noise":
                rows.append(dict(split=split, condition=cond, nms=nms, method="candidates",
                                 cand_per_frame_DL=cand, cand_per_frame_CA=cand_ca))
                continue

            for m in CFAR_METHODS:
                r = score(fr_of(ratio_maps[m]))
                rows.append(dict(split=split, condition=cond, nms=nms, method=m,
                                 cand_per_frame_CA=cand_ca, **r))

            per = [score(fr_of(lm)) for lm in logit_maps]
            agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            agg["pd_se"] = float(np.std([p["pd"] for p in per], ddof=1) / np.sqrt(len(per)))
            rows.append(dict(split=split, condition=cond, nms=nms, method="DL",
                             cand_per_frame_DL=cand, **agg))
            print("    nms %d (%dx%d) done" % (nms, 2 * nms + 1, 2 * nms + 1), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["bench2_tune", "bench2_val"])
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    valid = cfar_valid_mask(SHAPE, CfarConfig())
    models = load_models()
    print("%d frozen seeds loaded (epoch 21)" % len(models), flush=True)

    rows = []
    for sp in a.splits:
        print("\n=== %s ===" % sp, flush=True)
        rows += run_split(sp, models, valid)

    df = pd.DataFrame(rows)
    p = os.path.join(OUT, "nms_selection_sweep.csv")
    df.to_csv(p, index=False)
    print("\nwrote", os.path.relpath(p, ROOT))

    with open(os.path.join(OUT, "run_info.json"), "w") as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(), splits=a.splits,
                       radii=list(NMS_RADII), conditions=list(CONDITIONS),
                       cfar_methods=list(CFAR_METHODS), target_pfa=TARGET_PFA,
                       gate=[MC.tol_r, MC.tol_v], n_seeds=len(models)), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
