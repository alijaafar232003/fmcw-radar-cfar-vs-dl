"""Is `unresolved` pinned at Pd 0.5 by physics, or by `nms_size=5` merging the pair?

The claim on record is physics: two targets inside one mainlobe cannot both be found, so one
detection can match only one truth and Pd caps near 0.5. But NMS suppresses local maxima within
(2*nms_size+1) bins, and at `nms_size=5` that is an 11x11 window — 2.15 m in range, far wider than
the ~2-bin mainlobe. If NMS is what merges the pair, the cap is an artefact of a post-processing
choice, not a property of the radar.

`masking` is included for the same reason: PROGRESS.md step 6d documents `nms_size=5` capping that
stratum near Pd 0.69 by suppressing genuinely resolvable neighbours (only 38% of masking pairs keep
both targets on clean maps, 5.9% for pairs under 1 m).

Separations put a number on the expectation. `unresolved` spans 0.05–0.35 m = 0.26–1.79 range bins,
all below the measured 0.39 m equal-strength resolution limit, so even `nms_size=1` (a 3x3 window)
should merge them. `masking` spans 0.40–2.50 m = 2.0–12.8 bins, much of it resolvable, so that
stratum SHOULD respond to NMS width.

Also reports **detections per target**: `nms_size=1` was disqualified in step 6d because it returns
~1.20 of them, and the evaluator ignores unmatched detections inside a truth's guard region — so a
target gets several attempts and the extra ones cost nothing. Any Pd gain at nms=1 has to be read
against that.

Restricted to the two affected conditions to keep the cost down. Run on `bench2_val`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map                     # noqa: E402
from cfar.detect import all_peaks                                    # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                # noqa: E402
from dl.models import TinyCNN                                        # noqa: E402
from dl.stratified_eval import _at_measured_pfa                      # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,            # noqa: E402
                  roc_curve, roc_thresholds)

SPLIT = "bench2_val"
TARGET_PFA = 1e-4
MC = MatchConfig(1, 1)
SHAPE = (256, 64)
CFAR_METHODS = ("CA", "GO", "OS")
CV = ROOT / "results" / "cfar_val"
DLR = ROOT / "results" / "dl"


def score(frames):
    """Pd, and detections per target, at the matched-Pfa threshold."""
    roc = roc_curve(frames, roc_thresholds(frames, MC), MC)
    _, thr = _at_measured_pfa(roc, TARGET_PFA)

    evs, ndet, ntgt = [], 0, 0
    for p, t, s, v in frames:
        det = p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2))
        evs.append(evaluate_frame(det, t, s, MC, valid_mask=v))
        ndet += len(det)
        ntgt += len(t)
    a = aggregate(evs)
    return a["pd"], ndet / max(ntgt, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["unresolved", "masking"])
    ap.add_argument("--nms", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--pos-weight", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=4, help="DL seeds (4 is enough here)")
    args = ap.parse_args()

    valid = cfar_valid_mask(SHAPE, CfarConfig())
    frames_meta = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}

    # Separation per frame, to state what each stratum actually contains.
    sep = {}
    for f, g in truths.groupby("frame"):
        if len(g) == 2:
            sep[f] = abs(g.range_m.iloc[1] - g.range_m.iloc[0])

    ckpts = sorted(DLR.glob(f"tinycnn_pw{args.pos_weight}_s*_ep050.pt"))[:args.seeds]
    models = []
    for cp in ckpts:
        ck = torch.load(cp, weights_only=False)
        m = TinyCNN(width=ck["config"].get("width", 16))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    print(f"{len(models)} DL seeds, pos_weight={args.pos_weight}")

    rows = []
    for cond in args.conditions:
        ids = frames_meta.loc[frames_meta.condition == cond, "frame"].to_numpy()
        # `sep` only holds two-target frames, so single-target strata such as `isolated` have
        # none. Diagnostic print only -- scoring is unaffected.
        s = [sep[i] for i in ids if i in sep]
        if s:
            print(f"\n{cond}: {len(ids)} frames, separation "
                  f"{np.min(s):.2f}-{np.max(s):.2f} m ({np.min(s)/0.19518:.1f}-"
                  f"{np.max(s)/0.19518:.1f} range bins)")
        else:
            print(f"\n{cond}: {len(ids)} frames (no target pairs, so no separation range)")

        # Score maps computed ONCE per frame; NMS is applied afterwards at each width.
        ratio_maps = {m: {} for m in CFAR_METHODS}
        logit_maps = [{} for _ in models]
        for i in ids:
            mag = np.load(ROOT / "data" / SPLIT / "frames" / f"frame_{int(i):06d}" /
                          "rd_map_mag.npy")
            power = mag.astype(np.float64) ** 2
            for m in CFAR_METHODS:
                ratio_maps[m][int(i)] = cfar_ratio_map(power, CfarConfig(kind=m))[0]
            for k, mdl in enumerate(models):
                logit_maps[k][int(i)] = predict_heatmap(mdl, mag)

        for nms in args.nms:
            fr_of = lambda maps: [(all_peaks(maps[int(i)], valid, nms_size=nms),
                                   tb.get(i, np.zeros((0, 2))), SHAPE, valid) for i in ids]

            for m in CFAR_METHODS:
                pd_, dpt = score(fr_of(ratio_maps[m]))
                rows.append(dict(condition=cond, nms=nms, method=m, pd=pd_, det_per_target=dpt))

            vals = [score(fr_of(lm)) for lm in logit_maps]
            rows.append(dict(condition=cond, nms=nms, method="DL",
                             pd=np.mean([v[0] for v in vals]),
                             pd_se=np.std([v[0] for v in vals], ddof=1) / np.sqrt(len(vals)),
                             det_per_target=np.mean([v[1] for v in vals])))
            print(f"  nms_size {nms} (window {2*nms+1}x{2*nms+1}) done")

    df = pd.DataFrame(rows)
    out = DLR / "nms_sweep.csv"
    df.to_csv(out, index=False)

    for cond in args.conditions:
        d = df[df.condition == cond]
        print("\n" + "=" * 72)
        print(f"{cond} — Pd at matched measured Pfa 1e-4, by NMS width")
        print("=" * 72)
        print(d.pivot_table(index="method", columns="nms", values="pd").round(3).to_string())
        print("\ndetections per target (>1 means a target gets several attempts,")
        print("and the evaluator forgives the extras inside its guard region):")
        print(d.pivot_table(index="method", columns="nms",
                            values="det_per_target").round(2).to_string())

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
