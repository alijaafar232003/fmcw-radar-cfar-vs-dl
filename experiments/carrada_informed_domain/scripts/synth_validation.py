"""Phase 9 synthetic-validation report for the CARRADA-informed arm.

Scores the 8 new checkpoints on the NEW synthetic validation split at their own frozen thresholds,
per condition, and summarises the score distributions. This is a health check on the models and the
operating point, not a comparison with anything: the two arms are validated on DIFFERENT synthetic
domains, so their validation numbers are not comparable and are never presented side by side as a
performance claim.

Reads only carrada_informed_val. No CARRADA data of any split.

    python synth_validation.py --radius 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch                                                             # noqa: E402

from cfar.cfar import CfarConfig                                         # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap, predict_points    # noqa: E402
from dl.models import TinyCNN                                            # noqa: E402
from eval import MatchConfig, aggregate, evaluate_frame                  # noqa: E402

SPLIT = "carrada_informed_val"
assert "test" not in SPLIT and "carrada_informed" in SPLIT
SHAPE = (256, 64)
VALID = cfar_valid_mask(SHAPE, CfarConfig())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=5, choices=(2, 5))
    a = ap.parse_args()
    torch.set_num_threads(6)

    thr = {t["seed"]: t["threshold"] for t in json.load(
        open(os.path.join(EXP, "configs", "thresholds_r%d.json" % a.radius),
             encoding="utf-8"))["thresholds"]}

    d = os.path.join(ROOT, "data", SPLIT)
    md = pd.read_csv(os.path.join(d, "metadata.csv"))
    cond = dict(zip(md.frame_id, md.condition))
    names = sorted(os.listdir(os.path.join(d, "frames")))

    rows, score_stats = [], []
    for seed in range(8):
        ck = torch.load(os.path.join(EXP, "checkpoints",
                                     "tinycnn_ci_s%d_ep021.pt" % seed), weights_only=False)
        m = TinyCNN(width=16)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        t = thr[seed]

        per, allscores = {}, []
        for nm in names:
            fd = os.path.join(d, "frames", nm)
            mag = np.load(os.path.join(fd, "rd_map_mag.npy"))
            lab = pd.read_csv(os.path.join(fd, "labels.csv"))
            tru = (lab[["range_bin", "velocity_bin"]].to_numpy(float) if len(lab)
                   else np.zeros((0, 2)))
            pts = predict_points(m, mag, valid=VALID, nms_size=a.radius)
            if len(pts):
                allscores.append(pts[:, 2])
            det = pts[pts[:, 2] >= t][:, :2] if len(pts) else np.zeros((0, 2))
            ev = evaluate_frame(det, tru, SHAPE, MatchConfig(1, 1), valid_mask=VALID)
            per.setdefault(cond.get(nm, "all"), []).append(ev)

        for c, evs in sorted(per.items()):
            agg = aggregate(evs)
            rows.append(dict(seed=seed, condition=c, n_frames=len(evs), pd=agg["pd"],
                             pfa_per_cell=agg["pfa_per_cell"], fa_per_frame=agg["fa_per_frame"],
                             tp=agg["tp"], fn=agg["fn"], fp=agg["fp"]))
        s = np.concatenate(allscores) if allscores else np.zeros(0)
        score_stats.append(dict(seed=seed, threshold=t, n_candidates=int(s.size),
                                score_min=float(s.min()), score_p50=float(np.median(s)),
                                score_p99=float(np.percentile(s, 99)),
                                score_max=float(s.max()),
                                frac_above_threshold=float((s >= t).mean())))
        print("seed %d done (threshold %.4f)" % (seed, t), flush=True)

    R = pd.DataFrame(rows)
    S = pd.DataFrame(score_stats)
    R.to_csv(os.path.join(EXP, "results", "synth_val_r%d.csv" % a.radius), index=False)
    S.to_csv(os.path.join(EXP, "results", "synth_val_scores_r%d.csv" % a.radius), index=False)

    print("\nPd at the frozen threshold, per condition (mean over 8 seeds):")
    g = R.groupby("condition").agg(pd_mean=("pd", "mean"), pd_sd=("pd", "std"),
                                   pfa=("pfa_per_cell", "mean"))
    for c, r in g.iterrows():
        print("  %-14s Pd %.4f +/- %.4f   measured Pfa %.3e" % (c, r.pd_mean, r.pd_sd, r.pfa))
    pooled = R.groupby("seed").apply(
        lambda x: x.tp.sum() / max(1, (x.tp.sum() + x.fn.sum())), include_groups=False)
    print("\npooled Pd per seed: " + "  ".join("%.4f" % v for v in pooled))
    print("  mean %.4f  sd %.4f" % (pooled.mean(), pooled.std(ddof=1)))
    print("\nscore distributions:")
    print("  candidates/seed  %.0f" % S.n_candidates.mean())
    print("  score p50 %.3f  p99 %.3f  max %.3f" % (S.score_p50.mean(), S.score_p99.mean(),
                                                    S.score_max.mean()))
    print("  fraction above threshold %.5f" % S.frac_above_threshold.mean())
    print("\nwrote results/synth_val_r%d.csv" % a.radius)


if __name__ == "__main__":
    main()
