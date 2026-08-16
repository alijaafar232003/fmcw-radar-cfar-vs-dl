"""Does anything depend on training length, at the configuration we actually report?

Two gaps this closes.

**The flatness evidence was measured on the wrong arm.** "Nothing moves between epochs 15 and 30"
came from the `pos_weight=974` runs — the value later shown to be the wrong choice. It has never
been demonstrated at `pos_weight=100`, which is what gets reported.

**Localization has only ever been measured at epoch 50.** So we do not know whether it depends on
training length at all. If it degrades with longer training — the network growing more confident
across the whole 3x3 label block — that would matter more than Pd does, since localization is where
CFAR wins.

4 seeds x epochs {15, 25, 34, 50}, `bench2_val`, matched measured Pfa 1e-4, gate ±1. Peaks are
cached per checkpoint so a re-run costs nothing.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig                                     # noqa: E402
from cfar.detect import all_peaks                                    # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                # noqa: E402
from dl.models import TinyCNN                                        # noqa: E402
from dl.stratified_eval import NMS, _at_measured_pfa                 # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,            # noqa: E402
                  roc_curve, roc_thresholds)

SPLIT = "bench2_val"
TARGET_PFA = 1e-4
SHAPE = (256, 64)
RANGE_RES, VEL_RES = 0.19518, 0.42013
CV = ROOT / "results" / "cfar_val"
DLR = ROOT / "results" / "dl"


def peaks_for(cp: Path, valid) -> dict:
    cache = DLR / f"peaks_val_{cp.stem.replace('tinycnn_', '')}.npz"
    if cache.exists():
        a = np.load(cache)["peaks"]
        return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}

    ck = torch.load(cp, weights_only=False)
    model = TinyCNN(width=ck["config"].get("width", 16))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    pk, rows = {}, []
    for fd in sorted((ROOT / "data" / SPLIT / "frames").iterdir()):
        i = int(fd.name.split("_")[1])
        p = all_peaks(predict_heatmap(model, np.load(fd / "rd_map_mag.npy")), valid, nms_size=NMS)
        pk[i] = p
        if len(p):
            rows.append(np.column_stack([np.full(len(p), i, float), p]))
    np.savez_compressed(cache, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))
    print(f"  cached {cache.name}")
    return pk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="epflat")
    ap.add_argument("--tolerance", type=int, default=1)
    args = ap.parse_args()

    mc = MatchConfig(args.tolerance, args.tolerance)
    valid = cfar_valid_mask(SHAPE, CfarConfig())
    frames_meta = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}
    ids = frames_meta.frame.to_numpy()
    conds = sorted(c for c in frames_meta.condition.unique() if c != "noise")

    ckpts = sorted(DLR.glob(f"tinycnn_{args.tag}_s*_ep*.pt"))
    print(f"{len(ckpts)} checkpoints")

    rows = []
    for cp in ckpts:
        m = re.match(rf"tinycnn_{args.tag}_s(\d+)_ep(\d+)", cp.stem)
        seed, epoch = int(m.group(1)), int(m.group(2))
        pk = peaks_for(cp, valid)

        for gname, sub in [("ALL", ids)] + [(c, frames_meta.loc[frames_meta.condition == c,
                                                                "frame"].to_numpy())
                                            for c in conds]:
            fr = [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))), SHAPE, valid)
                  for i in sub]
            roc = roc_curve(fr, roc_thresholds(fr, mc), mc)
            _, thr = _at_measured_pfa(roc, TARGET_PFA)
            evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr)
                                  else np.zeros((0, 2)), t, s, mc, valid_mask=v)
                   for p, t, s, v in fr]
            a = aggregate(evs, range_res_m=RANGE_RES, vel_res_mps=VEL_RES)
            rows.append(dict(seed=seed, epoch=epoch, group=gname, pd=a["pd"],
                             dr_bins=a["mean_abs_dr_bins"], dv_bins=a["mean_abs_dv_bins"],
                             range_m=a["mean_abs_range_err_m"], vel_mps=a["mean_abs_vel_err_mps"]))
        print(f"  scored seed {seed} epoch {epoch}")

    df = pd.DataFrame(rows)
    out = DLR / f"epoch_flatness_{args.tag}.csv"
    df.to_csv(out, index=False)

    a = df[df.group == "ALL"]
    summ = a.groupby("epoch").agg(pd_mean=("pd", "mean"), pd_se=("pd", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                                  dr=("dr_bins", "mean"), dr_se=("dr_bins", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                                  dv=("dv_bins", "mean"), dv_se=("dv_bins", lambda x: x.std(ddof=1) / np.sqrt(len(x))))
    print("\n" + "=" * 72)
    print(f"pos_weight=100, {a.seed.nunique()} seeds, all conditions pooled, gate +/-{args.tolerance}")
    print("=" * 72)
    print(summ.round(4).to_string())

    print("\nPd by condition (seed mean):")
    print(df[df.group != "ALL"].pivot_table(index="group", columns="epoch",
                                            values="pd").round(3).to_string())
    print("\nDoppler localization error by condition (bins):")
    print(df[df.group != "ALL"].pivot_table(index="group", columns="epoch",
                                            values="dv_bins").round(3).to_string())

    rng = summ.pd_mean.max() - summ.pd_mean.min()
    print(f"\nPd range across epochs: {rng:.4f}   mean SE: {summ.pd_se.mean():.4f}")
    print(f"Doppler error range:    {summ.dv.max() - summ.dv.min():.4f}   "
          f"mean SE: {summ.dv_se.mean():.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
