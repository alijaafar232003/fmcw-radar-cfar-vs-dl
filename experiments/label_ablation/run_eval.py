"""Score the label-ablation arms on bench2_val. Writes only under experiments/label_ablation/.

Unmodified project protocol: Pd at matched measured Pfa 1e-4 swept per condition, shared
`all_peaks` at nms_size=5, Hungarian matching, guard-excluded Pfa denominator. Gates 0, +/-1, +/-3.

⚠️ bench2_test is not touched.

Stages (each resumable -- existing artefacts are reused):
  1  peaks for every (arm, seed, epoch) on bench2_val
  2  pooled Pd + localization vs epoch, all gates      -> epoch_curve.csv
  3  epoch-21 per condition, all gates                 -> per_condition.csv
  4  paired bootstrap vs the flat3x3 control           -> deltas.csv
  5  CFAR localization reference from cached peaks     -> cfar_reference.csv

    python experiments/label_ablation/run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from cfar.cfar import CfarConfig                                    # noqa: E402
from cfar.detect import all_peaks                                   # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap               # noqa: E402
from dl.models import TinyCNN                                       # noqa: E402
from dl.stratified_eval import _at_measured_pfa                     # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,           # noqa: E402
                  roc_curve, roc_thresholds)
from labels import ARMS                                             # noqa: E402

SPLIT = "bench2_val"
SHAPE = (256, 64)
NMS = 5
TARGET_PFA = 1e-4
EPOCHS = (5, 10, 15, 21)
GATES = (0, 1, 3)
SEEDS = range(8)
N_BOOT = 2000
RANGE_RES, VEL_RES = 0.19518, 0.42013

CKPT = HERE / "checkpoints"
PK = HERE / "cache" / "peaks"
RES = HERE / "results"
VALID = cfar_valid_mask(SHAPE, CfarConfig())


def load_peaks(p: Path) -> dict:
    a = np.load(p)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def save_peaks(pk: dict, p: Path) -> None:
    rows = [np.column_stack([np.full(len(v), i, float), v]) for i, v in sorted(pk.items()) if len(v)]
    np.savez_compressed(p, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))


def matched_threshold(frames, mc):
    return _at_measured_pfa(roc_curve(frames, roc_thresholds(frames, mc), mc), TARGET_PFA)[1]


def score(pk, ids, tb, mc):
    fr = [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))), SHAPE, VALID)
          for i in ids]
    thr = matched_threshold(fr, mc)
    evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2)),
                          t, s, mc, valid_mask=v) for p, t, s, v in fr]
    a = aggregate(evs, range_res_m=RANGE_RES, vel_res_mps=VEL_RES)
    hits = np.array([e.tp for e in evs], float)
    n = np.array([len(t) for _, t, _, _ in fr], float)
    return a, thr, hits, n


def main() -> None:
    PK.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    CV = ROOT / "results" / "cfar_val"
    fm = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    TB = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}
    ids_all = fm.frame.to_numpy()
    conds = sorted(c for c in fm.condition.unique() if c != "noise")

    # ---------------- 1. peaks
    todo = [(a, s, e) for a in ARMS for s in SEEDS for e in EPOCHS
            if not (PK / f"peaks_{a}_s{s}_ep{e:03d}.npz").exists()]
    print(f"stage 1: {len(todo)} checkpoints to score ({len(ARMS) * 8 * len(EPOCHS)} total)")
    for n, (a, s, e) in enumerate(todo):
        ck = torch.load(CKPT / f"tinycnn_{a}_s{s}_ep{e:03d}.pt", weights_only=False)
        m = TinyCNN(width=ck["config"].get("width", 16))
        m.load_state_dict(ck["state_dict"])
        m.eval()
        out = {}
        for i in ids_all:
            mag = np.load(ROOT / "data" / SPLIT / "frames" / f"frame_{int(i):06d}" /
                          "rd_map_mag.npy")
            out[int(i)] = all_peaks(predict_heatmap(m, mag), VALID, nms_size=NMS)
        save_peaks(out, PK / f"peaks_{a}_s{s}_ep{e:03d}.npz")
        print(f"    [{n + 1}/{len(todo)}] {a}_s{s}_ep{e:03d}", flush=True)

    peaks = {(a, s, e): load_peaks(PK / f"peaks_{a}_s{s}_ep{e:03d}.npz")
             for a in ARMS for s in SEEDS for e in EPOCHS}

    # ---------------- 2. pooled, vs epoch, all gates
    print("\nstage 2: pooled Pd + localization vs epoch")
    rows = []
    for g in GATES:
        mc = MatchConfig(g, g)
        for a in ARMS:
            for e in EPOCHS:
                per = [score(peaks[(a, s, e)], ids_all, TB, mc)[0] for s in SEEDS]
                f = lambda k: np.array([p[k] for p in per], float)
                rows.append(dict(
                    gate=g, arm=a, epoch=e, n_seeds=len(per),
                    pd=f("pd").mean(), pd_se=f("pd").std(ddof=1) / np.sqrt(len(per)),
                    dr_bins=f("mean_abs_dr_bins").mean(),
                    dr_se=f("mean_abs_dr_bins").std(ddof=1) / np.sqrt(len(per)),
                    dv_bins=f("mean_abs_dv_bins").mean(),
                    dv_se=f("mean_abs_dv_bins").std(ddof=1) / np.sqrt(len(per)),
                    range_m=f("mean_abs_range_err_m").mean(),
                    vel_mps=f("mean_abs_vel_err_mps").mean()))
                print(f"    gate {g} {a:<9} ep{e:03d}  Pd {rows[-1]['pd']:.4f}  "
                      f"dv {rows[-1]['dv_bins']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "epoch_curve.csv", index=False)

    # ---------------- 3. epoch 21, per condition, all gates
    print("\nstage 3: epoch 21 per condition")
    rows, hits_store = [], {}
    for g in GATES:
        mc = MatchConfig(g, g)
        for cond in conds:
            cid = fm.loc[fm.condition == cond, "frame"].to_numpy()
            for a in ARMS:
                per, hh = [], []
                for s in SEEDS:
                    aggd, thr, h, n = score(peaks[(a, s, 21)], cid, TB, mc)
                    per.append(aggd)
                    hh.append(h)
                hits_store[(g, cond, a)] = (np.mean(hh, axis=0), n)
                f = lambda k: np.array([p[k] for p in per], float)
                rows.append(dict(gate=g, condition=cond, arm=a,
                                 pd=f("pd").mean(),
                                 pd_se=f("pd").std(ddof=1) / np.sqrt(len(per)),
                                 dr_bins=f("mean_abs_dr_bins").mean(),
                                 dv_bins=f("mean_abs_dv_bins").mean(),
                                 measured_pfa=f("pfa_per_cell").mean()))
            print(f"    gate {g} {cond} done", flush=True)
    pd.DataFrame(rows).to_csv(RES / "per_condition.csv", index=False)

    # ---------------- 4. bootstrap vs the flat3x3 control
    print("\nstage 4: paired bootstrap vs flat3x3")
    rng = np.random.default_rng(0)
    rows = []
    for g in GATES:
        for cond in conds:
            base, n = hits_store[(g, cond, "flat3x3")]
            k = rng.integers(0, len(n), size=(N_BOOT, len(n)))
            tot = n[k].sum(axis=1)
            for a in ARMS:
                if a == "flat3x3":
                    continue
                h, _ = hits_store[(g, cond, a)]
                d = (h[k].sum(axis=1) - base[k].sum(axis=1)) / tot
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                rows.append(dict(gate=g, condition=cond, arm=a,
                                 pd_flat=base.sum() / n.sum(), pd_arm=h.sum() / n.sum(),
                                 delta=(h.sum() - base.sum()) / n.sum(),
                                 ci_lo=float(lo), ci_hi=float(hi), p_boot=float(p)))
    B = pd.DataFrame(rows)
    B["alpha_bonf"] = 0.05 / len(conds)
    B["sig"] = B.p_boot < B.alpha_bonf
    B.to_csv(RES / "deltas.csv", index=False)

    # ---------------- 5. CFAR reference (cached peaks, never recomputed)
    print("\nstage 5: CFAR localization reference")
    rows = []
    for g in GATES:
        mc = MatchConfig(g, g)
        for kind in ("CA", "GO", "OS", "SO"):
            pk = load_peaks(CV / f"peaks_{kind}.npz")
            a, *_ = score(pk, ids_all, TB, mc)
            rows.append(dict(gate=g, method=kind, pd=a["pd"],
                             dr_bins=a["mean_abs_dr_bins"], dv_bins=a["mean_abs_dv_bins"],
                             range_m=a["mean_abs_range_err_m"], vel_mps=a["mean_abs_vel_err_mps"]))
    pd.DataFrame(rows).to_csv(RES / "cfar_reference.csv", index=False)

    print(f"\nwrote {RES}")


if __name__ == "__main__":
    main()
