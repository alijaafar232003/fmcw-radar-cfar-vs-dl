"""Does the DL margin depend on the ±1-bin matching gate being the same 3×3 the CNN was trained on?

The evaluator credits a detection within |Δrange| ≤ 1 AND |Δdoppler| ≤ 1 of a truth. The label mask
is a 3×3 block, so the network is trained to fire on **exactly** the region the evaluator then
rewards. CFAR gets no such instruction — it responds wherever the physical peak lands, and on
`clutter` the peak sits within 1 bin of the label only 77% of the time (PROGRESS.md Milestone 22).

That is Milestone 16's open asymmetry #5, recorded and never tested:
    "±1-bin gate may measure localisation rather than detection
     (DL learns label bins; CFAR finds physical peaks)."

**The test.** Re-score everything at gates of ±1, ±2 and ±3 bins. Widening the gate removes the
localisation requirement — a detector that was merely placing peaks more accurately loses its
advantage, while a detector that genuinely finds more targets keeps it.

Note the gate is not only the match rule: it also sets the guard region excluded from the
false-alarm denominator, so each tolerance gets its own threshold sweep and its own operating point.
Absolute Pd therefore rises for everyone as the gate widens; only the DL-minus-CFAR *difference* is
comparable across tolerances.

DL peaks are cached to disk on first run so this and the bootstrap need not recompute them.
Run on `bench2_val`.
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

from cfar.cfar import CfarConfig                                      # noqa: E402
from cfar.detect import all_peaks                                     # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                 # noqa: E402
from dl.models import TinyCNN                                         # noqa: E402
from dl.stratified_eval import NMS, _at_measured_pfa                  # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,             # noqa: E402
                  roc_curve, roc_thresholds)

SPLIT = "bench2_val"
TARGET_PFA = 1e-4
SHAPE = (256, 64)
CFAR_METHODS = ("CA", "GO", "OS", "SO")
CV = ROOT / "results" / "cfar_val"
DLR = ROOT / "results" / "dl"


def dl_peak_cache(pos_weight: int, valid) -> list[dict]:
    """Peaks per seed, cached to disk — the expensive part of every DL evaluation."""
    tag = "n" if pos_weight == 974 else f"pw{pos_weight}_s"
    ckpts = sorted(DLR.glob(f"tinycnn_{tag}*_ep050.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints for pos_weight={pos_weight}")

    out = []
    for cp in ckpts:
        seed = torch.load(cp, weights_only=False)["config"]["seed"]
        cache = DLR / f"peaks_val_pw{pos_weight}_s{seed}.npz"

        if cache.exists():
            a = np.load(cache)["peaks"]
            out.append({int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])})
            continue

        ck = torch.load(cp, weights_only=False)
        model = TinyCNN(width=ck["config"].get("width", 16))
        model.load_state_dict(ck["state_dict"])
        model.eval()

        rows, pk = [], {}
        for fd in sorted((ROOT / "data" / SPLIT / "frames").iterdir()):
            i = int(fd.name.split("_")[1])
            p = all_peaks(predict_heatmap(model, np.load(fd / "rd_map_mag.npy")), valid,
                          nms_size=NMS)
            pk[i] = p
            if len(p):
                rows.append(np.column_stack([np.full(len(p), i, float), p]))
        np.savez_compressed(cache, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))
        out.append(pk)
        print(f"  cached {cache.name}")
    return out


def score(frames, mc) -> float:
    """Pd at the threshold where this subset's measured Pfa equals the target, at this gate."""
    roc = roc_curve(frames, roc_thresholds(frames, mc), mc)
    _, thr = _at_measured_pfa(roc, TARGET_PFA)
    evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2)),
                          t, s, mc, valid_mask=v) for p, t, s, v in frames]
    return aggregate(evs)["pd"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-weight", type=int, default=100)
    ap.add_argument("--tolerances", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    valid = cfar_valid_mask(SHAPE, CfarConfig())
    frames_meta = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}
    conds = sorted(c for c in frames_meta.condition.unique() if c != "noise")

    cfar_peaks = {}
    for m in CFAR_METHODS:
        a = np.load(CV / f"peaks_{m}.npz")["peaks"]
        cfar_peaks[m] = {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}

    print(f"caching DL peaks (pos_weight={args.pos_weight}) ...")
    dl_peaks = dl_peak_cache(args.pos_weight, valid)
    print(f"  {len(dl_peaks)} seeds ready")

    rows = []
    for tol in args.tolerances:
        mc = MatchConfig(tol, tol)
        for cond in conds:
            ids = frames_meta.loc[frames_meta.condition == cond, "frame"].to_numpy()
            fr_of = lambda pk: [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))),
                                 SHAPE, valid) for i in ids]

            cf = {m: score(fr_of(cfar_peaks[m]), mc) for m in CFAR_METHODS}
            best_m = max(cf, key=cf.get)
            dl = np.array([score(fr_of(pk), mc) for pk in dl_peaks])

            rows.append(dict(tol=tol, condition=cond, best_cfar=cf[best_m], best_method=best_m,
                             dl_mean=dl.mean(), dl_se=dl.std(ddof=1) / np.sqrt(len(dl)),
                             diff=dl.mean() - cf[best_m], **{f"cfar_{m}": cf[m] for m in cf}))
        print(f"  tolerance +/-{tol} done")

    df = pd.DataFrame(rows)
    out = DLR / f"tolerance_sweep_pw{args.pos_weight}.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 88)
    print("Pd at matched measured Pfa 1e-4, by matching tolerance")
    print("=" * 88)
    for col, name in (("best_cfar", "best CFAR"), ("dl_mean", "DL (8 seeds)"), ("diff", "DL - CFAR")):
        print(f"\n{name}")
        print(df.pivot_table(index="condition", columns="tol", values=col).round(3).to_string())

    d = df.pivot_table(index="condition", columns="tol", values="diff")
    d["change 1->3"] = d[3] - d[1]
    print("\n" + "=" * 88)
    print("Does the DL margin shrink as the gate widens?")
    print("=" * 88)
    print(d.round(3).to_string())
    print("\nA margin that COLLAPSES was localisation-to-the-label, not detection.")
    print("A margin that HOLDS is a genuine detection advantage.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
