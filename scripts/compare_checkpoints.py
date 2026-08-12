"""Do checkpoints from across the loss plateau differ in the metric we actually report?

The 50-epoch run's validation loss is statistically flat from ~epoch 26 (slope p = 0.17) while
train keeps falling. Loss is a proxy, so "where should we stop?" cannot be settled on it. This
scores several checkpoints on the metric that gets reported — per-condition Pd at matched MEASURED
Pfa — and the answer is informative either way:

* **They land on top of each other** → the stopping epoch does not matter, the question is closed,
  and a `pos_weight` ablation can be run single-seed with a clean conscience.
* **They scatter** → the evaluation is noisier than the loss curve suggests, and any ablation needs
  multiple seeds before it can claim anything.

Run on `bench2_val`. The test split stays untouched until the model is final.

    python scripts/compare_checkpoints.py --checkpoints ep030 best last
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

from cfar.cfar import CfarConfig                                        # noqa: E402
from cfar.detect import all_peaks                                       # noqa: E402
from dl.dataset import CachedRDDataset                                  # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                   # noqa: E402
from dl.models import TinyCNN                                           # noqa: E402
from dl.stratified_eval import NMS, _at_measured_pfa, frame_condition   # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,               # noqa: E402
                  roc_curve, roc_thresholds)

SPLIT = "bench2_val"
TARGET_PFA = 1e-4
MC = MatchConfig(1, 1)
DL = ROOT / "results" / "dl"


def resolve(name: str) -> list[Path]:
    """A name, a glob, or a path -> checkpoint paths.

    'best'/'last'/'ep030' match one file; 's*_ep030' matches one per seed, which is what a
    noise-floor comparison needs.
    """
    p = Path(name)
    if p.exists():
        return [p]
    hits = sorted(DL.glob(f"tinycnn_{name}.pt")) or sorted(DL.glob(f"tinycnn_*_{name}.pt"))
    if not hits:
        raise SystemExit(f"no checkpoint matching {name!r} in {DL}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["ep030", "best", "last"])
    ap.add_argument("--split", default=SPLIT)
    args = ap.parse_args()

    ds = CachedRDDataset(args.split)
    conds = [frame_condition(fd) for fd in ds.frame_dirs]
    truths = {}
    for i, fd in enumerate(ds.frame_dirs):
        lab = (pd.read_csv(fd / "labels.csv")
               if (fd / "labels.csv").stat().st_size > 5 else pd.DataFrame())
        truths[i] = (lab[["range_bin", "velocity_bin"]].to_numpy(float)
                     if len(lab) else np.zeros((0, 2)))

    shape = (256, 64)
    valid = cfar_valid_mask(shape, CfarConfig())
    print(f"{args.split}: {len(ds)} frames, conditions {sorted(set(conds))}")

    paths = [p for name in args.checkpoints for p in resolve(name)]
    print(f"comparing {len(paths)} checkpoints: {[p.name for p in paths]}")

    rows = []
    for path in paths:
        ck = torch.load(path, weights_only=False)
        model = TinyCNN(width=ck["config"].get("width", 16))
        model.load_state_dict(ck["state_dict"])
        model.eval()

        print(f"\n{path.name}  (epoch {ck['epoch']}, val loss {ck['val_loss']:.5f})")

        # One forward pass per frame; the operating point is chosen afterwards by sweeping.
        peaks = {}
        for i in range(len(ds)):
            mag = np.load(ds.frame_dirs[i] / "rd_map_mag.npy")
            peaks[i] = all_peaks(predict_heatmap(model, mag), valid, nms_size=NMS)

        for cond in sorted(set(conds)):
            idx = [i for i, c in enumerate(conds) if c == cond]
            fr = [(peaks[i], truths[i], shape, valid) for i in idx]

            roc = roc_curve(fr, roc_thresholds(fr, MC), MC)
            _, thr = _at_measured_pfa(roc, TARGET_PFA)

            evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr)
                                  else np.zeros((0, 2)), t, s, MC, valid_mask=v)
                   for p, t, s, v in fr]
            a = aggregate(evs)
            rows.append(dict(checkpoint=path.stem, seed=ck["config"].get("seed"),
                             epoch=ck["epoch"], val_loss=ck["val_loss"],
                             condition=cond, pd=a["pd"], measured_pfa=a["pfa_per_cell"],
                             threshold=thr))
        print("   scored")

    df = pd.DataFrame(rows)
    out = DL / "checkpoint_comparison.csv"
    df.to_csv(out, index=False)

    # Also append to a cumulative archive. The single-file output was overwritten by every run,
    # which lost the epoch 30/43/50 and 3-seed 15/30 numbers -- they survived only because they had
    # been written into PROGRESS.md by hand. Keyed by checkpoint+condition so re-runs replace
    # rather than duplicate.
    arch = DL / "checkpoint_history.csv"
    if arch.exists():
        prev = pd.read_csv(arch)
        df_all = (pd.concat([prev, df], ignore_index=True)
                    .drop_duplicates(subset=["checkpoint", "condition"], keep="last"))
    else:
        df_all = df
    df_all.to_csv(arch, index=False)
    print(f"archive now holds {df_all.checkpoint.nunique()} checkpoints -> {arch}")

    piv = df.pivot_table(index="condition", columns="epoch", values="pd")
    print("\nPd at matched MEASURED Pfa 1e-4, by training epoch")
    print(piv.round(3).to_string())

    spread = (piv.max(axis=1) - piv.min(axis=1)).drop("noise", errors="ignore")
    print("\nspread across checkpoints, per condition:")
    print(spread.round(3).to_string())
    print(f"\nmax spread {spread.max():.3f} | mean spread {spread.mean():.3f}")
    print("\nFor reference, the CFAR method-to-method spread on the clean conditions was"
          "\n0.003-0.013, and that was judged inside sampling noise at 400 frames.")
    print("If these checkpoints spread by a similar amount or more, the stopping epoch is not"
          "\nresolvable by this evaluation and any ablation needs multiple seeds.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
