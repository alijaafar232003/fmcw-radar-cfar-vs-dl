"""Pd vs cell SNR against theory — the validation figure, on `bench2_snr_sweep`.

THE POINT OF THIS FIGURE. Every other result in the project is a *relative* claim: DL against CFAR
at a matched operating point. This is the one **absolute** check — it asks whether either detector
is anywhere near the physics. The dashed line is the square-law detector that *knows* the noise
level, `Pd = Pfa^(1/(1+SNR))`; the gap below it is **CFAR loss**, the price of estimating that level
from a finite reference window instead of being told it.

A curve that tracks the theoretical shape and sits slightly below it is the expected, healthy
result. **A curve that crosses ABOVE theory is a scoring artefact, not a superior detector** —
theory is an upper bound for a detector with no extra information.

**AND IT DID CROSS, WHICH IS WHY THIS FIGURE HAS TWO PANELS.** At the project's ±1 gate every
method — all four CFAR variants *and* the network — sat ~0.3 Pd ABOVE theory and reached Pd 0.5
about 1.7–2.8 dB earlier than theory says is possible. That is not five superior detectors; it is a
mismatched comparison. `Pd = Pfa^(1/(1+SNR))` is the probability that **one specific cell** exceeds
the threshold, whereas the ±1 gate counts a hit if **any** detection lands in the 3×3 neighbourhood
— nine chances, and the target's mainlobe spans ~2 bins so the neighbours carry real target energy
too.

Re-scored at **gate 0** (exact cell, the like-for-like comparison) the anomaly disappears
completely and CFAR sits 0.47–0.78 dB below theory: textbook CFAR loss. So:

* **left panel, gate 0** — the valid comparison against theory, and the honest CFAR-loss number;
* **right panel, gate ±1** — the operating point every other result in this project uses, with
  theory drawn only as a *reference shape*. At this gate theory is **not** a bound and must never
  be described as one.

The gate-0 panel also states DL's localization problem in the bluntest possible terms: on
exact-cell matching the network barely detects anything, because its peak lands inside the 3×3
label block rather than on the target cell (§14–§16, Open/O4).

The split is built for exactly this: 1,300 frames, ONE target each, SNR swept 0–24 dB uniformly, no
interference and no clutter. So the curve is not confounded by masking or clutter-edge effects the
way a per-condition breakdown of `bench2_val` would be.

`bench2_snr_sweep` is an evaluation-only split — no training, no tuning, and it is not the held-out
test set. Reading it costs nothing.

    python scripts/pd_vs_snr_figure.py --tag frozen --epoch 21
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

from cfar.cfar import CfarConfig                                     # noqa: E402
from cfar.detect import all_peaks                                    # noqa: E402
from dl.dataset import normalize_rd                                  # noqa: E402
from dl.infer import cfar_valid_mask                                 # noqa: E402
from dl.models import TinyCNN                                        # noqa: E402
from dl.stratified_eval import NMS, _at_measured_pfa                 # noqa: E402
from eval import (MatchConfig, evaluate_frame, roc_curve,            # noqa: E402
                  roc_thresholds)

SPLIT = "bench2_snr_sweep"
TARGET_PFA = 1e-4
SHAPE = (256, 64)
CV = ROOT / "results" / "cfar_snr_sweep"
DLR = ROOT / "results" / "dl"
METHODS = ("CA", "GO", "OS", "SO")


def theory_pd(snr_db, pfa: float = TARGET_PFA) -> np.ndarray:
    """Square-law detector that KNOWS the noise level. The same curve as `audit_data_quality.py`."""
    snr = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    return pfa ** (1.0 / (1.0 + snr))


def load_peaks(p: Path) -> dict:
    a = np.load(p)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def dl_peaks(tag: str, epoch: int, seed: int, ids, valid) -> dict:
    cache = DLR / f"peaks_snr_sweep_{tag}_s{seed}_ep{epoch:03d}.npz"
    if cache.exists():
        return load_peaks(cache)

    ck = torch.load(DLR / f"tinycnn_{tag}_s{seed}_ep{epoch:03d}.pt", weights_only=False)
    model = TinyCNN(width=ck["config"].get("width", 16))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    pk, rows = {}, []
    with torch.no_grad():
        for i in ids:
            mag = np.load(ROOT / "data" / SPLIT / "frames" / f"frame_{int(i):06d}" / "rd_map_mag.npy")
            hm = model(torch.from_numpy(normalize_rd(mag))[None, None])[0, 0].numpy()
            p = all_peaks(hm, valid, nms_size=NMS)
            pk[int(i)] = p
            if len(p):
                rows.append(np.column_stack([np.full(len(p), i, float), p]))
    np.savez_compressed(cache, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))
    return pk


def hits_by_target(pk, ids, tb, valid, mc, thr):
    """Per TARGET: was it detected at this operating point? Returns a bool per truth row."""
    out = []
    for i in ids:
        det = pk.get(int(i), np.zeros((0, 3)))
        det = det[det[:, 2] >= thr][:, :2] if (len(det) and thr == thr) else np.zeros((0, 2))
        tru = tb.get(i, np.zeros((0, 2)))
        ev = evaluate_frame(det, tru, SHAPE, mc, valid_mask=valid)
        matched = set(ev.matched_truth_idx)
        out.extend(k in matched for k in range(len(tru)))
    return np.array(out, dtype=bool)


def matched_thr(pk, ids, tb, valid, mc):
    fr = [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))), SHAPE, valid) for i in ids]
    return _at_measured_pfa(roc_curve(fr, roc_thresholds(fr, mc), mc), TARGET_PFA)[1]


def curve(snr, hit, edges, min_n=20):
    d = pd.DataFrame(dict(snr=snr, hit=hit))
    d["bin"] = pd.cut(d.snr, edges)
    g = d.groupby("bin", observed=True).agg(n=("hit", "size"), snr=("snr", "mean"),
                                            pd=("hit", "mean")).reset_index(drop=True)
    return g[g.n >= min_n]


def snr_at_pd(c, target=0.5):
    """Interpolate the SNR at which a curve reaches `target` Pd — for the detection-loss number."""
    c = c.sort_values("snr")
    if c.pd.min() > target or c.pd.max() < target:
        return np.nan
    return float(np.interp(target, c.pd.to_numpy(), c.snr.to_numpy()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--epoch", type=int, default=21)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    valid = cfar_valid_mask(SHAPE, CfarConfig())
    fm = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv").sort_values(["frame", "target"])
    ids = fm.frame.to_numpy()
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}
    snr = truths.cell_snr_db.to_numpy(float)
    print(f"{len(fm)} frames, {len(truths)} targets, cell SNR {snr.min():.1f}..{snr.max():.1f} dB")

    edges = np.arange(np.floor(snr.min()), np.ceil(snr.max()) + 2, 2.0)
    s_line = np.linspace(edges[0], edges[-1], 300)
    th50 = float(np.interp(0.5, theory_pd(np.linspace(-5, 30, 400)), np.linspace(-5, 30, 400)))

    all_curves, rows = {}, []
    for tol in (0, 1):
        mc = MatchConfig(tol, tol)
        curves = {}
        for m in METHODS:
            pk = load_peaks(CV / f"peaks_{m}.npz")
            thr = matched_thr(pk, ids, tb, valid, mc)
            curves[m] = curve(snr, hits_by_target(pk, ids, tb, valid, mc, thr), edges)
        per_seed = []
        for sd in range(args.seeds):
            pk = dl_peaks(args.tag, args.epoch, sd, ids, valid)
            thr = matched_thr(pk, ids, tb, valid, mc)
            per_seed.append(curve(snr, hits_by_target(pk, ids, tb, valid, mc, thr), edges))
        dl = per_seed[0][["snr", "n"]].copy()
        P = np.vstack([c.pd.to_numpy() for c in per_seed])
        dl["pd"] = P.mean(axis=0)
        dl["se"] = P.std(axis=0, ddof=1) / np.sqrt(len(per_seed))
        curves["DL"] = dl
        all_curves[tol] = curves

        for k, c in curves.items():
            rows.append(dict(gate=tol, method=k, snr50=snr_at_pd(c),
                             loss_db=snr_at_pd(c) - th50, pd_max=float(c.pd.max()),
                             max_above_theory=float((c.pd - theory_pd(c.snr)).max())))
        print(f"  gate +/-{tol} done")

    summary = pd.DataFrame(rows)
    out_csv = DLR / f"pd_vs_snr_{args.tag}_ep{args.epoch:03d}.csv"
    pd.concat([c.assign(method=k, gate=g) for g, cs in all_curves.items()
               for k, c in cs.items()]).to_csv(out_csv, index=False)
    summary.to_csv(DLR / f"pd_vs_snr_{args.tag}_ep{args.epoch:03d}_summary.csv", index=False)

    print("\n" + "=" * 78)
    print(f"Pd vs cell SNR, {SPLIT}, matched measured Pfa {TARGET_PFA:.0e}")
    print("=" * 78)
    print(f"theory reaches Pd 0.5 at {th50:.2f} dB (a detector that KNOWS the noise level)")
    print(summary.round(3).to_string(index=False))
    print("\ngate 0 is the like-for-like comparison: theory is a single-cell test, so only there")
    print("is it a bound. At +/-1 a hit may land anywhere in 3x3 -- nine chances -- so every")
    print("method sits above theory and the comparison is not valid.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLOR = {"CA": "#4c72b0", "GO": "#dd8452", "OS": "#55a868", "SO": "#c44e52", "DL": "#8172b3"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    titles = {0: "gate 0 — exact cell: the VALID comparison",
              1: "gate ±1 — the project's operating point"}
    for ax, tol in zip(axes, (0, 1)):
        cs = all_curves[tol]
        ax.plot(s_line, theory_pd(s_line), "--", color="0.25", lw=1.6, zorder=5,
                label=r"theory: $P_d=P_{fa}^{1/(1+\mathrm{SNR})}$")
        for m in METHODS:
            c = cs[m]
            ax.plot(c.snr, c.pd, "o-", ms=4, lw=1.4, color=COLOR[m], alpha=0.85, label=f"{m}-CFAR")
        d = cs["DL"]
        ax.errorbar(d.snr, d.pd, yerr=d.se, fmt="s-", ms=5, lw=2.2, color=COLOR["DL"],
                    capsize=2.5, zorder=6, label=f"DL ({args.seeds} seeds)")
        ax.axhline(0.5, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel("target cell SNR (dB)")
        ax.set_title(titles[tol], fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(f"Pd at matched measured Pfa {TARGET_PFA:.0e}")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].text(0.97, 0.05, "theory is NOT a bound at this gate\n(3x3 = nine chances vs one)",
                 transform=axes[1].transAxes, ha="right", fontsize=8, color="#8a3a3a")
    fig.suptitle(f"Pd vs SNR against theory — {SPLIT}, one target per frame, no clutter",
                 fontsize=11)
    fig.tight_layout()
    out_png = ROOT / "results" / f"pd_vs_snr_{args.tag}_ep{args.epoch:03d}.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {out_png}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
