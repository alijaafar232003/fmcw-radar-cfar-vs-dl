"""Where should training stop? The whole curve, not four points.

`epoch_flatness.py` measured epochs {15, 25, 34, 50} and concluded "Pd flat, localization degrades".
Four points cannot distinguish a curve that is flat from one that is merely sampled where it
happens to be flat, and nothing below epoch 15 had ever been evaluated. This runs every epoch from
1 to 25 so the stopping point is read off a curve.

THE SELECTION RULE IS FIXED BEFORE THE NUMBERS EXIST (stated in PROGRESS.md and the notebook
section before any result is quoted):

    plateau  = max over epochs of pooled seed-mean Pd
    tolerance = the SE of the pooled seed-mean Pd at the plateau epoch
    chosen   = the EARLIEST epoch whose pooled seed-mean Pd >= plateau - tolerance

Pre-committing it is the point: "earliest epoch within 1 SE of the plateau" cannot be re-negotiated
once the curve is on screen, and it encodes the actual preference -- among epochs that detect
equally well, take the least-trained one, because localization is the metric that degrades with
training and it is where CFAR wins.

Two secondary questions this is also sized to answer:
  * does localization degrade monotonically from epoch 1, or start flat and turn? Monotone from
    the start supports the 3x3-label explanation (the network spreads confidence across the whole
    label block, and does so from the first epoch); a flat-then-turn shape does not.
  * how far back does Pd stay flat? Nothing below 15 had been tested.

4 seeds x 25 epochs, `bench2_val`, matched measured Pfa 1e-4, gate +/-1. Peaks are cached per
checkpoint so a re-run costs only the scoring.

Examples
--------
    python scripts/epoch_sweep.py --tag epsweep                 # score every checkpoint
    python scripts/epoch_sweep.py --tag epsweep --seeds 0,1     # one process per seed
    python scripts/epoch_sweep.py --tag epsweep --summarise     # table + plot from the CSV
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig                                     # noqa: E402
from cfar.detect import all_peaks                                    # noqa: E402
from dl.dataset import cache_paths                                   # noqa: E402
from dl.infer import cfar_valid_mask                                 # noqa: E402
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


def frame_ids() -> np.ndarray:
    """Frame id per row of the pre-normalised cache, in the cache's own (sorted) order.

    `build_cache` writes rows in `sorted(frames/)` order, so row k is `frame_{k:06d}`. Read the
    directory rather than assuming it: a missing frame would otherwise shift every label by one
    silently.
    """
    return np.array([int(p.name.split("_")[1])
                     for p in sorted((ROOT / "data" / SPLIT / "frames").iterdir()) if p.is_dir()])


def peaks_for(cp: Path, valid: np.ndarray, x_cache: np.ndarray, ids: np.ndarray,
              batch: int = 64) -> dict:
    """(K, 3) peaks per frame for one checkpoint, cached to disk.

    Inference runs off the pre-normalised cache in batches rather than re-reading and re-normalising
    1,600 .npy files per checkpoint. `normalize_rd` is deterministic and the cache was built by it,
    so this is the same input `predict_heatmap` would construct -- asserted against the raw path by
    `--verify-cache`. At 100 checkpoints the file-reading version dominates the run.
    """
    cache = DLR / f"peaks_val_{cp.stem.replace('tinycnn_', '')}.npz"
    if cache.exists():
        a = np.load(cache)["peaks"]
        return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}

    ck = torch.load(cp, weights_only=False)
    model = TinyCNN(width=ck["config"].get("width", 16))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    pk, rows = {}, []
    with torch.no_grad():
        for lo in range(0, len(ids), batch):
            xb = torch.from_numpy(np.array(x_cache[lo:lo + batch]))[:, None]
            out = model(xb)[:, 0].numpy()           # logits, as predict_heatmap returns
            for j, hm in enumerate(out):
                i = int(ids[lo + j])
                p = all_peaks(hm, valid, nms_size=NMS)
                pk[i] = p
                if len(p):
                    rows.append(np.column_stack([np.full(len(p), i, float), p]))

    np.savez_compressed(cache, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))
    return pk


def verify_cache_path(valid: np.ndarray, x_cache: np.ndarray, ids: np.ndarray) -> None:
    """Assert the batched cache path gives byte-identical peaks to the per-file path it replaces."""
    from dl.dataset import normalize_rd
    from dl.infer import predict_heatmap

    cp = sorted(DLR.glob("tinycnn_epflat_s0_ep*.pt"))[0]
    ck = torch.load(cp, weights_only=False)
    model = TinyCNN(width=ck["config"].get("width", 16))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    dirs = sorted(p for p in (ROOT / "data" / SPLIT / "frames").iterdir() if p.is_dir())
    for k in (0, 1, len(dirs) // 2, len(dirs) - 1):
        mag = np.load(dirs[k] / "rd_map_mag.npy")
        assert np.array_equal(normalize_rd(mag), np.asarray(x_cache[k])), f"cache != raw at {k}"

        a = all_peaks(predict_heatmap(model, mag), valid, nms_size=NMS)
        with torch.no_grad():
            hm = model(torch.from_numpy(np.asarray(x_cache[k]))[None, None])[0, 0].numpy()
        b = all_peaks(hm, valid, nms_size=NMS)
        assert np.array_equal(a, b), f"batched peaks != per-frame peaks at {k}"
    print(f"cache path verified against the per-file path on 4 frames of {cp.name}")


def score(pk: dict, groups, tb: dict, mc: MatchConfig, cache: Path | None = None) -> list[dict]:
    """Pd and localization error at the matched measured Pfa, per group.

    Cached per checkpoint. Scoring is ~60 s and cannot start until the epoch has trained, so the
    sweep is run repeatedly against a growing set of checkpoints while training is still going;
    without this every re-run would rescore everything already done.
    """
    if cache is not None and cache.exists():
        return pd.read_csv(cache).to_dict("records")

    out = []
    for gname, sub in groups:
        fr = [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))), SHAPE, valid_g)
              for i in sub]
        roc = roc_curve(fr, roc_thresholds(fr, mc), mc)
        _, thr = _at_measured_pfa(roc, TARGET_PFA)
        evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr)
                              else np.zeros((0, 2)), t, s, mc, valid_mask=v)
               for p, t, s, v in fr]
        a = aggregate(evs, range_res_m=RANGE_RES, vel_res_mps=VEL_RES)
        out.append(dict(group=gname, pd=a["pd"], dr_bins=a["mean_abs_dr_bins"],
                        dv_bins=a["mean_abs_dv_bins"], threshold=thr,
                        range_m=a["mean_abs_range_err_m"], vel_mps=a["mean_abs_vel_err_mps"]))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).to_csv(cache, index=False)
    return out


# ---------------------------------------------------------------- summary / selection / plot

def seed_average(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean and SE of the mean, per epoch and group."""
    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

    return (df.groupby(["group", "epoch"])
              .agg(n_seeds=("seed", "nunique"),
                   pd_mean=("pd", "mean"), pd_se=("pd", se),
                   dr_mean=("dr_bins", "mean"), dr_se=("dr_bins", se),
                   dv_mean=("dv_bins", "mean"), dv_se=("dv_bins", se))
              .reset_index())


def select_epoch(pooled: pd.DataFrame) -> dict:
    """THE PRE-REGISTERED RULE. Earliest epoch whose pooled Pd is within 1 SE of the plateau.

    `pooled` is the ALL-group seed-average, one row per epoch. The tolerance is the SE at the
    plateau epoch -- the sampling noise of the quantity the plateau is defined by.
    """
    p = pooled.sort_values("epoch").reset_index(drop=True)
    k = int(p.pd_mean.idxmax())
    plateau, tol = float(p.pd_mean[k]), float(p.pd_se[k])
    ok = p[p.pd_mean >= plateau - tol]
    row = ok.iloc[0]
    return {"plateau_epoch": int(p.epoch[k]), "plateau_pd": plateau, "tolerance_1se": tol,
            "chosen_epoch": int(row.epoch), "chosen_pd": float(row.pd_mean),
            "chosen_dr": float(row.dr_mean), "chosen_dv": float(row.dv_mean),
            "chosen_dr_se": float(row.dr_se), "chosen_dv_se": float(row.dv_se)}


def monotonicity(pooled: pd.DataFrame, col: str) -> dict:
    """Is the localization curve monotone from epoch 1, or flat then turning?

    Reported three ways rather than one: the fraction of consecutive steps that worsen, where the
    running minimum sits (a curve that degrades from the start has its minimum at epoch 1), and a
    Spearman correlation with epoch over the first third against the whole range.
    """
    from scipy import stats

    p = pooled.sort_values("epoch")
    y = p[col].to_numpy()
    e = p.epoch.to_numpy()
    d = np.diff(y)
    third = max(3, len(y) // 3)
    return {"argmin_epoch": int(e[int(np.argmin(y))]),
            "frac_steps_worsening": float((d > 0).mean()),
            "rho_all": float(stats.spearmanr(e, y).statistic),
            "rho_first_third": float(stats.spearmanr(e[:third], y[:third]).statistic),
            "delta_first_third": float(y[third - 1] - y[0]),
            "delta_rest": float(y[-1] - y[third - 1])}


def flatness_back(pooled: pd.DataFrame, sel: dict) -> pd.DataFrame:
    """How far back does Pd stay flat? Deficit from the plateau in units of the 1-SE tolerance."""
    p = pooled.sort_values("epoch").copy()
    p["deficit"] = sel["plateau_pd"] - p.pd_mean
    p["deficit_in_se"] = p.deficit / sel["tolerance_1se"]
    p["within_1se"] = p.deficit <= sel["tolerance_1se"]
    return p[["epoch", "pd_mean", "pd_se", "deficit", "deficit_in_se", "within_1se"]]


def plot(avg: pd.DataFrame, sel: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = avg[avg.group == "ALL"].sort_values("epoch")
    conds = sorted(c for c in avg.group.unique() if c != "ALL")

    # Four panels, not three. The pooled Pd needs its OWN axis: on a shared scale with the
    # per-condition curves (clutter 0.14 to isolated 0.73) the whole +0.059 rise that this
    # experiment exists to measure is a barely visible wiggle.
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1, 1, 1]})
    cmap = plt.get_cmap("tab10")

    ax = axes[0]
    ax.axhspan(sel["plateau_pd"] - sel["tolerance_1se"], sel["plateau_pd"],
               color="crimson", alpha=0.13, zorder=0,
               label="within 1 SE of the plateau")
    ax.errorbar(pooled.epoch, pooled.pd_mean, yerr=pooled.pd_se, color="k", lw=2.2,
                marker="o", ms=4, capsize=2, zorder=5)
    ax.set_ylabel("pooled Pd (zoomed)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    for ax, (mean, sem, label) in zip(axes[1:], [
            ("pd_mean", "pd_se", "Pd by condition"),
            ("dr_mean", "dr_se", "range error (bins)"),
            ("dv_mean", "dv_se", "Doppler error (bins)")]):
        for j, c in enumerate(conds):
            g = avg[avg.group == c].sort_values("epoch")
            ax.plot(g.epoch, g[mean], lw=1.0, alpha=0.5, color=cmap(j % 10), label=c)
        ax.errorbar(pooled.epoch, pooled[mean], yerr=pooled[sem], color="k", lw=2.2,
                    marker="o", ms=3.5, capsize=2, zorder=5, label="pooled")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    for ax in axes:
        ax.axvline(sel["chosen_epoch"], color="crimson", ls="--", lw=1.4, zorder=4)

    axes[0].set_title(f"Epoch sweep, pos_weight=100, 4 seeds, bench2_val, gate +/-1\n"
                      f"pre-registered rule selects epoch {sel['chosen_epoch']} "
                      f"(plateau {sel['plateau_pd']:.4f} at epoch {sel['plateau_epoch']}, "
                      f"1 SE = {sel['tolerance_1se']:.4f})", fontsize=10)
    axes[1].legend(fontsize=7, ncol=4, loc="lower right")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def collect(tag: str, tolerance: int) -> pd.DataFrame:
    """Assemble the per-checkpoint score caches into one frame (the seeds run as separate procs)."""
    rows = []
    for f in sorted((DLR / "epoch_sweep_cache").glob(f"{tag}_s*_ep*_tol{tolerance}.csv")):
        m = re.match(rf"{tag}_s(\d+)_ep(\d+)_tol{tolerance}$", f.stem)
        d = pd.read_csv(f)
        d.insert(0, "epoch", int(m.group(2)))
        d.insert(0, "seed", int(m.group(1)))
        rows.append(d)
    if not rows:
        raise SystemExit(f"no score caches for tag {tag!r} at tolerance {tolerance}")
    df = pd.concat(rows, ignore_index=True)
    out = DLR / f"epoch_sweep_{tag}.csv"
    df.to_csv(out, index=False)
    print(f"collected {df.seed.nunique()} seeds x {df.epoch.nunique()} epochs -> {out}")
    return df


def summarise(csv: Path, tag: str) -> None:
    df = pd.read_csv(csv)
    avg = seed_average(df)
    pooled = avg[avg.group == "ALL"].sort_values("epoch").reset_index(drop=True)
    sel = select_epoch(pooled)

    print("\n" + "=" * 78)
    print("SELECTION RULE, FIXED BEFORE THE RESULTS: earliest epoch whose pooled seed-mean Pd")
    print("is within 1 SE (of the plateau epoch's seed-mean) of the plateau.")
    print("=" * 78)
    print(f"plateau      Pd {sel['plateau_pd']:.4f} at epoch {sel['plateau_epoch']}")
    print(f"tolerance    1 SE = {sel['tolerance_1se']:.4f}  -> floor {sel['plateau_pd'] - sel['tolerance_1se']:.4f}")
    print(f"CHOSEN       epoch {sel['chosen_epoch']}   Pd {sel['chosen_pd']:.4f}   "
          f"range {sel['chosen_dr']:.3f}+/-{sel['chosen_dr_se']:.3f} bins   "
          f"Doppler {sel['chosen_dv']:.3f}+/-{sel['chosen_dv_se']:.3f} bins")

    print("\nPOOLED, seed-mean +/- SE (4 seeds)")
    t = pooled[["epoch", "n_seeds", "pd_mean", "pd_se", "dr_mean", "dr_se", "dv_mean", "dv_se"]]
    print(t.round(4).to_string(index=False))

    print("\nQ1  does localization degrade monotonically from epoch 1?")
    for col, name in (("dr_mean", "range  "), ("dv_mean", "Doppler")):
        m = monotonicity(pooled, col)
        print(f"  {name}  min at epoch {m['argmin_epoch']:>2} | "
              f"{m['frac_steps_worsening']:.0%} of steps worsen | "
              f"rho(all) {m['rho_all']:+.3f}  rho(first third) {m['rho_first_third']:+.3f} | "
              f"delta first third {m['delta_first_third']:+.3f}, rest {m['delta_rest']:+.3f}")

    print("\nQ2  where does Pd stop being flat going earlier? (deficit from plateau, in SE)")
    print(flatness_back(pooled, sel).round(4).to_string(index=False))

    print("\nPd by condition (seed mean)")
    print(avg[avg.group != "ALL"].pivot_table(index="group", columns="epoch",
                                              values="pd_mean").round(3).to_string())
    print("\nRange error by condition (bins)")
    print(avg[avg.group != "ALL"].pivot_table(index="group", columns="epoch",
                                              values="dr_mean").round(3).to_string())
    print("\nDoppler error by condition (bins)")
    print(avg[avg.group != "ALL"].pivot_table(index="group", columns="epoch",
                                              values="dv_mean").round(3).to_string())

    avg.to_csv(DLR / f"epoch_sweep_{tag}_seedavg.csv", index=False)
    pd.DataFrame([sel]).to_csv(DLR / f"epoch_sweep_{tag}_selection.csv", index=False)
    plot(avg, sel, ROOT / "results" / f"epoch_sweep_{tag}.png")
    print(f"wrote {DLR / f'epoch_sweep_{tag}_seedavg.csv'} and _selection.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="epsweep")
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("--seeds", default=None, help="comma-separated subset, for parallel runs")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--summarise", action="store_true", help="table + plot from an existing CSV")
    ap.add_argument("--verify-cache", action="store_true")
    args = ap.parse_args()

    global valid_g
    valid_g = cfar_valid_mask(SHAPE, CfarConfig())

    if args.summarise:
        collect(args.tag, args.tolerance)
        summarise(DLR / f"epoch_sweep_{args.tag}.csv", args.tag)
        return

    if args.threads:
        torch.set_num_threads(args.threads)

    xp, _ = cache_paths(SPLIT)
    x_cache = np.load(xp, mmap_mode="r")
    ids = frame_ids()
    assert len(ids) == len(x_cache) == 1600, f"{len(ids)} ids vs {len(x_cache)} cached rows"

    if args.verify_cache:
        verify_cache_path(valid_g, x_cache, ids)
        return

    frames_meta = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float) for f, g in truths.groupby("frame")}
    conds = sorted(c for c in frames_meta.condition.unique() if c != "noise")
    groups = [("ALL", frames_meta.frame.to_numpy())] + \
             [(c, frames_meta.loc[frames_meta.condition == c, "frame"].to_numpy()) for c in conds]
    mc = MatchConfig(args.tolerance, args.tolerance)

    want = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    ckpts = []
    for cp in sorted(DLR.glob(f"tinycnn_{args.tag}_s*_ep*.pt")):
        m = re.match(rf"tinycnn_{args.tag}_s(\d+)_ep(\d+)$", cp.stem)
        if m and (want is None or int(m.group(1)) in want):
            ckpts.append((int(m.group(1)), int(m.group(2)), cp))
    ckpts.sort()
    print(f"{len(ckpts)} checkpoints to score"
          f"{'' if want is None else f' (seeds {sorted(want)})'}", flush=True)

    sc_dir = DLR / "epoch_sweep_cache"
    rows, t0 = [], time.perf_counter()
    for n, (seed, epoch, cp) in enumerate(ckpts, 1):
        cache = sc_dir / f"{args.tag}_s{seed}_ep{epoch:03d}_tol{args.tolerance}.csv"
        pk = peaks_for(cp, valid_g, x_cache, ids) if not cache.exists() else {}
        for r in score(pk, groups, tb, mc, cache=cache):
            rows.append(dict(seed=seed, epoch=epoch, **r))
        el = time.perf_counter() - t0
        print(f"  [{n}/{len(ckpts)}] seed {seed} epoch {epoch:>2}  "
              f"({el / n:.0f}s/ckpt, {el / 60:.0f} min elapsed)", flush=True)

    suffix = "" if want is None else f"_s{'-'.join(str(s) for s in sorted(want))}"
    out = DLR / f"epoch_sweep_{args.tag}{suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
