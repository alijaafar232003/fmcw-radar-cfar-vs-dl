"""Why does GO lose `clutter_edge` at matched measured Pfa, when it is designed for exactly that?

Run on `bench2_tune` + `bench2_val` only — never on the test split, so anything this motivates
stays a legitimate tuning decision rather than a test-set fit.

Four questions, cheapest first:

  1. Does the stratum actually straddle? Targets sit at |v| 2.5-4.2 m/s and the Doppler half-width
     is 4 bins = 1.68 m/s. If some frames' windows never reach the ridge, the stratum is diluted
     and every method's edge behaviour is being averaged with plain-noise behaviour.
  2. What do the three noise estimates actually do at the target cell, as a function of how much
     of the window sits in the ridge?
  3. How much of GO's disadvantage is its homogeneous-noise bias (max of two estimates is biased
     high) rather than anything to do with edges?
  4. Is the 21x9 window simply wrong FOR GO? Step 6 tuned one geometry for all three methods,
     which silently assumes their optima coincide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import (CfarConfig, cfar_ratio_map, _ring_footprint,      # noqa: E402
                       _ring_mean, _half_ring_means, _os_ring_statistic)
from cfar.detect import all_peaks                                        # noqa: E402
from eval import MatchConfig, aggregate, evaluate_frame, roc_curve       # noqa: E402
from radar.config import RadarConfig                                     # noqa: E402
from radar.simulator import make_axes                                    # noqa: E402

SPLITS = ["bench2_tune", "bench2_val"]
COND = "clutter_edge"
TARGET_PFA = 1e-4
MC = MatchConfig(1, 1)

QUANTILES = np.clip(np.concatenate([
    np.linspace(0.0, 0.99, 40),
    1.0 - np.logspace(np.log10(0.01), np.log10(2e-5), 40)]), 0.0, 1.0)


def load(cond=COND):
    out = []
    for s in SPLITS:
        for fd in sorted((ROOT / "data" / s / "frames").iterdir()):
            md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
            if md.get("condition") != cond:
                continue
            lab = (pd.read_csv(fd / "labels.csv")
                   if (fd / "labels.csv").stat().st_size > 5 else pd.DataFrame())
            out.append((np.load(fd / "rd_map_mag.npy").astype(np.float64) ** 2, lab, md))
    return out


def pd_at(roc, target):
    d = roc.dropna(subset=["pfa_per_cell"])
    d = (d.groupby("pfa_per_cell", as_index=False)
           .agg(pd=("pd", "max"), threshold=("threshold", "min"), fp=("fp", "max"))
           .sort_values("pfa_per_cell"))
    pfa, fp = d.pfa_per_cell.to_numpy(float), d.fp.to_numpy(float)
    obs = (fp > 0) & (pfa > 0)
    if not obs.any():
        return np.nan, np.nan
    cells = float(np.round(np.median(fp[obs] / pfa[obs])))
    pfa = np.where(pfa > 0, pfa, 0.5 / cells)
    x, t = np.log10(pfa), np.log10(target)
    if len(x) < 2 or t < x.min() or t > x.max():
        return np.nan, np.nan
    return float(np.interp(t, x, d.pd.to_numpy())), float(np.interp(t, x, d.threshold.to_numpy()))


def main() -> None:
    frames = load()
    print(f"{len(frames)} {COND} frames from {' + '.join(SPLITS)}")

    _, _, _, vax = make_axes(RadarConfig())
    v0 = int(np.argmin(np.abs(vax)))
    cfg = CfarConfig()
    hw_r, hw_d, g_d = cfg.half_width_r, cfg.half_width_d, cfg.guard_d

    # ---------------------------------------------------------------- 1. where is the ridge?
    print("\n" + "=" * 78)
    print("1. RIDGE EXTENT, MEASURED (mean Doppler profile over these frames)")
    print("=" * 78)
    prof = np.mean([p.mean(axis=0) for p, _, _ in frames], axis=0)
    floor = np.median(prof[:v0 - 20])
    db = 10 * np.log10(prof / floor)
    ridge = np.where(db > 3.0)[0]
    lo, hi = int(ridge.min()), int(ridge.max())
    print(f"   zero-Doppler bin      {v0}")
    print(f"   >3 dB above floor     bins {lo}..{hi}  (half-width {max(v0 - lo, hi - v0)} bins)")
    for off in range(0, 12):
        print(f"      offset {off:>2} bins ({off * 0.42:4.2f} m/s):  {db[v0 + off]:6.2f} dB")

    # ------------------------------------------------------- 2. does every window straddle it?
    print("\n" + "=" * 78)
    print("2. DOES THE WINDOW ACTUALLY REACH THE RIDGE?")
    print("=" * 78)
    print(f"   Doppler half-width {hw_d} bins; guard {g_d}; so the TRAINING columns are")
    print(f"   offsets +/-{g_d + 1}..{hw_d} from the cell under test.")

    rows = []
    for p, lab, md in frames:
        for _, L in lab.iterrows():
            vb = int(L["velocity_bin"])
            train_lo, train_hi = vb - hw_d, vb + hw_d
            # training columns that fall inside the measured ridge
            cols = [c for c in range(train_lo, train_hi + 1)
                    if abs(c - vb) > g_d and lo <= c <= hi]
            near = [c for c in cols if (c - vb) * (v0 - vb) > 0]     # on the ridge side
            rows.append(dict(vb=vb, off=abs(vb - v0), n_train_in_ridge=len(cols),
                             n_near_half=len(near), snr=L["cell_snr_db"]))
    g = pd.DataFrame(rows)
    print(f"\n   targets: {len(g)}")
    print(f"   {'window training cols in ridge':<34}{'targets':>9}{'share':>8}")
    for k, sub in g.groupby("n_train_in_ridge"):
        print(f"   {k:<34}{len(sub):>9}{100 * len(sub) / len(g):>7.1f}%")
    print(f"\n   -> {100 * (g.n_train_in_ridge == 0).mean():.1f}% of targets have a window that "
          f"NEVER touches the ridge")
    print(f"   Doppler offset from zero: {g.off.min()}..{g.off.max()} bins "
          f"({0.42 * g.off.min():.2f}..{0.42 * g.off.max():.2f} m/s)")

    # ------------------------------------------- 3. the three estimates at the target cell
    print("\n" + "=" * 78)
    print("3. NOISE ESTIMATE AT THE TARGET CELL, BY RIDGE OVERLAP")
    print("=" * 78)
    print("   Ratio to CA. GO > 1 means GO is more conservative (higher threshold, lower Pd).")

    rank = int(round(cfg.os_rank_frac * cfg.num_ref_cells))
    est = []
    for p, lab, md in frames:
        ca = _ring_mean(p, hw_r, hw_d, cfg.guard_r, g_d)
        lead, lag = _half_ring_means(p, hw_r, hw_d, cfg.guard_r, g_d, axis=cfg.go_axis)
        go = np.maximum(lead, lag)
        os_ = _os_ring_statistic(p, hw_r, hw_d, cfg.guard_r, g_d, rank)
        for _, L in lab.iterrows():
            rb, vb = int(L["range_bin"]), int(L["velocity_bin"])
            cols = [c for c in range(vb - hw_d, vb + hw_d + 1)
                    if abs(c - vb) > g_d and lo <= c <= hi]
            est.append(dict(overlap=len(cols), ca=ca[rb, vb], go=go[rb, vb], os=os_[rb, vb],
                            go_ca=go[rb, vb] / ca[rb, vb], os_ca=os_[rb, vb] / ca[rb, vb]))
    e = pd.DataFrame(est)
    print(f"\n   {'overlap':<9}{'n':>6}{'GO/CA':>9}{'OS/CA':>9}")
    for k, sub in e.groupby("overlap"):
        print(f"   {k:<9}{len(sub):>6}{sub.go_ca.median():>9.2f}{sub.os_ca.median():>9.2f}")
    print(f"\n   homogeneous-noise reference: GO/CA = 1.09 (max of two half-means),")
    print(f"   OS/CA = 1.40 (0.75-quantile of exponential). Anything ABOVE those is the edge.")

    # ------------------------------------------------ 4. is 21x9 simply the wrong window for GO?
    print("\n" + "=" * 78)
    print("4. PER-METHOD WINDOW SWEEP ON THE TUNING DATA")
    print("=" * 78)
    print("   Step 6 chose ONE geometry for all three methods. That is only fair if their")
    print("   optima coincide. Sweeping the Doppler training depth per method:")
    print(f"\n   {'window':<10}{'CA':>8}{'GO':>8}{'OS':>8}")

    geoms = {"21x7": (2, 1), "21x9": (2, 2), "21x11": (2, 3),
             "21x13": (2, 4), "21x17": (2, 6), "21x21": (2, 8)}

    best = {}
    for label, (gv, tv) in geoms.items():
        out = {}
        for kind in ("CA", "GO", "OS"):
            c = CfarConfig(kind=kind, guard_v=gv, train_v=tv)
            scored = []
            for p, lab, md in frames:
                ratio, valid, _ = cfar_ratio_map(p, c)
                tru = (lab[["range_bin", "velocity_bin"]].to_numpy(float)
                       if len(lab) else np.zeros((0, 2)))
                scored.append((all_peaks(ratio, valid, nms_size=5), tru, p.shape, valid))
            pool = np.concatenate([s[0][:, 2] for s in scored if len(s[0])])
            roc = roc_curve(scored, np.unique(np.quantile(pool, QUANTILES)), MC)
            pdv, thr = pd_at(roc, TARGET_PFA)

            evs = [evaluate_frame(s[0][s[0][:, 2] >= thr][:, :2] if (len(s[0]) and thr == thr)
                                  else np.zeros((0, 2)), s[1], s[2], MC, valid_mask=s[3])
                   for s in scored]
            out[kind] = aggregate(evs)["pd"]
            best.setdefault(kind, []).append((label, out[kind]))
        print(f"   {label:<10}{out['CA']:>8.3f}{out['GO']:>8.3f}{out['OS']:>8.3f}")

    print("\n   best window per method:")
    for kind, lst in best.items():
        lab, v = max(lst, key=lambda t: (t[1] if t[1] == t[1] else -1))
        print(f"      {kind}: {lab}  Pd {v:.3f}")


if __name__ == "__main__":
    main()
