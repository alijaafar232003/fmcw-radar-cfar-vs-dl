"""Score the E4 patch arms on bench2_val through the project's unmodified evaluation pipeline.

The UNMASKED arm is scored densely -- verified equivalent to patch-centre inference to 1.9e-06 in
eval mode -- which is ~13 s per seed. The MASKED arm has no dense equivalent by construction, so
every valid CUT needs its own masked patch: ~13,216 patches per frame.

⚠️ bench2_test is never opened; asserted below.

    python experiments/e4_cut_guard_mask/scripts/evaluate_e4.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "src"))

from cfar.cfar import CfarConfig                                    # noqa: E402
from cfar.detect import all_peaks                                   # noqa: E402
from dl.dataset import normalize_rd                                 # noqa: E402
from dl.infer import cfar_valid_mask                                # noqa: E402
from dl.models import TinyCNN                                       # noqa: E402
from dl.stratified_eval import _at_measured_pfa                     # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,           # noqa: E402
                  roc_curve, roc_thresholds)
from patches import HW_D, HW_R, MASK, NEUTRAL                       # noqa: E402

SPLIT = "bench2_val"
assert "test" not in SPLIT, "E4 must never score bench2_test"
SHAPE, NMS, TARGET_PFA, N_BOOT = (256, 64), 5, 1e-4, 2000
GATES, SEEDS, EPOCH = (0, 1, 3), range(8), 21
RANGE_RES, VEL_RES = 0.19518, 0.42013
VALID = cfar_valid_mask(SHAPE, CfarConfig())
PKD = HERE / "cache" / "peaks"
RES = HERE / "results"


def masked_score_map(model, x: np.ndarray) -> np.ndarray:
    """Per-CUT guard-masked logits. No dense equivalent exists -- that is the premise of E4."""
    rs, vs = np.nonzero(VALID)
    out = np.zeros(SHAPE, dtype=np.float32)
    P = np.empty((len(rs), 2 * HW_R + 1, 2 * HW_D + 1), dtype=np.float32)
    for k, (r, v) in enumerate(zip(rs, vs)):
        P[k] = x[r - HW_R:r + HW_R + 1, v - HW_D:v + HW_D + 1]
    P[:, MASK] = NEUTRAL
    with torch.no_grad():
        vals = []
        for i in range(0, len(P), 8192):
            b = torch.from_numpy(P[i:i + 8192])[:, None]
            vals.append(model(b)[:, 0, HW_R, HW_D].numpy())
    out[rs, vs] = np.concatenate(vals)
    return out


def load_peaks(p):
    a = np.load(p)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def save_peaks(pk, p):
    rows = [np.column_stack([np.full(len(v), i, float), v]) for i, v in sorted(pk.items()) if len(v)]
    np.savez_compressed(p, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))


def score(pk, ids, tb, mc):
    fr = [(pk.get(int(i), np.zeros((0, 3))), tb.get(i, np.zeros((0, 2))), SHAPE, VALID) for i in ids]
    thr = _at_measured_pfa(roc_curve(fr, roc_thresholds(fr, mc), mc), TARGET_PFA)[1]
    evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2)),
                          t, s, mc, valid_mask=v) for p, t, s, v in fr]
    a = aggregate(evs, range_res_m=RANGE_RES, vel_res_mps=VEL_RES)
    return a, thr, np.array([e.tp for e in evs], float), np.array([len(t) for _, t, _, _ in fr],
                                                                  float)


def main() -> None:
    PKD.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    CV = ROOT / "results" / "cfar_val"
    fm = pd.read_csv(CV / "frames.csv")
    tb = {f: g[["range_bin", "velocity_bin"]].to_numpy(float)
          for f, g in pd.read_csv(CV / "truths.csv").groupby("frame")}
    ids = fm.frame.to_numpy()
    conds = sorted(c for c in fm.condition.unique() if c != "noise")

    # ---------------- peaks
    for arm in ("unmasked", "masked"):
        for s in SEEDS:
            out = PKD / f"peaks_{arm}_s{s}_ep{EPOCH:03d}.npz"
            if out.exists():
                continue
            ck = torch.load(HERE / "checkpoints" / f"tinycnn_{arm}_s{s}_ep{EPOCH:03d}.pt",
                            weights_only=False)
            m = TinyCNN(width=16)
            m.load_state_dict(ck["state_dict"])
            m.eval()
            t0, pk = time.perf_counter(), {}
            for i in ids:
                x = normalize_rd(np.load(ROOT / "data" / SPLIT / "frames" /
                                         f"frame_{int(i):06d}" / "rd_map_mag.npy"))
                if arm == "unmasked":
                    with torch.no_grad():
                        sm = m(torch.from_numpy(x)[None, None])[0, 0].numpy()
                else:
                    sm = masked_score_map(m, x)
                pk[int(i)] = all_peaks(sm, VALID, nms_size=NMS)
            save_peaks(pk, out)
            print(f"  {arm}_s{s}: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    peaks = {(a, s): load_peaks(PKD / f"peaks_{a}_s{s}_ep{EPOCH:03d}.npz")
             for a in ("unmasked", "masked") for s in SEEDS}

    # ---------------- per condition, per gate
    rows, hits = [], {}
    for g in GATES:
        mc = MatchConfig(g, g)
        for cond in conds:
            cid = fm.loc[fm.condition == cond, "frame"].to_numpy()
            for arm in ("unmasked", "masked"):
                per, hh, n = [], [], None
                for s in SEEDS:
                    a, thr, h, n = score(peaks[(arm, s)], cid, tb, mc)
                    per.append(a)
                    hh.append(h)
                hits[(g, cond, arm)] = (np.array(hh), n)
                f = lambda k: np.array([p[k] for p in per], float)
                rows.append(dict(gate=g, condition=cond, arm=arm, n_seeds=len(per),
                                 pd=f("pd").mean(), pd_se=f("pd").std(ddof=1) / np.sqrt(len(per)),
                                 measured_pfa=f("pfa_per_cell").mean(),
                                 dr_bins=f("mean_abs_dr_bins").mean(),
                                 dv_bins=f("mean_abs_dv_bins").mean()))
            print(f"  gate {g} {cond} done", flush=True)
    pd.DataFrame(rows).to_csv(RES / "per_condition.csv", index=False)

    # ---------------- paired bootstrap, seed-paired
    rng = np.random.default_rng(0)
    out = []
    for g in GATES:
        for cond in conds:
            hu, n = hits[(g, cond, "unmasked")]
            hm, _ = hits[(g, cond, "masked")]
            du = hu.mean(axis=0)
            dm = hm.mean(axis=0)
            k = rng.integers(0, len(n), size=(N_BOOT, len(n)))
            tot = n[k].sum(axis=1)
            d = (du[k].sum(axis=1) - dm[k].sum(axis=1)) / tot
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            per_seed = np.array([(hu[s].sum() - hm[s].sum()) / n.sum() for s in range(len(SEEDS))])
            out.append(dict(gate=g, condition=cond,
                            pd_unmasked=du.sum() / n.sum(), pd_masked=dm.sum() / n.sum(),
                            delta_pd=(du.sum() - dm.sum()) / n.sum(),
                            ci_lo=float(lo), ci_hi=float(hi), p_boot=float(p),
                            seed_paired_mean=per_seed.mean(),
                            seed_paired_se=per_seed.std(ddof=1) / np.sqrt(len(per_seed)),
                            seeds_positive=int((per_seed > 0).sum())))
    D = pd.DataFrame(out)
    D["alpha_bonf"] = 0.05 / len(conds)
    D["sig"] = D.p_boot < D.alpha_bonf
    D.to_csv(RES / "deltas.csv", index=False)

    d1 = D[D.gate == 1]
    print(f"\ngate +/-1: mean dPd {d1.delta_pd.mean():+.4f}, "
          f"positive on {int((d1.delta_pd > 0).sum())}/{len(d1)}, "
          f"significant on {int(d1.sig.sum())}/{len(d1)}")
    print(f"wrote {RES}")


if __name__ == "__main__":
    main()
