"""Phase 6f - complete E3 (label ablation) at the final suppression radius.

`e3_run_eval.py` re-extracted the 24 frozen E3 checkpoints at radius 2 for the REPORTED epoch only,
and `p11_e3_fast.py` reproduced the gate-0 bootstrap from those peaks. Two things Chapter 6 quotes
were still missing at the final radius: the tolerant-gate numbers (Pd and localisation error at
+/-1), and the epoch curve, which needs the three earlier checkpoints scored as well.

This completes both. It reproduces every E3 artefact the report reads:

    epoch_curve.csv      pooled Pd + localisation vs epoch, gates 0 / +/-1 / +/-3
    per_condition.csv    epoch 21, per condition, all gates
    deltas.csv           paired bootstrap of each arm against the flat 3x3 control
    cfar_reference.csv   CFAR localisation reference at the matched radius

Stage 1 runs inference for epochs 5, 10 and 15 from the EXISTING frozen checkpoints -- no model is
retrained, re-initialised or fine-tuned. Epoch 21 is already cached from `e3_run_eval.py` and is
reused. Stages 2-5 mirror `run_eval.py` exactly, in the same order and with the same bootstrap seed,
substituting `fast_eval.FrameScorer` for the per-threshold `evaluate_frame` call; `verify()` asserts
the two agree exactly before any result is computed.

The scoring calls are independent of one another -- each is one (arm, seed, epoch, gate) swept to
its own operating point -- so they are handed to a process pool. That is a scheduling change only:
each task computes exactly what the serial version computes, results are collected in the order the
tasks were submitted, and the one stage with a random number generator (the stage-4 bootstrap) stays
sequential and keeps its seed.

The CFAR reference is read from the radius-MATCHED caches in results/final/peaks rather than from
`results/cfar_val/peaks_*.npz`, which are radius-5. Comparing radius-2 detections against radius-5
baselines would be a mixed measurement that looks like a result.

Writes into results/p10_attribution/e3/ only. The original experiment folder is not touched.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(EXP))
LAB = os.path.join(ROOT, "experiments", "label_ablation")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, LAB)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfar.cfar import CfarConfig                                            # noqa: E402
from dl.infer import cfar_valid_mask                                        # noqa: E402
from dl.stratified_eval import _at_measured_pfa                             # noqa: E402
from eval import MatchConfig, roc_thresholds                                # noqa: E402

from fast_eval import FrameScorer, verify                                   # noqa: E402

SPLIT = "bench2_val"
assert "test" not in SPLIT, "E3 must never score bench2_test"
SHAPE = (256, 64)
RADIUS = 2
TARGET_PFA = 1e-4
N_BOOT = 2000
ARMS = ("point", "flat3x3", "gaussian")
SEEDS = tuple(range(8))
EPOCHS = (5, 10, 15, 21)
GATES = (0, 1, 3)
KINDS = ("CA", "GO", "OS", "SO")
RANGE_RES, VEL_RES = 0.19518, 0.42013
N_PROC = 7
VALID = cfar_valid_mask(SHAPE, CfarConfig())

RES = os.path.join(EXP, "results", "p10_attribution", "e3")
PKD = os.path.join(RES, "peaks")
R2PK = os.path.join(EXP, "results", "final", "peaks")


def load_peaks(path):
    a = np.load(path)["peaks"]
    return {int(f): a[a[:, 0] == f][:, 1:] for f in np.unique(a[:, 0])}


def save_peaks(pk, path):
    rows = [np.column_stack([np.full(len(v), i, float), v]) for i, v in sorted(pk.items()) if len(v)]
    np.savez_compressed(path, peaks=np.concatenate(rows) if rows else np.zeros((0, 4)))


def frames_of(pk, ids, tb):
    return [(pk.get(int(i), np.zeros((0, 3))), tb.get(int(i), np.zeros((0, 2))), SHAPE, VALID)
            for i in ids]


def sweep(scorers, thresholds):
    rows = []
    for thr in thresholds:
        tp = fp = fn = cells = 0
        for s in scorers:
            a, b, c, d, _, _, _ = s.at(thr)
            tp += a
            fp += b
            fn += c
            cells += d
        rows.append(dict(threshold=float(thr), pd=tp / (tp + fn) if (tp + fn) else np.nan,
                         pfa_per_cell=fp / cells if cells else np.nan,
                         fa_per_frame=fp / len(scorers), tp=tp, fp=fp, fn=fn))
    return pd.DataFrame(rows)


def score_arm(pk, ids, tb, tol):
    """Sweep to the matched operating point and score there, as `run_eval.score` does."""
    fr = frames_of(pk, ids, tb)
    grid = roc_thresholds(fr, MatchConfig(tol, tol))
    scorers = [FrameScorer(f[0], f[1], SHAPE, tol, tol, VALID) for f in fr]
    _, thr = _at_measured_pfa(sweep(scorers, grid), TARGET_PFA)

    hits, ntru, fp, cells, dr, dv = [], [], 0, 0, [], []
    for s in scorers:
        a, b, c, d, x, y, _ = s.at(thr)
        hits.append(a)
        ntru.append(a + c)
        fp += b
        cells += d
        dr.extend(x)
        dv.extend(y)
    hits = np.array(hits, float)
    ntru = np.array(ntru, float)
    adr = np.abs(np.asarray(dr, float))
    adv = np.abs(np.asarray(dv, float))
    return dict(threshold=float(thr), hits=hits, n=ntru,
                pd=float(hits.sum() / ntru.sum()) if ntru.sum() else np.nan,
                pfa=fp / cells if cells else np.nan,
                dr_bins=float(adr.mean()) if adr.size else np.nan,
                dv_bins=float(adv.mean()) if adv.size else np.nan)


# ------------------------------------------------------------------ process pool
_TB = None
_CACHE = {}


def _init():
    """Each worker loads the truth table once and caches the peak files it is handed."""
    global _TB
    truths = pd.read_csv(os.path.join(ROOT, "results", "cfar_val", "truths.csv"))
    _TB = {int(f): g[["range_bin", "velocity_bin"]].to_numpy(float)
           for f, g in truths.groupby("frame")}


def _task(args):
    """One (peak file, frame subset, gate) scoring job. Deterministic; no RNG."""
    path, ids, tol, want_hits = args
    pk = _CACHE.get(path)
    if pk is None:
        if len(_CACHE) > 6:
            _CACHE.clear()
        pk = _CACHE[path] = load_peaks(path)
    r = score_arm(pk, np.asarray(ids), _TB, tol)
    if not want_hits:
        r.pop("hits")
        r.pop("n")
    return r


def main():
    t_start = time.perf_counter()
    os.makedirs(PKD, exist_ok=True)
    print("=" * 78)
    print("E3 LABEL ABLATION - radius %d on %s" % (RADIUS, SPLIT))
    print("=" * 78)

    fm = pd.read_csv(os.path.join(ROOT, "results", "cfar_val", "frames.csv"))
    truths = pd.read_csv(os.path.join(ROOT, "results", "cfar_val", "truths.csv"))
    TB = {int(f): g[["range_bin", "velocity_bin"]].to_numpy(float)
          for f, g in truths.groupby("frame")}
    ids_all = fm.frame.to_numpy()
    conds = sorted(c for c in fm.condition.unique() if c != "noise")

    def pkpath(a, s, e):
        return os.path.join(PKD, "peaks_%s_s%d_ep%03d.npz" % (a, s, e))

    # ---------------- 1. peaks for the epochs that are not yet cached at this radius
    todo = [(a, s, e) for a in ARMS for s in SEEDS for e in EPOCHS
            if not os.path.exists(pkpath(a, s, e))]
    print("stage 1: %d checkpoints to score at radius %d (%d already cached)"
          % (len(todo), RADIUS, len(ARMS) * len(SEEDS) * len(EPOCHS) - len(todo)))
    t0 = time.perf_counter()
    if todo:
        import torch
        from cfar.detect import all_peaks
        from dl.infer import predict_heatmap
        from dl.models import TinyCNN
        torch.set_num_threads(6)
        for n, (a, s, e) in enumerate(todo):
            ck = torch.load(os.path.join(LAB, "checkpoints",
                                         "tinycnn_%s_s%d_ep%03d.pt" % (a, s, e)),
                            weights_only=False)
            m = TinyCNN(width=ck["config"].get("width", 16))
            m.load_state_dict(ck["state_dict"])
            m.eval()
            out = {}
            for i in ids_all:
                mag = np.load(os.path.join(ROOT, "data", SPLIT, "frames",
                                           "frame_%06d" % int(i), "rd_map_mag.npy"))
                out[int(i)] = all_peaks(predict_heatmap(m, mag), VALID, nms_size=RADIUS)
            save_peaks(out, pkpath(a, s, e))
            print("    [%d/%d] %s_s%d_ep%03d" % (n + 1, len(todo), a, s, e), flush=True)
    t_infer = time.perf_counter() - t0

    # ---------------- prove the fast path is the same protocol
    fr_chk = frames_of(load_peaks(pkpath("flat3x3", 0, 21)), ids_all[:200], TB)
    g = roc_thresholds(fr_chk, MatchConfig(1, 1))
    nchk = verify(fr_chk, g[::15], 1, 1, SHAPE, VALID, n=8)
    nchk += verify(fr_chk, g[::15], 0, 0, SHAPE, VALID, n=8)
    print("fast_eval verified bit-identical to evaluate_frame on %d frame-threshold pairs" % nchk,
          flush=True)

    pool = mp.Pool(N_PROC, initializer=_init)
    ids_t = tuple(int(i) for i in ids_all)

    # ---------------- 2. pooled, vs epoch, all gates
    print("\nstage 2: pooled Pd + localisation vs epoch (%d workers)" % N_PROC, flush=True)
    t0 = time.perf_counter()
    keys = [(g_, a, e) for g_ in GATES for a in ARMS for e in EPOCHS]
    tasks = [(pkpath(a, s, e), ids_t, g_, False) for (g_, a, e) in keys for s in SEEDS]
    got = pool.map(_task, tasks, chunksize=1)
    rows = []
    for j, (g_, a, e) in enumerate(keys):
        per = got[j * len(SEEDS):(j + 1) * len(SEEDS)]
        f = lambda k: np.array([p[k] for p in per], float)
        rows.append(dict(
            gate=g_, arm=a, epoch=e, n_seeds=len(per),
            pd=f("pd").mean(), pd_se=f("pd").std(ddof=1) / np.sqrt(len(per)),
            dr_bins=f("dr_bins").mean(), dr_se=f("dr_bins").std(ddof=1) / np.sqrt(len(per)),
            dv_bins=f("dv_bins").mean(), dv_se=f("dv_bins").std(ddof=1) / np.sqrt(len(per)),
            range_m=f("dr_bins").mean() * RANGE_RES,
            vel_mps=f("dv_bins").mean() * VEL_RES))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "epoch_curve.csv"), index=False)
    print("    %.1f min" % ((time.perf_counter() - t0) / 60), flush=True)

    # ---------------- 3. epoch 21, per condition, all gates
    print("\nstage 3: epoch 21 per condition", flush=True)
    t0 = time.perf_counter()
    cids = {c: tuple(int(i) for i in fm.loc[fm.condition == c, "frame"].to_numpy())
            for c in conds}
    keys = [(g_, c, a) for g_ in GATES for c in conds for a in ARMS]
    tasks = [(pkpath(a, s, 21), cids[c], g_, True) for (g_, c, a) in keys for s in SEEDS]
    got = pool.map(_task, tasks, chunksize=1)
    rows, hits_store = [], {}
    for j, (g_, c, a) in enumerate(keys):
        per = got[j * len(SEEDS):(j + 1) * len(SEEDS)]
        hits_store[(g_, c, a)] = (np.mean([p["hits"] for p in per], axis=0), per[0]["n"])
        f = lambda k: np.array([p[k] for p in per], float)
        rows.append(dict(gate=g_, condition=c, arm=a,
                         pd=f("pd").mean(), pd_se=f("pd").std(ddof=1) / np.sqrt(len(per)),
                         dr_bins=f("dr_bins").mean(), dv_bins=f("dv_bins").mean(),
                         measured_pfa=f("pfa").mean()))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "per_condition.csv"), index=False)
    print("    %.1f min" % ((time.perf_counter() - t0) / 60), flush=True)

    # ---------------- 4. bootstrap vs the flat3x3 control (sequential; keeps its seed)
    print("\nstage 4: paired bootstrap vs flat3x3", flush=True)
    rng = np.random.default_rng(0)
    rows = []
    for g_ in GATES:
        for cond in conds:
            base, n = hits_store[(g_, cond, "flat3x3")]
            k = rng.integers(0, len(n), size=(N_BOOT, len(n)))
            tot = n[k].sum(axis=1)
            for a in ARMS:
                if a == "flat3x3":
                    continue
                h, _ = hits_store[(g_, cond, a)]
                d = (h[k].sum(axis=1) - base[k].sum(axis=1)) / tot
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                rows.append(dict(gate=g_, condition=cond, arm=a,
                                 pd_flat=base.sum() / n.sum(), pd_arm=h.sum() / n.sum(),
                                 delta=(h.sum() - base.sum()) / n.sum(),
                                 ci_lo=float(lo), ci_hi=float(hi), p_boot=float(p)))
    B = pd.DataFrame(rows)
    B["alpha_bonf"] = 0.05 / len(conds)
    B["sig"] = B.p_boot < B.alpha_bonf
    B.to_csv(os.path.join(RES, "deltas.csv"), index=False)

    # ---------------- 5. CFAR reference, from the radius-MATCHED caches
    print("\nstage 5: CFAR localisation reference (radius-matched)", flush=True)
    t0 = time.perf_counter()
    keys = [(g_, k) for g_ in GATES for k in KINDS]
    tasks = [(os.path.join(R2PK, "bench2_val_r2_%s.npz" % k), ids_t, g_, False)
             for (g_, k) in keys]
    got = pool.map(_task, tasks, chunksize=1)
    rows = [dict(gate=g_, method=k, pd=r["pd"], dr_bins=r["dr_bins"], dv_bins=r["dv_bins"],
                 range_m=r["dr_bins"] * RANGE_RES, vel_mps=r["dv_bins"] * VEL_RES)
            for (g_, k), r in zip(keys, got)]
    pd.DataFrame(rows).to_csv(os.path.join(RES, "cfar_reference.csv"), index=False)
    print("    %.1f min" % ((time.perf_counter() - t0) / 60), flush=True)

    pool.close()
    pool.join()

    ec = pd.read_csv(os.path.join(RES, "epoch_curve.csv"))
    g1 = ec[(ec.gate == 1) & (ec.epoch == 21)].set_index("arm")
    g0 = ec[(ec.gate == 0) & (ec.epoch == 21)].set_index("arm")
    print("")
    for a in ("flat3x3", "gaussian", "point"):
        print("  %-9s  dv %.4f   Pd(+/-1) %.4f   Pd(exact) %.4f"
              % (a, g1.loc[a, "dv_bins"], g1.loc[a, "pd"], g0.loc[a, "pd"]))
    print("\ninference %.1f min, total %.1f min -> %s"
          % (t_infer / 60, (time.perf_counter() - t_start) / 60, RES))


if __name__ == "__main__":
    mp.freeze_support()
    main()
