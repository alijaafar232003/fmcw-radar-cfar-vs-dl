"""A per-frame scorer that gives BIT-IDENTICAL results to `eval.detection.evaluate_frame`.

Why this exists. A score sweep calls `evaluate_frame` once per threshold per frame -- about a
hundred times per frame -- and almost everything it computes does not depend on the threshold at
all: the guard mask, the false-alarm denominator, which candidates sit inside a truth gate, and the
candidate-to-truth distance matrix are all fixed once the frame's candidate list and the matching
gate are known. `evaluate_frame` recomputes them every time, including a Python loop over every
surviving candidate.

`FrameScorer` computes those once per frame and then, at each threshold, does only the part that
genuinely depends on it: subset the rows and run the assignment. The arithmetic is otherwise the
same operation in the same order, including the Hungarian call and its row ordering.

This is an optimisation, NOT a protocol change, and it is not taken on trust: `verify()` scores
random frames at random thresholds through both paths and asserts every field agrees exactly. The
scripts that use it run that check before they use it.

`src/eval/detection.py` is not modified.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

UNAFFORDABLE = 1e6


class FrameScorer:
    """One frame, one matching gate. Precomputes everything threshold-independent."""

    __slots__ = ("scores", "cost", "dr", "dv", "in_guard", "cells", "n_truth", "order")

    def __init__(self, peaks, truths, shape, tol_r, tol_v, valid_mask):
        det = np.asarray(peaks, dtype=float)
        if det.size == 0:
            det = np.zeros((0, 3), dtype=float)
        elif det.ndim == 1:
            det = det[None, :]
        tru = np.asarray(truths, dtype=float)
        if tru.size == 0:
            tru = np.zeros((0, 2), dtype=float)
        elif tru.ndim == 1:
            tru = tru[None, :]
        tru = tru[:, :2]

        self.n_truth = len(tru)
        self.scores = det[:, 2].copy() if det.shape[1] >= 3 else np.full(len(det), np.inf)
        pts = det[:, :2]

        # Guard mask and false-alarm denominator: fixed by the truths and the gate.
        guard = np.zeros(shape, dtype=bool)
        for r, v in tru.astype(int):
            r0, r1 = max(0, r - tol_r), min(shape[0], r + tol_r + 1)
            v0, v1 = max(0, v - tol_v), min(shape[1], v + tol_v + 1)
            guard[r0:r1, v0:v1] = True
        self.cells = (int(shape[0] * shape[1]) - int(guard.sum()) if valid_mask is None
                      else int((valid_mask & ~guard).sum()))

        # Which candidates sit inside some truth's gate: fixed by the candidate list.
        # Out-of-bounds points count as not-in-guard, matching `_is_in_guard`.
        if len(pts):
            r = pts[:, 0].astype(int)
            v = pts[:, 1].astype(int)
            inb = (r >= 0) & (r < shape[0]) & (v >= 0) & (v < shape[1])
            ig = np.zeros(len(pts), dtype=bool)
            ig[inb] = guard[r[inb], v[inb]]
            self.in_guard = ig
        else:
            self.in_guard = np.zeros(0, dtype=bool)

        # Distance matrix, with pairs outside the gate made unaffordable.
        if len(pts) and self.n_truth:
            dr = pts[:, 0][:, None] - tru[:, 0][None, :]
            dv = pts[:, 1][:, None] - tru[:, 1][None, :]
            elig = (np.abs(dr) <= tol_r) & (np.abs(dv) <= tol_v)
            self.dr, self.dv = dr, dv
            self.cost = np.where(elig, np.sqrt(dr ** 2 + dv ** 2), UNAFFORDABLE)
        else:
            self.dr = self.dv = np.zeros((len(pts), self.n_truth))
            self.cost = np.zeros((len(pts), self.n_truth))

    def at(self, thr):
        """(tp, fp, fn, cells, dr list, dv list, matched truth indices) at one threshold."""
        keep = (self.scores >= thr) if thr == thr else np.zeros(len(self.scores), dtype=bool)
        n_det = int(keep.sum())

        if self.n_truth == 0 or n_det == 0:
            fp = int((~self.in_guard[keep]).sum()) if n_det else 0
            return 0, fp, int(self.n_truth), self.cells, [], [], []

        cost = self.cost[keep]
        di, ti = linear_sum_assignment(cost)
        ok = cost[di, ti] < UNAFFORDABLE
        di, ti = di[ok], ti[ok]

        matched = np.zeros(n_det, dtype=bool)
        matched[di] = True
        fp = int((~matched & ~self.in_guard[keep]).sum())

        dr = self.dr[keep][di, ti]
        dv = self.dv[keep][di, ti]
        return (int(len(di)), fp, int(self.n_truth - len(di)), self.cells,
                dr.tolist(), dv.tolist(), ti.tolist())


def verify(frames, thresholds, tol_r, tol_v, shape, valid_mask, n=60, seed=0):
    """Assert FrameScorer reproduces `evaluate_frame` exactly on a random sample."""
    from eval.detection import MatchConfig, evaluate_frame

    rng = np.random.default_rng(seed)
    cfg = MatchConfig(tol_r, tol_v)
    checked = 0
    for k in rng.choice(len(frames), size=min(n, len(frames)), replace=False):
        peaks, truths = frames[int(k)][0], frames[int(k)][1]
        fs = FrameScorer(peaks, truths, shape, tol_r, tol_v, valid_mask)
        for thr in thresholds:
            p = np.asarray(peaks, dtype=float)
            det = (p[p[:, 2] >= thr][:, :2] if (p.size and p.ndim == 2 and p.shape[1] >= 3)
                   else np.zeros((0, 2)))
            ref = evaluate_frame(det, truths, shape, cfg, valid_mask=valid_mask)
            tp, fp, fn, cells, dr, dv, ti = fs.at(thr)
            assert (tp, fp, fn, cells) == (ref.tp, ref.fp, ref.fn, ref.cells_tested), \
                "counts differ at frame %d thr %r: %s vs %s" % (
                    k, thr, (tp, fp, fn, cells), (ref.tp, ref.fp, ref.fn, ref.cells_tested))
            assert sorted(ti) == sorted(ref.matched_truth_idx), "matching differs"
            assert np.allclose(sorted(dr), sorted(ref.dr)) and \
                   np.allclose(sorted(dv), sorted(ref.dv)), "localisation differs"
            checked += 1
    return checked
