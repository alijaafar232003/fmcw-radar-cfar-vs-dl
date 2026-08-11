"""Detection scoring — the shared rulebook for CFAR and deep learning.

Both detectors are reduced to the SAME output type — a list of detected points in
(range_bin, velocity_bin) — and scored here with identical rules. That is what makes
the CFAR-vs-DL comparison fair; neither method gets its own scoring logic.

Rules implemented (see docs/benchmarking_conventions.md, Part 2):

* **Matching**: a detection matches a truth if it lies inside the tolerance gate
  (|dr| <= tol_r AND |dv| <= tol_v). Matching is **one-to-one and optimal**
  (Hungarian assignment), so one detection cannot claim two targets.
* **Counting**: matched -> TP; truth with no match -> FN; detection matched to
  nothing -> FP.
* **Guard region**: cells inside the gate of any truth are excluded from the
  false-alarm denominator, and an unmatched detection sitting there is *ignored*
  rather than counted as an FP (it is a near-miss of a real target, not a
  spurious detection).
* **Metrics**: Pd, per-cell Pfa, false alarms per frame, localization error.
* **ROC**: sweep a detection score threshold and trace (Pfa, Pd).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class MatchConfig:
    """Detection-to-truth matching tolerance, in bins.

    Defaults to +/-1 bin, i.e. one resolution cell in each dimension.
    """

    tol_r: int = 1
    tol_v: int = 1


# ----------------------------------------------------------------------
# Per-frame result
# ----------------------------------------------------------------------

@dataclass
class FrameEval:
    tp: int
    fp: int
    fn: int
    cells_tested: int
    dr: list[int] = field(default_factory=list)              # TP localization error
    dv: list[int] = field(default_factory=list)
    matched_truth_idx: list[int] = field(default_factory=list)


def _as_points(a) -> np.ndarray:
    """Normalise input to an (N, 2) float array of (range_bin, velocity_bin)."""
    arr = np.asarray(a, dtype=float)

    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)

    if arr.ndim == 1:
        arr = arr[None, :]

    return arr[:, :2]


def _guard_mask(truths: np.ndarray, shape: tuple[int, int], cfg: MatchConfig) -> np.ndarray:
    """Boolean map of cells within the tolerance gate of any true target."""
    mask = np.zeros(shape, dtype=bool)

    for r, v in truths.astype(int):
        r0, r1 = max(0, r - cfg.tol_r), min(shape[0], r + cfg.tol_r + 1)
        v0, v1 = max(0, v - cfg.tol_v), min(shape[1], v + cfg.tol_v + 1)
        mask[r0:r1, v0:v1] = True

    return mask


def evaluate_frame(
    detections,
    truths,
    shape: tuple[int, int],
    cfg: MatchConfig = MatchConfig(),
    valid_mask: np.ndarray | None = None,
) -> FrameEval:
    """Score one frame's detections against its ground truth.

    detections : (N, 2+) array of (range_bin, velocity_bin, [score])
    truths     : (M, 2)  array of (range_bin, velocity_bin)
    shape      : RD map shape, needed for the false-alarm denominator
    valid_mask : optional bool map of cells the detector actually tested. When given
                 (e.g. CFAR excludes a border), the false-alarm denominator counts only
                 valid, non-guard cells — otherwise Pfa would be understated.
    """
    det = _as_points(detections)
    tru = _as_points(truths)

    guard = _guard_mask(tru, shape, cfg) if len(tru) else np.zeros(shape, dtype=bool)

    if valid_mask is None:
        cells_tested = int(shape[0] * shape[1]) - int(guard.sum())
    else:
        cells_tested = int((valid_mask & ~guard).sum())

    def _is_in_guard(point) -> bool:
        r, v = int(point[0]), int(point[1])
        if 0 <= r < shape[0] and 0 <= v < shape[1]:
            return bool(guard[r, v])
        return False

    # No targets, or no detections: nothing to match.
    if len(tru) == 0 or len(det) == 0:
        fp = sum(1 for p in det if not _is_in_guard(p))
        return FrameEval(tp=0, fp=int(fp), fn=int(len(tru)), cells_tested=cells_tested)

    # Optimal one-to-one assignment. Pairs outside the gate are made unaffordable.
    dr = det[:, 0][:, None] - tru[:, 0][None, :]
    dv = det[:, 1][:, None] - tru[:, 1][None, :]

    eligible = (np.abs(dr) <= cfg.tol_r) & (np.abs(dv) <= cfg.tol_v)
    unaffordable = 1e6
    cost = np.where(eligible, np.sqrt(dr ** 2 + dv ** 2), unaffordable)

    det_idx, tru_idx = linear_sum_assignment(cost)
    keep = cost[det_idx, tru_idx] < unaffordable
    det_idx, tru_idx = det_idx[keep], tru_idx[keep]

    tp = int(len(det_idx))
    fn = int(len(tru) - tp)

    # Unmatched detections inside the guard region are near-misses of a real target,
    # not false alarms -> ignore them.
    matched = set(det_idx.tolist())
    fp = sum(
        1 for i in range(len(det))
        if i not in matched and not _is_in_guard(det[i])
    )

    return FrameEval(
        tp=tp,
        fp=int(fp),
        fn=fn,
        cells_tested=cells_tested,
        dr=[int(det[a, 0] - tru[b, 0]) for a, b in zip(det_idx, tru_idx)],
        dv=[int(det[a, 1] - tru[b, 1]) for a, b in zip(det_idx, tru_idx)],
        matched_truth_idx=tru_idx.tolist(),
    )


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate(frame_evals: list[FrameEval], range_res_m=None, vel_res_mps=None) -> dict:
    """Combine per-frame results into the headline metrics."""
    tp = sum(f.tp for f in frame_evals)
    fp = sum(f.fp for f in frame_evals)
    fn = sum(f.fn for f in frame_evals)
    cells = sum(f.cells_tested for f in frame_evals)
    frames = len(frame_evals)

    dr = np.array([x for f in frame_evals for x in f.dr], dtype=float)
    dv = np.array([x for f in frame_evals for x in f.dv], dtype=float)

    out = {
        "frames": frames,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pd": tp / (tp + fn) if (tp + fn) else np.nan,
        "pfa_per_cell": fp / cells if cells else np.nan,
        "fa_per_frame": fp / frames if frames else np.nan,
        "cells_tested": cells,
        "mean_abs_dr_bins": float(np.abs(dr).mean()) if dr.size else np.nan,
        "mean_abs_dv_bins": float(np.abs(dv).mean()) if dv.size else np.nan,
        "rmse_bins": float(np.sqrt((dr ** 2 + dv ** 2).mean())) if dr.size else np.nan,
    }

    if range_res_m is not None and dr.size:
        out["mean_abs_range_err_m"] = float(np.abs(dr).mean() * range_res_m)
    if vel_res_mps is not None and dv.size:
        out["mean_abs_vel_err_mps"] = float(np.abs(dv).mean() * vel_res_mps)

    return out


# ----------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------

def roc_thresholds(frames, cfg: MatchConfig = MatchConfig(), n_fa: int = 60,
                   n_quantile: int = 40) -> np.ndarray:
    """Threshold grid that resolves the FALSE-ALARM axis, where the operating point lives.

    A grid built from quantiles of the whole score pool samples the wrong place. On dense strata
    the top few percent of peaks are all genuine targets — above some threshold there are no false
    alarms at all — so a tail-dense quantile grid spends most of its points in a region where Pfa
    is exactly 0, and leaves the interesting span unsampled. On `multi` that produced a ROC whose
    smallest non-zero false-alarm count was 371: nothing between 1 and 370 was ever measured, and
    the curve appeared to start abruptly at Pfa 7e-5.

    So the grid is anchored to the false alarms themselves. Scores of peaks that are NOT within the
    gate of any truth are pooled and sorted, and thresholds are taken at geometrically spaced
    ranks — giving roughly equal resolution per decade of Pfa, from a single false alarm upward.
    A coarse quantile grid is unioned in to keep the high-Pfa end covered.

    Using ground truth to CHOOSE where to sample is not using it to score: every point on the
    resulting curve is still measured by `evaluate_frame` in the ordinary way.
    """
    fa_scores, all_scores = [], []

    for frame in frames:
        det = np.asarray(frame[0], dtype=float)
        truths = _as_points(frame[1])

        if not det.size or det.ndim != 2 or det.shape[1] < 3:
            continue

        all_scores.append(det[:, 2])

        if len(truths):
            dr = np.abs(det[:, 0][:, None] - truths[:, 0][None, :])
            dv = np.abs(det[:, 1][:, None] - truths[:, 1][None, :])
            near = ((dr <= cfg.tol_r) & (dv <= cfg.tol_v)).any(axis=1)
        else:
            near = np.zeros(len(det), dtype=bool)

        fa_scores.append(det[~near, 2])

    if not all_scores:
        return np.array([0.0])

    pool = np.concatenate(all_scores)
    grid = list(np.quantile(pool, np.linspace(0.0, 0.99, n_quantile)))

    fa = np.concatenate(fa_scores) if fa_scores else np.array([])
    if fa.size:
        fa = np.sort(fa)[::-1]                     # strongest false alarm first
        ranks = np.unique(np.round(
            np.geomspace(1, fa.size, min(n_fa, fa.size))).astype(int)) - 1
        # Step just below each chosen false alarm so that it is actually included.
        grid.extend(fa[ranks] * (1 - 1e-9))

    return np.unique(np.asarray(grid, dtype=float))


def roc_curve(frames, thresholds, cfg: MatchConfig = MatchConfig()) -> pd.DataFrame:
    """Sweep a detection-score threshold and trace (Pfa, Pd).

    frames : list of (detections, truths, shape), where detections is (N, 3):
             (range_bin, velocity_bin, score). Detections with score >= threshold
             are kept at each step.
    """
    rows = []

    for thr in thresholds:
        evals = []

        for frame in frames:
            detections, truths, shape = frame[0], frame[1], frame[2]
            valid_mask = frame[3] if len(frame) > 3 else None

            det = np.asarray(detections, dtype=float)

            if det.size and det.ndim == 2 and det.shape[1] >= 3:
                det = det[det[:, 2] >= thr][:, :2]

            evals.append(evaluate_frame(det, truths, shape, cfg, valid_mask=valid_mask))

        row = aggregate(evals)
        row["threshold"] = float(thr)
        rows.append(row)

    cols = ["threshold", "pd", "pfa_per_cell", "fa_per_frame", "tp", "fp", "fn"]
    return pd.DataFrame(rows)[cols]


def pd_vs_snr(frames, snr_edges, *, threshold: float,
              cfg: MatchConfig = MatchConfig()) -> pd.DataFrame:
    """Pd binned by each target's own cell SNR, at ONE fixed operating point.

    frames : list of (detections, truths, shape, truth_snr_db[, valid_mask]), where
             detections is (N, 3) — (range_bin, velocity_bin, score) — and truth_snr_db
             holds one cell-SNR value per truth row.
    threshold : score threshold defining the operating point. **Required, and keyword-only.**

    Why the threshold is not optional: every producer in this project (`all_peaks`, and both
    detectors through it) returns EVERY local maximum, unthresholded, so the caller can sweep.
    Scoring that list directly would count a target as detected whenever any local maximum lands
    within the gate — i.e. Pd measured at "detect everything", which rises towards 1 regardless of
    detector quality. A Pd-vs-SNR curve is only meaningful at a fixed operating point (see
    docs/benchmarking_conventions.md, "Aggregation for curves"), normally the threshold that puts
    the detector at the matched measured Pfa.
    """
    snr_all: list[float] = []
    hit_all: list[bool] = []

    for frame in frames:
        detections, truths, shape, truth_snr = frame[0], frame[1], frame[2], frame[3]
        valid_mask = frame[4] if len(frame) > 4 else None

        det = np.asarray(detections, dtype=float)

        if det.size:
            if det.ndim != 2 or det.shape[1] < 3:
                raise ValueError(
                    "pd_vs_snr needs scored detections (N, 3) so it can apply `threshold`; "
                    f"got shape {det.shape}. Pass the unthresholded output of all_peaks()."
                )
            det = det[det[:, 2] >= threshold][:, :2]

        ev = evaluate_frame(det, truths, shape, cfg, valid_mask=valid_mask)
        matched = set(ev.matched_truth_idx)

        for i, snr in enumerate(np.asarray(truth_snr, dtype=float)):
            snr_all.append(float(snr))
            hit_all.append(i in matched)

    df = pd.DataFrame({"cell_snr_db": snr_all, "detected": hit_all})
    df["snr_bin"] = pd.cut(df["cell_snr_db"], bins=snr_edges)

    out = (
        df.groupby("snr_bin", observed=True)
        .agg(n=("detected", "size"),
             mean_snr_db=("cell_snr_db", "mean"),
             pd=("detected", "mean"))
        .reset_index()
    )

    return out
