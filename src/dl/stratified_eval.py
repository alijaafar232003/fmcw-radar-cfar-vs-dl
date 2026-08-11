"""Per-condition ("stratified") evaluation of CFAR and DL on a stratified benchmark split.

Why stratified: a neutral aggregate answers "who is best overall" but not "who is best in each
regime". The benchmark datasets record a `condition` per frame (see docs/dataset_coverage_study.md),
so per-condition estimates have guaranteed sample counts instead of being inferred from label
geometry with a handful of examples.

Fairness: every method is reduced to a point list by CFAR's own `all_peaks` with the same
`nms_size`, restricted to the same border-excluded `valid` mask, and scored by the same `src/eval`
matching.

**Operating point: matched MEASURED Pfa, per condition.** Each method's score threshold is swept on
the condition being scored, and Pd is read off where the *achieved* per-cell false-alarm rate equals
the common target. This is the same protocol as `scripts/cfar_geometry_percondition.py`, which
produced the CFAR baseline these numbers sit beside — the two halves of the comparison have to be
the same experiment.

It replaces Monte-Carlo calibration on pure noise, which matched the *design* Pfa and was Milestone
16's open asymmetry #3. Step 5 measured why that is not good enough: clutter drives the achieved
rate 60-125x above design, by different factors per method, so "everyone at design 1e-4" is in fact
CA at 1.2e-2 against GO at 8.7e-3 against OS at 3.6e-2 — three different operating points. The
noise-calibrated thresholds are still computed and returned, but as a *diagnostic* (the
design-vs-measured gap is itself a documented finding), never as the operating point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfar.calibrate import calibrate_all, make_noise_power_frames
from cfar.cfar import CfarConfig, cfar_ratio_map
from cfar.detect import all_peaks
from dl.dataset import project_root
from dl.infer import cfar_valid_mask, predict_heatmap
from eval import MatchConfig, aggregate, evaluate_frame, roc_curve
from radar.config import RadarConfig
from radar.simulator import make_axes

NMS = 5
METHODS = ("CA", "OS", "GO", "DL")


def calibrate_dl_threshold(model, radar_cfg: RadarConfig, target_pfa: float,
                           n_frames: int = 30, seed: int = 4242) -> float:
    """Monte-Carlo threshold for the network, using the same protocol as CFAR.

    The dB-above-median input normalization is scale-invariant, so unit-variance noise frames are
    representative of any noise level.
    """
    pool = []
    for power in make_noise_power_frames(radar_cfg, n_frames, seed=seed):
        prob = predict_heatmap(model, np.sqrt(power))
        pool.append(prob[cfar_valid_mask(prob.shape)])

    return float(np.quantile(np.concatenate(pool), 1.0 - target_pfa))


def frame_condition(frame_dir: Path) -> str | None:
    return json.loads((frame_dir / "metadata.json").read_text()).get("condition")


def score_frame(method: str, mag: np.ndarray, model):
    """EVERY local maximum with its score, for one method on one frame.

    Unthresholded on purpose: the operating point is chosen later, by sweeping. Running the
    detector once and sweeping the score is also what makes the per-condition ROC affordable.
    """
    cfg = CfarConfig(kind=method if method != "DL" else "CA")

    if method == "DL":
        valid = cfar_valid_mask(mag.shape, cfg)
        return all_peaks(predict_heatmap(model, mag), valid, nms_size=NMS), valid

    # Geometry comes from CfarConfig's defaults, which ARE the 21x9 window selected in step 6.
    # Re-stating guard/train here would silently pin the range axis while leaving the Doppler
    # overrides to the defaults -- the kind of drift that makes a re-run disagree with its config.
    ratio, valid, _ = cfar_ratio_map(mag ** 2, cfg)
    return all_peaks(ratio, valid, nms_size=NMS), valid


def _at_measured_pfa(roc: pd.DataFrame, target_pfa: float) -> tuple[float, float]:
    """(Pd, score threshold) where the MEASURED per-cell Pfa equals `target_pfa`.

    Returns NaN if the target lies outside the swept range — that is a real answer (the method
    never reaches that false-alarm rate on this condition), not something to extrapolate through.
    """
    d = roc.dropna(subset=["pfa_per_cell"])

    if not len(d):
        return float("nan"), float("nan")

    # Collapse tied Pfa values before interpolating. On clean strata a whole span of thresholds
    # yields the SAME measured Pfa (often exactly 0 -- no false alarm survives any of them), and
    # np.interp on duplicated x picks an arbitrary partner, which recovered a threshold 4x too
    # high on `isolated`. Pd rises and Pfa rises as the threshold falls, so the representative of
    # a tie is its lowest threshold / highest Pd: the boundary just before Pfa moves.
    d = (d.groupby("pfa_per_cell", as_index=False)
           .agg(pd=("pd", "max"), threshold=("threshold", "min"), fp=("fp", "max"))
           .sort_values("pfa_per_cell"))

    # Interpolate against LOG Pfa. Pfa falls roughly exponentially with the threshold, so a linear
    # fit between two bracketing grid points overshoots and lands the operating point below target
    # (measured 3.5-8.7e-5 against a 1e-4 target).
    #
    # Zero-Pfa rows cannot be dropped, though they have no logarithm: on dense strata every
    # surviving peak sits inside a guard region, so Pfa steps from ~1.2e-4 to exactly 0 and the
    # zero row IS the upper bracket -- dropping it made all of `multi` unscoreable. They are placed
    # at the half-count floor, the usual convention for an unobserved rate: 0 false alarms over N
    # cells is "below 0.5/N", not "zero".
    pfa = d["pfa_per_cell"].to_numpy(dtype=float)
    fp = d["fp"].to_numpy(dtype=float)

    obs = (fp > 0) & (pfa > 0)
    if not obs.any():
        return float("nan"), float("nan")

    cells = float(np.round(np.median(fp[obs] / pfa[obs])))
    pfa = np.where(pfa > 0, pfa, 0.5 / max(cells, 1.0))

    x = np.log10(pfa)
    t = np.log10(target_pfa)

    if len(x) < 2 or t < x.min() or t > x.max():
        return float("nan"), float("nan")

    thr = float(np.interp(t, x, d["threshold"].to_numpy()))

    # `noise` has no targets, so Pd is undefined there -- but its threshold is still meaningful
    # and is what the false-alarm columns are reported at.
    y = d["pd"].to_numpy(dtype=float)
    pd_hat = float(np.interp(t, x, y)) if np.isfinite(y).all() else float("nan")

    return pd_hat, thr


def evaluate_stratified(model, split: str = "bench_test", target_pfa: float = 1e-4,
                        per_condition: int | None = None, radar_cfg: RadarConfig | None = None):
    """Per-condition metrics for every method at matched MEASURED Pfa.

    Returns (table, thresholds) where thresholds has two keys: "matched", the operating point
    actually used, keyed by (condition, method); and "design", the noise-calibrated thresholds,
    kept only so the design-vs-measured gap can be reported. `per_condition` caps frames per
    stratum (None = all).

    A NaN `pd` means the method never reached `target_pfa` on that condition within its swept score
    range — a real result (usually clutter holding the achieved rate above the target), not a bug.
    """
    radar_cfg = radar_cfg or RadarConfig()
    root = project_root()
    _, _, _, vax = make_axes(radar_cfg)
    v0 = int(np.argmin(np.abs(vax)))
    band = set(range(v0 - 8, v0 + 9))          # clutter ridge +/- 8 velocity bins

    # Diagnostic only -- the design-vs-measured gap is a reported finding, not the operating point.
    design_thresholds = dict(calibrate_all(radar_cfg, target_pfa, guard=2, train=8,
                                           n_frames=40, seed=1))
    if "DL" in METHODS and model is not None:
        design_thresholds["DL"] = calibrate_dl_threshold(model, radar_cfg, target_pfa)

    # group frames by their recorded condition
    frames_by_cond: dict[str, list[Path]] = {}
    for fd in sorted((root / "data" / split / "frames").iterdir()):
        frames_by_cond.setdefault(frame_condition(fd), []).append(fd)

    if per_condition:
        frames_by_cond = {k: v[:per_condition] for k, v in frames_by_cond.items()}

    mc = MatchConfig(1, 1)
    rows, matched_thresholds = [], {}

    for cond, fds in sorted(frames_by_cond.items()):
        # Load once; every method and every threshold is scored on the same frames.
        loaded = []
        for fd in fds:
            mag = np.load(fd / "rd_map_mag.npy")
            lab = pd.read_csv(fd / "labels.csv")
            tru = (lab[["range_bin", "velocity_bin"]].to_numpy()
                   if len(lab) else np.zeros((0, 2)))
            loaded.append((mag, tru))

        for method in METHODS:
            scored = [(*score_frame(method, mag, model), tru, mag.shape)
                      for mag, tru in loaded]
            frames = [(pk, tru, shape, valid) for pk, valid, tru, shape in scored]

            # Sweep this method's own score distribution on THIS condition.
            pool = np.concatenate([f[0][:, 2] for f in frames if len(f[0])]) \
                if any(len(f[0]) for f in frames) else np.zeros(1)

            # Dense in the UPPER TAIL, because that is where the operating point sits. A uniform
            # quantile grid puts ~2 points in the top 1%, and Pfa 1e-4 is inside it: the sweep then
            # steps straight over the target and the recovered threshold lands in the gap (measured
            # 4x too high on `isolated`). Pd tolerates that gap; a threshold does not, and the
            # false-alarm columns are evaluated AT the threshold.
            q = np.concatenate([
                np.linspace(0.0, 0.99, 40),
                1.0 - np.logspace(np.log10(0.01), np.log10(2e-5), 40),
            ])
            grid = np.quantile(pool, np.clip(q, 0.0, 1.0))

            roc = roc_curve(frames, np.unique(grid), mc)
            pd_matched, thr = _at_measured_pfa(roc, target_pfa)
            matched_thresholds[(cond, method)] = thr

            # Mechanism metrics are only meaningful AT the matched operating point.
            evs, band_fa, npair, n_weak = [], 0, 0, 0
            for pk, tru, shape, valid in frames:
                det = pk[pk[:, 2] >= thr][:, :2] if (len(pk) and thr == thr) else np.zeros((0, 2))
                ev = evaluate_frame(det, tru, shape, mc, valid_mask=valid)
                evs.append(ev)

                matched = set(ev.matched_truth_idx)
                if len(tru) >= 2:                     # pair scenarios: track the WEAK target
                    npair += 1
                    n_weak += (1 in matched)

                for d in det:                          # false alarms inside the clutter band
                    r, v = int(d[0]), int(d[1])
                    is_fa = (all(abs(r - t[0]) > 1 or abs(v - t[1]) > 1 for t in tru)
                             if len(tru) else True)
                    if is_fa and v in band:
                        band_fa += 1

            a = aggregate(evs)
            rows.append({
                "condition": cond,
                "method": method,
                "frames": len(fds),
                "pd": round(pd_matched, 3) if pd_matched == pd_matched else np.nan,
                "pd_weak": round(n_weak / npair, 3) if npair else np.nan,
                "fa_per_frame": round(a["fa_per_frame"], 2),
                "clutter_band_fa": round(band_fa / len(fds), 2),
                "measured_pfa": a["pfa_per_cell"],
                "threshold": thr,
            })

    return pd.DataFrame(rows), {"matched": matched_thresholds, "design": design_thresholds}
