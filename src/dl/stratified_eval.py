"""Per-condition ("stratified") evaluation of CFAR and DL on a stratified benchmark split.

Why stratified: a neutral aggregate answers "who is best overall" but not "who is best in each
regime". The benchmark datasets record a `condition` per frame (see docs/dataset_coverage_study.md),
so per-condition estimates have guaranteed sample counts instead of being inferred from label
geometry with a handful of examples.

Fairness: every method is reduced to a point list by CFAR's own `all_peaks` with the same
`nms_size`, restricted to the same border-excluded `valid` mask, and scored by the same `src/eval`
matching. Thresholds are Monte-Carlo calibrated per method to a COMMON target Pfa on pure noise, so
the methods are compared at a matched false-alarm rate rather than at arbitrary operating points.
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
from eval import MatchConfig, aggregate, evaluate_frame
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


def detect(method: str, mag: np.ndarray, model, alpha: float):
    """Point list for one method on one frame (identical post-processing for all)."""
    power = mag ** 2

    if method == "DL":
        valid = cfar_valid_mask(power.shape)
        peaks = all_peaks(predict_heatmap(model, mag), valid, nms_size=NMS)
    else:
        cfg = CfarConfig(kind=method, guard=2, train=8)
        ratio, valid, _ = cfar_ratio_map(power, cfg)
        peaks = all_peaks(ratio, valid, nms_size=NMS)

    kept = peaks[peaks[:, 2] >= alpha][:, :2] if len(peaks) else peaks
    return kept, valid


def evaluate_stratified(model, split: str = "bench_test", target_pfa: float = 1e-4,
                        per_condition: int | None = None, radar_cfg: RadarConfig | None = None):
    """Per-condition metrics for every method at a matched Pfa.

    Returns (table, thresholds). `per_condition` caps frames per stratum (None = all).
    """
    radar_cfg = radar_cfg or RadarConfig()
    root = project_root()
    _, _, _, vax = make_axes(radar_cfg)
    v0 = int(np.argmin(np.abs(vax)))
    band = set(range(v0 - 8, v0 + 9))          # clutter ridge +/- 8 velocity bins

    thresholds = dict(calibrate_all(radar_cfg, target_pfa, guard=2, train=8, n_frames=40, seed=1))
    thresholds["DL"] = calibrate_dl_threshold(model, radar_cfg, target_pfa)

    # group frames by their recorded condition
    frames_by_cond: dict[str, list[Path]] = {}
    for fd in sorted((root / "data" / split / "frames").iterdir()):
        frames_by_cond.setdefault(frame_condition(fd), []).append(fd)

    if per_condition:
        frames_by_cond = {k: v[:per_condition] for k, v in frames_by_cond.items()}

    mc = MatchConfig(1, 1)
    rows = []

    for cond, fds in sorted(frames_by_cond.items()):
        for method in METHODS:
            evs, band_fa, npair, n_weak = [], 0, 0, 0

            for fd in fds:
                mag = np.load(fd / "rd_map_mag.npy")
                lab = pd.read_csv(fd / "labels.csv")
                tru = (lab[["range_bin", "velocity_bin"]].to_numpy()
                       if len(lab) else np.zeros((0, 2)))

                det, valid = detect(method, mag, model, thresholds[method])
                ev = evaluate_frame(det, tru, mag.shape, mc, valid_mask=valid)
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
                "pd": round(a["pd"], 3) if a["pd"] == a["pd"] else np.nan,
                "pd_weak": round(n_weak / npair, 3) if npair else np.nan,
                "fa_per_frame": round(a["fa_per_frame"], 2),
                "clutter_band_fa": round(band_fa / len(fds), 2),
                "measured_pfa": a["pfa_per_cell"],
            })

    return pd.DataFrame(rows), thresholds
