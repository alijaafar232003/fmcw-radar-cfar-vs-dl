"""Inference: heatmap -> detection point list, and evaluation of a whole split.

FAIRNESS (see docs/dl_design.md §8). Three requirements, enforced here:

1. **Identical post-processing.** The peak extraction below imports `all_peaks` from
   `src/cfar/detect.py` — literally the same function CFAR uses, with the same `nms_size`.
   Calling the same code makes divergent post-processing structurally impossible.
2. **Identical evaluation region.** CFAR cannot score cells within one window half-width of the
   border (no full 21x21 window fits), so it is scored on the interior only and its Pfa
   denominator counts only those cells. The same `valid` mask is applied to the network, so both
   are scored on the same cells with the same denominator.
3. **Identical metrics.** Both go through `src/eval` (one-to-one matching in a +/-1-bin gate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfar.cfar import CfarConfig
from cfar.detect import all_peaks
from dl.dataset import RDDataset, normalize_rd
from eval import MatchConfig, aggregate, evaluate_frame, roc_curve

NMS_SIZE = 5  # must match the value used for CFAR


def cfar_valid_mask(shape: tuple[int, int], guard: int = 2, train: int = 8) -> np.ndarray:
    """The region CFAR can actually test — used for the network too, for a like-for-like Pfa."""
    hw = CfarConfig(guard=guard, train=train).half_width
    valid = np.zeros(shape, dtype=bool)
    valid[hw:-hw, hw:-hw] = True
    return valid


@torch.no_grad()
def predict_heatmap(model, mag: np.ndarray, device="cpu") -> np.ndarray:
    """RD magnitude -> per-cell detection probability."""
    model.eval()
    x = torch.from_numpy(normalize_rd(mag))[None, None].to(device)
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def predict_points(model, mag: np.ndarray, valid: np.ndarray | None = None,
                   nms_size: int = NMS_SIZE, device="cpu") -> np.ndarray:
    """RD magnitude -> (K, 3) array of (range_bin, velocity_bin, score).

    Uses CFAR's own `all_peaks`, so both detectors share one post-processing path.
    """
    prob = predict_heatmap(model, mag, device=device)

    if valid is None:
        valid = cfar_valid_mask(prob.shape)

    return all_peaks(prob, valid, nms_size=nms_size)


def collect_split(model, split: str, limit: int | None = None, device="cpu"):
    """Run the model over a split -> frames ready for src/eval.

    Returns (frames, snrs) where frames = [(points, truths, shape, valid), ...] and snrs holds
    the per-target cell SNR for Pd-vs-SNR analysis.
    """
    ds = RDDataset(split, limit=limit)
    frames, snrs = [], []

    for fd in ds.frame_dirs:
        mag = np.load(fd / "rd_map_mag.npy")
        lab = pd.read_csv(fd / "labels.csv")

        truths = lab[["range_bin", "velocity_bin"]].to_numpy() if len(lab) else np.zeros((0, 2))
        valid = cfar_valid_mask(mag.shape)
        points = predict_points(model, mag, valid=valid, device=device)

        frames.append((points, truths, mag.shape, valid))
        snrs.append(lab["cell_snr_db"].to_numpy() if len(lab) else np.zeros(0))

    return frames, snrs


def evaluate_at_threshold(frames, threshold: float, cfg: MatchConfig = MatchConfig(1, 1)) -> dict:
    """Score a collected split at one detection threshold."""
    evals = []

    for points, truths, shape, valid in frames:
        keep = points[points[:, 2] >= threshold][:, :2] if len(points) else points
        evals.append(evaluate_frame(keep, truths, shape, cfg, valid_mask=valid))

    return aggregate(evals)


def dl_roc(frames, thresholds=None, cfg: MatchConfig = MatchConfig(1, 1)) -> pd.DataFrame:
    """Sweep the detection threshold -> (Pfa, Pd) curve, analogous to sweeping CFAR's alpha."""
    if thresholds is None:
        thresholds = np.concatenate([
            np.linspace(0.01, 0.9, 30),
            1 - np.logspace(np.log10(0.1), np.log10(1e-4), 20),
        ])
        thresholds = np.unique(np.clip(thresholds, 1e-6, 1 - 1e-9))

    return roc_curve(frames, thresholds=thresholds, cfg=cfg)
