"""PyTorch dataset for RD-map target detection.

Input normalization (the highest-leverage decision — see docs/dl_design.md §4):

    x = 10*log10( power / median(power) )   ->  clip [-10, +60] dB  ->  / 20

i.e. **dB above the frame's own noise floor**. Rationale:

* raw power spans ~7 orders of magnitude (unusable: exploding gradients);
* the absolute scale varies 57x between frames (noise_var 554..31830), because scene SNR is
  randomised per frame — so a fixed global reference would force the network to waste capacity
  renormalising;
* the stored `rd_map_db` is normalised by each frame's own MAX, which destroys absolute SNR (a
  weak-target frame and a strong-target frame become identical);
* dividing by the median gives a measured std of 0.000 across frames (perfectly stable
  reference) while preserving "how far above noise is this cell" — the actual detection cue;
* it mirrors CFAR, which also compares each cell to an estimated noise level, so it grants no
  information CFAR lacks.

The median is robust here: targets occupy ~0.064% of cells and the clutter ridge ~3%.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CLIP_LO_DB = -10.0
CLIP_HI_DB = 60.0
SCALE_DB = 20.0


def project_root(start: Path | None = None) -> Path:
    """Walk up to the project root.

    Requires BOTH data/ and configs/ to be present: `src/data/` is a Python package in this
    repo, so testing for data/ alone would wrongly stop at src/.
    """
    p = (start or Path(__file__).resolve()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data").is_dir() and (cand / "configs").is_dir():
            return cand
    raise FileNotFoundError("could not locate the project root (need both data/ and configs/)")


def normalize_rd(mag: np.ndarray) -> np.ndarray:
    """RD magnitude -> network input: dB above the frame's noise floor, clipped and scaled."""
    power = np.asarray(mag, dtype=np.float64) ** 2
    median = float(np.median(power))

    x = 10.0 * np.log10(power / (median + 1e-30) + 1e-30)
    x = np.clip(x, CLIP_LO_DB, CLIP_HI_DB) / SCALE_DB

    return x.astype(np.float32)


class RDDataset(Dataset):
    """Frames from one dataset split.

    Returns (x, y) with x = normalized RD map [1, R, V] and y = label mask [1, R, V].
    `frame_dirs` is kept public so evaluation code can read labels.csv for the same frames.
    """

    def __init__(self, split: str, root: Path | None = None, limit: int | None = None):
        self.root = Path(root) if root is not None else project_root()
        self.split_dir = self.root / "data" / split

        if not (self.split_dir / "frames").exists():
            raise FileNotFoundError(f"no frames/ in {self.split_dir}")

        self.frame_dirs = sorted(p for p in (self.split_dir / "frames").iterdir() if p.is_dir())

        if limit is not None:
            self.frame_dirs = self.frame_dirs[:limit]

    def __len__(self) -> int:
        return len(self.frame_dirs)

    def __getitem__(self, i: int):
        fd = self.frame_dirs[i]

        x = normalize_rd(np.load(fd / "rd_map_mag.npy"))
        y = np.load(fd / "label_mask.npy").astype(np.float32)

        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]
