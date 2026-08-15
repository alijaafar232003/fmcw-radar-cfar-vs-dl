"""Do train and val share the same TARGET, not just the same frame?

`verify_splits.py` proved no frame-level leakage by content hash. That does not exclude a subtler
failure: the same target parameters (range, velocity, amplitude) drawn in both splits from
different master seeds. With continuous draws exact repeats should be impossible, so this is a
falsification check -- if it finds matches, the generator is not drawing independently.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["range_m", "velocity_mps", "amplitude_db"]


def load(split):
    rows = []
    for fd in sorted((ROOT / "data" / split / "frames").iterdir()):
        p = fd / "labels.csv"
        if p.stat().st_size > 5:
            rows.append(pd.read_csv(p)[COLS])
    return pd.concat(rows, ignore_index=True)


tr, va = load("bench2_train"), load("bench2_val")
print(f"train targets {len(tr):,} | val targets {len(va):,}")

# Exact duplicates at full float precision.
merged = tr.merge(va, on=COLS, how="inner")
print(f"exact parameter matches: {len(merged)}")

# Nearest neighbour in normalised parameter space -- how close does the closest pair get?
from scipy.spatial import cKDTree
scale = tr[COLS].std().to_numpy()
d, _ = cKDTree(tr[COLS].to_numpy() / scale).query(va[COLS].to_numpy() / scale, k=1)
print(f"nearest-neighbour distance (normalised): min {d.min():.5f}  "
      f"p1 {np.percentile(d,1):.4f}  median {np.median(d):.4f}")
print(f"val targets with a train neighbour closer than 0.01: {(d < 0.01).sum()} "
      f"({100*(d<0.01).mean():.2f}%)")
print("\nFor scale, a 0.01 normalised distance is about "
      f"{0.01*scale[0]:.3f} m in range and {0.01*scale[1]:.3f} m/s in velocity -- "
      "well inside one bin (0.195 m, 0.420 m/s), i.e. visually identical targets.")
