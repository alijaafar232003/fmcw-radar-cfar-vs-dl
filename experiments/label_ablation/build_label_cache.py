"""Build the three arms' label caches and derive each arm's pos_weight. Writes only here.

Reads the project's frame labels read-only. Reuses `data/<split>/nn_cache_x.npy` (the normalised
INPUT) read-only -- it is identical across arms, so there is no reason to duplicate 550 MB.

    python experiments/label_ablation/build_label_cache.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from dl.dataset import RDDataset, cache_paths                       # noqa: E402
from labels import (ARMS, REFERENCE_ARM, REFERENCE_POS_WEIGHT,      # noqa: E402
                    build_label, pos_weight_for, psf_weights, sigma_from_psf)

SPLITS = ("bench2_train", "bench2_val")
SHAPE = (256, 64)
CACHE = HERE / "cache"
RES = HERE / "results"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    sr, sd = sigma_from_psf()
    W = psf_weights(sr, sd)
    print(f"sigma from measured PSF (off-bin worst case): range {sr:.4f}, Doppler {sd:.4f} bins")
    print("gaussian 3x3 weights:")
    print(pd.DataFrame(W, index=[-1, 0, 1], columns=[-1, 0, 1]).round(4).to_string())

    masses = {}
    for split in SPLITS:
        xp, _ = cache_paths(split)
        assert xp.exists(), f"input cache missing for {split}; run the project's build_cache first"
        ds = RDDataset(split)
        n = len(ds)
        print(f"\n{split}: {n} frames  (reusing {xp.name} read-only)")

        arrays = {a: np.lib.format.open_memmap(
            CACHE / f"y_{a}_{split}.npy", mode="w+", dtype=np.float32, shape=(n, *SHAPE))
            for a in ARMS}

        for i, fd in enumerate(ds.frame_dirs):
            lab = pd.read_csv(fd / "labels.csv")
            for a in ARMS:
                arrays[a][i] = build_label(a, lab, SHAPE, weights=W)
            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{n}")

        for a in ARMS:
            arrays[a].flush()
            if split == "bench2_train":
                masses[a] = float(np.asarray(arrays[a]).sum())

        # The flat arm must be bit-identical to the project's own supervision.
        if split == "bench2_train":
            _, yp = cache_paths(split)
            proj = np.load(yp, mmap_mode="r")
            ours = np.load(CACHE / f"y_flat3x3_{split}.npy", mmap_mode="r")
            k = np.linspace(0, len(ds) - 1, 40).astype(int)
            same = all(np.array_equal(proj[j].astype(np.float32), ours[j]) for j in k)
            print(f"  flat3x3 identical to the project's nn_cache_y on {len(k)} sampled frames: "
                  f"{same}")
            assert same, "the flat3x3 control does not reproduce the frozen run's labels"

    ref = masses[REFERENCE_ARM]
    rows = []
    for a in ARMS:
        pw = pos_weight_for(masses[a], ref)
        n_pos_equiv = masses[a]
        rows.append(dict(arm=a, positive_mass=masses[a],
                         mass_ratio_to_reference=masses[a] / ref,
                         pos_weight=round(pw, 2),
                         mass_x_pos_weight=masses[a] * pw))
        print(f"{a:<9} positive mass {masses[a]:>12,.1f}  "
              f"({masses[a] / ref:.4f}x reference)  ->  pos_weight {pw:8.2f}")

    PW = pd.DataFrame(rows)
    PW.to_csv(RES / "pos_weight_derivation.csv", index=False)
    (RES / "sigma.json").write_text(json.dumps(
        dict(sigma_range_bins=sr, sigma_doppler_bins=sd,
             source="results/psf_profile.csv, off-bin (worst case), -3 dB width / 2.3548",
             weights=W.tolist()), indent=2), encoding="utf-8")

    prod = PW.mass_x_pos_weight
    print(f"\nmass x pos_weight across arms: {prod.min():.4g} to {prod.max():.4g} "
          f"(spread {100 * (prod.max() / prod.min() - 1):.3f}%) -- constant by construction")
    print(f"\nwrote {CACHE} and {RES}")


if __name__ == "__main__":
    main()
