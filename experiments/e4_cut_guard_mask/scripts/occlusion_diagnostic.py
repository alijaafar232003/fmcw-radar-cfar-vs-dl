"""Descriptive occlusion sensitivity of the FROZEN TinyCNN. No training. Reads only.

⚠️ DESCRIPTIVE DIAGNOSTIC, NOT THE CAUSAL EXPERIMENT. Occluding at inference time creates inputs
the network never saw in training, so a large sensitivity does not prove the network would fail
without that information -- only that its current function depends on it. The causal question needs
a retrained masked arm.

For each sampled CUT: take its 21x9 receptive field, set one cell at a time to the neutral value
(0.0 = the frame's own median power after `normalize_rd`), and record the change in the centre
logit. Averaged over samples, this gives a 21x9 importance map.

Also checks the §9 assumption that 0.0 is the neutral / noise-floor value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig                     # noqa: E402
from dl.dataset import normalize_rd                  # noqa: E402
from dl.models import TinyCNN                        # noqa: E402

CFG = CfarConfig()
G_R, G_D = CFG.guard_r, CFG.guard_d
HW_R, HW_D = CFG.half_width_r, CFG.half_width_d
OUT = HERE / "results"
N_FRAMES = 120
SEEDS = range(8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- the neutral-value assumption
    fd0 = sorted((ROOT / "data" / "bench2_val" / "frames").iterdir())[0]
    mag = np.load(fd0 / "rd_map_mag.npy")
    power = mag.astype(np.float64) ** 2
    x = normalize_rd(mag)
    at_median = float(10 * np.log10(np.median(power) / np.median(power)) / 20.0)
    print(f"normalize_rd at the frame median: {at_median:.6f}  (0.0 by construction)")
    print(f"observed input percentiles: p50 {np.percentile(x, 50):+.4f}  "
          f"p25 {np.percentile(x, 25):+.4f}  p75 {np.percentile(x, 75):+.4f}")
    print("-> 0.0 is the frame's own noise floor, so it is the neutral fill. Confirmed.\n")

    frames = sorted((ROOT / "data" / "bench2_val" / "frames").iterdir())[:N_FRAMES]

    heat_t = np.zeros((2 * HW_R + 1, 2 * HW_D + 1))
    heat_b = np.zeros_like(heat_t)
    n_t = n_b = 0

    for seed in SEEDS:
        ck = torch.load(ROOT / "results" / "dl" / f"tinycnn_frozen_s{seed}_ep021.pt",
                        weights_only=False)
        model = TinyCNN(width=ck["config"].get("width", 16))
        model.load_state_dict(ck["state_dict"])
        model.eval()

        rng = np.random.default_rng(seed)
        for fd in frames:
            xi = torch.from_numpy(normalize_rd(np.load(fd / "rd_map_mag.npy")))
            lab = pd.read_csv(fd / "labels.csv")
            cuts = []
            for r, v in lab[["range_bin", "velocity_bin"]].to_numpy():
                r, v = int(r), int(v)
                if HW_R <= r < 256 - HW_R and HW_D <= v < 64 - HW_D:
                    cuts.append((r, v, True))
            for _ in range(len(cuts) or 1):
                r = int(rng.integers(HW_R, 256 - HW_R))
                v = int(rng.integers(HW_D, 64 - HW_D))
                cuts.append((r, v, False))

            for r, v, is_target in cuts:
                patch = xi[r - HW_R:r + HW_R + 1, v - HW_D:v + HW_D + 1].clone()
                variants = patch.repeat(patch.numel() + 1, 1, 1)
                k = 1
                for i in range(patch.shape[0]):
                    for j in range(patch.shape[1]):
                        variants[k, i, j] = 0.0
                        k += 1
                with torch.no_grad():
                    logits = model(variants[:, None])[:, 0, HW_R, HW_D].numpy()
                delta = np.abs(logits[1:] - logits[0]).reshape(patch.shape)
                if is_target:
                    heat_t += delta
                    n_t += 1
                else:
                    heat_b += delta
                    n_b += 1
        print(f"  seed {seed} done ({n_t} target CUTs, {n_b} background CUTs so far)", flush=True)

    heat_t /= max(n_t, 1)
    heat_b /= max(n_b, 1)
    np.savez(OUT / "occlusion_heatmaps.npz", target=heat_t, background=heat_b,
             n_target=n_t, n_background=n_b)

    def split(h):
        gr = slice(HW_R - G_R, HW_R + G_R + 1)
        gd = slice(HW_D - G_D, HW_D + G_D + 1)
        guard = h[gr, gd].copy()
        cut = float(guard[G_R, G_D])
        guard[G_R, G_D] = 0.0
        return cut, float(guard.sum()), float(h.sum() - cut - guard.sum())

    rows = []
    for name, h in (("target CUT", heat_t), ("background CUT", heat_b)):
        cut, guard, ref = split(h)
        tot = cut + guard + ref
        rows.append(dict(population=name, cut=cut, guard_excl_cut=guard, reference=ref,
                         pct_cut=100 * cut / tot, pct_guard=100 * guard / tot,
                         pct_reference=100 * ref / tot))
        print(f"\n{name}: total |dlogit| mass {tot:.4f}")
        print(f"   CUT itself           {cut:.4f}  ({100 * cut / tot:5.1f}%)")
        print(f"   guard (24 cells)     {guard:.4f}  ({100 * guard / tot:5.1f}%)")
        print(f"   reference (164 cells){ref:.4f}  ({100 * ref / tot:5.1f}%)")

    pd.DataFrame(rows).to_csv(OUT / "occlusion_summary.csv", index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
