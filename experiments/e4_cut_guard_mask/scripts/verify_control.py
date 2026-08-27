"""E4 feasibility + control-equivalence checks. Reads only; writes only under this experiment.

Three questions, answered numerically before any training is considered:

  1. Is CUT-relative guard masking implementable on a DENSE fully-convolutional network?
  2. Does the frozen dense TinyCNN's logit at a cell equal the CENTRE logit of the 21x9 patch
     centred on that cell (in eval mode)?
  3. If a patch formulation is required, what exactly changes besides guard access?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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
OUT.mkdir(parents=True, exist_ok=True)

report = {}

# ------------------------------------------------------------------ 1. the contradiction
print("=" * 78)
print("1. Is CUT-relative guard masking possible on a DENSE network?")
print("=" * 78)
print(f"Guard is +/-{G_R} range, +/-{G_D} Doppler, so cell (r+1, v) lies inside the guard of CUT")
print(f"(r, v), and cell (r, v) lies inside the guard of CUT (r+1, v).")
print()
print("A dense pass computes every output from ONE input map. For output (r, v) the cell")
print("(r+1, v) must be HIDDEN; for output (r+1, v) that same cell must be VISIBLE, because it")
print("is its own CUT. One map cannot satisfy both.")
neighbours_in_guard = sum(1 for dr in range(-G_R, G_R + 1) for dd in range(-G_D, G_D + 1)
                          if (dr, dd) != (0, 0))
print(f"\n-> every cell is simultaneously a CUT (must be visible) and a guard neighbour of "
      f"{neighbours_in_guard} other CUTs (must be hidden).")
report["dense_masking_possible"] = False
report["guard_cells_masked_per_cut"] = neighbours_in_guard

# how bad is the "mask everyone's guard at once" approximation?
print("\nWhat if we mask on a decimated lattice so guards do not overlap?")
step = 2 * G_R + 1
n_cuts = (256 // step) * (64 // (2 * G_D + 1))
masked = n_cuts * neighbours_in_guard
print(f"  CUTs spaced {step} apart: {n_cuts} per pass, masking {masked:,} of {256 * 64:,} cells "
      f"= {100 * masked / (256 * 64):.0f}% of the map")
print("  -> each CUT's own context is then also ~96% masked. That masks far more than its guard,")
print("     so it answers a different question. Rejected.")
print(f"\nExact dense masking needs one pass per CUT offset: "
      f"{(2 * HW_R + 1) * (2 * HW_D + 1)} passes for non-overlapping receptive fields.")
report["passes_for_exact_dense"] = (2 * HW_R + 1) * (2 * HW_D + 1)

# ------------------------------------------------------------------ 2. patch equivalence
print("\n" + "=" * 78)
print("2. Does dense logit == patch-centre logit for the FROZEN model (eval mode)?")
print("=" * 78)

ck = torch.load(ROOT / "results" / "dl" / "tinycnn_frozen_s0_ep021.pt", weights_only=False)
model = TinyCNN(width=ck["config"].get("width", 16))
model.load_state_dict(ck["state_dict"])
model.eval()

fd = sorted((ROOT / "data" / "bench2_val" / "frames").iterdir())[0]
x = normalize_rd(np.load(fd / "rd_map_mag.npy"))
X = torch.from_numpy(x)[None, None]

with torch.no_grad():
    dense = model(X)[0, 0].numpy()

rng = np.random.default_rng(0)
rows = rng.integers(HW_R, 256 - HW_R, 40)
cols = rng.integers(HW_D, 64 - HW_D, 40)
diffs = []
with torch.no_grad():
    for r, v in zip(rows, cols):
        patch = X[:, :, r - HW_R:r + HW_R + 1, v - HW_D:v + HW_D + 1]
        centre = model(patch)[0, 0, HW_R, HW_D].item()
        diffs.append(abs(centre - float(dense[r, v])))
diffs = np.array(diffs)
print(f"  |dense - patch centre| over 40 interior cells:")
print(f"    max {diffs.max():.3e}   mean {diffs.mean():.3e}")
equivalent = bool(diffs.max() < 1e-4)
print(f"  -> {'EQUIVALENT' if equivalent else 'NOT equivalent'} in eval mode")
print("  (expected: the centre output of a 21x9 patch draws on exactly those 21x9 inputs and")
print("   never on padding, and eval-mode BatchNorm is a fixed affine map)")
report["patch_equals_dense_eval"] = equivalent
report["patch_dense_max_abs_diff"] = float(diffs.max())

# ------------------------------------------------------------------ 3. what changes in training
print("\n" + "=" * 78)
print("3. What changes if the MASKED arm is trained as a patch model?")
print("=" * 78)

m_train = TinyCNN(width=16).train()
m_eval = TinyCNN(width=16).eval()
m_eval.load_state_dict(m_train.state_dict())
with torch.no_grad():
    big = m_train(X.repeat(4, 1, 1, 1))[0, 0, 128, 32].item()
patches = torch.stack([X[0, :, 128 - HW_R:128 + HW_R + 1, 32 - HW_D:32 + HW_D + 1]
                       for _ in range(4)])
with torch.no_grad():
    small = m_train(patches)[0, 0, HW_R, HW_D].item()
print(f"  same weights, TRAIN mode (batch BatchNorm):")
print(f"    dense map  centre logit {big:+.4f}")
print(f"    21x9 patch centre logit {small:+.4f}")
print(f"    difference {abs(big - small):.4f}")
print("  BatchNorm normalises over batch AND spatial extent. Dense sees 256x64 per channel;")
print("  a patch sees 21x9. The training-time statistics differ, so a patch-trained model is")
print("  not the same learner as the dense control even with identical weights and data.")
report["train_mode_dense_vs_patch_diff"] = float(abs(big - small))

n_cells_dense = 256 * 64
n_cells_patch = 1
print(f"\n  Loss semantics: dense supervises {n_cells_dense:,} cells per frame; a patch model")
print(f"  supervises {n_cells_patch} (its centre). Covering every eligible CUT would need")
print(f"  {(256 - 2 * HW_R) * (64 - 2 * HW_D):,} patches per frame -- "
      f"{(2 * HW_R + 1) * (2 * HW_D + 1)}x the dense compute.")
report["patches_per_frame_for_full_coverage"] = int((256 - 2 * HW_R) * (64 - 2 * HW_D))

(OUT / "feasibility_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'feasibility_report.json'}")
