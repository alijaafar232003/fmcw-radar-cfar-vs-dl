"""Step 7 verification — gate the datasets before anything trains on them.

Six checks. The leakage check is the one that would invalidate everything downstream if it failed,
and it is checked by frame CONTENT rather than by seed, because trusting the seeds is trusting the
thing under test.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPLITS = ["bench2_train", "bench2_val", "bench2_test", "bench2_tune", "bench2_snr_sweep"]
BORDER_R, BORDER_D = 10, 4          # half-widths of the chosen 21x9 window


def scan(split: str) -> dict:
    root = ROOT / "data" / split / "frames"
    rows, hashes, bad = [], [], []

    for fd in sorted(root.iterdir()):
        md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
        mag = np.load(fd / "rd_map_mag.npy")
        z = np.load(fd / "rd_map_complex.npy") if (fd / "rd_map_complex.npy").exists() else None

        if not np.isfinite(mag).all() or (z is not None and not np.isfinite(z).all()):
            bad.append(fd.name)

        # Content hash of the magnitude map -> detects duplicate frames across splits.
        hashes.append(hashlib.md5(mag.tobytes()).hexdigest())

        cl = md.get("clutter", {})
        lab = (pd.read_csv(fd / "labels.csv")
               if (fd / "labels.csv").stat().st_size > 5 else pd.DataFrame())

        for _, L in lab.iterrows():
            rb, vb = int(L["range_bin"]), int(L["velocity_bin"])
            w = mag[max(0, rb - 2):rb + 3, max(0, vb - 2):vb + 3]
            pk = np.unravel_index(np.argmax(w), w.shape)
            rows.append(dict(
                cond=md.get("condition", "?"), snr=L["cell_snr_db"],
                rb=rb, vb=vb,
                err=max(abs(pk[0] - min(rb, 2)), abs(pk[1] - min(vb, 2))),
                cnr=cl.get("cnr_db") if cl.get("enabled") else np.nan,
                shape=cl.get("shape") if cl.get("enabled") else np.nan,
            ))

    return dict(split=split, targets=pd.DataFrame(rows), hashes=hashes, bad=bad,
                meta=pd.read_csv(ROOT / "data" / split / "metadata.csv"))


def main() -> None:
    data = {s: scan(s) for s in SPLITS}

    print("1. COUNTS PER STRATUM")
    for s in SPLITS:
        vc = data[s]["meta"]["condition"].value_counts()
        bal = "balanced" if vc.nunique() == 1 else f"UNEVEN {dict(vc)}"
        print(f"   {s:<20}{len(data[s]['meta']):>6} frames  {vc.size} strata  {bal}")

    print("\n2. LEAKAGE (frame-content hashes shared between splits)")
    worst = 0
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            n = len(set(data[a]["hashes"]) & set(data[b]["hashes"]))
            worst = max(worst, n)
            if n:
                print(f"   {a} <-> {b}: {n} SHARED  <-- LEAK")
    print(f"   -> {'no shared frames between any pair' if worst == 0 else 'LEAK DETECTED'}")

    print("\n3. ALIGNMENT (targets >= 15 dB, peak within 1 bin of label)")
    for s in SPLITS:
        t = data[s]["targets"]
        g = t[t.snr >= 15]
        if not len(g):
            continue
        print(f"   {s:<20}{100 * (g.err <= 1).mean():5.1f}%  (n={len(g)})")
    t = data["bench2_test"]["targets"]
    g = t[t.snr >= 15]
    print("   bench2_test by condition:")
    print("      " + "  ".join(f"{c} {v:.0f}%" for c, v in
                               (100 * g.groupby("cond").err.apply(lambda e: (e <= 1).mean())).items()))

    print("\n4. DIFFICULTY")
    for s in SPLITS:
        v = data[s]["targets"].snr.values
        if not len(v):
            continue
        print(f"   {s:<20}median {np.median(v):5.1f} dB   <5dB {100*np.mean(v<5):4.1f}%   "
              f">30dB {100*np.mean(v>=30):4.1f}%")

    print("\n5. SWEPT PARAMETERS RECORDED")
    for s in ("bench2_train", "bench2_test"):
        t = data[s]["targets"].dropna(subset=["cnr"])
        print(f"   {s:<20}cnr {t.cnr.min():.1f}..{t.cnr.max():.1f} dB   "
              f"weibull shape {t['shape'].min():.2f}..{t['shape'].max():.2f}")

    print("\n6. TARGETS INSIDE THE TESTABLE REGION (21x9 borders)")
    for s in SPLITS:
        t = data[s]["targets"]
        if not len(t):
            continue
        oob = ((t.rb < BORDER_R) | (t.rb > 255 - BORDER_R) |
               (t.vb < BORDER_D) | (t.vb > 63 - BORDER_D)).sum()
        print(f"   {s:<20}{oob} of {len(t)} in the untested border")

    nbad = sum(len(d["bad"]) for d in data.values())
    print(f"\n7. NON-FINITE FRAMES: {nbad}")


if __name__ == "__main__":
    main()
