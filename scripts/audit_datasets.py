"""Deeper audit of the generated splits — does each stratum contain what its config claims?

verify_splits.py answers "is anything broken?". This answers "is it what we asked for?", which is
the failure mode that passes every structural check and still invalidates the study: a stratum that
is silently mis-specified produces perfectly valid frames of the wrong scenario.

Sampled rather than exhaustive, so it runs in seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.config import RadarConfig, derive_radar_params    # noqa: E402
from radar.simulator import make_axes                        # noqa: E402

SPLITS = ["bench2_train", "bench2_val", "bench2_test", "bench2_tune", "bench2_snr_sweep"]
N_SAMPLE = 400


def sample_frames(split: str, n: int, seed: int = 0):
    """Random sample, not evenly spaced.

    The generator assigns conditions cyclically (frame i gets condition i % n_strata), so an
    evenly-spaced sample aliases against that cycle and can under-represent or miss whole strata
    depending on how the stride divides. Random sampling is stride-independent.
    """
    fds = sorted((ROOT / "data" / split / "frames").iterdir())
    if n >= len(fds):
        return fds
    idx = np.random.default_rng(seed).choice(len(fds), n, replace=False)
    return [fds[i] for i in sorted(idx)]


def main() -> None:
    cfg = RadarConfig()
    d = derive_radar_params(cfg)
    _, _, range_axis, vel_axis = make_axes(cfg)

    print("=" * 78)
    print("1. RADAR CONFIG RECORDED IN EACH DATASET MATCHES THE CODE")
    print("=" * 78)
    for s in SPLITS:
        p = ROOT / "data" / s / "radar_config_used.json"
        if not p.exists():
            print(f"   {s:<20} MISSING radar_config_used.json")
            continue
        saved = json.loads(p.read_text(encoding="utf-8"))
        same = all(abs(saved[k] - getattr(cfg, k)) < 1e-9 for k in ("fc", "B", "Tc")) \
            and saved["N"] == cfg.N and saved["M"] == cfg.M
        print(f"   {s:<20} {'OK' if same else 'MISMATCH ' + str(saved)}")

    print()
    print("=" * 78)
    print("2. DOES EACH STRATUM CONTAIN WHAT ITS CONFIG SPECIFIES?")
    print("=" * 78)
    rows = []
    for s in SPLITS:
        for fd in sample_frames(s, N_SAMPLE):
            md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
            lab = (pd.read_csv(fd / "labels.csv")
                   if (fd / "labels.csv").stat().st_size > 5 else pd.DataFrame())
            cl = md.get("clutter", {})
            sep = (abs(lab.range_m.iloc[1] - lab.range_m.iloc[0])
                   if len(lab) == 2 else np.nan)
            rows.append(dict(
                split=s, cond=md.get("condition", "sweep"), n=len(lab),
                rmin=lab.range_m.min() if len(lab) else np.nan,
                rmax=lab.range_m.max() if len(lab) else np.nan,
                vabs_min=lab.velocity_mps.abs().min() if len(lab) else np.nan,
                vabs_max=lab.velocity_mps.abs().max() if len(lab) else np.nan,
                sep=sep,
                clutter=bool(cl.get("enabled", False)),
                cnr=cl.get("cnr_db") if cl.get("enabled") else np.nan,
                shape=cl.get("shape") if cl.get("enabled") else np.nan,
            ))
    df = pd.DataFrame(rows)

    EXPECT = {
        "isolated":     "1 target, no clutter",
        "masking":      "2 targets, sep 0.40-2.50 m, no clutter",
        "unresolved":   "2 targets, sep 0.05-0.35 m, no clutter",
        "clutter":      "1-3 targets, |v| < 1.2, clutter ON",
        "clutter_edge": "1-2 targets, |v| 2.5-4.2, clutter ON",
        "multi":        "4-5 targets, no clutter",
        "noise":        "0 targets",
        "mixed":        "0-5 targets, clutter ON",
        "sweep":        "1 target, no clutter, no propagation",
    }
    g = df.groupby("cond")
    print(f"   {'stratum':<14}{'n tgt':>12}{'range m':>14}{'|v| m/s':>14}"
          f"{'sep m':>14}{'clutter':>9}")
    for cond, sub in g:
        ntxt = f"{sub.n.min()}-{sub.n.max()}"
        rtxt = (f"{sub.rmin.min():.1f}-{sub.rmax.max():.1f}"
                if sub.rmin.notna().any() else "-")
        vtxt = (f"{sub.vabs_min.min():.2f}-{sub.vabs_max.max():.2f}"
                if sub.vabs_min.notna().any() else "-")
        stxt = f"{sub.sep.min():.2f}-{sub.sep.max():.2f}" if sub.sep.notna().any() else "-"
        ctxt = f"{100*sub.clutter.mean():.0f}%"
        print(f"   {cond:<14}{ntxt:>12}{rtxt:>14}{vtxt:>14}{stxt:>14}{ctxt:>9}")
    print()
    for cond in sorted(EXPECT):
        if cond in g.groups:
            print(f"      expected {cond:<14} {EXPECT[cond]}")

    print()
    print("=" * 78)
    print("3. COMPLEX MAP CONSISTENT WITH MAGNITUDE (spot check)")
    print("=" * 78)
    for s in SPLITS:
        errs, shapes_ok = [], True
        for fd in sample_frames(s, 25):
            mag = np.load(fd / "rd_map_mag.npy")
            z = np.load(fd / "rd_map_complex.npy")
            shapes_ok &= (z.shape == mag.shape)
            errs.append(float(np.abs(np.abs(z) - mag).max()))
        print(f"   {s:<20}orientation {'OK' if shapes_ok else 'MISMATCH'}   "
              f"max ||z| - mag| = {max(errs):.3e}")

    print()
    print("=" * 78)
    print("4. LABEL BINS AGREE WITH THE PHYSICAL COORDINATES")
    print("=" * 78)
    bad = tot = 0
    for s in SPLITS:
        for fd in sample_frames(s, 120):
            if (fd / "labels.csv").stat().st_size <= 5:
                continue
            lab = pd.read_csv(fd / "labels.csv")
            for _, L in lab.iterrows():
                tot += 1
                rb = int(np.argmin(np.abs(range_axis - L["range_m"])))
                vb = int(np.argmin(np.abs(vel_axis - L["velocity_mps"])))
                if abs(rb - int(L["range_bin"])) > 0 or abs(vb - int(L["velocity_bin"])) > 0:
                    bad += 1
    print(f"   {tot - bad}/{tot} labels have range_bin/velocity_bin matching range_m/velocity_mps")

    print()
    print("=" * 78)
    print("5. CLASS BALANCE (what the loss sees)")
    print("=" * 78)
    for s in ("bench2_train", "bench2_val", "bench2_test"):
        pos = tot_c = 0
        for fd in sample_frames(s, 200):
            m = np.load(fd / "label_mask.npy")
            pos += int(m.sum()); tot_c += m.size
        r = pos / tot_c
        print(f"   {s:<20}positive rate {r:.6f}  ->  1:{int(1/r)}")

    print()
    print("=" * 78)
    print("6. METADATA COMPLETENESS")
    print("=" * 78)
    need = ["condition", "num_targets", "snr_db", "noise_var", "clutter", "multipath", "seed"]
    for s in SPLITS:
        md = json.loads((sample_frames(s, 1)[0] / "metadata.json").read_text(encoding="utf-8"))
        miss = [k for k in need if k not in md]
        print(f"   {s:<20}{'all present' if not miss else 'MISSING ' + str(miss)}"
              f"   ({len(md)} fields)")


if __name__ == "__main__":
    main()
