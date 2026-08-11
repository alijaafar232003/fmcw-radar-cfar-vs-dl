"""Data-quality audit of the frozen bench2 splits against docs/data_quality_criteria.md.

The existing scripts each answer one narrow question:
  verify_splits.py   -- is anything structurally broken (leakage, alignment, non-finite)?
  audit_datasets.py  -- does each stratum contain what its config asked for?
  verify_clutter.py  -- does the clutter MODEL behave (run on freshly simulated frames)?

None of them measures the statistical criteria on the frames that will actually be trained and
tested on. This script does that, criterion by criterion, reading the frozen data only.

Criteria covered here: 3 (statistical realism), 4 (difficulty calibration), 5 (coverage),
6 (no exploitable shortcuts), 7 (benchmark neutrality, composition only), 9 (reproducibility).
Criteria 1, 2 and 8 are physics/label/literature checks answered elsewhere and are reported
in the write-up rather than re-derived here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPLITS = ["bench2_train", "bench2_val", "bench2_test", "bench2_tune", "bench2_snr_sweep"]
PFA_DESIGN = 1e-4
N_SAMPLE = 300          # frames per split for the map-level (expensive) checks


def frame_dirs(split: str, n: int | None = None, seed: int = 0) -> list[Path]:
    fds = sorted((ROOT / "data" / split / "frames").iterdir())
    if n is None or n >= len(fds):
        return fds
    idx = np.random.default_rng(seed).choice(len(fds), n, replace=False)
    return [fds[i] for i in sorted(idx)]


def load_labels(split: str) -> pd.DataFrame:
    """Every label in the split, with its frame's condition attached."""
    meta = pd.read_csv(ROOT / "data" / split / "metadata.csv")
    rows = []
    for fid, cond in zip(meta.frame_id, meta.condition):
        p = ROOT / "data" / split / "frames" / fid / "labels.csv"
        if p.stat().st_size <= 5:
            continue
        lab = pd.read_csv(p)
        lab["condition"] = cond
        lab["frame_id"] = fid
        rows.append(lab)
    out = pd.concat(rows, ignore_index=True)
    for c in ("range_m", "velocity_mps", "amplitude_db", "cell_snr_db",
              "nearest_range_m", "nearest_velocity_mps"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def theory_pd(snr_db, pfa: float = PFA_DESIGN) -> np.ndarray:
    """Swerling-0/1 square-law Pd = Pfa^(1/(1+SNR)) -- the same curve used in the notebook."""
    snr = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    return pfa ** (1.0 / (1.0 + snr))


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------------------
# Criterion 3 -- statistical realism
# --------------------------------------------------------------------------------------
def criterion_3() -> None:
    rule("CRITERION 3  STATISTICAL REALISM")

    # (a) thermal noise: cell POWER should be exponential after the square law.
    # Measured on target-free frames of the `noise` stratum, so nothing else contaminates it.
    print("\n(a) receiver noise -- cell power vs exponential (noise stratum, no targets)")
    meta = pd.read_csv(ROOT / "data" / "bench2_test" / "metadata.csv")
    noise_ids = meta.loc[meta.condition == "noise", "frame_id"].tolist()[:40]
    ks_stats = []
    for fid in noise_ids:
        mag = np.load(ROOT / "data" / "bench2_test" / "frames" / fid / "rd_map_mag.npy")
        p = (mag.astype(float) ** 2).ravel()
        p = p / p.mean()
        ks_stats.append(stats.kstest(p, "expon").statistic)
    print(f"    frames tested         {len(ks_stats)}")
    print(f"    KS statistic          mean {np.mean(ks_stats):.4f}   max {np.max(ks_stats):.4f}")
    print("    (<0.02 = indistinguishable from exponential at this sample size)")

    # (b) clutter: heavy tail in the frozen data, not just in a fresh simulation.
    # The centre Doppler column of clutter frames, normalised per frame so that frames with
    # different CNR can be pooled without the mixture faking a heavy tail.
    print("\n(b) ground clutter -- Weibull shape of the ridge amplitudes (clutter stratum)")
    for split in ("bench2_train", "bench2_test"):
        m = pd.read_csv(ROOT / "data" / split / "metadata.csv")
        ids = m.loc[m.condition == "clutter", "frame_id"].tolist()[:120]
        cols, ratios = [], []
        for fid in ids:
            mag = np.load(ROOT / "data" / split / "frames" / fid / "rd_map_mag.npy").astype(float)
            c = mag[:, mag.shape[1] // 2]
            c = c / c.mean()
            cols.append(c)
            ratios.append(np.percentile(c, 99) / np.percentile(c, 50))
        a = np.concatenate(cols)
        shape, _, _ = stats.weibull_min.fit(a, floc=0)
        print(f"    {split:<16} c = {shape:.2f}   p99/p50 = {np.mean(ratios):.2f}   (n={len(ids)})")
    print("    c ~ 2 = Rayleigh (Gaussian-like);  c < 2 = heavy-tailed, the realistic regime")

    # (c) radar equation: does received strength actually fall with range?
    print("\n(c) range dependence -- does cell SNR obey the 1/R^4 law?")
    lab = load_labels("bench2_test")
    lab = lab[lab.condition.isin(["isolated", "multi"])]        # clean strata only
    r = lab.range_m.to_numpy(float)
    amp = lab.amplitude_db.to_numpy(float)
    x = 40 * np.log10(r / 25.0)                                  # ref_range_m = 25
    # `amplitude_db` in labels.csv is the POST-propagation amplitude, so regressing it on the
    # loss term recovers the exponent directly: slope -1 means the 40log10 law was applied once.
    # The residual is then the drawn amplitude, whose spread must match the config's window.
    slope, icept, rval, _, _ = stats.linregress(x, amp)
    resid = amp - slope * x - icept
    print(f"    corr(range, cell SNR)       {np.corrcoef(r, lab.cell_snr_db.to_numpy(float))[0,1]:+.3f}")
    print(f"    corr(range, amplitude_db)   {np.corrcoef(r, amp)[0,1]:+.3f}")
    print(f"    fit of amplitude_db on 40log10(R/25):  slope {slope:+.3f}  r {rval:+.3f}")
    print("    slope should be -1.00 (the loss is applied exactly once, at the right exponent)")
    print(f"    residual (= the drawn amplitude) spans {resid.min()+icept:.1f}..{resid.max()+icept:.1f} dB"
          f"   config draws 4..16 dB")
    for lo, hi in ((8, 18), (18, 28), (28, 38), (38, 46)):
        s = lab[(lab.range_m >= lo) & (lab.range_m < hi)]
        if len(s):
            print(f"      {lo:>2}-{hi:<3} m   mean cell SNR {s.cell_snr_db.mean():6.2f} dB  (n={len(s)})")


# --------------------------------------------------------------------------------------
# Criterion 4 -- difficulty calibration
# --------------------------------------------------------------------------------------
def criterion_4() -> None:
    rule("CRITERION 4  DIFFICULTY CALIBRATION")
    print("\nfraction of targets in the informative band (theory Pd 0.1-0.9 at Pfa 1e-4)")
    print(f"    {'split':<20}{'median SNR':>11}{'Pd<0.1':>9}{'0.1-0.9':>9}{'Pd>0.9':>9}")
    for s in SPLITS:
        lab = load_labels(s)
        pd_ = theory_pd(lab.cell_snr_db.values)
        print(f"    {s:<20}{np.median(lab.cell_snr_db):>10.1f} "
              f"{100*np.mean(pd_ < 0.1):>8.1f}%{100*np.mean((pd_ >= 0.1) & (pd_ <= 0.9)):>8.1f}%"
              f"{100*np.mean(pd_ > 0.9):>8.1f}%")

    print("\nper condition (bench2_test)")
    lab = load_labels("bench2_test")
    lab["pd"] = theory_pd(lab.cell_snr_db.values)
    print(f"    {'condition':<16}{'n':>6}{'median SNR':>12}{'in band':>9}{'saturated':>11}")
    for cond, sub in lab.groupby("condition"):
        print(f"    {cond:<16}{len(sub):>6}{sub.cell_snr_db.median():>11.1f} "
              f"{100*np.mean((sub.pd >= 0.1) & (sub.pd <= 0.9)):>8.1f}%"
              f"{100*np.mean(sub.pd > 0.9):>10.1f}%")


# --------------------------------------------------------------------------------------
# Criterion 5 -- coverage and diversity
# --------------------------------------------------------------------------------------
def criterion_5() -> None:
    rule("CRITERION 5  COVERAGE AND DIVERSITY")
    for s in ("bench2_train", "bench2_test"):
        lab = load_labels(s)
        meta = pd.read_csv(ROOT / "data" / s / "metadata.csv")
        print(f"\n  {s}")
        print(f"    targets per frame   {dict(meta.num_targets.value_counts().sort_index())}")
        for col, unit in (("range_m", "m"), ("velocity_mps", "m/s"),
                          ("amplitude_db", "dB"), ("cell_snr_db", "dB")):
            v = lab[col]
            print(f"    {col:<16}{v.min():8.2f} .. {v.max():7.2f} {unit:<5}"
                  f" median {v.median():7.2f}")
        # Gaps: an empty decile of the range span would mean the sampler is not covering it.
        h, edges = np.histogram(lab.range_m, bins=10, range=(8, 46))
        print(f"    range deciles       {list(h)}"
              f"{'   <-- EMPTY BIN' if (h == 0).any() else ''}")
        h, _ = np.histogram(lab.velocity_mps, bins=10, range=(-8, 8))
        print(f"    velocity deciles    {list(h)}")
        # Accidental correlations between the factors we sample independently.
        c = lab[["range_m", "velocity_mps", "amplitude_db"]].corr().values
        print(f"    |corr| among range/velocity/amplitude, max off-diagonal "
              f"{np.abs(c - np.eye(3)).max():.3f}")


# --------------------------------------------------------------------------------------
# Criterion 6 -- no exploitable shortcuts
# --------------------------------------------------------------------------------------
def criterion_6() -> None:
    rule("CRITERION 6  NO EXPLOITABLE SHORTCUTS")
    ax = np.load(ROOT / "data" / "bench2_test" / "axes.npz")
    dr = float(np.diff(ax["range_axis"])[0])
    dv = float(np.diff(ax["velocity_axis"])[0])

    print("\n(a) off-grid placement -- targets must not sit on bin centres")
    for s in ("bench2_train", "bench2_test"):
        lab = load_labels(s)
        fr = (lab.range_m - lab.nearest_range_m) / dr
        fv = (lab.velocity_mps - lab.nearest_velocity_mps) / dv
        # Uniform(-0.5, 0.5) has std 0.289; a grid-snapped generator would give ~0.
        print(f"    {s:<16}range frac  std {fr.std():.3f}  |mean| {abs(fr.mean()):.3f}   "
              f"doppler frac  std {fv.std():.3f}  |mean| {abs(fv.mean()):.3f}")
        ks = stats.kstest((fr + 0.5).values, "uniform").statistic
        print(f"    {'':<16}range fraction vs Uniform(-0.5,0.5): KS {ks:.3f}"
              f"   (0.289 std = ideal uniform)")

    print("\n(b) does a trivial frame statistic predict the answer?")
    print("    (a CNN could exploit any of these instead of learning detection)")
    for s in ("bench2_train",):
        meta = pd.read_csv(ROOT / "data" / s / "metadata.csv")
        fds = frame_dirs(s, N_SAMPLE)
        keep = {fd.name for fd in fds}
        sub = meta[meta.frame_id.isin(keep)].set_index("frame_id")
        means, medians, maxes, npos = [], [], [], []
        for fd in fds:
            mag = np.load(fd / "rd_map_mag.npy").astype(float)
            means.append(mag.mean()); medians.append(np.median(mag)); maxes.append(mag.max())
            npos.append(int(np.load(fd / "label_mask.npy").sum()))
        n = sub.loc[[fd.name for fd in fds], "num_targets"].values
        for name, v in (("frame mean", means), ("frame median", medians), ("frame max", maxes)):
            print(f"    corr({name:<13}, num_targets) = {np.corrcoef(v, n)[0,1]:+.3f}")
        print(f"    corr(frame median , positive cells) = {np.corrcoef(medians, npos)[0,1]:+.3f}")
        print(f"    per-frame median spread across frames: "
              f"{np.min(medians):.0f} .. {np.max(medians):.0f} "
              f"(a constant would mean per-frame normalisation had baked the answer in)")

    print("\n(c) seed -> content leakage")
    meta = pd.read_csv(ROOT / "data" / "bench2_train" / "metadata.csv")
    print(f"    unique seeds {meta.seed.nunique()} of {len(meta)} frames")
    print(f"    corr(seed, num_targets)  {np.corrcoef(meta.seed, meta.num_targets)[0,1]:+.3f}")
    print(f"    corr(seed, snr_db)       {np.corrcoef(meta.seed, meta.snr_db)[0,1]:+.3f}")

    print("\n(d) condition ordering -- is the stratum inferable from the frame index?")
    for s in ("bench2_train", "bench2_test"):
        order = pd.read_csv(ROOT / "data" / s / "metadata.csv").condition.values
        k = len(set(order))                     # train excludes `unresolved`, so k differs
        cyc = bool((order[:4 * k] == order[k:5 * k]).all())
        print(f"    {s:<16}{k} strata, assigned cyclically: {cyc}")
    print("    -> harmless for training (shuffled by the loader) but never index-split a set")


# --------------------------------------------------------------------------------------
# Criterion 7 (composition) and 9 (reproducibility)
# --------------------------------------------------------------------------------------
def criterion_7_9() -> None:
    rule("CRITERION 7  BENCHMARK COMPOSITION   /   CRITERION 9  REPRODUCIBILITY")
    print("\ntest-set composition (what the headline number averages over)")
    meta = pd.read_csv(ROOT / "data" / "bench2_test" / "metadata.csv")
    vc = meta.condition.value_counts().sort_index()
    for c, n in vc.items():
        print(f"    {c:<16}{n:>6}  {100*n/len(meta):5.1f}%")
    print("    -> equal strata by design; report PER-CONDITION, and label the adversarial ones")

    print("\nreproducibility artefacts stored with each split")
    for s in SPLITS:
        d = ROOT / "data" / s
        have = [f for f in ("dataset_config_used.yaml", "radar_config_used.json",
                            "metadata.csv", "axes.npz") if (d / f).exists()]
        m = pd.read_csv(d / "metadata.csv")
        txt = (d / "dataset_config_used.yaml").read_text(encoding="utf-8")
        master = (txt.split("random_seed:")[1].split("\n")[0].split("#")[0].strip()
                  if "random_seed:" in txt else "?")
        print(f"    {s:<20}{len(have)}/4 files   master seed {master:<7}"
              f"{m.seed.nunique()} unique frame seeds")


def main() -> None:
    """Run every criterion, or just the ones named on the command line (e.g. `... 3 6`)."""
    todo = sys.argv[1:] or ["3", "4", "5", "6", "7"]
    if "3" in todo:
        criterion_3()
    if "4" in todo:
        criterion_4()
    if "5" in todo:
        criterion_5()
    if "6" in todo:
        criterion_6()
    if "7" in todo or "9" in todo:
        criterion_7_9()


if __name__ == "__main__":
    main()
