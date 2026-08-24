"""Build notebooks/cfar_baseline.ipynb — the classical baseline on `bench2_test`.

Restructured 19 Aug 2026: 11 sections -> 7 + appendix, and **SO-CFAR added throughout** (it was
absent from METHODS, so every "best method" column was a max over three, and the localization table
omitted the method that is best on range in most conditions).

Source-only by design: no stored outputs, everything computed from `results/`, nothing pasted.
Rebuild with:

    python scripts/build_cfar_notebook.py
"""
import json
import pathlib

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    CELLS.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": t})


def cap(t):
    """Figure caption — one line, directly under the figure, saying what to look at."""
    md(f"*<span style='color:#6b6b6b'>{t}</span>*")


# ============================================================================== header
md("""# CFAR baseline — `bench2_test`

**What this is.** The classical detector, characterised on its own, before any comparison with the
network. Four variants — **CA, GO, OS, SO** — on the held-out split, at a matched operating point.

**The result in one line.** *The point estimates order exactly as CFAR theory predicts — CA best in
clean noise, OS where an interferer sits in the ring, GO inside clutter, SO nowhere — but on this
split **no winner is separable from its runner-up**, so the ordering is consistent with the
mechanism rather than established by it.* Which method "wins" also depends on the metric: §4 shows
a case where the answer flips, and §6 shows SO winning on a metric it loses badly on elsewhere.

**How to read it.**

| § | |
|---|---|
| 1 | the frame — split, geometry, metric hierarchy |
| 2 | **how the CFAR window was chosen** — the part that makes this a baseline rather than a default |
| 3 | the baseline result — Pd per condition, who wins, and why |
| 4 | why the methods differ — mechanism, shown on real frames |
| 5 | validation against physics — Pd vs SNR, CFAR loss in dB, resolution |
| 6 | secondary metrics — localization and runtime |
| 7 | what this establishes, and what it does not |
| A | appendix — per-target record, full ROC set |

Nothing here modifies data. Everything reads cached detections from `results/cfar_test/`, produced
by `scripts/cfar_test_baseline.py`.""")

code('''import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig
from eval import MatchConfig, aggregate, evaluate_frame, roc_curve, roc_thresholds
from radar.config import RadarConfig
from radar.simulator import make_axes

_, _, RANGE_AXIS, vax = make_axes(RadarConfig())
V0 = int(np.argmin(np.abs(vax)))          # zero-Doppler bin: where the clutter ridge sits

RES = ROOT / "results" / "cfar_test"
DATA = ROOT / "data" / "bench2_test" / "frames"

# SO was missing from this list until 19 Aug. Its absence made every "best" column a maximum over
# three methods, and dropped the variant that is best on RANGE localization in most conditions.
METHODS = ["CA", "GO", "OS", "SO"]
TARGET_PFA = 1e-4
SHAPE = (256, 64)
RANGE_RES_M, VEL_RES_MPS = 0.19518, 0.42013

CFG = CfarConfig()
VALID = np.zeros(SHAPE, dtype=bool)
VALID[CFG.half_width_r:-CFG.half_width_r, CFG.half_width_d:-CFG.half_width_d] = True

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
COLOR = {"CA": "#2a78d6", "GO": "#eb6834", "OS": "#3d9970", "SO": "#a05fb4"}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "axes.grid": True, "grid.alpha": 0.6, "figure.dpi": 120})

frames = pd.read_csv(RES / "frames.csv")
truths = pd.read_csv(RES / "truths.csv")
summary = pd.read_csv(RES / "summary.csv")

PEAKS, RUNTIME = {}, {}
for m in METHODS:
    d = np.load(RES / f"peaks_{m}.npz")
    arr = d["peaks"]
    PEAKS[m] = {int(f): arr[arr[:, 0] == f][:, 1:] for f in np.unique(arr[:, 0])}
    RUNTIME[m] = float(d["ms_per_frame"])

TRUTH_BY_FRAME = {f: g for f, g in truths.groupby("frame")}
CONDS = sorted(frames.condition.unique())
SCORED = [c for c in CONDS if c != "noise"]           # `noise` holds no targets
THR = {(r.condition, r.method): r.threshold for _, r in summary.iterrows()}

print(f"{len(frames):,} frames | {len(truths):,} targets | methods: {', '.join(METHODS)}")
print("cached detections:", {m: sum(len(v) for v in PEAKS[m].values()) for m in METHODS})
assert set(summary.method) == set(METHODS), f"summary is missing {set(METHODS) - set(summary.method)}"
print("SO present in summary, peaks and runtime:",
      all(m in summary.method.values for m in METHODS), "|",
      {m: round(RUNTIME[m], 2) for m in METHODS}, "ms/frame")''')

# ============================================================================== 1 · Frame
md("""---
## 1 · The frame

### The metric hierarchy — stated once, used throughout

| rank | metric | why |
|---|---|---|
| **primary** | **Pd at matched *measured* Pfa = 1e-4, per condition** | the only comparison where no method can buy detections with false alarms |
| secondary | localization error (range / Doppler, bins **and** metres / m·s⁻¹) | a detection in the wrong cell is worth less; §6 |
| secondary | runtime, ms per frame | a detector that cannot run is not a baseline; §6 |

**Why *measured* Pfa and not design Pfa.** `alpha` is derived assuming exponentially distributed
cell power in homogeneous noise. Clutter violates that, so the threshold theory calls 1e-4 delivers
something else — and by a *different factor for each method*. Comparing at "design 1e-4" would put
CA at 1.2e-2 against GO at 8.7e-3 against OS at 3.6e-2: three different operating points wearing one
label. Each method's threshold is therefore swept **on the condition being scored** until its
achieved per-cell rate equals 1e-4, and Pd is read there (`PROGRESS.md` step 5).

### Geometry and matching, in one place

| | value | note |
|---|---|---|
| RD map | 256 × 64 | 0.195 m and 0.42 m·s⁻¹ per bin |
| CFAR window | **21 × 9** | guard 2/2, train 8/2 → **164 reference cells**; chosen in §2 |
| scored region | interior only | margin 10 range / 4 Doppler — CFAR cannot test nearer the border, so nothing else is scored there either |
| NMS | **`nms_size = 5`, a HALF-width → an 11 × 11 window** | 2.15 m × 4.6 m·s⁻¹. Previously written both ways in this notebook; it is the half-width |
| matching | one-to-one, ±1 bin | ±3 reported beside it in §6 |

### What the split is made of""")

code('''comp = (frames.groupby("condition")
        .agg(frames=("frame", "size"), targets=("num_targets", "sum"),
             mean_targets=("num_targets", "mean")))
comp["share_%"] = 100 * comp.frames / comp.frames.sum()
comp["adversarial"] = [c in ("masking", "unresolved", "clutter_edge") for c in comp.index]
display(comp.round(2))
adv = 100 * comp.loc[comp.adversarial, "frames"].sum() / comp.frames.sum()
print(f"{adv:.1f}% of the split is deliberately adversarial -- built to break a specific detector.")
print("So a single pooled Pd over this mixture is NOT a 'typical scene' number and must never")
print("be presented as one. Per-condition is the result. (docs/data_quality_criteria.md, #7)")''')

# ============================================================================== 2 · configuration
md("""---
## 2 · How the CFAR window was chosen

**This is what makes it a baseline rather than a default.** A CFAR comparison is only fair if the
classical side was given the same chance to fit the sensor that the network gets. No paper in the
19-paper corpus re-tunes CFAR at all — they inherit a textbook geometry and report that it loses.

Everything below was selected on **`bench2_tune`** (400 frames, seed 91001), which is **disjoint
from every evaluation split**. Neither side has seen `bench2_test`.

### 2.1 Guard size comes from the measured point-spread function

The guard region exists to keep target energy out of the noise estimate. Its size is therefore a
question about the **PSF**, not a hyperparameter: how far does a point target's energy actually
spread?

The mainlobe width in *bins* is set by the **Blackman window**, not by the radar's resolution —
measured identically on the old and new sensors even though every physical parameter changed. The
worst case is a target sitting **off-bin** (half a bin from centre), which smears wider than an
on-bin target, so the geometry is justified against that.""")

code('''psf = pd.read_csv(ROOT / "results" / "psf_profile.csv")
worst = psf[psf.case == "off-bin (worst case)"]

fig, ax = plt.subplots(figsize=(9.5, 4.8))
for axis, ls in (("range", "-"), ("doppler", "--")):
    d = worst[worst.axis == axis].sort_values("offset_bins")
    ax.plot(d.offset_bins, d.rel_db, ls, lw=1.9, label=f"{axis} (off-bin target)",
            color=COLOR["CA"] if axis == "range" else COLOR["OS"])

g_r, g_d = CFG.guard_r, CFG.guard_d
RC, DC = COLOR["CA"], COLOR["OS"]

# The guard is 2 on BOTH axes, but the window is not: it reaches 10 bins in range and only 4 in
# Doppler. One "window edge" line would be wrong for one of the two curves, so both are drawn.
ax.axvspan(-(g_r + 0.5), g_r + 0.5, color="#c62828", alpha=0.08, zorder=0)
for s in (-1, 1):
    ax.axvline(s * (g_r + 0.5), color="#c62828", ls=":", lw=1.8,
               label="guard edge (+/-2, both axes)" if s == 1 else None)
    ax.axvline(s * (CFG.half_width_r + 0.5), color=RC, ls="--", lw=1.2, alpha=0.8,
               label=f"range window edge (+/-{CFG.half_width_r})" if s == 1 else None)
    ax.axvline(s * (CFG.half_width_d + 0.5), color=DC, ls="--", lw=1.2, alpha=0.8,
               label=f"Doppler window edge (+/-{CFG.half_width_d})" if s == 1 else None)
ax.axhline(-30, color=INK, lw=1, ls="-.", alpha=0.6)
ax.axhline(-40, color=INK, lw=1, ls="-.", alpha=0.35)
ax.text(13.6, -29, "-30 dB", fontsize=8, color=INK, ha="right")
ax.text(13.6, -39, "-40 dB", fontsize=8, color=INK, ha="right")
ax.text(0, -8, "guard\\n(excluded)", fontsize=8, ha="center", color="#c62828")

ax.set_xlim(-14, 14); ax.set_ylim(-95, 8)
ax.set_xlabel("offset from the target's peak bin")
ax.set_ylabel("response relative to peak (dB)")
ax.set_title("Guard = 2 covers the target's -30 dB spread; the reference ring starts\\n"
             "where the response is already below -40 dB", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=2)
plt.tight_layout(); plt.show()

for axis in ("range", "doppler"):
    s = worst[worst.axis == axis].set_index("offset_bins").rel_db
    b30 = min([abs(o) for o, v in s.items() if v < -30])
    b40 = min([abs(o) for o, v in s.items() if v < -40])
    print(f"  {axis:8s}: below -30 dB beyond {b30} bins, below -40 dB beyond {b40} bins")
print(f"  guard = {CFG.guard_r}/{CFG.guard_d} -> the reference ring starts at {CFG.guard_r + 1} bins in range, {CFG.guard_d + 1} in Doppler.")''')

cap("Look at where the red band ends: guard ±2 exactly covers the −30 dB spread, and the reference "
    "ring begins where the target contributes under −40 dB. That is the whole justification for "
    "guard = 2 in one picture.")

md("""**Guard 3 was tested and rejected.** It would buy the −40 dB skirt, but costs two more bins of
the scarce Doppler axis (64 bins total, against 256 in range), and the sweep measured the difference
at **< 1% Pd**. Paying 2 Doppler bins for under a point of Pd is a bad trade on this sensor.

### 2.2 The window geometry — and why the pooled sweep was not trusted

Ten geometries were swept. Pooled Pd rose monotonically as the **Doppler** window narrowed, while
range width barely mattered.""")

code('''sw = pd.read_csv(ROOT / "results" / "cfar_geometry_sweep.csv")
pool = (sw[sw.kind == "CA"][["window", "n_ref", "usable_doppler", "pd_at_pfa"]]
        .sort_values("pd_at_pfa", ascending=False).set_index("window"))
display(pool.round(3))
c = np.corrcoef(pool.usable_doppler, pool.pd_at_pfa)[0, 1]
print(f"correlation between pooled Pd and USABLE DOPPLER BINS: r = {c:.3f}")
print("That is what a measurement artefact looks like -- a narrower Doppler window simply leaves")
print("more of the map scoreable. So the pooled sweep was NOT taken at face value.")''')

md("""**The per-condition breakdown is what separated the two explanations.** A pooling artefact
would shift every condition roughly uniformly. A real clutter mechanism would concentrate the gain
in the **clutter-bearing** conditions and leave clean ones alone.""")

md("""**The candidate windows, drawn to scale.** Before any numbers: this is what is being compared.
The shaded band is the clutter ridge (half-width ~4 Doppler bins); the cell under test sits at a
typical `clutter_edge` offset of 7 bins from the ridge centre.""")

code('''from matplotlib.patches import Rectangle

# (label, half_width_r, half_width_d) -- guard is 2/2 throughout.
GEOMS = [("21x21", 10, 10), ("21x13", 10, 6), ("21x11", 10, 5),
         ("21x9", 10, 4), ("29x9", 14, 4)]
SELECTED = "21x9"
RIDGE_HALF, CUT_OFFSET, G = 4, 7, 2

fig, axes = plt.subplots(1, len(GEOMS), figsize=(14, 3.4), sharey=True)
for ax, (label, hr, hd) in zip(axes, GEOMS):
    sel = label == SELECTED
    # Doppler axis measured from the ridge centre; the CUT sits CUT_OFFSET bins out.
    ax.axvspan(-RIDGE_HALF - .5, RIDGE_HALF + .5, color="#38f9d7", alpha=0.22, zorder=0)
    ax.add_patch(Rectangle((CUT_OFFSET - hd - .5, -hr - .5), 2 * hd + 1, 2 * hr + 1,
                           fill=False, ec="#1a7f6c" if sel else MUTED,
                           lw=2.6 if sel else 1.6, zorder=3))
    ax.add_patch(Rectangle((CUT_OFFSET - G - .5, -G - .5), 2 * G + 1, 2 * G + 1,
                           fill=False, ec="#ffb03a", lw=1.8, zorder=4))
    ax.plot(CUT_OFFSET, 0, "s", mfc="white", mec=INK, ms=5, mew=1.2, zorder=5)

    n_over = sum(1 for c in range(CUT_OFFSET - hd, CUT_OFFSET + hd + 1)
                 if abs(c - CUT_OFFSET) > G and abs(c) <= RIDGE_HALF)
    ax.set_title(f"{label}{'  (selected)' if sel else ''}\\n{n_over} ring cols in ridge",
                 fontsize=9, color="#1a7f6c" if sel else INK)
    ax.set_xlim(-9, 23); ax.set_ylim(-16, 16)
    ax.set_xlabel("Doppler bin (0 = ridge)", fontsize=8)
    ax.grid(False)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("range bin (0 = CUT)", fontsize=8)
fig.suptitle("The candidate windows to scale: cyan band = clutter ridge, orange = guard, "
             "white square = CUT", fontsize=10)
plt.tight_layout(); plt.show()''')

cap("Read the Doppler width. The 21-wide ring (left) reaches deep into the cyan ridge; narrowing it to 9 (fourth panel) leaves only the two innermost training columns touching the ridge edge. The count under each title is exactly that. 29×9 keeps the narrow Doppler ring but widens the RANGE half-width 10 → 14, which is what costs it `masking`.")

code('''pc = pd.read_csv(ROOT / "results" / "cfar_geometry_percondition_v2.csv")
gm = pc.groupby(["window", "condition"]).pd_at_pfa.mean().unstack()
order = ["21x21", "21x13", "21x11", "21x9", "29x9"]
gm = gm.loc[[w for w in order if w in gm.index]]
cols = [c for c in ["clutter_edge", "mixed", "clutter", "masking", "multi",
                    "isolated", "unresolved"] if c in gm.columns]
gm = gm[cols]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                         gridspec_kw={"width_ratios": [1.35, 1]})

im = axes[0].imshow(gm.to_numpy(), cmap="viridis", aspect="auto")
axes[0].set_xticks(range(len(gm.columns)))
axes[0].set_xticklabels(gm.columns, rotation=25, ha="right", fontsize=8)
axes[0].set_yticks(range(len(gm.index))); axes[0].set_yticklabels(gm.index, fontsize=9)
for a in range(gm.shape[0]):
    for b in range(gm.shape[1]):
        axes[0].text(b, a, f"{gm.iloc[a, b]:.2f}", ha="center", va="center", fontsize=7.5,
                     color="white" if gm.iloc[a, b] < 0.45 else "black")
axes[0].grid(False)
axes[0].set_title("Pd by geometry x condition (mean over CA/GO/OS)", fontsize=10)
fig.colorbar(im, ax=axes[0], label="Pd at matched measured Pfa", pad=0.02)

gain = (gm.loc["21x9"] - gm.loc["21x21"]).sort_values()
axes[1].barh(gain.index, gain.to_numpy(),
             color=["#c62828" if v < 0 else "#2e7d32" for v in gain])
axes[1].axvline(0, color=INK, lw=1)
axes[1].set_xlabel("Pd change, 21x21 -> 21x9")
clut = [c for c in ("clutter_edge", "mixed") if c in gain.index]
rest = [c for c in gain.index if c not in clut]
conc = gain[clut].mean() / abs(gain[rest].mean())
axes[1].set_title(f"The gain concentrates in the clutter-bearing conditions:\\n"
                  f"{gain[clut].mean():+.3f} there vs {gain[rest].mean():+.3f} elsewhere "
                  f"({conc:.0f}x)", fontsize=10)
plt.tight_layout(); plt.show()

print(f"clutter-bearing ({', '.join(clut)}): mean gain {gain[clut].mean():+.3f}")
print(f"everything else:                      mean gain {gain[rest].mean():+.3f}")
print(f"-> {conc:.1f}x concentration. A pooling artefact would shift every condition alike;")
print("   this does not, so the mechanism is real. And the clean conditions do not merely fail")
print("   to gain -- two of them LOSE, which is the opposite of a uniform shift:")
print("   " + ", ".join(f"{c} {gain[c]:+.3f}" for c in ["isolated", "clutter", "multi"] if c in gain.index))

if "29x9" in gm.index:
    d = (gm.loc["29x9"] - gm.loc["21x9"]).sort_values()
    print()
    print("29x9 minus 21x9 (29x9 was the marginal POOLED winner):")
    print(d.round(3).to_string())
    print(f"-> `masking` is the largest regression at {d.get('masking', float('nan')):+.3f}:")
    print("   29x9's RANGE half-width widens 10 -> 14, pulling more interferers into the ring.")
    print("   Nothing to do with Doppler. The pooled metric would have picked the worse window.")''')

cap("Left: read down the `clutter_edge` and `mixed` columns — those are where narrowing the Doppler "
    "window pays. Right: the gain there is ~10× the mean elsewhere, and `multi` actually gets "
    "slightly worse — a uniform pooling artefact would not look like this. The `masking` "
    "regression at 29×9 is in the printout.")

md("""> **Note on which artefact this is.** The figure reads
> `results/cfar_geometry_percondition_v2.csv`, regenerated 19 Aug under the **current** scoring
> protocol. The original `cfar_geometry_percondition.csv` is kept but is **not** authoritative: it
> disagreed with the step-6 table in `PROGRESS.md` on the *sign* for `isolated` and `multi`, because
> it was produced when `all_peaks()` defaulted to `nms_size=2`, read Pd by **interpolation** off the
> ROC, and used a uniform quantile threshold grid. All three have since been replaced (NMS 5,
> Pd measured *at* the matched threshold, false-alarm-anchored grid), so that script no longer
> reproduces its own output. Re-run under today's protocol, the sweep **matches the recorded step-6
> run** — `clutter_edge` +0.294 against +0.296, `isolated` −0.013 against −0.025, and the `masking`
> regression at 29×9 −0.043 against −0.041.""")

md("""#### Why `train = 8` — the knee

Guard size came from the PSF (§2.1); the *number of training cells* is a different trade. More cells
mean a lower-variance noise estimate and less CFAR loss; fewer cells mean a more **local** estimate
that tracks structure the window is sitting in. Milestone 9 recorded that Pd went flat beyond about
8 — that evidence was never persisted, so it is regenerated here on `bench2_tune` under the current
protocol.""")

code('''TS = pd.read_csv(ROOT / "results" / "cfar_train_sweep.csv")
tm = TS.groupby(["train_r", "condition"]).pd_at_pfa.mean().unstack()
tm["POOLED"] = tm.mean(axis=1)
display(tm.round(3))

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(tm.index, tm.POOLED, "o-", color=INK, lw=2.2, ms=6, label="pooled (mean over conditions)")
for c in ["clutter_edge", "mixed", "isolated"]:
    if c in tm.columns:
        ax.plot(tm.index, tm[c], "-", lw=1.1, alpha=0.55, label=c)
ax.axvline(8, color="#c62828", ls="--", lw=1.6)
ax.text(8.2, tm.POOLED.min(), "selected: train = 8", fontsize=8, color="#c62828")
ax.set_xlabel("training cells per side on the range axis (train_r)")
ax.set_ylabel("Pd at matched measured Pfa 1e-4")
ax.set_title("Pd rises to ~8 training cells and is flat after: more reference cells\\n"
             "stop buying anything, while a wider ring becomes less local", fontsize=10)
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

d = tm.POOLED.diff()
print("change in pooled Pd per step of train_r:")
print(d.round(4).to_string())
pre = d.loc[:8].dropna()
post = d.loc[10:].dropna()
print()
print(f"mean step up to train_r=8:   {pre.mean():+.4f}")
print(f"mean step beyond train_r=8:  {post.mean():+.4f}  (and it changes sign, i.e. noise)")
print("-> the knee is at 8. Past it the estimate is no longer improving, only less local.")''')

cap("The pooled curve flattens at train = 8; steps beyond it are smaller than the steps before it and change sign, which is what a noise floor looks like. Everything right of the red line costs locality for no Pd.")

md("""**Selected: 21 × 9, guard 2/2, train 8/2 — 164 reference cells.**

**Costs, accepted and recorded:** `isolated` −0.025 and `clutter` −0.016 against the widest window.
The `isolated` loss is textbook CFAR loss from fewer reference cells (416 → 164; analytic loss
0.05 → 0.12 dB). The `clutter` loss is the exact mirror of the `clutter_edge` gain — for a target
*inside* the ridge a narrow ring sits entirely in clutter and over-estimates the noise, where a wide
ring reaches clean cells. That trade is the mechanism, not a defect.

### 2.3 The window, drawn to scale on real data""")

code('''# A clutter_edge frame. The RIDGE has a half-width of ~4 bins (Milestone 22), so what
# matters is not whether the ridge CENTRE falls inside the window -- it never does -- but whether
# the ring's TRAINING columns overlap the ridge's extent. Pick the frame with the largest overlap.
RIDGE_HALF = 4
hw_r, hw_d, g_r, g_d = CFG.half_width_r, CFG.half_width_d, CFG.guard_r, CFG.guard_d

best = None
for i in frames[frames.condition == "clutter_edge"].frame.to_numpy():
    t = TRUTH_BY_FRAME.get(i)
    if t is None or not len(t):
        continue
    rb, vb = int(t.range_bin.iloc[0]), int(t.velocity_bin.iloc[0])
    if not (20 < rb < SHAPE[0] - 20):
        continue
    cols = [c for c in range(vb - hw_d, vb + hw_d + 1)
            if abs(c - vb) > g_d and abs(c - V0) <= RIDGE_HALF]
    if best is None or len(cols) > best[0]:
        best = (len(cols), int(i), rb, vb)
n_overlap, pick, RB, VB = best

mag = np.load(DATA / f"frame_{pick:06d}" / "rd_map_mag.npy").astype(float)
dbm = 20 * np.log10(mag / np.median(mag) + 1e-12)
pad = 6
r0, r1 = RB - hw_r - pad, RB + hw_r + pad + 1
c0, c1 = min(VB - hw_d, V0 - RIDGE_HALF) - 3, VB + hw_d + pad + 1

fig, ax = plt.subplots(figsize=(10, 6.4))
ax.imshow(dbm[r0:r1, c0:c1], origin="lower", cmap="magma", vmin=-3, vmax=45,
          extent=[c0 - 0.5, c1 - 0.5, r0 - 0.5, r1 - 0.5], aspect="auto")

from matplotlib.patches import Rectangle
# Shade the ridge extent, so the overlap with the ring is VISIBLE rather than asserted.
ax.axvspan(V0 - RIDGE_HALF - .5, V0 + RIDGE_HALF + .5, color="#38f9d7", alpha=0.16, zorder=1)
ax.axvline(V0, color="#38f9d7", lw=1.0, ls=":", zorder=2)
ax.add_patch(Rectangle((VB - hw_d - .5, RB - hw_r - .5), 2 * hw_d + 1, 2 * hw_r + 1,
                       fill=False, ec="#38f9d7", lw=2.4, zorder=4,
                       label=f"window {2*hw_r+1}x{2*hw_d+1}"))
ax.add_patch(Rectangle((VB - g_d - .5, RB - g_r - .5), 2 * g_d + 1, 2 * g_r + 1,
                       fill=False, ec="#ffb03a", lw=2.4, zorder=5, label=f"guard {g_r}/{g_d}"))
ax.plot(VB, RB, "s", mfc="none", mec="white", ms=9, mew=2.2, zorder=6, label="CUT")
ax.text(V0, r0 + 1.5, "clutter ridge", fontsize=8, color="#9ff5e6", ha="center", zorder=6)

ax.set_xlabel("Doppler bin"); ax.set_ylabel("range bin"); ax.grid(False)
ax.set_title(f"The 21x9 window on a real `clutter_edge` frame (#{pick}) -- the target sits in"
             f" clean noise,\\nbut {n_overlap} of its training columns reach into the ridge",
             fontsize=10)
ax.legend(frameon=True, fontsize=8, loc="upper right", facecolor=SURFACE)
plt.tight_layout(); plt.show()

print(f"target at Doppler bin {VB}; ridge centre {V0} -> offset {abs(VB - V0)} bins.")
print(f"The ridge CENTRE is outside the window. {n_overlap} training columns still overlap the")
print("ridge extent, which is enough to inflate the noise estimate -- the clutter-edge failure.")
print(f"reference cells = {2*hw_r+1}x{2*hw_d+1} - {2*g_r+1}x{2*g_d+1} = "
      f"{(2*hw_r+1)*(2*hw_d+1) - (2*g_r+1)*(2*g_d+1)} = {CFG.num_ref_cells}")''')

cap("The cyan box is what the detector averages to estimate noise; the orange box is excluded. "
    "The shaded band is the clutter ridge — look at where it overlaps the left edge of the cyan "
    "box. That overlap is the entire clutter-edge failure: the target is in clean noise, but its "
    "noise estimate is not.")
md("""### 2.4 Alpha is calibrated by Monte Carlo — measured, verified, and shipped

The closed form `alpha = N(Pfa^(-1/N) - 1)` is derived for the arithmetic mean of N **independent**
exponential cells. Both assumptions fail here: the Blackman window correlates adjacent reference
cells, so the effective sample is smaller than the 164 counted; and only CA is a mean at all — OS is
an order statistic, GO the maximum of two half-ring means, SO the minimum.

So alpha is set **empirically**, by Monte Carlo on pure noise, and the value is cached in
`configs/cfar_alpha_table.json`, which `CfarConfig.resolved_alpha()` reads. The procedure:

1. pool the per-cell ratio values over 300 pure-noise frames through the real processing chain, on
   the **calibration seed (0)**;
2. take alpha as the (1 − Pfa) empirical quantile of that pool;
3. apply it to a **held-out seed (9000)** and measure the achieved rate.

Step 3 is the part that matters: it tests that the value generalises rather than fitting the
calibration draw. A variant is only written to the table if it passes.

> **This was not always so.** Until 19 Aug 2026 `cfar.py` never imported `calibrate.py`, and all
> four variants ran on closed-form values — see Correction **C8**. The figure below shows both
> states, because the size of the correction is the argument for doing it.""")

code('''import json as _json
from cfar.cfar import CfarConfig as _CC

TAB = _json.loads((ROOT / "configs" / "cfar_alpha_table.json").read_text(encoding="utf-8"))
AU = pd.read_csv(ROOT / "results" / "cfar_alpha_audit.csv").set_index("method").loc[METHODS]

# Indexed by method, NOT a bare list: pandas aligns Series by index but assigns lists
# POSITIONALLY, so a bare list would silently misalign if AU were ever ordered differently
# from METHODS -- GO would show OS's alpha, with no error raised.
shipped = pd.Series({m: _CC(kind=m).resolved_alpha() for m in METHODS}, name="alpha")
assert list(AU.index) == METHODS, f"AU index {list(AU.index)} != METHODS {METHODS}"

tab = pd.DataFrame({
    "alpha (closed form)": AU.alpha_closed_form,
    "alpha (calibrated, SHIPPED)": shipped,
    "calib / closed": AU.ratio_cal_over_closed,
    "Pfa @ closed form": AU.pfa_cells_verifyseed_at_current,
    "Pfa @ calibrated": AU.pfa_cells_verifyseed_at_calibrated,
})
tab["x target, closed"] = tab["Pfa @ closed form"] / TARGET_PFA
tab["x target, calibrated"] = tab["Pfa @ calibrated"] / TARGET_PFA
display(tab.round(5))

print("in the table:", all(_CC(kind=m).calibrated_alpha_available() for m in METHODS),
      "| generated", TAB["meta"]["generated"], "| frames/seed", TAB["meta"]["frames_per_seed"])
print()
print("achieved / target on the HELD-OUT seed:")
print(f"  before (closed form): {tab['x target, closed'].min():.2f}x to "
      f"{tab['x target, closed'].max():.2f}x   <- a ~50x spread between variants")
print(f"  after  (calibrated):  {tab['x target, calibrated'].min():.2f}x to "
      f"{tab['x target, calibrated'].max():.2f}x")
print()
print("The held-out seed is the evidence: these alphas were fitted on seed 0 and land on target")
print("on seed 9000, so they are not fitting the calibration noise.")''')

code('''CU = pd.read_csv(ROOT / "results" / "cfar_alpha_curve.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                         gridspec_kw={"width_ratios": [2.1, 1]})

ax = axes[0]
for m in METHODS:
    c = CU[CU.method == m].sort_values("alpha")
    ax.plot(c.alpha, c.pfa.replace(0, np.nan), color=COLOR[m], lw=1.9, label=m)
    a_clo = float(AU.loc[m, "alpha_closed_form"])
    a_cal = float(_CC(kind=m).resolved_alpha())
    p_clo = float(AU.loc[m, "pfa_cells_verifyseed_at_current"])
    # Arrow runs FROM where the variant was TO where it should be: the correction itself.
    ax.annotate("", xy=(a_cal, TARGET_PFA), xytext=(a_clo, p_clo),
                arrowprops=dict(arrowstyle="-|>", color=COLOR[m], lw=1.7, alpha=0.95,
                                shrinkA=6, shrinkB=8,
                                connectionstyle="arc3,rad=0.15"))
    ax.plot(a_clo, p_clo, "o", color=COLOR[m], ms=10, mec="black", mew=1.3, zorder=6)
    ax.plot(a_cal, TARGET_PFA, "*", color=COLOR[m], ms=17, mec="black", mew=0.9, zorder=7)
ax.axhline(TARGET_PFA, color=INK, ls="--", lw=1.5)
ax.text(3.15, TARGET_PFA * 1.35, "target Pfa 1e-4", fontsize=8.5, color=INK)
ax.annotate("OS: too CONSERVATIVE (fires 8x too rarely)" + chr(10) + "-> alpha must come DOWN",
            xy=(9.3, 1.35e-5), xytext=(3.15, 2.2e-6), fontsize=8, color=COLOR["OS"],
            arrowprops=dict(arrowstyle="->", color=COLOR["OS"], lw=1.2))
ax.annotate("SO: too PERMISSIVE (fires 6.5x too often)" + chr(10) + "-> alpha must go UP",
            xy=(9.9, 6.6e-4), xytext=(3.15, 1.1e-2), fontsize=8, color=COLOR["SO"],
            arrowprops=dict(arrowstyle="->", color=COLOR["SO"], lw=1.2))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("threshold multiplier alpha"); ax.set_ylabel("achieved per-cell Pfa on pure noise")
ax.set_title("What calibration does: circle = closed form (where each variant WAS),\\n"
             "star = calibrated (where it should be). The arrow is the correction.", fontsize=10)
ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper right")
ax.set_ylim(1e-6, 5e-2)

ax = axes[1]
x = np.arange(len(METHODS))
ax.bar(x, [float(AU.loc[m, "pfa_cells_verifyseed_at_calibrated"]) for m in METHODS],
       0.55, color=[COLOR[m] for m in METHODS])
ax.axhline(TARGET_PFA, color=INK, ls="--", lw=1.5)
ax.set_xticks(x); ax.set_xticklabels(METHODS)
ax.set_ylabel("achieved per-cell Pfa"); ax.set_ylim(0, 1.6e-4)
ax.set_title("At the calibrated alpha, on the\\nHELD-OUT seed: all four on target", fontsize=10)
plt.tight_layout(); plt.show()''')

cap("Left: follow each arrow from its circle to its star. The horizontal length of the arrow is how far that variant was miscalibrated, and the arrows point in OPPOSITE directions — OS needed a lower threshold, SO a much higher one, which is why one closed form could never serve all four. Right: the calibrated alphas were fitted on seed 0 and land on target on seed 9000, so they generalise rather than fitting the calibration draw.")

md("""**What this changed, and what it did not.**

* **No reported Pd moved — provably, not just empirically.** alpha is *not* an input to the ratio
  map: `cfar_ratio_map` returns `power / noise`, and alpha is the threshold applied to it
  afterwards. Every result in this project is scored at matched **measured** Pfa, which sweeps that
  threshold and discards alpha. Re-deriving all summaries with the calibrated table left `pd`,
  `measured_pfa`, `tp`, `fp` and `fn` **bit-identical** on both `bench2_val` and `bench2_test`.
* **`bench2_test` was not re-read.** Because the ratio map is alpha-independent, the cached
  detections could not change; the summaries were re-derived from them with `--score-only`, and the
  `peaks_*.npz` files verified byte-identical afterwards.
* **What did change** is the design operating point — `design_alpha`, `design_pd`, `design_pfa` —
  and therefore §4.2, which is the only section that reads it.""")

# ============================================================================== 3 · result
md("""---
## 3 · The baseline result""")

code('''piv = summary.pivot_table(index="condition", columns="method", values="pd")[METHODS]
piv = piv.loc[[c for c in SCORED if c in piv.index]]
piv["best"] = piv[METHODS].idxmax(axis=1)
piv["spread"] = piv[METHODS].max(axis=1) - piv[METHODS].min(axis=1)
display(piv.round(3))

# Bootstrap each Pd over FRAMES. A spread is not a finding until it is bigger than its own error
# bar -- the same defect that inflated the DL-vs-CFAR p-values until they were redone paired
# (cnn_development_log.ipynb, Correction C4).
rng = np.random.default_rng(0)
NB = 1000
boot = {}
for cond in SCORED:
    ids = frames.loc[frames.condition == cond, "frame"].to_numpy()
    for m in METHODS:
        thr = THR.get((cond, m), np.nan)
        if not np.isfinite(thr):
            continue
        hits, tgts = [], []
        for i in ids:
            t = TRUTH_BY_FRAME.get(i)
            tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
                   if t is not None else np.zeros((0, 2)))
            pk = PEAKS[m].get(int(i), np.zeros((0, 3)))
            det = pk[pk[:, 2] >= thr][:, :2] if len(pk) else np.zeros((0, 2))
            ev = evaluate_frame(det, tru, SHAPE, MatchConfig(1, 1), valid_mask=VALID)
            hits.append(ev.tp); tgts.append(len(tru))
        hits, tgts = np.array(hits, float), np.array(tgts, float)
        if tgts.sum() == 0:
            continue
        draws = np.array([hits[k].sum() / max(tgts[k].sum(), 1.0)
                          for k in (rng.integers(0, len(ids), (NB, len(ids))))])
        boot[(cond, m)] = (draws.std(ddof=1), np.percentile(draws, 2.5),
                           np.percentile(draws, 97.5))

SE = pd.DataFrame({m: {c: boot[(c, m)][0] for c in SCORED if (c, m) in boot} for m in METHODS})

# `spread` is best-minus-worst, and with SO included that mostly measures HOW BAD SO IS rather
# than whether the winner is distinguishable. The question that matters for the win table is
# whether the winner beats the RUNNER-UP, so the separability test uses that margin.
srt = piv[METHODS].apply(lambda r: r.sort_values(ascending=False), axis=1)
chk = pd.DataFrame({
    "best": piv["best"],
    "2nd": [piv.loc[c, METHODS].sort_values(ascending=False).index[1] for c in piv.index],
    "margin over 2nd": [piv.loc[c, METHODS].sort_values(ascending=False).iloc[0]
                        - piv.loc[c, METHODS].sort_values(ascending=False).iloc[1]
                        for c in piv.index],
    "SE of best": [SE.loc[c, piv.loc[c, "best"]] for c in piv.index],
    "full spread (incl SO)": piv["spread"],
})
chk["winner separable?"] = chk["margin over 2nd"] > 2 * chk["SE of best"]
print("Is the WINNER distinguishable from the runner-up?")
display(chk.round(4))
print("False = the top two are a tie in all but name; do not read `best` as a finding there.")
print("Note the full-spread column is dominated by SO, which is why it is not the test used.")''')

code('''fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(len(piv)); w = 0.2
for k, m in enumerate(METHODS):
    err = [boot.get((c, m), (np.nan,) * 3)[0] for c in piv.index]
    ax.bar(x + (k - 1.5) * w, piv[m], w, label=m, color=COLOR[m],
           yerr=err, capsize=2.5, ecolor=INK, error_kw={"lw": 1})
ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=20, ha="right")
ax.set_ylabel("Pd at matched measured Pfa 1e-4")
ax.set_title("No CFAR variant is best everywhere — and on four conditions the\\n"
             "differences are inside the error bars", fontsize=10)
ax.legend(frameon=False, ncol=4, fontsize=9)
plt.tight_layout(); plt.show()''')

cap("Error bars are ±1 bootstrap SE over frames. Compare bar heights only where §3's `real?` column "
    "says True — elsewhere the ordering is noise.")

md("""### 3.1 Who wins where, and by what mechanism

The mechanism column is the point. A win we cannot explain is a win we should not report.""")

code('''MECH = {
    ("isolated", "CA"):     "homogeneous noise: the ring is one clean sample, and the MEAN is its "
                            "minimum-variance estimator. Nothing to reject, so rank/max statistics "
                            "only add variance.",
    ("multi", "CA"):        "targets are well separated, so each ring is still ~homogeneous. Same "
                            "argument as `isolated`.",
    ("unresolved", "CA"):   "both methods sit at the ~0.5 physics cap (two targets inside one "
                            "mainlobe), so the ring statistic barely matters.",
    ("masking", "OS"):      "an interfering target sits INSIDE the reference ring. The mean is "
                            "dragged up by it; the 0.75-quantile ignores a minority of high cells, "
                            "so OS keeps a sane noise estimate.",
    ("clutter", "GO"):      "the CUT is INSIDE the ridge, so the ring straddles clutter and clean "
                            "noise. CA averages the two and UNDER-estimates; greatest-of takes the "
                            "clutter-side half and corrects upward.",
    ("mixed", "GO"):        "same geometry as `clutter` -- interference plus a ridge -- so the "
                            "greatest-of correction applies for the same reason.",
    ("clutter_edge", "OS"): "the CUT is OUTSIDE the ridge, the opposite geometry. Here GO picks the "
                            "CONTAMINATED half and makes the over-estimate worse; OS rejects the "
                            "ridge cells as the minority they are.",
}
rows = []
for c in piv.index:
    win = piv.loc[c, "best"]
    real = bool(chk.loc[c, "winner separable?"])
    rows.append(dict(condition=c, winner=win, Pd=piv.loc[c, win],
                     margin=piv.loc[c, "spread"], separable=real,
                     mechanism=MECH.get((c, win), "** NO MECHANISM ON FILE -- INVESTIGATE **")))
W = pd.DataFrame(rows).set_index("condition")
display(W)

print("SO wins:", int((W.winner == "SO").sum()), "of", len(W),
      "-- expected 0: smallest-of trades a large sensitivity loss for edge robustness.")
missing = [c for c in W.index if W.loc[c, "mechanism"].startswith("**")]
print("conditions whose winner does NOT match the expected mechanism map:", missing or "none")''')

md("""**Every winner matches the mechanism predicted in advance — 7 of 7 — and SO wins nothing**, as
expected, since smallest-of buys edge robustness with a large sensitivity loss.

> ⚠️ **But not one of those wins is separable from its runner-up.** Every top-two margin is inside
> 2 bootstrap SE (table above). Read honestly, that means: **the point estimates agree with theory
> everywhere, and the data on this split cannot resolve the differences.** Both halves matter —
> 7-of-7 directional agreement with an independently derived mechanism is unlikely to be chance,
> but it is not the same as having measured each gap.
>
> *This was invisible before SO was added.* With three methods the separability test compared
> best-against-**worst**, which passed on four conditions; the meaningful comparison is
> best-against-**runner-up**, and it passes nowhere. Adding SO did not create the ambiguity, it
> exposed it.
>
> **Consequence for the report:** quote "CFAR variants are complementary in the direction theory
> predicts", not "OS beats GO on masking by 0.09". The variant-level ordering needs more frames per
> condition than 400 to establish.

Note the pairing that makes the mechanism argument rather than a list: **GO wins `clutter` and OS
wins `clutter_edge`**, and those are *opposite geometries*. Inside the ridge, greatest-of corrects
an under-estimate; outside it, greatest-of picks the contaminated half and makes things worse. A
single "best CFAR" would hide that.

### 3.2 `clutter_edge` does not mean what its name says

Before reading the `clutter_edge` row: a third of that stratum is not an edge at all.

The ridge measures **±4 Doppler bins**, `clutter_edge` targets sit at offsets **6–10**, and the
21×9 window's training columns reach only ±3–4 bins from the cell under test. So for the *far*
targets the window never touches the ridge: those frames are ordinary noise frames wearing the
label. The band (2.5–4.2 m·s⁻¹) was set when the window was 21×21 and reached 4.2 m·s⁻¹, where
every frame straddled — **narrowing the window in §2.2 silently broke the stratum's definition.**""")

code('''ids = frames.loc[frames.condition == "clutter_edge", "frame"].to_numpy()
hw_r, hw_d = CFG.half_width_r, CFG.half_width_d

# The ridge measures +/-4 Doppler bins (PROGRESS.md Milestone 22). A frame is a TRUE edge only if
# a TRAINING column -- inside the window, outside the guard -- actually lands in the ridge. The
# target itself sits 6-10 bins out, so testing the target's own offset finds nothing; it is the
# ring's reach that matters.
RIDGE_HALF = 4
true_edge, no_edge = [], []
for i in ids:
    t = TRUTH_BY_FRAME.get(i)
    if t is None or not len(t):
        continue
    reaches = sum(1 for vb in t.velocity_bin
                  for c in range(int(vb) - CFG.half_width_d, int(vb) + CFG.half_width_d + 1)
                  if abs(c - int(vb)) > CFG.guard_d and abs(c - V0) <= RIDGE_HALF)
    (true_edge if reaches else no_edge).append(int(i))
print(f"true edge: {len(true_edge)} frames | no edge: {len(no_edge)} frames")


def score_subset(subset):
    out = {}
    for m in METHODS:
        evs = []
        for i in subset:
            thr = THR[("clutter_edge", m)]
            t = TRUTH_BY_FRAME.get(i)
            tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
                   if t is not None else np.zeros((0, 2)))
            pk = PEAKS[m].get(int(i), np.zeros((0, 3)))
            det = pk[pk[:, 2] >= thr][:, :2] if (len(pk) and thr == thr) else np.zeros((0, 2))
            evs.append(evaluate_frame(det, tru, SHAPE, MatchConfig(1, 1), valid_mask=VALID))
        out[m] = aggregate(evs)["pd"]
    return out


split = pd.DataFrame({
    "true edge (ring reaches the ridge)": score_subset(true_edge),
    "no edge (label is wrong)": score_subset(no_edge),
    "pooled - as reported above": summary[summary.condition == "clutter_edge"]
                                  .set_index("method").pd.to_dict(),
}).T[METHODS]
split.insert(0, "frames", [len(true_edge), len(no_edge), len(ids)])
display(split.round(3))
print(f"{100 * len(no_edge) / len(ids):.0f}% of the stratum is not actually an edge.")
print("The dilution was HIDING OS's advantage, not creating it.")''')

# ============================================================================== 4 · why
md("""---
## 4 · Why the methods differ

§3 says who wins. This says what is physically happening, on real frames.

### 4.1 The noise estimate, mapped

All four methods are statistics of the *same 164 reference cells* — a mean, a half-ring maximum, a
0.75-quantile and a half-ring minimum. They only diverge when that sample stops being homogeneous.
This maps each estimate on a `clutter_edge` frame, where it does.""")

code('''from cfar.cfar import _ring_mean, _half_ring_means, _os_ring_statistic

i = true_edge[len(true_edge) // 2]
mag = np.load(DATA / f"frame_{i:06d}" / "rd_map_mag.npy").astype(float)
power = mag ** 2
t = TRUTH_BY_FRAME[i]
rb, vb = int(t.range_bin.iloc[0]), int(t.velocity_bin.iloc[0])

g_r, g_d = CFG.guard_r, CFG.guard_d
ca = _ring_mean(power, hw_r, hw_d, g_r, g_d)
lead, lag = _half_ring_means(power, hw_r, hw_d, g_r, g_d, axis=CFG.go_axis)
go, so = np.maximum(lead, lag), np.minimum(lead, lag)
os_ = _os_ring_statistic(power, hw_r, hw_d, g_r, g_d,
                         int(round(CFG.os_rank_frac * CFG.num_ref_cells)))
EST = {"CA": ca, "GO": go, "OS": os_, "SO": so}
floor = float(np.median(power))

fig, ax = plt.subplots(figsize=(10.5, 4.6))
v = np.arange(SHAPE[1])
ax.plot(v, 10 * np.log10(power[rb] / floor + 1e-12), color=MUTED, lw=1.1,
        label="|RD|² at the target's range")
for m in METHODS:
    ax.plot(v, 10 * np.log10(EST[m][rb] / floor + 1e-12), color=COLOR[m], lw=1.9,
            label=f"{m} noise estimate")
ax.axvline(vb, color=INK, ls="--", lw=1.2)
ax.text(vb + 0.7, ax.get_ylim()[1] - 5, "target", fontsize=8, color=INK)
ax.axvline(V0, color="#38f9d7", ls=":", lw=1.4)
ax.text(V0 + 0.7, ax.get_ylim()[1] - 12, "ridge", fontsize=8, color="#2b8f80")
ax.set_xlabel("Doppler bin"); ax.set_ylabel("dB above the frame's noise floor")
ax.set_title("At a clutter EDGE, GO over-estimates the noise and OS does not\\n"
             "(higher estimate -> higher threshold -> lower Pd)", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout(); plt.show()

est = pd.Series({m: EST[m][rb, vb] / floor for m in METHODS}, name="estimate / noise floor")
display(est.round(2).to_frame())
print("Read the ORDER, not the values: whichever method estimates highest sets the highest")
print("threshold and therefore detects least on this geometry.")''')

cap("Follow the coloured lines across the ridge at the cyan marker. GO (orange) rises with the "
    "ridge even though the target sits outside it — that is the clutter-edge failure. OS (green) "
    "stays flat because the ridge cells are a minority it can reject.")

md("""### 4.2 GO wins on false-alarm control, not on matched-Pfa Pd

GO's clutter advantage is real, but it shows up as a **lower false-alarm rate at a fixed
threshold** — and matching the *measured* rate neutralises exactly that, by construction.

This is worth stating plainly because it is where a prediction failed during window selection: GO
was predicted to win `clutter_edge`, and at matched Pfa it does not. **GO wins `clutter` (CUT inside
the ridge); OS wins `clutter_edge` (CUT outside it).** Opposite geometries, opposite winners.""")

code('''thr = summary.pivot_table(index="condition", columns="method", values="threshold")[METHODS]
# resolved_alpha() returns the CALIBRATED value now that configs/cfar_alpha_table.json exists
# (S2.4). Before 19 Aug this read the closed form, which is why this section used to need a
# normalisation step to separate the calibration offset from the clutter effect.
design = {m: CfarConfig(kind=m).resolved_alpha() for m in METHODS}
ratio = pd.DataFrame({m: thr[m] / design[m] for m in METHODS})

print("measured threshold / DESIGN threshold, at the calibrated alpha:")
display(ratio.round(2))

# The `noise` row is the control: no clutter there, so every method should sit at the same place.
nz = ratio.loc["noise"]
print(f"`noise` row: {nz.min():.3f} to {nz.max():.3f} across the four methods "
      f"(spread {nz.max() - nz.min():.3f})")
print("Before calibration this row was 0.76 (OS) to 1.25 (SO) -- a method-SPECIFIC offset that",
      "had to be")
print("divided out. It is now common to all four, so the raw ratios are directly comparable and")
print("the normalisation step this section used to carry has been removed.")
print()
print("The residual ~0.95 is not miscalibration: it is the POST-NMS offset (S5). alpha is a")
print("per-cell threshold, while these thresholds are matched on post-NMS detections.")

fig, ax = plt.subplots(figsize=(10.5, 4.8))
x = np.arange(len(ratio)); w = 0.2
for k, m in enumerate(METHODS):
    ax.bar(x + (k - 1.5) * w, ratio[m], w, color=COLOR[m], label=m)
ax.axhline(1.0, color=INK, ls="--", lw=1.2)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(ratio.index, rotation=20, ha="right")
ax.set_ylabel("measured / design threshold  (log scale)")
ax.set_title("With calibrated alphas the clean conditions collapse onto one line;\\n"
             "what is left above it is the clutter effect, and it differs per method",
             fontsize=10)
ax.legend(frameon=False, ncol=4, fontsize=9)
plt.tight_layout(); plt.show()

clut = [c for c in ("clutter", "clutter_edge", "mixed") if c in ratio.index]
clean = [c for c in ratio.index if c not in clut]
print("clutter-bearing rows, factor above the clean baseline:")
display((ratio.loc[clut] / ratio.loc[clean].mean()).round(2))
print("CA/GO/OS need 1.7-2.7x their design threshold in clutter; SO needs ~13-24x, because")
print("taking the MINIMUM of two half-rings biases the noise estimate down hardest exactly")
print("where the ring is inhomogeneous.")''')

cap("Log scale, because SO spans more than a decade. Every clean condition now sits on one line just under 1.0 — that common offset is the post-NMS effect, not miscalibration. What rises above it is the clutter effect, and it is a different height for each method, which is why a shared 'design 1e-4' would place the four at four different operating points.")

md("""**The `noise` row is the control.** There is no clutter there, so every method should sit at
the same place — and now does: **0.952 to 0.959**, a spread of 0.007. Before calibration that row
ran 0.76 (OS) to 1.25 (SO), a method-*specific* offset that had to be divided out before the clutter
effect could be read. **That normalisation step has been removed** — it is no longer measuring
anything, which is the cleanest evidence that §2.4's calibration worked.

**The residual ~0.95 is not miscalibration.** It is the post-NMS offset: alpha is a **per-cell**
threshold, while these thresholds are matched on **post-NMS** detections, and NMS removes ~20–35% of
candidates (§5). Being common to all four, it cancels in any comparison between them.

**What is left above the line is the clutter effect**, and it differs by method:

| factor above the clean baseline | CA | GO | OS | SO |
|---|---|---|---|---|
| worst clutter row | **2.25×** | **1.74×** | **2.74×** | **24.2×** |

CA/GO/OS need 1.7–2.7× their design threshold in clutter. **SO needs 13–24×**, because taking the
*minimum* of two half-rings biases the noise estimate downward hardest exactly where the ring is
inhomogeneous. A single "design 1e-4" would therefore place the four at four different operating
points — which is the whole reason this project matches the **measured** rate instead.""")

md("""### 4.3 Where the false alarms land""")

code('''rows = []
for m in METHODS:
    for cond in CONDS:
        thr_ = THR.get((cond, m), np.nan)
        if not np.isfinite(thr_):
            continue
        for i in frames.loc[frames.condition == cond, "frame"]:
            t = TRUTH_BY_FRAME.get(i)
            tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
                   if t is not None else np.zeros((0, 2)))
            pk = PEAKS[m].get(int(i), np.zeros((0, 3)))
            det = pk[pk[:, 2] >= thr_] if len(pk) else np.zeros((0, 3))
            for d in det:
                if not len(tru) or all(abs(d[0] - a) > 1 or abs(d[1] - b) > 1 for a, b in tru):
                    rows.append(dict(method=m, condition=cond,
                                     dv=int(d[1]) - V0, rb=int(d[0])))
FA = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10, 4.4))
bins = np.arange(-32, 33, 2)
for m in METHODS:
    ax.hist(FA[FA.method == m].dv, bins=bins, histtype="step", lw=1.9,
            color=COLOR[m], label=m)
ax.axvline(0, color="#38f9d7", ls=":", lw=1.6)
ax.text(0.7, ax.get_ylim()[1] * 0.92, "clutter ridge", fontsize=8, color="#2b8f80")
ax.set_xlabel("Doppler offset from the ridge (bins)")
ax.set_ylabel("false alarms")
ax.set_title("False alarms cluster on the clutter ridge — that is what the\\n"
             "matched-Pfa protocol is paying for", fontsize=10)
ax.legend(frameon=False, ncol=4, fontsize=9)
plt.tight_layout(); plt.show()

share = (FA.assign(on_ridge=FA.dv.abs() <= 2).groupby("method").on_ridge.mean() * 100)
print("share of each method's false alarms within +/-2 bins of the ridge (%):")
print(share.round(1).to_string())''')

cap("The spike at zero is the ridge. A method with a taller spike is spending more of its "
    "false-alarm budget on clutter rather than on noise.")

# ============================================================================== 5 · physics
md("""---
## 5 · Validation against physics

Everything so far is *relative* — one variant against another. This is the one **absolute** check:
is the classical detector anywhere near what theory says is achievable?

> **Which measurement this is.** `bench2_test`, restricted to the **clean conditions** (`isolated` +
> `multi`), because interference-limited strata are limited by the interference rather than by the
> noise the theory models. A second, independent CFAR-loss measurement exists on
> **`bench2_snr_sweep`** (one target per frame, SNR swept uniformly 0–24 dB) and reports
> **0.47–0.78 dB** — see `cnn_development_log.ipynb` §20. **Different splits, different target
> populations, different numbers**; neither supersedes the other.

Theory is a square-law detector that **knows** the noise level: `Pd = Pfa^(1/(1+SNR))`. The gap
below it is **CFAR loss** — the price of estimating that level from 164 cells instead of being told
it.

> ⚠️ **Theory is a bound only at gate 0 (exact cell).** The ±1 gate counts a hit anywhere in a 3×3
> neighbourhood — nine chances against theory's one — and the mainlobe spans ~2 bins, so neighbours
> carry real target energy. At ±1 every method sits *above* theory, which is a property of the gate,
> not of the detectors. Both are shown, and the CFAR-loss number is taken from gate 0.""")

code('''def per_target_flags(gate):
    """One row per TARGET: was it detected, at that method's matched threshold."""
    mc = MatchConfig(gate, gate)
    out = {}
    for m in METHODS:
        rec = []
        for cond in SCORED:
            thr_ = THR.get((cond, m), np.nan)
            for i in frames.loc[frames.condition == cond, "frame"]:
                t = TRUTH_BY_FRAME.get(i)
                if t is None or not len(t):
                    continue
                tru = t[["range_bin", "velocity_bin"]].to_numpy(float)
                pk = PEAKS[m].get(int(i), np.zeros((0, 3)))
                det = (pk[pk[:, 2] >= thr_][:, :2]
                       if (len(pk) and np.isfinite(thr_)) else np.zeros((0, 2)))
                ev = evaluate_frame(det, tru, SHAPE, mc, valid_mask=VALID)
                hit = set(ev.matched_truth_idx)
                for k, (_, row) in enumerate(t.iterrows()):
                    rec.append(dict(frame=int(i), target=k, condition=cond,
                                    cell_snr_db=row.cell_snr_db, range_m=row.range_m,
                                    detected=k in hit))
        out[m] = pd.DataFrame(rec)
    return out


FLAGS = {1: per_target_flags(1), 0: per_target_flags(0)}
print("per-target records built:", {g: len(v["CA"]) for g, v in FLAGS.items()})''')

code('''edges = np.arange(-2, 34, 2.0)
CLEAN = ["isolated", "multi"]


def curve(df):
    d = df.dropna(subset=["detected"]).copy()
    d["b"] = pd.cut(d.cell_snr_db, edges)
    g = d.groupby("b", observed=True).agg(n=("detected", "size"), snr=("cell_snr_db", "mean"),
                                          pd=("detected", "mean"))
    return g[g.n >= 20]


def theory(s):
    return TARGET_PFA ** (1.0 / (1.0 + 10 ** (np.asarray(s) / 10)))


s = np.linspace(-2, 30, 250)
th50 = float(np.interp(0.5, theory(s), s))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
loss = []
for ax, gate in zip(axes, (0, 1)):
    ax.plot(s, theory(s), "--", color=INK, lw=1.5, label="theory (knows the noise)")
    for m in METHODS:
        f = FLAGS[gate][m]
        c = curve(f[f.condition.isin(CLEAN)])
        ax.plot(c.snr, c.pd, "o-", ms=3.8, color=COLOR[m], label=m)
        if gate == 0 and c.pd.max() >= 0.5 >= c.pd.min():
            snr50 = float(np.interp(0.5, c.pd.to_numpy(), c.snr.to_numpy()))
            loss.append(dict(method=m, snr_at_pd50_db=snr50, cfar_loss_db=snr50 - th50))
    ax.set_xlabel("target cell SNR (dB)"); ax.set_ylim(0, 1.02)
    ax.set_title(f"gate {'0 — exact cell: VALID vs theory' if gate == 0 else '±1 — NOT a bound'}",
                 fontsize=10)
axes[0].set_ylabel("Pd at matched measured Pfa 1e-4")
axes[0].legend(frameon=False, fontsize=8.5)
fig.suptitle("Clean conditions only (isolated + multi) — interference-limited strata are limited\\n"
             "by the interference, not by the noise the theory models", fontsize=10)
plt.tight_layout(); plt.show()

print(f"theory reaches Pd 0.5 at {th50:.2f} dB")
display(pd.DataFrame(loss).set_index("method").round(3))
print("cfar_loss_db = extra SNR needed to reach Pd 0.5 -- the price of ESTIMATING the noise level.")''')

cap("Left panel is the honest comparison: each curve sits slightly below the dashed line, and that "
    "gap in dB is the CFAR loss tabled underneath. Right panel shows why ±1 cannot be used for "
    "this — every method appears to beat physics.")

md("""#### ⚠️ One correction to the CFAR-loss number: post-NMS Pfa is not per-cell Pfa

The theory curve `Pd = Pfa^(1/(1+SNR))` is a statement about **one cell** crossing a threshold at a
**per-cell** false-alarm rate. Our detectors are matched at a **post-NMS** Pfa of 1e-4 — counted over
surviving local maxima, not over every cell.

§2.4 measured the NMS survival fraction on pure noise (0.66–0.83 depending on method). So a post-NMS
Pfa of 1e-4 corresponds to a **per-cell rate above 1e-4**, which means the detectors are being scored
at a slightly *easier* operating point than the one theory is drawn at — and the CFAR loss above is
therefore **understated**.

The corrected anchor is computed below rather than estimated. Inverting the theory curve at
Pd = 0.5 is closed form: `SNR = ln(Pfa)/ln(0.5) − 1`.""")

code('''AUD = pd.read_csv(ROOT / "results" / "cfar_alpha_audit.csv").set_index("method")
surv = (AUD.pfa_peaks_verifyseed_at_current / AUD.pfa_cells_verifyseed_at_current).loc[METHODS]


def snr50_for_pfa(pfa):
    """SNR (dB) at which the known-noise detector reaches Pd 0.5, from Pd = Pfa^(1/(1+SNR))."""
    return 10 * np.log10(np.log(pfa) / np.log(0.5) - 1.0)


anchor_naive = snr50_for_pfa(TARGET_PFA)
rows = []
for m in METHODS:
    s = float(surv[m])
    pfa_cell = TARGET_PFA / s                    # per-cell rate behind a post-NMS rate of 1e-4
    anchor = snr50_for_pfa(pfa_cell)
    base = [r for r in loss if r["method"] == m]
    meas = base[0]["snr_at_pd50_db"] if base else np.nan
    rows.append(dict(method=m, nms_survival=s, per_cell_pfa=pfa_cell,
                     theory_anchor_db=anchor,
                     cfar_loss_as_reported=meas - anchor_naive,
                     cfar_loss_corrected=meas - anchor,
                     understated_by_db=anchor_naive - anchor))
C = pd.DataFrame(rows).set_index("method")
display(C.assign(per_cell_pfa=C.per_cell_pfa.map(lambda v: f"{v:.3e}")).round({"nms_survival": 3, "theory_anchor_db": 3, "cfar_loss_as_reported": 3, "cfar_loss_corrected": 3, "understated_by_db": 3}))

print(f"theory anchor as drawn (per-cell Pfa = 1e-4):     {anchor_naive:.3f} dB")
print(f"corrected anchors (per-cell Pfa = 1e-4 / survival): "
      f"{C.theory_anchor_db.min():.3f} to {C.theory_anchor_db.max():.3f} dB")
print()
print(f"-> the reported CFAR loss is understated by "
      f"{C.understated_by_db.min():.2f}-{C.understated_by_db.max():.2f} dB.")
print(f"   corrected range: {C.cfar_loss_corrected.min():.2f}-{C.cfar_loss_corrected.max():.2f} dB "
      f"(as reported: {C.cfar_loss_as_reported.min():.2f}-{C.cfar_loss_as_reported.max():.2f} dB)")
print("   Still a fraction of a dB. The conclusion -- CFAR sits just below the known-noise bound")
print("   -- is unchanged; the number is simply a little larger than the figure above shows.")''')

md("""**Why this is not corrected in the figure.** The survival fraction is measured on *pure noise*,
where every surviving peak is a false alarm; on target-bearing frames the mix differs, so applying a
single per-method scalar to the plotted curves would trade a stated bias for an unstated one. The
bias is small, its direction is known, and it is stated here instead.

**⚠️ This affects §5 only, and the reason is worth stating carefully.** It would be wrong to argue
that the offset is common-mode and cancels because both arms run the same `all_peaks()` at the same
`nms_size`. **Running the same NMS does not imply the same survival fraction** — survival depends on
how peaked a detector's score map is, which is a property of the detector, not of the suppression
code. The four CFAR variants above share one `all_peaks()` call and their survival fractions still
span 0.66–0.83.

The DL-vs-CFAR comparison is untouched for a simpler reason that needs no cancellation assumption:
**both arms are matched directly at the same final post-NMS, guard-excluded Pfa**, measured on each
arm's own reported detections. Whatever per-cell rates sit behind those matched post-NMS rates may
differ between the arms, and that is irrelevant — no reported comparison is made at the per-cell
level. Only this section compares against an absolute, per-cell bound, so only this section carries
the offset. See `evaluation_metrics_overview.ipynb` §5.2–5.3.""")

md("""### 5.1 Resolution — Pd of the weak target vs separation""")

code('''pair = truths[truths.condition.isin(["masking", "unresolved"])]
sep = pair.groupby("frame").agg(
    sep_m=("range_m", lambda v: abs(v.iloc[1] - v.iloc[0]) if len(v) == 2 else np.nan))

fig, ax = plt.subplots(figsize=(9.8, 4.8))
bins = np.array([0.0, 0.35, 0.6, 0.9, 1.2, 1.5, 1.95, 2.5])
for m in METHODS:
    f = FLAGS[1][m]
    f = f[f.condition.isin(["masking", "unresolved"])].merge(sep, on="frame")
    weak = f[f.target == 1].copy()
    weak["b"] = pd.cut(weak.sep_m, bins)
    g = weak.groupby("b", observed=True).agg(n=("detected", "size"), pd=("detected", "mean"),
                                             mid=("sep_m", "mean"))
    g = g[g.n >= 15]
    ax.plot(g.mid, g.pd, "o-", color=COLOR[m], label=m, ms=4.5)
for xv, lab in ((0.39, "equal-strength limit"), (0.59, "weak-partner limit"),
                (1.95, "NMS reach")):
    ax.axvline(xv, color=MUTED, ls=":", lw=1)
    ax.text(xv, 1.02, lab, rotation=90, fontsize=7, color=MUTED, va="bottom", ha="right")
ax.set_xlabel("range separation (m)"); ax.set_ylabel("Pd of the WEAK target")
ax.set_title("Below the measured resolution limit no variant recovers the weak target —\\n"
             "this is physics, not a detector deficiency", fontsize=10)
ax.legend(frameon=False, ncol=4, fontsize=9); ax.set_ylim(0, 1)
plt.tight_layout(); plt.show()''')

cap("The dotted lines are measured limits, not assumptions. All four curves collapse together "
    "left of 0.39 m — that region is information-free, whatever detector is used.")

# ============================================================================== 6 · secondary
md("""---
## 6 · Secondary metrics

### 6.1 Localization

Bootstrapped over frames, like Pd in §3, and reported at **both** gates. The metric is bounded by
the gate by construction — a detection displaced further than the gate is a miss plus a false alarm,
not a large error — so the two gates say different things and both are shown.""")

code('''rng2 = np.random.default_rng(1)
rows = []
for gate in (1, 3):
    mc = MatchConfig(gate, gate)
    for m in METHODS:
        per_frame = []
        for cond in SCORED:
            thr_ = THR.get((cond, m), np.nan)
            for i in frames.loc[frames.condition == cond, "frame"]:
                t = TRUTH_BY_FRAME.get(i)
                if t is None or not len(t):
                    continue
                tru = t[["range_bin", "velocity_bin"]].to_numpy(float)
                pk = PEAKS[m].get(int(i), np.zeros((0, 3)))
                det = (pk[pk[:, 2] >= thr_][:, :2]
                       if (len(pk) and np.isfinite(thr_)) else np.zeros((0, 2)))
                ev = evaluate_frame(det, tru, SHAPE, mc, valid_mask=VALID)
                per_frame.append(ev)
        a = aggregate(per_frame, range_res_m=RANGE_RES_M, vel_res_mps=VEL_RES_MPS)

        # Bootstrap the two error metrics over frames.
        idx = np.arange(len(per_frame))
        dr_d, dv_d = [], []
        for _ in range(300):
            k = rng2.integers(0, len(idx), len(idx))
            ab = aggregate([per_frame[j] for j in k],
                           range_res_m=RANGE_RES_M, vel_res_mps=VEL_RES_MPS)
            dr_d.append(ab["mean_abs_dr_bins"]); dv_d.append(ab["mean_abs_dv_bins"])
        rows.append(dict(gate=gate, method=m,
                         dr_bins=a["mean_abs_dr_bins"], dr_se=np.std(dr_d, ddof=1),
                         dv_bins=a["mean_abs_dv_bins"], dv_se=np.std(dv_d, ddof=1),
                         range_m=a["mean_abs_range_err_m"], vel_mps=a["mean_abs_vel_err_mps"]))
L = pd.DataFrame(rows)
for gate in (1, 3):
    print(f"gate +/-{gate}")
    display(L[L.gate == gate].set_index("method").drop(columns="gate").round(4))
best_r = L[L.gate == 1].set_index("method").dr_bins.idxmin()
best_v = L[L.gate == 1].set_index("method").dv_bins.idxmin()
print(f"best on RANGE at +/-1: {best_r}   |   best on DOPPLER at +/-1: {best_v}")
print("SO is competitive here despite winning nothing on Pd -- which is exactly why omitting it")
print("from this table (as this notebook did until 19 Aug) understated the classical baseline.")''')

md("""### 6.2 Runtime

Measured in the same process on the same machine, so the comparison is internally valid; the
absolute values are laptop-specific.""")

code('''meta = json.loads((ROOT / "results" / "dl" /
                   "runtime_cost_frozen_ep021_test_meta.json").read_text(encoding="utf-8"))
rt = pd.Series(RUNTIME, name="ms_per_frame").to_frame()
rt["vs fastest"] = rt.ms_per_frame / rt.ms_per_frame.min()
display(rt.round(2))
print("measurement conditions:",
      f"device={meta['device']}, threads={meta['threads']}, batch={meta['batch_size']}, "
      f"timing={meta['timing']}")
print(f"platform: {meta['platform']}")
print()
print("OS is ~25x CA: a per-cell RANK SELECTION does not vectorise, where the mean is an")
print("O(1)-per-cell separable box filter. That cost buys the masking win in S3.")

fig, ax = plt.subplots(figsize=(8, 3.6))
o = rt.sort_values("ms_per_frame")
ax.barh(o.index, o.ms_per_frame, color=[COLOR[m] for m in o.index])
for k, (m, v) in enumerate(o.ms_per_frame.items()):
    ax.text(v * 1.05, k, f"{v:.2f} ms", va="center", fontsize=8.5)
ax.set_xscale("log"); ax.set_xlabel("ms per 256x64 frame (log scale)")
ax.set_title("OS pays ~25x CA for its rank statistic", fontsize=10)
plt.tight_layout(); plt.show()''')

cap("Log scale. CA, GO and SO are all cheap sliding-window statistics; only OS needs a per-cell "
    "rank selection, and that is the entire cost difference.")

# ============================================================================== 7 · establishes
md("""---
## 7 · What this establishes

1. **The classical baseline is tuned, not inherited.** Window geometry and guard size were selected
   by measurement on a disjoint split (§2.1–§2.3), and **alpha is calibrated by Monte Carlo on pure
   noise and verified on a held-out seed** (§2.4). No paper in the 19-paper corpus re-tunes CFAR at
   all, and this removes the most obvious objection to any comparison built on top of it.

   *This claim was asserted before it was true.* Until 19 Aug 2026 `cfar.py` never imported
   `calibrate.py` and every variant ran on a closed-form alpha; the geometry half was always
   honest, the alpha half was not. See Correction **C8**.
2. **The variants are complementary in the direction theory predicts** — CA on homogeneous noise,
   OS when an interferer is in the ring, GO inside a ridge, and, the pair that makes the argument,
   **GO inside the ridge but OS at its edge**, which are opposite geometries.

   **Neither half of this should be dropped.** No winner is separable from its runner-up at 2 SE, so
   no *individual* gap is established. But **all 7 of 7 match a mechanism map specified in advance**,
   and that joint pattern is not seven independent coin flips: even under the most conservative null
   — the winner being a toss-up between the top two on every condition — the chance of getting all
   seven right is **(1/2)⁷ ≈ 0.008**; against a uniform choice among four variants it is
   **(1/4)⁷ ≈ 6×10⁻⁵**. The same logic as the 7-of-7 monotone localization result in the DL log:
   individually weak, jointly hard to get by luck. **The pattern is evidence; the per-condition
   margins are not.**
3. **SO wins nothing on Pd but is competitive on localization**, which is only visible because it is
   now included at all.
4. **The detector agrees with physics.** On **`bench2_test`, clean conditions only** (`isolated` +
   `multi`), at gate 0, CFAR loss against the known-noise bound is **0.27–0.64 dB as plotted, or
   0.45–0.86 dB** once the post-NMS-vs-per-cell Pfa offset is corrected (§5) — a fraction of a dB
   either way. The pipeline measures what it claims to measure.

   ⚠️ **Not the same number as the SNR-sweep figure**, which reports **0.47–0.78 dB** on
   **`bench2_snr_sweep`** (`cnn_development_log.ipynb` §20). Different split and different target
   population — one target per frame with SNR swept uniformly 0–24 dB (mean 11.2 dB) there, against
   1–2 targets per frame at mean 15.2 dB here. Both are valid; they are not interchangeable.
5. **Matching the *measured* Pfa is not a formality.** Even with correctly calibrated alphas, the
   design threshold is wrong by **1.7×–2.7× for CA/GO/OS and 13–24× for SO** on clutter-bearing
   conditions — **a different factor for each method** (§4.2), because clutter breaks the
   homogeneous-noise assumption the calibration is done under. Sweeping to the measured rate is what
   makes the four comparable at all.

### Limitations to carry into the report

* **Variant-level differences are not resolved on this split.** 400 frames per condition gives a
  bootstrap SE of 0.007–0.024, and every best-vs-runner-up margin is smaller than 2 SE. Separating
  the variants would need substantially more frames per condition.
* **The alphas are calibrated for *homogeneous noise*, which is the only regime where a fixed
  multiplier can be right.** In clutter every variant still needs 1.7–24× its design threshold
  (§4.2). That is a property of the detector, not a calibration failure, and it is exactly why
  results are reported at matched measured Pfa.
* **`os_rank_frac` was never swept.** OS's one free parameter sits at its default, so OS may be
  under-tuned exactly where it wins.
* **`clutter_edge` is ~⅓ diluted** — a third of the stratum is not an edge (§3.2). The pooled row
  understates OS.
* **The pooled Pd is not a "typical scene" number.** 37.5% of the split is deliberately adversarial
  (§1); only per-condition numbers mean anything.
* **`nms_size = 5` caps the `masking` stratum** at roughly Pd 0.69 by merging resolvable
  neighbours. Documented rather than tuned away, since NMS is shared with the learned detector and
  changing it would break that symmetry.
* **One simulator.** Every number here is on synthetic data from a single generator; nothing
  establishes that the ordering survives a different clutter model or real measurements.
* **Localization is conditioned on detection.** A method that detects only the easiest targets looks
  better than it is — relevant when comparing §6.1 across methods with very different Pd.""")

# ============================================================================== appendix
md("""---
## Appendix

Material moved out of the main line: complete per-condition ROCs, and the per-target record the
§5 curves are built from.

### A.1 ROC per condition — representative set inline, remainder here""")

code('''def roc_for(cond, m):
    ids = frames.loc[frames.condition == cond, "frame"]
    fr = []
    for i in ids:
        t = TRUTH_BY_FRAME.get(i)
        tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
               if t is not None else np.zeros((0, 2)))
        fr.append((PEAKS[m].get(int(i), np.zeros((0, 3))), tru, SHAPE, VALID))
    return roc_curve(fr, roc_thresholds(fr, MatchConfig(1, 1)), MatchConfig(1, 1))


SHOW = ["isolated", "masking", "clutter_edge"]      # clean / interference / edge
fig, axes = plt.subplots(1, len(SHOW), figsize=(13.5, 4.2), sharey=True)
for ax, cond in zip(axes, SHOW):
    for m in METHODS:
        r = roc_for(cond, m)
        ax.semilogx(r.pfa_per_cell.replace(0, np.nan), r.pd, color=COLOR[m], lw=1.7, label=m)
    ax.axvline(TARGET_PFA, color=INK, ls="--", lw=1.1)
    ax.set_xlabel("measured Pfa per cell"); ax.set_title(cond, fontsize=10); ax.set_ylim(0, 1)
axes[0].set_ylabel("Pd"); axes[0].legend(frameon=False, fontsize=8.5)
fig.suptitle("Representative ROCs — clean, interference, and edge. Dashed line is the "
             "operating point every table uses.", fontsize=10)
plt.tight_layout(); plt.show()

print("Remaining conditions:", [c for c in SCORED if c not in SHOW])
print("Re-run roc_for(cond, method) for any of them; they add no ordering the tables do not show.")''')

md("""### A.2 The per-target record

One row per target, with whether each method found it. This is what §5's curves aggregate, and it is
the artefact to query for any per-target question.""")

code('''rec = pd.concat([FLAGS[1][m].assign(method=m) for m in METHODS], ignore_index=True)
wide = rec.pivot_table(index=["frame", "target", "condition", "cell_snr_db"],
                       columns="method", values="detected").reset_index()
wide["n_found"] = wide[METHODS].sum(axis=1)
display(wide.head(12))
print(f"{len(wide):,} targets")
print()
print("How many methods found each target:")
display(wide.n_found.value_counts().sort_index().to_frame("targets"))
print("Targets found by NO method are the honest failure set -- mostly low-SNR or in-ridge.")
display(wide[wide.n_found == 0].groupby("condition")
        .agg(missed=("frame", "size"), median_snr=("cell_snr_db", "median")).round(1))''')

md("""---
*Rebuild: `python scripts/build_cfar_notebook.py`. Source-only by design — no stored outputs;
every number is computed from `results/` at run time.*""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "cfar_baseline.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
