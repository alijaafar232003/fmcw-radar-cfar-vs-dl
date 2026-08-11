"""Build notebooks/cfar_baseline.ipynb. Idempotent; re-run after scripts/cfar_test_baseline.py."""
import json
import pathlib

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    CELLS.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": t})


# ----------------------------------------------------------------------------------- 0. header
md("""# CFAR Baseline — `bench2_test`

**The classical baseline, measured once on the full test split (3,200 frames).**

This is *pre-registered*: CFAR has no learnable parameters, so measuring it before the network
exists means the numbers cannot be unconsciously tuned against, and it exercises the whole
evaluation path while there is no network to blame if something breaks.

| | |
|---|---|
| Split | `bench2_test` — 3,200 frames, 8 conditions x 400, seed 94001 |
| Radar | 77 GHz, B = 768 MHz, Tc = 72.4 us, N = 512, M = 64 -> 256 x 64 |
| CFAR window | **21 x 9** (guard 2/2, train 8/2), 164 reference cells |
| NMS | 11 x 11 (`nms_size=5`), shared by every detector |
| Matching | one-to-one (Hungarian) inside a +/-1-bin gate |
| Operating point | **matched MEASURED Pfa = 1e-4**, per condition |

**Why matched *measured* Pfa.** Clutter drives the achieved false-alarm rate 60-125x above design,
by different factors per method. A comparison at "design 1e-4" would compare CA at 1.2e-2 against
GO at 8.7e-3 against OS at 3.6e-2 - three different operating points. See `PROGRESS.md` step 5.

Run top to bottom. Nothing here modifies data; everything reads cached detections from
`results/cfar_test/`, produced by `scripts/cfar_test_baseline.py`.""")

code('''import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig
from eval import MatchConfig, aggregate, evaluate_frame, roc_curve
from dl.stratified_eval import _at_measured_pfa

RES = ROOT / "results" / "cfar_test"
METHODS = ["CA", "GO", "OS"]
TARGET_PFA = 1e-4
SHAPE = (256, 64)

CFG = CfarConfig()
VALID = np.zeros(SHAPE, dtype=bool)
VALID[CFG.half_width_r:-CFG.half_width_r, CFG.half_width_d:-CFG.half_width_d] = True

# House style, shared with the other figures in this project.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
COLOR = {"CA": "#2a78d6", "GO": "#eb6834", "OS": "#3d9970"}
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

print(f"{len(frames):,} frames | {len(truths):,} targets | conditions: {', '.join(CONDS)}")
print("detections cached:", {m: sum(len(v) for v in PEAKS[m].values()) for m in METHODS})''')

# ----------------------------------------------------------------------------- 1. composition
md("""## 1. What the test set is made of

The headline number averages over whatever mixture this is, so the mixture is stated first. Three
of the eight conditions (`masking`, `unresolved`, `clutter_edge`) are built to break a *specific*
detector — **37.5% of the split is deliberately adversarial**.

That is good experimental design and a bad headline: a single pooled Pd over this mixture is not a
"typical scene" number and must never be presented as one. Per-condition is the result; anything
pooled carries this table beside it. (Data-quality criterion 7 — see
`docs/data_quality_criteria.md`.)""")

code('''comp = (frames.groupby("condition")
        .agg(frames=("frame", "size"), targets=("num_targets", "sum"),
             mean_targets=("num_targets", "mean")))
comp["share_%"] = 100 * comp.frames / comp.frames.sum()
comp["adversarial"] = [c in ("masking", "unresolved", "clutter_edge") for c in comp.index]
display(comp.round(2))

snr = truths.groupby("condition").cell_snr_db.describe()[["count", "mean", "50%", "min", "max"]]
display(snr.round(1))''')

# --------------------------------------------------------------------------------- 2. headline
md("""## 2. The headline — Pd per condition, at matched measured Pfa

Each method's threshold is swept **on the condition being scored** until its achieved per-cell
false-alarm rate equals 1e-4; Pd is read off there. No method is allowed to buy Pd with false
alarms.""")

code('''piv = summary.pivot_table(index="condition", columns="method", values="pd")[METHODS]
piv["best"] = piv.idxmax(axis=1)
piv["spread"] = (piv[METHODS].max(axis=1) - piv[METHODS].min(axis=1))
display(piv.round(3))

fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(piv))
for k, m in enumerate(METHODS):
    ax.bar(x + (k - 1) * 0.26, piv[m], 0.26, label=m, color=COLOR[m])
ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=20, ha="right")
ax.set_ylabel("Pd at matched measured Pfa 1e-4")
ax.set_title("CFAR on bench2_test - per condition, 21x9 window")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()

print("Spread between best and worst method, per condition:")
print(piv["spread"].round(3).sort_values(ascending=False).to_string())
print("\\nA spread of ~0.01 at 400 frames/condition is inside sampling noise;")
print("only clearly separated rows support a claim about which method is better.")''')

# ------------------------------------------------------------------------------------- 3. ROC
md("""## 3. ROC per condition

The single Pd number above is one point on each of these curves. Plotting the whole curve shows
whether the ranking is a property of the detector or an artefact of the operating point — if curves
cross, "which method is best" has no answer without naming a Pfa.""")

code('''QUANTILES = np.clip(np.concatenate([
    np.linspace(0.0, 0.99, 40),
    1.0 - np.logspace(np.log10(0.01), np.log10(2e-5), 40)]), 0.0, 1.0)


def frames_for(cond, method):
    ids = frames.loc[frames.condition == cond, "frame"].to_numpy()
    out = []
    for i in ids:
        t = TRUTH_BY_FRAME.get(i)
        tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
               if t is not None else np.zeros((0, 2)))
        out.append((PEAKS[method].get(int(i), np.zeros((0, 3))), tru, SHAPE, VALID))
    return out


ROCS = {}
for cond in CONDS:
    for m in METHODS:
        fr = frames_for(cond, m)
        pool = np.concatenate([f[0][:, 2] for f in fr if len(f[0])])
        ROCS[(cond, m)] = roc_curve(fr, np.unique(np.quantile(pool, QUANTILES)), MatchConfig(1, 1))

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
for ax, cond in zip(axes.ravel(), CONDS):
    for m in METHODS:
        r = ROCS[(cond, m)]
        r = r[r.pfa_per_cell > 0]
        ax.semilogx(r.pfa_per_cell, r.pd, color=COLOR[m], label=m, lw=1.6)
    ax.axvline(TARGET_PFA, color=MUTED, ls=":", lw=1)
    ax.set_title(cond, fontsize=10)
    ax.set_ylim(0, 1)
axes[0, 0].legend(frameon=False, fontsize=9)
for ax in axes[1]:
    ax.set_xlabel("measured Pfa per cell")
for ax in axes[:, 0]:
    ax.set_ylabel("Pd")
fig.suptitle("ROC per condition (dotted line = the 1e-4 operating point)", y=1.0)
plt.tight_layout(); plt.show()''')

# ------------------------------------------------------------------------- 4. design vs measured
md("""## 4. Design Pfa is not measured Pfa — the reason this protocol exists

`alpha` is derived from theory assuming exponentially distributed cell power in *homogeneous*
noise. Clutter violates that assumption, so the threshold that theory says gives 1e-4 delivers
something else entirely. The column to read is **threshold**: how far each method had to be pushed
beyond its design value to actually achieve 1e-4 on that condition.""")

code('''thr = summary.pivot_table(index="condition", columns="method", values="threshold")[METHODS]
# GO's alpha is derived over ONE half-ring (74 cells on this window, not 164/2 = 82 -- the
# centre Doppler column belongs to neither half).
design = {m: (CfarConfig(kind=m).alpha_go() if m == "GO" else CfarConfig(kind=m).alpha())
          for m in METHODS}

show = thr.copy()
for m in METHODS:
    show[f"{m} xdesign"] = thr[m] / design[m]
display(show.round(2))

print("design alpha (homogeneous-noise theory):",
      {m: round(v, 2) for m, v in design.items()})
print("\\nA ratio near 1 means theory was right for that condition.")
print("Ratios far above 1 are the clutter conditions: the same alpha there would have")
print("produced far more false alarms than designed, which is exactly what makes a")
print("design-Pfa comparison misleading.")

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(thr))
for k, m in enumerate(METHODS):
    ax.bar(x + (k - 1) * 0.26, thr[m] / design[m], 0.26, label=m, color=COLOR[m])
ax.axhline(1.0, color=INK, lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(thr.index, rotation=20, ha="right")
ax.set_ylabel("threshold needed / design alpha")
ax.set_title("How far theory misses, per condition")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()''')

# --------------------------------------------------------- 4b. the metric decides the winner
md("""## 4b. Which method "wins" depends on the metric — and GO is the case in point

**GO wins the clutter edge. Just not on Pd at matched Pfa.**

GO takes the *greatest* of the two half-windows either side of the cell, split along Doppler. At a
clutter edge, one half sits in the ridge and the other in clean noise, so the max deliberately picks
the contaminated one. Verified directly on a synthetic edge (unit noise, ridge 20 dB up): three bins
outside the ridge, the noise estimates are **CA 26.1, GO 56.7, OS 4.8** — GO is the most
conservative by a factor of two, OS the least by a factor of five.

That conservatism is the *entire design goal* of GO: it stops the false-alarm rate exploding when
the window straddles an edge. It buys that with detections — a higher noise estimate is a higher
threshold is a lower Pd.

**Matching the measured Pfa erases exactly that benefit by construction.** Forcing every method to
the same false-alarm rate means GO's caution buys nothing it isn't already given for free, and only
its Pd cost remains. So OS leads the matched-Pfa table while GO leads the fixed-threshold one. Both
are true; they answer different questions.

Report both, and say which question each answers:

- *"Given a false-alarm budget you must hold, who detects most?"* → matched measured Pfa (§2).
- *"Deploy the textbook threshold and see what happens"* → fixed design alpha (below).""")

code('''fa_d = summary.pivot_table(index="condition", columns="method",
                           values="design_fa_per_frame")[METHODS]
pd_d = summary.pivot_table(index="condition", columns="method", values="design_pd")[METHODS]
pfa_d = summary.pivot_table(index="condition", columns="method", values="design_pfa")[METHODS]

print("FALSE ALARMS PER FRAME at the design threshold  (lower is better -- GO's metric)")
display(fa_d.round(2).style.highlight_min(axis=1, color="#cfe8cf"))

print("Pd at the same fixed threshold (NOT comparable across methods -- "
      "they sit at different false-alarm rates)")
display(pd_d.round(3))

print("Measured Pfa at the design threshold -- the drift that makes the above incomparable")
display(pfa_d.map(lambda v: f"{v:.1e}"))

# `noise` has no targets, so it appears in the false-alarm table but not the Pd one.
pd_matched = (summary.pivot_table(index="condition", columns="method", values="pd")
              .reindex(fa_d.index)[METHODS])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
x = np.arange(len(fa_d))
for k, m in enumerate(METHODS):
    axes[0].bar(x + (k - 1) * 0.26, fa_d[m].to_numpy(), 0.26, label=m, color=COLOR[m])
    axes[1].bar(x + (k - 1) * 0.26, pd_matched[m].to_numpy(), 0.26, label=m, color=COLOR[m])
axes[0].set_title("Fixed design threshold: false alarms/frame (GO wins)")
axes[0].set_ylabel("FA per frame")
axes[1].set_title("Matched measured Pfa: Pd (OS wins)")
axes[1].set_ylabel("Pd")
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(fa_d.index, rotation=25, ha="right")
    ax.legend(frameon=False)
plt.tight_layout(); plt.show()

print("Same detectors, same data, opposite rankings. This is a methodological result")
print("worth reporting, not an inconsistency to hide: it is why the operating point")
print("has to be stated before any CFAR comparison means anything.")''')

# --------------------------------------------------------------------------- 5. per-target flags
md("""## 5. Which targets were found — the per-target record

Everything below (Pd vs SNR, vs separation, vs clutter spikiness) needs to know *which individual
targets* each method detected at its matched operating point, not just how many. This builds that
record once.""")

code('''def detection_flags(method):
    """One row per target: was it matched, at this condition's matched threshold?"""
    rows = []
    thr_by_cond = summary[summary.method == method].set_index("condition").threshold

    for cond in CONDS:
        thr = thr_by_cond[cond]
        for i in frames.loc[frames.condition == cond, "frame"]:
            t = TRUTH_BY_FRAME.get(i)
            if t is None or not len(t):
                continue
            tru = t[["range_bin", "velocity_bin"]].to_numpy(float)
            pk = PEAKS[method].get(int(i), np.zeros((0, 3)))
            det = pk[pk[:, 2] >= thr][:, :2] if (len(pk) and thr == thr) else np.zeros((0, 2))

            ev = evaluate_frame(det, tru, SHAPE, MatchConfig(1, 1), valid_mask=VALID)
            hit = set(ev.matched_truth_idx)
            for k in range(len(t)):
                rows.append((i, k, k in hit))

    f = pd.DataFrame(rows, columns=["frame", "target", "detected"])
    return truths.merge(f, on=["frame", "target"], how="left")


FLAGS = {m: detection_flags(m) for m in METHODS}
print({m: f"{100 * FLAGS[m].detected.mean():.1f}% of all targets" for m in METHODS})''')

# ------------------------------------------------------------------------------ 6. Pd vs SNR
md("""## 6. Pd vs cell SNR, against theory

The validation figure. The dashed line is the textbook square-law detector at this Pfa,
`Pd = Pfa^(1/(1+SNR))` — the best any detector could do if it *knew* the noise level. The gap
between measured and theory is **CFAR loss**: the price of estimating that level from a finite
reference window instead of knowing it.

A curve that tracks the theoretical shape and sits slightly below it is the expected result. A
curve that crosses above it would indicate a scoring error, not a superior detector.""")

code('''edges = np.arange(-2, 34, 2.0)


def pd_vs_snr_curve(df):
    d = df.dropna(subset=["detected"]).copy()
    d["bin"] = pd.cut(d.cell_snr_db, edges)
    g = d.groupby("bin", observed=True).agg(n=("detected", "size"),
                                            snr=("cell_snr_db", "mean"),
                                            pd=("detected", "mean"))
    return g[g.n >= 20]


fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

s = np.linspace(-2, 32, 200)
theory = TARGET_PFA ** (1.0 / (1.0 + 10 ** (s / 10)))

for m in METHODS:
    c = pd_vs_snr_curve(FLAGS[m])
    axes[0].plot(c.snr, c.pd, "o-", ms=3.5, color=COLOR[m], label=m)
axes[0].plot(s, theory, "--", color=INK, lw=1.2, label="theory (known noise)")
axes[0].set_xlabel("target cell SNR (dB)"); axes[0].set_ylabel("Pd")
axes[0].set_title("All conditions pooled"); axes[0].legend(frameon=False); axes[0].set_ylim(0, 1)

clean = FLAGS["CA"][FLAGS["CA"].condition.isin(["isolated", "multi"])]
for m in METHODS:
    f = FLAGS[m]
    c = pd_vs_snr_curve(f[f.condition.isin(["isolated", "multi"])])
    axes[1].plot(c.snr, c.pd, "o-", ms=3.5, color=COLOR[m], label=m)
axes[1].plot(s, theory, "--", color=INK, lw=1.2, label="theory")
axes[1].set_xlabel("target cell SNR (dB)")
axes[1].set_title("Clean conditions only (isolated + multi)")
axes[1].legend(frameon=False); axes[1].set_ylim(0, 1)
plt.tight_layout(); plt.show()

print("The clean panel is the fair comparison against theory: interference-limited")
print("conditions are limited by the interference, not by the noise the theory models.")''')

# ------------------------------------------------------------------- 7. separation (masking)
md("""## 7. Pd vs target separation — the resolution ablation

`masking` sweeps the separation of a strong/weak pair from 0.40 m to 2.50 m, and `unresolved`
covers 0.05-0.35 m. Because separation was **swept per frame and recorded**, the resolution limit
emerges from the data rather than being asserted.

Two boundaries to look for: the measured resolution limit (~0.39 m equal-strength, ~0.59 m for a
weak partner) and the CFAR window's reach (1.95 m in range), beyond which the interferer is outside
the training ring.

This is the proposal's *close-target separation* ablation.""")

code('''pair = truths[truths.condition.isin(["masking", "unresolved"])].copy()
sep = (pair.groupby("frame")
       .agg(sep_m=("range_m", lambda v: abs(v.iloc[1] - v.iloc[0]) if len(v) == 2 else np.nan)))

fig, ax = plt.subplots(figsize=(9.5, 4.8))
bins = np.array([0.0, 0.35, 0.6, 0.9, 1.2, 1.5, 1.95, 2.5])

for m in METHODS:
    f = FLAGS[m]
    f = f[f.condition.isin(["masking", "unresolved"])].merge(sep, on="frame")
    weak = f[f.target == 1]                      # target 1 is the weak partner
    weak = weak.assign(b=pd.cut(weak.sep_m, bins))
    g = weak.groupby("b", observed=True).agg(n=("detected", "size"), pd=("detected", "mean"),
                                             mid=("sep_m", "mean"))
    g = g[g.n >= 15]
    ax.plot(g.mid, g.pd, "o-", color=COLOR[m], label=m, ms=4)

for xv, lab in ((0.39, "equal-strength limit"), (0.59, "weak-partner limit"), (1.95, "window reach")):
    ax.axvline(xv, color=MUTED, ls=":", lw=1)
    ax.text(xv, 1.02, lab, rotation=90, fontsize=7, color=MUTED, va="bottom", ha="right")

ax.set_xlabel("range separation (m)"); ax.set_ylabel("Pd of the WEAK target")
ax.set_title("Weak-target detection vs separation")
ax.legend(frameon=False); ax.set_ylim(0, 1)
plt.tight_layout(); plt.show()''')

# ----------------------------------------------------------------- 8. clutter spikiness sweep
md("""## 8. Measured Pfa vs clutter spikiness — where CA-CFAR's assumption breaks

CA-CFAR's `alpha` assumes exponentially distributed cell power. Heavy-tailed clutter breaks that,
so this measures **how fast the false-alarm rate runs away from design as the clutter gets
spikier** — at a *fixed* threshold, which is the only way the drift is visible.

Weibull texture shape was swept 1.1-2.0 per frame and recorded (lower = spikier). This is the
proposal's *clutter strength* ablation, and it is the mechanism behind the whole clutter result.""")

code('''cl = frames[frames.condition.isin(["clutter", "clutter_edge", "mixed"])].dropna(subset=["shape"])
cl = cl.assign(sbin=pd.cut(cl["shape"], np.arange(1.0, 2.2, 0.2)),
               cbin=pd.cut(cl.cnr_db, np.arange(5, 30, 5)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

for m in METHODS:
    alpha = CfarConfig(kind=m).alpha()          # the DESIGN threshold, deliberately
    rows = []
    for key, g in cl.groupby("sbin", observed=True):
        evs = []
        for i in g.frame:
            t = TRUTH_BY_FRAME.get(i)
            tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
                   if t is not None else np.zeros((0, 2)))
            pk = PEAKS[m].get(int(i), np.zeros((0, 3)))
            det = pk[pk[:, 2] >= alpha][:, :2] if len(pk) else np.zeros((0, 2))
            evs.append(evaluate_frame(det, tru, SHAPE, MatchConfig(1, 1), valid_mask=VALID))
        a = aggregate(evs)
        rows.append((key.mid, a["pfa_per_cell"], a["fa_per_frame"]))
    r = pd.DataFrame(rows, columns=["shape", "pfa", "fa"])
    axes[0].semilogy(r["shape"], r.pfa, "o-", color=COLOR[m], label=m, ms=4)
    axes[1].plot(r["shape"], r.fa, "o-", color=COLOR[m], label=m, ms=4)

axes[0].axhline(TARGET_PFA, color=INK, ls="--", lw=1)
axes[0].set_xlabel("Weibull texture shape (lower = spikier)")
axes[0].set_ylabel("measured Pfa at the DESIGN threshold")
axes[0].set_title("False-alarm rate runs away from design"); axes[0].legend(frameon=False)
axes[0].invert_xaxis()

axes[1].set_xlabel("Weibull texture shape (lower = spikier)")
axes[1].set_ylabel("false alarms per frame")
axes[1].set_title("Same thing, in human units"); axes[1].invert_xaxis()
plt.tight_layout(); plt.show()

print("Dashed line = the 1e-4 the threshold was designed for. The vertical distance")
print("is how badly homogeneous-noise theory fails on real clutter.")
print("NOTE: the config's `shape` is the TEXTURE parameter; the measured amplitude")
print("shape is lower (speckle compounds it). Quote measured values in the report.")''')

# ------------------------------------------------------------------------ 9. false alarms
md("""## 9. False alarms — where they land

Pd is only half the story. At matched Pfa the *rate* is equal by construction, so what differs is
**where** the false alarms fall. Concentration in the zero-Doppler clutter ridge is the classical
failure mode, and it is the thing a learned detector is expected to improve on.""")

code('''from radar.config import RadarConfig
from radar.simulator import make_axes

_, _, _, vax = make_axes(RadarConfig())
v0 = int(np.argmin(np.abs(vax)))
BAND = set(range(v0 - 8, v0 + 9))

rows = []
for m in METHODS:
    thr_by_cond = summary[summary.method == m].set_index("condition").threshold
    for cond in CONDS:
        thr, band, total, n = thr_by_cond[cond], 0, 0, 0
        for i in frames.loc[frames.condition == cond, "frame"]:
            t = TRUTH_BY_FRAME.get(i)
            tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
                   if t is not None else np.zeros((0, 2)))
            pk = PEAKS[m].get(int(i), np.zeros((0, 3)))
            det = pk[pk[:, 2] >= thr][:, :2] if (len(pk) and thr == thr) else np.zeros((0, 2))
            n += 1
            for d in det:
                is_fa = (all(abs(d[0] - a) > 1 or abs(d[1] - b) > 1 for a, b in tru)
                         if len(tru) else True)
                if is_fa:
                    total += 1
                    band += int(d[1]) in BAND
        rows.append(dict(method=m, condition=cond, fa_per_frame=total / n,
                         band_fa_per_frame=band / n,
                         pct_in_ridge=100 * band / total if total else np.nan))

fa = pd.DataFrame(rows)
display(fa.pivot_table(index="condition", columns="method",
                       values="band_fa_per_frame")[METHODS].round(2))
display(fa.pivot_table(index="condition", columns="method",
                       values="pct_in_ridge")[METHODS].round(1))
print("Second table: what share of each method's false alarms land in the")
print("zero-Doppler ridge (+/-8 bins). The ridge is ~26% of the testable Doppler axis,")
print("so anything well above that is clutter-driven, not chance.")''')

# ---------------------------------------------------------------------- 10. localisation + cost
md("""## 10. Localisation accuracy and runtime

Two secondary metrics the proposal asks for. Localisation is measured over true positives only.
Runtime is the detector itself (ratio map + peak extraction), excluding I/O.""")

code('''rows = []
for m in METHODS:
    thr_by_cond = summary[summary.method == m].set_index("condition").threshold
    evs = []
    for cond in CONDS:
        thr = thr_by_cond[cond]
        for i in frames.loc[frames.condition == cond, "frame"]:
            t = TRUTH_BY_FRAME.get(i)
            tru = (t[["range_bin", "velocity_bin"]].to_numpy(float)
                   if t is not None else np.zeros((0, 2)))
            pk = PEAKS[m].get(int(i), np.zeros((0, 3)))
            det = pk[pk[:, 2] >= thr][:, :2] if (len(pk) and thr == thr) else np.zeros((0, 2))
            evs.append(evaluate_frame(det, tru, SHAPE, MatchConfig(1, 1), valid_mask=VALID))
    a = aggregate(evs, range_res_m=0.19518, vel_res_mps=0.42013)
    rows.append(dict(method=m, ms_per_frame=RUNTIME[m],
                     mean_dr_bins=a["mean_abs_dr_bins"], mean_dv_bins=a["mean_abs_dv_bins"],
                     rmse_bins=a["rmse_bins"],
                     mean_range_err_m=a.get("mean_abs_range_err_m"),
                     mean_vel_err_mps=a.get("mean_abs_vel_err_mps")))

display(pd.DataFrame(rows).set_index("method").round(3))
print("Localisation is bounded by the +/-1-bin matching gate, so these numbers say")
print("'where inside the gate' - they cannot exceed 1 bin by construction.")''')

# ------------------------------------------------------------------------------ 11. conclusions
md("""## 11. What this establishes

Fill in from the tables above when the notebook is run — the wording below is the frame, the
numbers are whatever came out.

**The baseline itself.** Per-condition Pd for CA/GO/OS on 3,200 unseen frames at a matched measured
false-alarm rate, at the window selected on a disjoint tuning split. This is what the network has to
beat, and it was fixed before the network existed.

**Complementarity — the CFAR headline.** No single classical method wins everywhere: OS is built to
survive interfering targets and pays for it at clutter edges; GO leans on whichever half-window is
clear of the ridge; CA is optimal in homogeneous noise and cheapest. That absence of a universal
winner is precisely what motivates a learned detector.

**Read the spreads before the ranking.** At 400 frames per condition, differences of ~0.01 are
inside sampling noise. Only clearly separated rows support a claim.

### Limitations to carry into the report

- **`unresolved` is capped near Pd 0.5 by physics**, not by the detector — two targets inside one
  mainlobe cannot both be found. It is a sanity check, not a comparison.
- **`nms_size=5` caps the masking stratum** near Pd 0.69 by suppressing resolvable neighbours.
  Documented in `src/cfar/detect.py`; shrinking it measurably costs Pd everywhere else.
- **Pfa is counted after NMS** over all valid non-guard cells. Applied identically to every
  detector, so the comparison holds, but it is not the same quantity as CFAR theory's per-cell Pfa.
- **The test set is 37.5% adversarial by construction** — see §1.""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "cfar_baseline.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
