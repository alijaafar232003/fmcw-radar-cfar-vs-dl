"""Build experiments/label_ablation/label_ablation_experiment.ipynb. Writes only here."""
import json
import pathlib

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": t})


def cap(t):
    md(f"*<span style='color:#6b6b6b'>{t}</span>*")


md("""# Label ablation — does a better label fix the localization deficit?

**The question**

The 3 × 3 label block is the documented suspect behind three separate findings:

* DL localizes **3.83× worse than CFAR in Doppler** (`bench2_test`, gate ±1);
* localization **degrades monotonically with training** while Pd plateaus;
* at **gate 0** DL collapses to Pd ≈ 0.41 where CFAR reaches ≈ 0.85.

All three share one proposed explanation: *the network is told nine cells are equally correct, so it
never learns which one is the peak.*

> **Does a better label fix this?**

**Three arms, 8 seeds, 21 epochs, everything else identical to `configs/frozen_dl.yaml`.** Scored on
`bench2_val`; `bench2_test` is not touched. Pre-registered in `PREREGISTRATION.md` before training
started.

**Isolation.** Nothing in `src/`, `scripts/`, `data/`, `notebooks/` or `results/` was modified. The
`flat3x3` control is built by the project's own `make_label_mask` and its checkpoints reproduce the
frozen run **bit-for-bit** — see §3.""")

code('''import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path.cwd() if Path.cwd().name == "label_ablation" else Path.cwd() / "experiments/label_ablation"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))

from labels import build_label, psf_weights, sigma_from_psf

RES = HERE / "results"
EPOCH_CURVE = pd.read_csv(RES / "epoch_curve.csv")
PER_COND = pd.read_csv(RES / "per_condition.csv")
DELTAS = pd.read_csv(RES / "deltas.csv")
CFARREF = pd.read_csv(RES / "cfar_reference.csv")
PW = pd.read_csv(RES / "pos_weight_derivation.csv")
SIG = json.loads((RES / "sigma.json").read_text())

ARMS = ["flat3x3", "gaussian", "point"]
COLOR = {"flat3x3": "#d1495b", "gaussian": "#c9a227", "point": "#1baf7a"}
LABEL = {"flat3x3": "flat 3x3 (incumbent)", "gaussian": "Gaussian (measured PSF)",
         "point": "point (1x1)"}
ORDER = ["isolated", "masking", "unresolved", "clutter", "clutter_edge", "multi", "mixed"]

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "grid.color": GRID, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "font.size": 10, "figure.dpi": 110})


def tidy(ax):
    ax.grid(axis="y", linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


print("artefacts loaded")''')

# ------------------------------------------------------------------ 1
md("""---
## 1 · The three labels, on a real target

Identical support for two of them — the Gaussian occupies the same 3 × 3 footprint as the incumbent
and differs **only** in the weights. That is the cleanest possible contrast.""")

code('''fd = next(f for f in sorted((ROOT / "data" / "bench2_val" / "frames").iterdir())
          if json.loads((f / "metadata.json").read_text())["condition"] == "isolated")
mag = np.load(fd / "rd_map_mag.npy")
lab = pd.read_csv(fd / "labels.csv")
r0, v0 = int(lab.range_bin.iloc[0]), int(lab.velocity_bin.iloc[0])

sr, sd = SIG["sigma_range_bins"], SIG["sigma_doppler_bins"]
W = psf_weights(sr, sd)
sub = slice(r0 - 4, r0 + 5), slice(v0 - 4, v0 + 5)
ext = [v0 - 4.5, v0 + 4.5, r0 - 4.5, r0 + 4.5]

fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9))
axes[0].imshow(10 * np.log10(mag[sub] ** 2 / np.median(mag ** 2)), origin="lower", cmap="magma",
               extent=ext)
axes[0].set_title("the RD response (dB above median)", loc="left", fontsize=10)
for ax, arm in zip(axes[1:], ARMS):
    y = build_label(arm, lab, mag.shape, weights=W)
    im = ax.imshow(y[sub], origin="lower", cmap="viridis", extent=ext, vmin=0, vmax=1)
    for i in range(-1, 2):
        for j in range(-1, 2):
            val = y[r0 + i, v0 + j]
            if val > 0:
                ax.text(v0 + j, r0 + i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white")
    ax.set_title(LABEL[arm], loc="left", fontsize=10)
for ax in axes:
    ax.scatter(v0, r0, marker="x", s=110, color="#00b4d8", linewidths=2.0)
    ax.set_xlabel("Doppler bin"); ax.grid(False)
axes[0].set_ylabel("Range bin")
plt.tight_layout(); plt.show()

print(f"sigma derived from the measured PSF: range {sr:.4f}, Doppler {sd:.4f} bins")
print(f"  (results/psf_profile.csv, off-bin worst case, -3 dB width / 2.3548)")
print(f"Gaussian weights: centre 1.000, edge {W[0,1]:.3f}/{W[1,0]:.3f}, corner {W[0,0]:.3f}")''')

cap("The flat label asserts nine cells are equally correct. The Gaussian occupies the same nine "
    "cells but keeps a unique maximum at the true bin. The point label marks one cell.")

# ------------------------------------------------------------------ 2
md("""---
## 2 · `pos_weight` per arm — the confound that had to be removed

Changing the label changes how many positive cells exist, which changes the class imbalance. Reusing
one `pos_weight` would confound label *shape* with class *weighting* and make the experiment
uninterpretable.

**Rule, fixed in advance:** `BCEWithLogitsLoss(pos_weight=w)` weights each cell's positive term by
`w · y`, so the loss's total emphasis on "this is a target" is proportional to `w · Σy`. That
product is held constant across arms, anchored to the incumbent:

$$w_\\text{arm} = 100 \\cdot \\frac{\\Sigma y_\\text{flat3x3}}{\\Sigma y_\\text{arm}}$$

**Anchored to 100, not re-derived per arm.** Exact class balancing (`w = N_neg/N_pos`) has already
been *measured* worse in this project — `pos_weight` 974 lost the three-way ablation on 6 of 7
conditions and manufactured a `clutter` deficit that was retracted (Correction C3).""")

code('''t = PW.copy()
t["positive mass"] = t.positive_mass.map(lambda v: f"{v:,.0f}")
t["ratio to reference"] = t.mass_ratio_to_reference.round(4)
display(t[["arm", "positive mass", "ratio to reference", "pos_weight"]]
        .style.hide(axis="index"))
p = PW.mass_x_pos_weight
print(f"mass x pos_weight is constant to {100 * (p.max() / p.min() - 1):.3f}% across arms.")
print(f"The point arm is {PW.set_index('arm').loc['point','mass_ratio_to_reference']:.4f}x rather "
      f"than exactly 1/9 = 0.1111:")
print("3x3 blocks clip at the frame border and overlap between close targets, so it is measured.")''')

# ------------------------------------------------------------------ 3
md("""---
## 3 · The control is verified, twice

Before any comparison is trusted, the `flat3x3` arm must genuinely be the frozen configuration —
otherwise the ablation compares two things at once.

1. **Labels** — built by the project's own `make_label_mask`, asserted byte-identical to
   `data/bench2_train/nn_cache_y.npy` on 40 sampled frames.
2. **Weights** — retraining `flat3x3` at `threads=2` reproduces the frozen checkpoints
   **bit-for-bit at epoch 21**, on all 8 seeds (max |ΔW| = 0.000e+00).

The second check also validates the training driver itself: it writes into this folder rather than
`results/dl/`, yet produces numerically identical models.""")

code('''import torch
worst = 0.0
for s in range(8):
    a = torch.load(HERE / "checkpoints" / f"tinycnn_flat3x3_s{s}_ep021.pt",
                   weights_only=False)["state_dict"]
    b = torch.load(ROOT / "results" / "dl" / f"tinycnn_frozen_s{s}_ep021.pt",
                   weights_only=False)["state_dict"]
    worst = max(worst, max(float((a[k].float() - b[k].float()).abs().max()) for k in a))
print(f"max |dW| between flat3x3 and the frozen arm, over all 8 seeds: {worst:.3e}")
assert worst == 0.0, "the control is NOT the frozen arm -- the ablation would be confounded"
print("-> the control IS the frozen arm")''')

# ------------------------------------------------------------------ 4
md("""---
## 4 · The headline: localization vs epoch

**This is the sharpest test of the mechanism.** Localization is known to degrade monotonically with
training under the flat label. If flat supervision *causes* that drift, weighted supervision should
reduce it and point supervision should remove it.""")

code('''E1 = EPOCH_CURVE[EPOCH_CURVE.gate == 1]

fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6))
for ax, col, se, name in ((axes[0], "dv_bins", "dv_se", "Doppler"),
                          (axes[1], "dr_bins", "dr_se", "Range")):
    for arm in ARMS:
        d = E1[E1.arm == arm].sort_values("epoch")
        ax.errorbar(d.epoch, d[col], yerr=d[se], color=COLOR[arm], lw=2.0, marker="o",
                    ms=5, capsize=3, label=LABEL[arm])
    ax.set_xlabel("epoch"); ax.set_ylabel(f"{name} localization error (bins)")
    ax.set_title(f"{name}: error vs training", loc="left", fontsize=10.5)
    ax.set_ylim(0, None); tidy(ax)
axes[0].legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()

print("Doppler error, epoch 5 -> 21 (8-seed mean, gate +/-1):")
for arm in ARMS:
    d = E1[E1.arm == arm].sort_values("epoch")
    a, b = d.dv_bins.iloc[0], d.dv_bins.iloc[-1]
    print(f"  {LABEL[arm]:<26} {a:.4f} -> {b:.4f}   ({b / a:.2f}x, {b - a:+.4f} bins)")''')

cap("The drift is monotone in how flat the supervision is. The incumbent nearly doubles its Doppler "
    "error over training; the Gaussian drifts gently; the point label is essentially flat. This is "
    "the mechanism confirmed — flat supervision causes the degradation.")

# ------------------------------------------------------------------ 5
md("""---
## 5 · What it costs, and what it buys

Pd is the constraint: a localization fix that destroyed detection would be no fix at all.""")

code('''E0 = EPOCH_CURVE[EPOCH_CURVE.gate == 0]
fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.4))
for ax, E, name in ((axes[0], E1, "gate +/-1 (headline)"), (axes[1], E0, "gate 0 (exact cell)")):
    for arm in ARMS:
        d = E[E.arm == arm].sort_values("epoch")
        ax.errorbar(d.epoch, d.pd, yerr=d.pd_se, color=COLOR[arm], lw=2.0, marker="o", ms=5,
                    capsize=3, label=LABEL[arm])
    ax.set_xlabel("epoch"); ax.set_ylabel("Pd at matched measured Pfa")
    ax.set_title(name, loc="left", fontsize=10.5); tidy(ax)
axes[0].legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()

e21 = EPOCH_CURVE[EPOCH_CURVE.epoch == 21]
c1 = CFARREF[CFARREF.gate == 1].set_index("method")
br, bv = c1.dr_bins.idxmin(), c1.dv_bins.idxmin()
rows = []
for arm in ARMS:
    a1 = e21[(e21.arm == arm) & (e21.gate == 1)].iloc[0]
    a0 = e21[(e21.arm == arm) & (e21.gate == 0)].iloc[0]
    rows.append(dict(arm=LABEL[arm], **{
        "Pd gate1": round(a1.pd, 4), "+/-SE": round(a1.pd_se, 4),
        "Pd gate0": round(a0.pd, 4),
        "range (bins)": round(a1.dr_bins, 4), "range/CFAR": round(a1.dr_bins / c1.loc[br, "dr_bins"], 2),
        "Doppler (bins)": round(a1.dv_bins, 4), "Dopp/CFAR": round(a1.dv_bins / c1.loc[bv, "dv_bins"], 2),
        "range (m)": round(a1.range_m, 4), "vel (m/s)": round(a1.vel_mps, 4)}))
display(pd.DataFrame(rows).set_index("arm"))
print(f"best CFAR on this split (gate +/-1): range {c1.loc[br,'dr_bins']:.4f} ({br}), "
      f"Doppler {c1.loc[bv,'dv_bins']:.4f} ({bv})")''')

cap("At gate ±1 the Gaussian matches the incumbent's Pd while more than halving the Doppler error. "
    "At gate 0 — the exact-cell criterion — the incumbent actually gets WORSE with training while "
    "both alternatives improve sharply.")

code('''fig, ax = plt.subplots(figsize=(11.0, 4.2))
D0 = DELTAS[DELTAS.gate == 0]
D1 = DELTAS[DELTAS.gate == 1]
x = np.arange(len(ORDER))
for j, (D, hatch, tag) in enumerate([(D0, "", "gate 0"), (D1, "//", "gate +/-1")]):
    for i, arm in enumerate(["gaussian", "point"]):
        d = D[D.arm == arm].set_index("condition").reindex(ORDER)
        ax.bar(x + (2 * j + i - 1.5) * 0.2, d.delta, width=0.2, color=COLOR[arm],
               alpha=1.0 if j == 0 else 0.45, hatch=hatch,
               label=f"{LABEL[arm]} - {tag}")
ax.axhline(0, color=INK, lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Pd difference vs flat 3x3")
ax.set_title("Change in Pd against the incumbent label, per condition", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, ncol=2); tidy(ax)
plt.tight_layout(); plt.show()

for g in (0, 1):
    d = DELTAS[DELTAS.gate == g]
    print(f"gate {g}: significant after Bonferroni (alpha {d.alpha_bonf.iloc[0]:.4f}) --")
    for arm in ("gaussian", "point"):
        a = d[d.arm == arm]
        print(f"   {arm:<9} {int(a.sig.sum())}/{len(a)} conditions, "
              f"delta {a.delta.min():+.4f} to {a.delta.max():+.4f}")''')

# ------------------------------------------------------------------ 6
md("""---
## 6 · Conclusion

### Does a PSF-weighted label recover the localization deficit, and at what cost to Pd?

**Yes, and at essentially no cost.**

| | Doppler error vs best CFAR | Pd at gate ±1 |
|---|---|---|
| flat 3 × 3 (incumbent) | **4.57×** | 0.5848 |
| Gaussian | **1.99×** | 0.5871 (**+0.0023**) |
| point | **1.27×** | 0.5750 (−0.0098) |

The Gaussian label **more than halves** the Doppler localization deficit while matching the
incumbent's Pd to within one standard error. The point label cuts the deficit further — to within
27% of CFAR — at a small but real Pd cost of about one percentage point.

**The mechanism is confirmed.** Doppler error over training (epoch 5 → 21) rises ×1.98 under the
flat label, ×1.25 under the Gaussian, and ×1.04 under the point label. The drift is monotone in how
flat the supervision is, which is exactly what the hypothesis predicted and what the pre-registered
"sharpest test" was designed to detect.

**At gate 0 the incumbent gets worse with training.** Its exact-cell Pd *falls* from 0.278 to 0.191
across 21 epochs, while the Gaussian holds ~0.33 and the point label ~0.39. The label is not merely
limiting exact-cell accuracy — under flat supervision, more training actively destroys it.

### The registered predictions

**1. "Gaussian matches 3 × 3 on Pd while improving localization" — CONFIRMED.** Pd +0.0023 (within
noise, significant on 1 of 7 conditions), Doppler localization 4.57× → 1.99×.

**2. "Point may train poorly or unstably at ~1:8500; if so, that is the first evidence the 3 × 3
choice was reasonable" — FAILED.** Point trained stably on all 8 seeds, and delivered the *best*
localization and the *best* gate-0 Pd. **The 3 × 3 choice is not vindicated — the opposite.** Its
only cost is −0.0098 Pd at gate ±1, concentrated in the clutter-bearing conditions (`clutter`
−0.025, `mixed` −0.024, `clutter_edge` −0.022, each significant), which is consistent with sparser
supervision giving less gradient signal exactly where the task is hardest.

**3. "A Pd gain from Gaussian would be unexpected — flag it as such."** Flagged. At gate ±1 the
+0.0023 is tiny and significant on only `multi`; it should not be reported as a detection win. At
gate 0 both alternatives gain hugely (+0.06 to +0.31, significant on all 7), **but that is the
localization improvement expressed as exact-cell Pd — the same quantity measured differently, not an
independent detection gain.**

### What this means for the project

The localization deficit is **caused by the label and is fixable**. That converts a documented
limitation into a design choice with a measured alternative, and it retires the leading competing
explanations — it is not the network's capacity, and not the matching gate.

⚠️ **Scope.** Validation split only; `bench2_test` untouched. This does **not** restate any reported
result, and the frozen configuration is unchanged. Whether to adopt a different label — and whether
that would warrant a second test read — is a separate decision, to be recorded before it is taken.

---

*Reproduce: `build_label_cache.py` → `run_all.py` → `run_eval.py` → `build_notebook.py`.
All artefacts stay under `experiments/label_ablation/`.*""")

for _i, _c in enumerate(CELLS):
    if _c["cell_type"] == "code":
        try:
            compile(_c["source"], f"<cell {_i}>", "exec")
        except SyntaxError as _e:
            raise SystemExit(f"cell {_i} does not parse: {_e}\n---\n{_c['source']}\n---")

nb = {"cells": [dict(c, source=c["source"].splitlines(keepends=True)) for c in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "label_ablation_experiment.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
