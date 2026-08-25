"""Build experiments/integrated_cfar/integrated_cfar_experiment.ipynb.

Reads only the artefacts written by run_integrated_cfar.py plus cached project baselines.
Modifies nothing outside this folder.
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
    md(f"*<span style='color:#6b6b6b'>{t}</span>*")


md("""# Integrated CFAR — an attribution experiment

**The question**

> How much of TinyCNN's advantage over CFAR is explained by the CNN integrating neighbouring
> mainlobe energy that single-cell CFAR discards?

Standard CFAR tests **one cell** against a noise estimate. A CNN with a 21 × 9 receptive field can
combine a target's whole mainlobe. If that difference is where the margin comes from, then giving
CFAR the same energy should close most of the gap.

This notebook builds the intermediate detector — the **same CFAR noise estimate, an integrated CUT**
— and measures what it buys.

**Scope and isolation.** Nothing in `src/`, `scripts/`, `notebooks/` or the project's `results/` is
modified. The noise estimators are imported from `src/cfar/cfar.py` and called unchanged; only the
numerator differs. Scored on **`bench2_val`** — `bench2_test` is spent and is not touched. This
experiment does **not** alter the project's primary results or conclusions.

**Pre-registered** in `PREREGISTRATION.md`, written before any result was computed.""")

code('''import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path.cwd() if Path.cwd().name == "integrated_cfar" else Path.cwd() / "experiments/integrated_cfar"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from cfar.cfar import CfarConfig
from integrated_cfar import uniform_kernel

RES = HERE / "results"
CFG = CfarConfig()

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, PURPLE, RED, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#8172b3", "#d1495b", "#c9a227"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "grid.color": GRID, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "font.size": 10, "figure.dpi": 110})


def tidy(ax):
    ax.grid(axis="y", linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


ORDER = ["isolated", "masking", "unresolved", "clutter", "clutter_edge", "multi", "mixed"]
PATCH = np.load(RES / "psf_patch.npy")
CAPTURE = pd.read_csv(RES / "psf_energy_capture.csv")
KERNELS = dict(np.load(RES / "kernels.npz"))
CAL = pd.read_csv(RES / "calibration.csv")
SUMMARY = pd.read_csv(RES / "summary_pd.csv")
DELTAS = pd.read_csv(RES / "deltas.csv")
LOC = pd.read_csv(RES / "localization.csv")
print("artefacts loaded from", RES)''')

# ---------------------------------------------------------------- 1 PSF
md("""---
## 1 · The point-spread function, measured

**The question this answers:** how far does a target's energy actually spread, and therefore what
region is worth integrating?

Measured from **`isolated` targets only** — one target per frame, so no neighbour contaminates the
patch — on the top SNR quartile, so the average reflects the response rather than the noise. The
1-D profiles were already measured by `scripts/measure_psf.py`: Blackman window, −3 dB width
1.62 / 1.65 bins, −30 dB half-width 2 bins, −40 dB 3 bins, at the worst-case half-bin offset.""")

code('''fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
c = PATCH.shape[0] // 2
ext = [-c - 0.5, c + 0.5, -c - 0.5, c + 0.5]

im = axes[0].imshow(PATCH, origin="lower", cmap="magma", extent=ext)
axes[0].set_title("averaged normalised target patch (power)", loc="left", fontsize=10.5)
fig.colorbar(im, ax=axes[0], pad=0.02)
im2 = axes[1].imshow(10 * np.log10(PATCH + 1e-6), origin="lower", cmap="magma", extent=ext,
                     vmin=-40, vmax=0)
axes[1].set_title("the same, in dB", loc="left", fontsize=10.5)
fig.colorbar(im2, ax=axes[1], label="dB below peak", pad=0.02)
for ax in axes:
    ax.add_patch(Rectangle((-1.5, -1.5), 3, 3, facecolor="none", edgecolor="#00b4d8",
                           linewidth=2.0, linestyle="--"))
    ax.set_xlabel("Doppler offset (bins)"); ax.grid(False)
axes[0].set_ylabel("Range offset (bins)")
plt.tight_layout(); plt.show()

sq = CAPTURE[(CAPTURE.half_r == CAPTURE.half_d)].copy()
sq["energy %"] = (100 * sq.energy_fraction).round(1)
display(sq[["region", "cells", "energy %"]].style.hide(axis="index"))''')

cap("The dashed box is the 3×3 region chosen below. Almost all of the response is inside it, and "
    "the dB panel shows how fast it falls away beyond ±1 bin.")

md("""### Why 3 × 3, derived rather than assumed

The region is chosen from the measured energy table, not by convention:

* **1 × 1 captures only ~27%** of the target energy — single-cell CFAR is genuinely discarding most
  of it, which is what makes this experiment worth running.
* **3 × 3 captures ~94%** in 9 cells.
* **5 × 5 captures ~99.7%** — but buys the last ~5 percentage points with **16 more noise cells**.

There is also a hard constraint: the CFAR guard region is ±2, so an integration region larger than
5 × 5 would draw on cells the **reference ring** also uses, correlating numerator with denominator.
3 × 3 sits strictly inside the guard. `check_kernel_fits_guard()` enforces this.""")

# ---------------------------------------------------------------- 2 statistics
md("""---
## 2 · Two integrated statistics

Both replace the single-cell CUT with a weighted sum over the 3 × 3 region. **Guard and reference
logic are untouched** — the denominator is computed by the imported, unmodified project estimators.

* **(a) uniform** — a plain sum. Every cell in the region weighted equally.
* **(b) PSF-weighted** — weights proportional to the measured normalised PSF. This is the **matched
  filter** for a known signal shape in white noise, and the statistic with a theoretical reason to
  help.

Both kernels are normalised to sum 1, so the ratio keeps its "multiples of the noise level" meaning
and the two are directly comparable.""")

code('''fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
for ax, (name, k) in zip(axes, [("uniform", KERNELS["uniform"]), ("PSF-weighted", KERNELS["psf"])]):
    ax.imshow(k, origin="lower", cmap="Blues", extent=[-1.5, 1.5, -1.5, 1.5], vmin=0,
              vmax=max(KERNELS["psf"].max(), KERNELS["uniform"].max()))
    for i in range(3):
        for j in range(3):
            ax.text(j - 1, i - 1, f"{k[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"{name}  (centre weight {k[1, 1]:.3f})", loc="left", fontsize=10)
    ax.set_xlabel("Doppler offset"); ax.grid(False)
axes[0].set_ylabel("Range offset")
plt.tight_layout(); plt.show()

print("The two differ only in how sharply they concentrate on the centre cell:")
print(f"  uniform      centre weight {KERNELS['uniform'][1, 1]:.3f}")
print(f"  PSF-weighted centre weight {KERNELS['psf'][1, 1]:.3f}")
print(f"  single-cell CFAR is the limiting case: centre weight 1.000")''')

md("""### ⚠️ The prediction registered before running

> With a −3 dB width of ~1.6 bins the target energy is concentrated in one or two cells, so
> **uniform summation adds mostly noise to the numerator and may perform worse than single-cell
> CFAR**. PSF-weighted is the statistic with a theoretical reason to help. If uniform loses, that is
> a result, not a failure.

The competing consideration, also registered: summing K cells raises the signal while the noise
standard deviation would grow only as √K *if the cells were independent* — but the Blackman window
**correlates adjacent cells**, so the noise sum grows faster than that and the gain erodes.""")

# ---------------------------------------------------------------- 3 geometry
md("""---
## 3 · The geometry, to scale on a real frame

**The question this answers:** what does each part of the statistic actually cover?""")

code('''FD = next(fd for fd in sorted((ROOT / "data" / "bench2_val" / "frames").iterdir())
          if json.loads((fd / "metadata.json").read_text())["condition"] == "isolated")
MAG = np.load(FD / "rd_map_mag.npy")
LAB = pd.read_csv(FD / "labels.csv")
tr, tv = int(LAB.range_bin.iloc[0]), int(LAB.velocity_bin.iloc[0])

hw_r, hw_d, g_r, g_d = CFG.half_width_r, CFG.half_width_d, CFG.guard_r, CFG.guard_d
W = 14
sub = slice(tr - W, tr + W + 1), slice(tv - 8, tv + 9)
ext = [tv - 8.5, tv + 8.5, tr - W - 0.5, tr + W + 0.5]

fig, ax = plt.subplots(figsize=(7.0, 6.4))
ax.imshow(10 * np.log10(MAG[sub] ** 2 / np.median(MAG ** 2)), origin="lower", cmap="magma",
          extent=ext, aspect="auto")
for rr in range(-hw_r, hw_r + 1):
    for dd in range(-hw_d, hw_d + 1):
        inner = abs(rr) <= g_r and abs(dd) <= g_d
        ax.add_patch(Rectangle((tv + dd - 0.5, tr + rr - 0.5), 1, 1,
                               facecolor=(GOLD if inner else BLUE), alpha=0.28,
                               edgecolor="white", linewidth=0.3))
ax.add_patch(Rectangle((tv - 1.5, tr - 1.5), 3, 3, facecolor=AQUA, alpha=0.55,
                       edgecolor="#00b4d8", linewidth=2.2))
ax.add_patch(Rectangle((tv - 0.5, tr - 0.5), 1, 1, facecolor=ORANGE, alpha=0.95,
                       edgecolor=INK, linewidth=1.0))
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_xlabel("Doppler bin"); ax.set_ylabel("Range bin"); ax.grid(False)
ax.set_title("orange = CUT · teal = 3x3 integration region · gold = guard · blue = reference ring",
             loc="left", fontsize=9.5)
plt.tight_layout(); plt.show()

print(f"CFAR window {2 * hw_r + 1}x{2 * hw_d + 1} = {(2 * hw_r + 1) * (2 * hw_d + 1)} cells")
print(f"  guard (excluded from the noise estimate): {(2 * g_r + 1) * (2 * g_d + 1)} cells")
print(f"  reference ring: {CFG.num_ref_cells} cells")
print(f"  integration region: 9 cells, entirely inside the guard")''')

cap("Standard CFAR tests only the orange cell. The integrated variants sum the teal 3×3. The gold "
    "guard and blue reference ring are unchanged, which is why the numerator can never overlap the "
    "denominator.")

# ---------------------------------------------------------------- 4 calibration
md("""---
## 4 · Calibration — mandatory for these statistics

Summing K **correlated** cells changes the null distribution: it is no longer exponential, so the
closed form `alpha = N(Pfa^(-1/N) − 1)` does not apply and the project's cached alphas are
meaningless here. Alpha is therefore calibrated by Monte Carlo on pure noise and **verified on a
held-out seed**, exactly as `scripts/cfar_calibrate_table.py` does.""")

code('''display(CAL.assign(alpha=CAL.alpha.round(4),
                   achieved_pfa_heldout=CAL.achieved_pfa_heldout.map(lambda v: f"{v:.3e}"),
                   ratio_to_target=CAL.ratio_to_target.round(3)).style.hide(axis="index"))
print(f"achieved / target on the held-out seed: {CAL.ratio_to_target.min():.2f}x to "
      f"{CAL.ratio_to_target.max():.2f}x")
print()
print("Note: alpha fixes the DESIGN operating point only. Every Pd below is read at matched")
print("MEASURED Pfa, where the threshold is swept, so no result depends on these values.")
print("They are reported because an uncalibrated design point would be a silent error.")''')

# ---------------------------------------------------------------- 5 results
md("""---
## 5 · Results

`bench2_val`, Pd at matched measured Pfa = 1e-4 swept per condition, gate ±1, shared `all_peaks`
at `nms_size=5`, Hungarian matching, guard-excluded Pfa denominator. Cached baselines for standard
CFAR and TinyCNN are **read, never recomputed**. Best-of-four within each family.""")

code('''D1 = DELTAS[DELTAS.gate == 1].copy()
piv = D1.pivot(index="condition", columns="statistic",
               values=["pd_int", "delta_integration", "di_p"])
tab = pd.DataFrame({
    "best CFAR": D1[D1.statistic == "uniform"].set_index("condition").best_cfar,
    "Pd CFAR": D1[D1.statistic == "uniform"].set_index("condition").pd_cfar,
    "Pd INT-uniform": piv[("pd_int", "uniform")],
    "Pd INT-psf": piv[("pd_int", "psf")],
    "Pd TinyCNN": D1[D1.statistic == "uniform"].set_index("condition").pd_dl,
    "d_int uniform": piv[("delta_integration", "uniform")],
    "d_int psf": piv[("delta_integration", "psf")],
}).reindex([c for c in ORDER if c in piv.index])
display(tab.round(4))

alpha_b = float(D1.alpha_bonf.iloc[0])
for s in ("uniform", "psf"):
    d = D1[D1.statistic == s]
    print(f"{s:<8} delta_integration: mean {d.delta_integration.mean():+.4f}, "
          f"negative on {int((d.delta_integration < 0).sum())}/{len(d)} conditions, "
          f"significant on {int((d.di_p < alpha_b).sum())}/{len(d)} "
          f"(Bonferroni alpha {alpha_b:.4f})")
    print(f"         delta_remaining:   mean {d.delta_remaining.mean():+.4f}")''')

code('''fig, ax = plt.subplots(figsize=(11.4, 4.6))
order = [c for c in ORDER if c in tab.index]
x = np.arange(len(order))
for j, (col, c, lab) in enumerate([("Pd CFAR", BLUE, "CFAR (best of 4)"),
                                   ("Pd INT-uniform", RED, "Integrated, uniform"),
                                   ("Pd INT-psf", GOLD, "Integrated, PSF-weighted"),
                                   ("Pd TinyCNN", AQUA, "TinyCNN")]):
    ax.bar(x + (j - 1.5) * 0.21, tab.loc[order, col], width=0.21, color=c, label=lab)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Pd at matched measured Pfa")
ax.set_title("Integration does not move CFAR toward TinyCNN - it moves it away",
             loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, ncol=2)
tidy(ax)
plt.tight_layout(); plt.show()''')

cap("Both integrated variants sit BELOW standard CFAR on every condition, while TinyCNN sits well "
    "above all three. The gap the experiment set out to explain is not merely unexplained by "
    "integration — integration widens it.")

md("""### Localization

Reported because if integration improved Pd but worsened placement, that would bear directly on the
CNN's own localization deficit. In fact it worsens **both**.""")

code('''l1 = LOC[LOC.gate == 1].copy()
l1["family"] = l1.arm.str.replace(r"-(CA|GO|OS|SO)$", "", regex=True)
pooled = l1.groupby("family")[["dr_bins", "dv_bins"]].mean().round(4)
pooled["dv vs CFAR"] = (pooled.dv_bins / pooled.loc["CFAR", "dv_bins"]).round(2)
pooled["dr vs CFAR"] = (pooled.dr_bins / pooled.loc["CFAR", "dr_bins"]).round(2)
display(pooled.loc[["CFAR", "INT-psf", "INT-uniform"]])

print("Ordering is monotone in how much the kernel smooths:")
print(f"  centre weight 1.000 (single cell) -> Doppler error "
      f"{pooled.loc['CFAR', 'dv_bins']:.4f} bins")
print(f"  centre weight {KERNELS['psf'][1, 1]:.3f} (PSF)         -> "
      f"{pooled.loc['INT-psf', 'dv_bins']:.4f} bins")
print(f"  centre weight {KERNELS['uniform'][1, 1]:.3f} (uniform)     -> "
      f"{pooled.loc['INT-uniform', 'dv_bins']:.4f} bins")''')

cap("The flatter the kernel, the worse the placement — and Doppler degrades more than range, "
    "under a perfectly symmetric 3×3 kernel. See the conclusion.")

# ---------------------------------------------------------------- 6 conclusion
md("""---
## 6 · Conclusion

**Against the three pre-registered outcomes:**

| outcome | reading | observed? |
|---|---|---|
| Integrated ≈ TinyCNN | the advantage is largely mainlobe energy | no |
| Integrated between | mainlobe explains part of it | no |
| **Integrated ≈ CFAR** | **integration explains none of it; the learned decision rule is the live explanation** | **yes — and integration is in fact worse than CFAR** |

**The registered prediction was correct, and stronger than predicted.** Uniform summation was
predicted to possibly perform worse than single-cell CFAR; it does, on every condition. But the
PSF-weighted matched filter — the variant with the theoretical reason to help — **also loses**,
by less. The ordering is monotone in how sharply the kernel concentrates on the centre cell, with
single-cell CFAR best.

**Why integration loses here.** The measured −3 dB width is ~1.6 bins, so the response is
concentrated in very few cells: 1 × 1 already captures 27% of the energy and the centre cell
dominates. Summing nine cells therefore adds a little signal and a lot of noise — and because the
Blackman window correlates adjacent cells, the noise sum grows faster than √K, so the deflection
falls rather than rises. The matched filter loses less precisely because it down-weights the
noisy outer cells.

**What this means for the project's central question.** The margin TinyCNN holds over CFAR is
**not** explained by access to mainlobe energy that CFAR discards. Giving the classical detector
that energy makes it worse. The **learned decision rule** therefore remains the live explanation,
and the attribution programme should concentrate there.

⚠️ **What this does not show.** It does not show the CNN gains nothing from its receptive field —
only that *this* form of energy integration, with the CFAR noise estimate held fixed, does not
reproduce the CNN's advantage. A network is free to weight its 21 × 9 field in ways a fixed 3 × 3
kernel cannot, including adaptively. The complementary control — a detector *denied* those cells —
is a separate experiment (`PLAN.md`, E4).

### An unplanned finding worth recording

A **symmetric** 3 × 3 kernel degrades **Doppler localization roughly twice as much as range**
(pooled, uniform: Doppler ×2.50, range ×1.37 against single-cell CFAR). The project has an open
question — Open/O4 — asking why the CNN's localization degrades asymmetrically between range and
Doppler when its 3 × 3 label is symmetric.

This shows the same asymmetry arising in a **purely classical detector with no learning involved**,
from a symmetric operator. That makes "the CNN's training is responsible for the asymmetry" less
likely, and points instead at something about the RD geometry itself.

⚠️ Suggestive, not conclusive: a fixed 3 × 3 box filter is not what the network does, and this was
not the experiment's question. Recorded as a lead for O4, not as an answer.

---

*Reproduce with `python experiments/integrated_cfar/run_integrated_cfar.py`, then rebuild this
notebook with `python experiments/integrated_cfar/build_notebook.py`. Scored on `bench2_val`;
`bench2_test` untouched.*""")

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

out = pathlib.Path(__file__).resolve().parent / "integrated_cfar_experiment.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
