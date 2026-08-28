"""Build experiments/e4_cut_guard_mask/e4_cut_guard_mask.ipynb. Loads artefacts; trains nothing."""
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


md("""# E4 — CUT-centred guard-masked CNN attribution

**The question**

> Does TinyCNN obtain part of its detection advantage because it can use target/mainlobe
> information inside the cells that classical CFAR treats as **guard** cells?

CFAR excludes its guard region by design — target energy leaking into the noise estimate would raise
the threshold and mask the target. A CNN with a 21 × 9 receptive field has no such restriction. E4
removes that access and measures what changes.

**This is the complement of E2.** E2 *gave* CFAR integrated mainlobe energy and found it did not
recover the DL advantage — integration made CFAR worse on 7/7 conditions. E4 *denies* the CNN the
same information.

Pre-registered in `PREREGISTRATION.md` before any masked model was trained. Scored on `bench2_val`;
`bench2_test` is never opened.""")

code('''import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path.cwd() if Path.cwd().name == "e4_cut_guard_mask" else Path.cwd() / "experiments/e4_cut_guard_mask"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE / "src"))
from patches import HW_R, HW_D, G_R, G_D, MASK, verify_mask

RES = HERE / "results"
DELTAS = pd.read_csv(RES / "deltas.csv")
PERC = pd.read_csv(RES / "per_condition.csv")
FEAS = json.loads((RES / "feasibility_report.json").read_text())
OCC = np.load(RES / "occlusion_heatmaps.npz")
OCCS = pd.read_csv(RES / "occlusion_summary.csv")
ISO = json.loads((RES / "isolation_verified.json").read_text())

ORDER = ["isolated", "masking", "unresolved", "clutter", "clutter_edge", "multi", "mixed"]
BLUE, ORANGE, AQUA, RED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#d1495b", "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": "#c3c2b7", "grid.color": GRID, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "font.size": 10, "figure.dpi": 110})


def tidy(ax):
    ax.grid(axis="y", linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


print("artefacts loaded")''')

md("""---
## 1 · Why this needed a patch model — a feasibility finding

**CUT-relative guard masking is impossible on a dense fully-convolutional network.** The guard is
±2, so cell (r+1, v) lies inside CUT (r, v)'s guard *while being its own CUT*. One input map cannot
hold it hidden and visible at once.""")

code('''print(f"guard cells masked per CUT: {FEAS['guard_cells_masked_per_cut']}")
print(f"dense masking possible: {FEAS['dense_masking_possible']}")
print(f"passes needed for exact dense masking: {FEAS['passes_for_exact_dense']} per frame")
print()
print(f"frozen dense logit vs 21x9 patch-centre logit (eval mode): "
      f"max |diff| {FEAS['patch_dense_max_abs_diff']:.2e}")
print(f"  -> the patch formulation describes the frozen model exactly at INFERENCE")
print(f"same weights in TRAIN mode, dense vs patch centre logit: "
      f"{FEAS['train_mode_dense_vs_patch_diff']:.4f}")
print("  -> but BatchNorm normalises over 256x64 vs 21x9, so patch TRAINING differs")
display(pd.DataFrame([verify_mask()]).T.rename(columns={0: "value"}))''')

md("""**Consequence for the design.** A masked patch model differs from the frozen dense model in three
ways — guard access, BatchNorm statistics, and loss population. So the primary contrast is
**masked-patch vs unmasked-patch**, both trained identically, which makes the last two common-mode.
The frozen dense TinyCNN was **never retrained**; it remains a read-only reference.""")

md("""---
## 2 · What is removed

24 cells — the 5 × 5 CFAR guard minus the CUT. **The CUT stays visible**, because CFAR tests it;
masking it would answer a different question. All 164 reference cells stay visible. Masked cells are
set to 0.0, which `normalize_rd` makes the frame's own median power. No mask channel is added.""")

code('''fig, ax = plt.subplots(figsize=(5.2, 8.0))
for i in range(2 * HW_R + 1):
    for j in range(2 * HW_D + 1):
        if MASK[i, j]:
            c, lab = RED, "masked guard"
        elif (i, j) == (HW_R, HW_D):
            c, lab = ORANGE, "CUT (visible)"
        else:
            c, lab = BLUE, "reference (visible)"
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c, alpha=0.55,
                               edgecolor="white", linewidth=0.5))
ax.set_xlim(-0.5, 2 * HW_D + 0.5); ax.set_ylim(-0.5, 2 * HW_R + 0.5)
ax.set_xlabel("Doppler offset"); ax.set_ylabel("Range offset")
ax.set_title("orange = CUT · red = masked guard (24) · blue = reference (164)", loc="left",
             fontsize=10)
ax.grid(False); ax.set_aspect("equal")
plt.tight_layout(); plt.show()''')

md("""---
## 3 · A descriptive prior: where the frozen model is sensitive

Occlusion sensitivity of the **frozen** model's CUT logit, 8 seeds. ⚠️ **Descriptive only** —
occluding at inference creates inputs the network never trained on, so this cannot establish
causation. It was registered as a *prior*, not a prediction.""")

code('''display(OCCS.round(4).style.hide(axis="index"))
t = OCCS[OCCS.population == "target CUT"].iloc[0]
print(f"target CUTs: the 24 guard cells carry {t.pct_guard:.1f}% of the sensitivity mass,")
print(f"the 164 reference cells {t.pct_reference:.1f}% -- "
      f"{(t.guard_excl_cut / 24) / (t.reference / 164):.1f}x more influence per cell.")

fig, axes = plt.subplots(1, 2, figsize=(9.0, 6.4))
for ax, key, name in ((axes[0], "target", "target CUTs"), (axes[1], "background", "background CUTs")):
    im = ax.imshow(OCC[key], origin="lower", cmap="magma",
                   extent=[-HW_D - .5, HW_D + .5, -HW_R - .5, HW_R + .5])
    ax.add_patch(Rectangle((-G_D - .5, -G_R - .5), 2 * G_D + 1, 2 * G_R + 1, facecolor="none",
                           edgecolor="#00b4d8", linewidth=2.0, linestyle="--"))
    ax.set_title(f"|d logit| — {name}", loc="left", fontsize=10)
    ax.set_xlabel("Doppler offset"); ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.02)
axes[0].set_ylabel("Range offset")
plt.tight_layout(); plt.show()''')

cap("Dashed box is the CFAR guard. The frozen model's sensitivity is concentrated almost entirely "
    "inside it — the region CFAR discards by design.")

md("""---
## 4 · The result

Both arms scored through the project's unmodified pipeline at matched **measured** post-NMS
Pfa ≈ 1e-4, swept per condition, 8 seeds each, seed-paired.

**The two gates answer different questions, and they disagree.**""")

code('''for g, name in ((1, "gate +/-1 — the pre-registered headline"), (0, "gate 0 — exact cell")):
    d = DELTAS[DELTAS.gate == g].set_index("condition").reindex(ORDER)
    print(f"=== {name} ===")
    print(d[["pd_unmasked", "pd_masked", "delta_pd", "ci_lo", "ci_hi", "p_boot",
             "seeds_positive", "sig"]].round(4).to_string())
    print(f"  mean dPd {d.delta_pd.mean():+.4f} | positive on {int((d.delta_pd > 0).sum())}/7 | "
          f"significant on {int(d.sig.sum())}/7\\n")

p = PERC[PERC.gate == 1]
print(f"measured Pfa / target across all arms and conditions: "
      f"{(p.measured_pfa / 1e-4).min():.3f} to {(p.measured_pfa / 1e-4).max():.3f}")''')

code('''fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), sharey=False)
for ax, g, name in ((axes[0], 1, "gate +/-1 (headline)"), (axes[1], 0, "gate 0 (exact cell)")):
    d = DELTAS[DELTAS.gate == g].set_index("condition").reindex(ORDER)
    x = np.arange(len(d))
    cols = [AQUA if s else MUTED for s in d.sig]
    ax.bar(x, d.delta_pd, color=cols, width=0.6)
    ax.errorbar(x, d.delta_pd, yerr=[d.delta_pd - d.ci_lo, d.ci_hi - d.delta_pd],
                fmt="none", ecolor=INK, capsize=3, linewidth=1.0)
    ax.axhline(0, color=INK, linewidth=1.4)
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=25, ha="right", fontsize=9)
    ax.set_title(f"{name}\\ndPd = Pd(unmasked) - Pd(masked)", loc="left", fontsize=10.5)
    tidy(ax)
axes[0].set_ylabel("dPd")
plt.tight_layout(); plt.show()''')

cap("Teal bars survive Bonferroni. Left: guard access gives a small but consistent detection gain. "
    "Right: at the exact-cell criterion the sign flips completely — the MASKED model is far more "
    "accurate, on every condition.")

md("""---
## 5 · Interpretation

### Against the pre-registered outcomes

The registered primary metric was ΔPd at gate ±1. Result: **ΔPd > 0 on 7/7 conditions**, mean
**+0.0150**, significant on 3 (`multi` +0.038, `masking` +0.025, `clutter` +0.024).

That is **outcome A — guard/mainlobe information does contribute** — but the effect is *small*.
Removing 24 of 189 input cells, the ones carrying ~64% of the frozen model's sensitivity, costs
about 1.5 percentage points of Pd.

### The unregistered finding, which is larger than the registered one

**At gate 0 the sign reverses on every condition**, mean ΔPd **−0.157**, significant on all 7. The
masked model is *dramatically* better at putting its peak on the exact target cell — on `isolated`,
0.511 against 0.260, nearly double.

The mechanism is the same one E3 identified from the other side. Mainlobe energy in the guard cells
makes the network's response **broad**: it fires across the whole blob, which is enough to be
matched within ±1 but not to identify the peak cell. Deny it those cells and the only evidence at a
CUT is the CUT itself against its reference ring — so the response sharpens.

**E3 and E4 have now found the same failure from opposite directions.** E3: flat 3 × 3 *labels*
broaden the response. E4: access to 3 × 3-ish mainlobe *input* broadens it too. Both are fine at
±1 and both collapse at gate 0. The localization deficit has **two independent contributing causes**,
one in the supervision and one in the input.

### Combined with E2

E2 gave CFAR the mainlobe energy and it **hurt** (7/7 conditions worse). E4 takes it away from the
CNN and detection **drops slightly** while localization **improves sharply**.

Putting them together:

> The guard/mainlobe cells carry information the learned detector genuinely uses for detection — but
> that information is not simply *energy*, because integrating the same energy into CFAR degrades it.
> And the network's use of those cells is what makes it place targets imprecisely.

**Explicitly not supported:** "the CNN wins because it sums more target energy." E2 rules that out,
and E4 shows the guard cells buy only ~1.5 pp of Pd while costing a great deal of localization.

### Limitations

* Both patch arms are **undertrained relative to the frozen dense model** — uniform 256-CUT sampling
  preserves the class distribution but gives ~67× fewer positives per epoch. Common-mode, so ΔPd is
  unbiased, but the absolute Pd of both arms is below the dense model's.
* The patch regime differs from dense training in BatchNorm statistics and loss population. That is
  why the frozen model is a reference and not the differenced arm.
* Validation split only. Nothing here restates a reported result.""")

md("""---
## 6 · Isolation verification""")

code('''print(f"protected files re-hashed: {ISO['n_protected']}")
print(f"protected files unchanged: {ISO['protected_unchanged']}")
print(f"git changes outside experiments/e4_cut_guard_mask/: "
      f"{ISO['changes_outside_e4'] or 'NONE'}")
assert ISO["protected_unchanged"] and not ISO["changes_outside_e4"]
print("\\nThe frozen TinyCNN was never retrained; 16 new patch models were trained "
      "(8 masked + 8 unmasked).")''')

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
out = pathlib.Path(__file__).resolve().parent / "e4_cut_guard_mask.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
