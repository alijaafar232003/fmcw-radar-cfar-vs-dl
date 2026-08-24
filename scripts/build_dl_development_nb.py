"""Build notebooks/dl_development.ipynb — how the learned detector was built, and why.

The polished companion to `cnn_development_log.ipynb`, which is KEPT UNCHANGED as the dated
research log. This notebook answers one question end to end:

    what does a small learned local decision rule add over classical CFAR when both operate over
    comparable Range-Doppler context, and what trade-offs does that learned rule introduce?

It is a design-and-development notebook, not a results notebook: the full matched-Pfa, bootstrap and
statistical machinery lives in `evaluation_metrics_overview.ipynb`, and the headline comparison in
`cfar_vs_dl_comparison.ipynb`. Numbers appear here only where they explain a design decision.

Every value is computed at run time from `src/dl/`, `configs/frozen_dl.yaml`, or a cached result
CSV. Nothing is pasted, and nothing is trained or re-scored here.

Idempotent; re-run after changing the model or the frozen config.
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
    """One line under a figure: what to look at."""
    md(f"*<span style='color:#6b6b6b'>{t}</span>*")


def plain(t):
    """Plain-English explanation after a technical section."""
    md(f"> **In plain English.** {t}")


# ============================================================================== header
md("""# TinyCNN — how the learned detector was built, and why

The deep-learning half of this project, from first principles to the frozen model. It is written to
be read months later by someone who has forgotten the details, and to be read *instead of* the source
code.

**The question the whole DL effort exists to answer:**

> *What does a small learned local decision rule add over classical CFAR when both operate over
> comparable Range–Doppler context, and what trade-offs does that learned rule introduce?*

Every design decision below follows from that sentence. In particular, **TinyCNN was deliberately not
built to be a good detector in the general sense** — it was built to make the comparison
interpretable. §0 explains why that distinction matters more than it might sound.

**What this notebook is not.** It is not the results notebook. The matched-Pfa protocol, the paired
bootstrap, the multiplicity correction and the localization analysis live in
`evaluation_metrics_overview.ipynb`; the headline CFAR-vs-DL comparison lives in
`cfar_vs_dl_comparison.ipynb`. Numbers appear here only where they explain a *decision*.

**What it replaces.** Nothing. `cnn_development_log.ipynb` remains the dated research log, with every
experiment in the order it happened. This notebook is the same material reorganised as an
explanation rather than a chronology; where the two differ, the log is the record of *when*, and this
notebook is the record of *why*.

Run top to bottom. Nothing here trains, scores, or modifies anything.""")

code('''import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import torch
import yaml

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map
from cfar.detect import all_peaks
from dl.dataset import normalize_rd, CLIP_LO_DB, CLIP_HI_DB, SCALE_DB
from dl.infer import cfar_valid_mask, predict_heatmap
from dl.models import TinyCNN, TinyCNNPool, count_params, measure_receptive_field
from radar.config import RadarConfig, derive_radar_params
from radar.labels import make_label_mask

CFG = RadarConfig()
DERIVED = derive_radar_params(CFG)
CFAR_CFG = CfarConfig()
FROZEN = yaml.safe_load((ROOT / "configs" / "frozen_dl.yaml").read_text(encoding="utf-8"))

SHAPE = (CFG.N // 2, CFG.M)
RES_R, RES_V = DERIVED.range_res, DERIVED.vel_res
DLR = ROOT / "results" / "dl"
NMS = 5

ORDER = ["isolated", "masking", "unresolved", "clutter", "clutter_edge", "multi", "mixed"]

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8172b3"
RED, GREY, GOLD = "#d1495b", "#b8b6ae", "#c9a227"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "grid.color": GRID,
                     "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "font.size": 10, "figure.dpi": 110})


def tidy(ax):
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


MODEL_CFG = FROZEN["model"]
TRAIN_CFG = FROZEN["training"]

MODEL = TinyCNN()
N_PARAMS = count_params(MODEL)
RF_ANALYTIC = MODEL.receptive_field
RF_MEASURED = measure_receptive_field(MODEL, size=(129, 129))

print(f"TinyCNN: {N_PARAMS:,} parameters   receptive field {RF_ANALYTIC[0]}x{RF_ANALYTIC[1]} "
      f"(analytic) / {RF_MEASURED[0]}x{RF_MEASURED[1]} (measured by backprop)")
print(f"CFAR window: {2 * CFAR_CFG.half_width_r + 1}x{2 * CFAR_CFG.half_width_d + 1}"
      f"   RD map: {SHAPE[0]}x{SHAPE[1]}   frozen epoch: {FROZEN['training']['epochs']}")''')

# ============================================================================== 0
md("""---
## 0 · Deep learning at a glance

Both detectors solve the same problem — score every cell of a Range–Doppler map, then reduce the
score map to a short list of detections. They differ in **exactly one place**: how the per-cell score
is computed.""")

code('''fig, axes = plt.subplots(2, 1, figsize=(15.0, 6.6))

flows = [
    ("CFAR", [("RD power map", GREY), ("sliding 21x9 window\\nCUT / guard / reference", BLUE),
              ("ratio = CUT / noise est.", BLUE), ("all_peaks()\\nNMS radius 5", AQUA),
              ("score threshold", AQUA), ("detections", ORANGE),
              ("shared evaluator", PURPLE)]),
    ("TinyCNN", [("RD magnitude map", GREY), ("normalize_rd()\\ndB above frame median", BLUE),
                 ("TinyCNN\\n4 dilated convs, RF 21x9", BLUE), ("all_peaks()\\nNMS radius 5", AQUA),
                 ("score threshold", AQUA), ("detections", ORANGE),
                 ("shared evaluator", PURPLE)]),
]

for ax, (name, stages) in zip(axes, flows):
    ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")
    w, gap = 11.8, 1.9
    for i, (label, c) in enumerate(stages):
        x = 3.0 + i * (w + gap)
        ax.add_patch(Rectangle((x, 2.6), w, 5.0, facecolor=c, alpha=0.14,
                               edgecolor=c, linewidth=1.5))
        ax.text(x + w / 2, 5.1, label, ha="center", va="center", fontsize=8.6)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.2, 5.1), (x + w + gap - 0.2, 5.1),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color=MUTED, linewidth=1.1))
    ax.text(0.5, 9.0, name, fontsize=13, weight="bold")

axes[0].annotate("", xy=(52, 1.6), xytext=(3, 1.6),
                 arrowprops=dict(arrowstyle="-", color=RED, linewidth=5, alpha=0.35))
axes[0].text(3, 0.4, "DIFFERENT: how each cell is scored", fontsize=9, color=RED, style="italic")
axes[1].annotate("", xy=(97, 1.6), xytext=(54, 1.6),
                 arrowprops=dict(arrowstyle="-", color=AQUA, linewidth=5, alpha=0.35))
axes[1].text(54, 0.4, "IDENTICAL: same functions, same parameters, same evaluator",
             fontsize=9, color="#0f6b4d", style="italic")
plt.tight_layout(); plt.show()''')

cap("The two arms diverge only in the first half. From `all_peaks()` onward they call the same "
    "functions with the same parameters — this is enforced structurally, not by convention: "
    "`dl.infer` imports `all_peaks` from `cfar.detect`.")

md("""### What is controlled, and what is not

| | CFAR | TinyCNN | controlled? |
|---|---|---|---|
| input | RD power | RD magnitude → dB above frame median | different representation, same frame |
| spatial extent per decision | 21 × 9 window | 21 × 9 receptive field | **matched by design** |
| decision rule | hand-derived statistic (mean / order-statistic / max / min of a reference ring) | learned function of the receptive field | **this is the variable under study** |
| peak extraction | `all_peaks()`, NMS radius 5 | the same call | **identical** |
| valid region | CFAR-testable interior | the same mask | **identical** |
| threshold | swept to matched measured Pfa | the same procedure | **identical** |
| matching / metrics | shared evaluator | the same code | **identical** |

⚠️ **Matching the spatial extent is not the same as matching the information used.** Both decide
each cell from a 21 × 9 neighbourhood, but CFAR imposes a fixed structure on that neighbourhood — a
cell under test, a guard ring excluded on purpose, and a reference ring that supplies the noise
estimate — while the network may use every cell in its receptive field in whatever way it learns.
**In particular, CFAR deliberately discards the mainlobe energy in the guard region and the network
does not have to.** This is a real and currently unresolved attribution question, revisited in §4
and §12; it is not settled anywhere in this project.""")

md("""### Why a *tiny* network, and not a normal one

**The short answer: a large CNN would have won for reasons we could not separate.**

A conventional detection CNN would improve on CFAR through several mechanisms at once — a much
larger spatial context, pooling and downsampling, far more capacity, and often its own
post-processing. If such a model beat CFAR, the result would be "a big network beats a classical
detector", which is both unsurprising and uninformative: there would be no way to say *which* of
those advantages produced the margin, and no way to rule out that the classical baseline was simply
given less context.

TinyCNN is built the other way round. Its receptive field is set to the CFAR window rather than
chosen for the network's convenience, it uses no pooling, it keeps full RD resolution, and it shares
every post-processing step with the classical arm. What is left free is the **decision rule** — and
that is precisely the thing the project set out to study.

The cost of this choice is honest and worth stating: TinyCNN is almost certainly not the strongest
network that could be built for this task. It is not meant to be. A larger model is a reasonable
*next* experiment, but it would inherit every attribution question below rather than answer one.""")

plain("""A big neural network would probably beat CFAR, but we would not learn much from that,
because it would be winning for several reasons at once and we could not tell them apart. TinyCNN is
deliberately handicapped to match CFAR's field of view and post-processing, so that the only real
difference left is *how the decision is made*. That makes the comparison mean something specific.""")

# ============================================================================== 1
md("""---
## 1 · What does TinyCNN actually receive?

**The question:** the raw RD map spans orders of magnitude and its absolute scale changes from frame
to frame. What reaches the first convolution?""")

md("""The transform is one line in `src/dl/dataset.py`:

$$x = \\mathrm{clip}\\!\\left(10\\log_{10}\\frac{|\\mathrm{RD}|^2}{\\mathrm{median}(|\\mathrm{RD}|^2)},\\;
-10,\\; +60\\right) \\Big/ 20$$

That is **decibels above the frame's own noise floor**, clipped and rescaled. Three reasons, in
order of importance:

1. **Raw power is unusable directly** — it spans roughly seven orders of magnitude, which produces
   exploding gradients.
2. **The absolute scale is not meaningful across frames.** Scene SNR is randomised per frame, so the
   noise level varies by a large factor between frames. A fixed global reference would force the
   network to spend capacity undoing that variation.
3. **The stored `rd_map_db` could not be used** — it is normalised by each frame's *maximum*, which
   destroys absolute SNR: a weak-target frame and a strong-target frame become identical.

Dividing by the **median** power gives a stable reference (targets occupy ~0.06% of cells and the
clutter ridge ~3%, so the median is essentially the noise floor) while preserving the quantity that
actually indicates a target: *how far above the noise is this cell?*""")

code('''FRAME_DIR = next(fd for fd in sorted((ROOT / "data" / "bench2_val" / "frames").iterdir())
                 if json.loads((fd / "metadata.json").read_text())["condition"] == "masking")
MAG = np.load(FRAME_DIR / "rd_map_mag.npy")
LAB = pd.read_csv(FRAME_DIR / "labels.csv")
TRUTHS = LAB[["range_bin", "velocity_bin"]].to_numpy(float)
MASK = np.load(FRAME_DIR / "label_mask.npy")

POWER = MAG.astype(np.float64) ** 2
MED = float(np.median(POWER))
DB = 10.0 * np.log10(POWER / MED)
X = normalize_rd(MAG)

rows = [
    ("frame", f"{FRAME_DIR.name} (bench2_val, condition=masking)"),
    ("raw power range", f"{POWER.min():.3e} to {POWER.max():.3e}"),
    ("dynamic range", f"{10 * np.log10(POWER.max() / max(POWER.min(), 1e-30)):.0f} dB "
                      f"({POWER.max() / max(POWER.min(), 1e-30):.1e}x)"),
    ("median power (the reference)", f"{MED:.4g}"),
    ("dB above median", f"{DB.min():+.1f} to {DB.max():+.1f} dB"),
    ("clip window", f"[{CLIP_LO_DB:+.0f}, {CLIP_HI_DB:+.0f}] dB, then / {SCALE_DB:.0f}"),
    ("network input range", f"{X.min():+.3f} to {X.max():+.3f}"),
    ("cells clipped low", f"{100 * (DB < CLIP_LO_DB).mean():.2f}%"),
    ("cells clipped high", f"{100 * (DB > CLIP_HI_DB).mean():.2f}%"),
]
display(pd.DataFrame(rows, columns=["", "value"]).style.hide(axis="index"))''')

code('''fig, axes = plt.subplots(2, 3, figsize=(15.4, 7.6))

for ax, (IMG, title, cmap, unit) in zip(axes[0], [
        (POWER, "1 - raw power |RD|^2", "viridis", "power (linear)"),
        (DB, "2 - dB above the frame median", "magma", "dB"),
        (X, "3 - network input (clipped / 20)", "magma", "input units")]):
    im = ax.imshow(IMG, aspect="auto", origin="lower", cmap=cmap,
                   norm="log" if unit.startswith("power") else None)
    ax.scatter(TRUTHS[:, 1], TRUTHS[:, 0], s=150, facecolors="none",
               edgecolors="#00b4d8", linewidths=1.5)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Doppler bin"); ax.grid(False)
    fig.colorbar(im, ax=ax, label=unit, pad=0.02)
axes[0][0].set_ylabel("Range bin")

axes[1][0].hist(POWER.ravel(), bins=np.geomspace(max(POWER.min(), 1e-6), POWER.max(), 60),
                color=GREY)
axes[1][0].set_xscale("log"); axes[1][0].set_title("raw power (log x)", loc="left", fontsize=10)
axes[1][1].hist(DB.ravel(), bins=60, color=BLUE, alpha=0.85)
for v, c in ((CLIP_LO_DB, RED), (CLIP_HI_DB, RED)):
    axes[1][1].axvline(v, color=c, linestyle="--", linewidth=1.2)
axes[1][1].set_title("dB above median (red = clip limits)", loc="left", fontsize=10)
axes[1][2].hist(X.ravel(), bins=60, color=ORANGE, alpha=0.85)
axes[1][2].set_title("network input", loc="left", fontsize=10)
for ax in axes[1]:
    ax.set_yscale("log"); tidy(ax); ax.set_ylabel("cells")
plt.tight_layout(); plt.show()''')

cap("The raw histogram is unusable as a network input — most cells sit in a narrow spike with a "
    "tail seven decades long. Taking dB relative to the frame's own median turns that into a "
    "bounded, roughly noise-centred distribution in which 0 means 'at the noise floor'.")

md("""### ⚠️ One consequence worth stating: a global scalar enters a local model

The network is convolutional and its receptive field is local (21 × 9). But the **median is computed
over the whole frame**, so every input cell is divided by a number that depends on all 16,384 cells.

That means TinyCNN is not *purely* local: it is a local function of a globally-normalised map. In
practice this is the same information CFAR uses — CFAR also compares each cell to an estimated noise
level — but CFAR estimates that level *locally*, from the reference ring, while the network is handed
a *global* estimate for free.

**Where this could matter:** on frames where the noise level is not uniform across the map, a global
median is a worse reference than a local one, and the network has no mechanism to notice. It is
recorded as a design property, not a defect, and it is not something the current experiments
isolate.""")

plain("""The network never sees raw radar power. Each frame is converted to "how many dB is this
cell above this frame's own background", which is stable from frame to frame and is essentially the
same cue CFAR uses. One subtlety: the background level is computed from the whole frame, so a global
number sneaks into an otherwise local model.""")

# ============================================================================== 2
md("""---
## 2 · What is the network trained to predict?

**The question:** the output is one number per cell. What is the target?""")

md("""**Dense per-cell supervision.** The network predicts a full 256 × 64 map — one value per RD
cell — and is trained against a binary mask of the same size. This mirrors the detection task
directly: CFAR also produces a decision per cell, so a per-cell target keeps the two arms comparable
and avoids introducing a box-regression or anchor mechanism that CFAR has no counterpart for.

**The positive region is a 3 × 3 block.** `make_label_mask(labels, shape, radius_r=1, radius_d=1)`
sets a 3 × 3 block of ones centred on each target's bin — not a single cell.""")

code('''r0, v0 = TRUTHS[0].astype(int)
sub = slice(r0 - 6, r0 + 7), slice(v0 - 6, v0 + 7)
ext = [v0 - 6.5, v0 + 6.5, r0 - 6.5, r0 + 6.5]

single = make_label_mask([{"range_bin": int(r0), "velocity_bin": int(v0)}], SHAPE,
                         radius_r=0, radius_d=0)
three = make_label_mask([{"range_bin": int(r0), "velocity_bin": int(v0)}], SHAPE,
                        radius_r=1, radius_d=1)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
for ax, (IMG, title) in zip(axes, [(X[sub], "network input (dB above median)"),
                                 (single[sub], "exact-cell label (1x1) - NOT used"),
                                 (three[sub], "the label actually used (3x3)")]):
    ax.imshow(IMG, aspect="auto", origin="lower", cmap="magma", extent=ext)
    ax.scatter(v0, r0, marker="x", s=150, color="#00b4d8", linewidths=2.4, zorder=4)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Doppler bin"); ax.grid(False)
axes[0].set_ylabel("Range bin")
plt.tight_layout(); plt.show()

pos = int(MASK.sum())
print(f"this frame: {len(LAB)} target(s) -> {pos} positive cells of {MASK.size:,} "
      f"({100 * pos / MASK.size:.3f}%)")
print(f"cells per target: {pos // max(len(LAB), 1)}  (a 3x3 block = 9, minus any overlap/clipping)")
print(f"class ratio on this frame: 1 positive per {MASK.size / max(pos, 1):.0f} cells")''')

cap("The centre panel is the label the project does NOT use; the right panel is the one it does. "
    "The difference is nine cells against one, and it is the leading suspect behind the "
    "localization behaviour in §9.")

md("""**Why a block rather than the exact cell?** Two reasons, both practical:

* **Trainability.** One positive cell in ~16,000 is an extreme imbalance. A 3 × 3 block multiplies
  the positive count by nine, which makes the gradient signal far less sparse.
* **Label precision.** The labelled bin is the nearest bin to a continuous target position, and the
  physical peak does not always land exactly there — measured alignment is ~93%, not 100%. A block
  is tolerant of that quantisation.

**⚠️ And the consequence, stated as a hypothesis.** If all nine cells around a target are labelled
positive, the network is being told that nine adjacent locations are equally correct. It is therefore
plausible that it learns a broad response rather than a sharp one, and places its maximum anywhere
inside that block.

**This is a hypothesis, not a demonstrated cause.** The evidence for it is circumstantial and is laid
out in §9: the localization deficit is real and measured, the 3 × 3 label is the obvious candidate,
and no experiment in this project has yet varied the label to test it. That experiment is Priority 1
in `PLAN.md`.""")

plain("""For every target we mark a 3×3 patch of cells as "target", not just the one true cell. That
makes training work better, but it may also teach the network that being one cell off is perfectly
fine — which would explain why it finds targets well but places them less precisely than CFAR. We
have not yet run the experiment that would prove that link.""")

# ============================================================================== 3
md("""---
## 3 · The architecture

**The question:** what is actually inside the model?

Four 3 × 3 convolutions, stride 1, no pooling, no downsampling. The only unusual feature is that the
**dilation differs between the two axes**, which is how the receptive field is shaped to match the
CFAR window (§4).""")

code('''x_probe = torch.zeros(1, 1, *SHAPE)
h, rows, rf_r, rf_d, conv_i = x_probe, [], 1, 1, 0

for layer in MODEL.net:
    h = layer(h)
    name = type(layer).__name__
    p = sum(q.numel() for q in layer.parameters())
    if name == "Conv2d":
        conv_i += 1
        rf_r += 2 * layer.dilation[0]
        rf_d += 2 * layer.dilation[1]
        rows.append(dict(layer=f"conv{conv_i}", type="Conv2d",
                         **{"in": layer.in_channels, "out": layer.out_channels},
                         kernel="3x3",
                         dil_range=layer.dilation[0], dil_dop=layer.dilation[1],
                         padding=f"{layer.padding[0]},{layer.padding[1]}",
                         feature_map=f"{tuple(h.shape)[1]}x{tuple(h.shape)[2]}x{tuple(h.shape)[3]}",
                         params=p, RF=f"{rf_r}x{rf_d}"))
    else:
        rows.append(dict(layer=f"  {name.lower()}{conv_i}", type=name, **{"in": "", "out": ""},
                         kernel="", dil_range="", dil_dop="", padding="",
                         feature_map=f"{tuple(h.shape)[1]}x{tuple(h.shape)[2]}x{tuple(h.shape)[3]}",
                         params=p, RF=""))

ARCH = pd.DataFrame(rows)
display(ARCH.style.hide(axis="index"))
print(f"total parameters: {ARCH.params.sum():,}   (count_params: {N_PARAMS:,})")
assert ARCH.params.sum() == N_PARAMS
print(f"input {tuple(x_probe.shape)} -> output {tuple(h.shape)}  "
      f"(one logit per RD cell, full resolution preserved)")''')

code('''fig, ax = plt.subplots(figsize=(15.0, 4.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")

convs = ARCH[ARCH.type == "Conv2d"].reset_index(drop=True)
blocks = [("input\\n1 x 256 x 64", GREY, "")]
for i, r in convs.iterrows():
    last = i == len(convs) - 1
    blocks.append((f"conv{i + 1}  3x3\\ndil ({r.dil_range},{r.dil_dop})\\n"
                   f"{r['in']} -> {r['out']} ch",
                   ORANGE if last else BLUE,
                   f"{r.params:,} params\\nRF {r.RF}" + ("" if last else "\\n+ BN + ReLU")))
blocks.append(("logit map\\n1 x 256 x 64", PURPLE, ""))

w, gap = 12.0, 2.4
for i, (label, c, note) in enumerate(blocks):
    x = 1.5 + i * (w + gap)
    ax.add_patch(Rectangle((x, 4.0), w, 4.4, facecolor=c, alpha=0.15,
                           edgecolor=c, linewidth=1.6))
    ax.text(x + w / 2, 6.2, label, ha="center", va="center", fontsize=8.8)
    if note:
        ax.text(x + w / 2, 3.1, note, ha="center", va="top", fontsize=7.6, color=MUTED,
                family="monospace")
    if i < len(blocks) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.3, 6.2), (x + w + gap - 0.3, 6.2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=MUTED, linewidth=1.2))
ax.text(1.5, 9.3, f"TinyCNN - {N_PARAMS:,} parameters, no pooling, full RD resolution throughout",
        fontsize=11.5, weight="bold")
ax.text(1.5, 1.2, "spatial size is identical at every layer: padding = dilation on each axis",
        fontsize=8.6, color=MUTED, style="italic")
plt.tight_layout(); plt.show()''')

cap("Note where the parameters are: the two middle 16→16 convolutions hold almost all of them, "
    "while the first and last layers are tiny because they touch only one channel.")

md("""### Three design choices, and what each buys

**No pooling, no downsampling.** The output is the same size as the input, and every layer preserves
the spatial size exactly (`padding = dilation` on each axis). This matters for a specific reason:
down- and up-sampling can displace a peak by a bin, and a one-bin error is exactly what the
localization metric measures. Keeping full resolution removes that failure mode by construction.

*(A control for this — `TinyCNNPool`, same parameter budget with a pool/upsample stage — is
implemented in `src/dl/models.py` but has **never been trained or scored**. Its receptive field also
differs, 13 × 13 measured, so it would not be a valid arm of the CFAR comparison anyway. It is
Priority 4 in `PLAN.md`.)*

**One logit per cell.** The final convolution outputs a single channel with no activation. The value
is a **logit**: a real number, unbounded in both directions, which the loss function converts to a
probability internally via a sigmoid. Positive means "more likely target than not"; large positive
means "confident".

**The network is scored on logits, never on probabilities.** The sigmoid is monotone, so no ranking,
Pd or Pfa changes — but in float32 it saturates to exactly 1.0 above a logit of about 9.7, and a
confident network puts hundreds of cells there. Those ties would destroy resolution at exactly the
low-Pfa end where the operating point sits.""")

plain("""The model is four small convolution layers stacked in a row, with no shrinking of the image
at any point, so the output is a grid the same size as the input with one number per cell. That
number is a "logit" — a score where higher means more target-like, and zero corresponds to a 50/50
call. We keep the raw scores rather than converting them to probabilities, because the conversion
squashes the very confident values together and we need to tell them apart.""")

# ============================================================================== 4
md("""---
## 4 · The receptive field, and why it is 21 × 9

**The question:** how much of the map does one output cell actually depend on, and why that much?

For stacked 3 × 3 convolutions at stride 1, the receptive field grows by twice the dilation at each
layer, independently on each axis:

$$\\mathrm{RF} = 1 + \\sum_i 2 d_i$$

With range dilations (1, 2, 3, 4) and Doppler dilations (1, 1, 1, 1) that gives 21 and 9.""")

code('''growth = (ARCH[ARCH.type == "Conv2d"][["layer", "dil_range", "dil_dop", "RF"]]
          .reset_index(drop=True).copy())
growth["RF_r"] = [int(s.split("x")[0]) for s in growth.RF]
growth["RF_d"] = [int(s.split("x")[1]) for s in growth.RF]
display(growth.style.hide(axis="index"))

print(f"analytic:  {RF_ANALYTIC[0]} x {RF_ANALYTIC[1]}")
print(f"measured (backprop from one output cell): {RF_MEASURED[0]} x {RF_MEASURED[1]}")
assert RF_ANALYTIC == RF_MEASURED, "analytic and measured receptive fields disagree"
print(f"CFAR window: {2 * CFAR_CFG.half_width_r + 1} x {2 * CFAR_CFG.half_width_d + 1}")
print(f"physical extent: {RF_MEASURED[0] * RES_R:.2f} m in range, "
      f"{RF_MEASURED[1] * RES_V:.2f} m/s in Doppler")''')

code('''fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.4))

ax = axes[0]
for i, r in growth.iterrows():
    hw_r, hw_d = r.RF_r // 2, r.RF_d // 2
    ax.add_patch(Rectangle((-hw_d - 0.5, -hw_r - 0.5), r.RF_d, r.RF_r, facecolor="none",
                           edgecolor=[BLUE, AQUA, PURPLE, ORANGE][i], linewidth=2.0))
    ax.text(hw_d + 0.6, hw_r - 0.2, f"after {r.layer}: {r.RF_r}x{r.RF_d}",
            fontsize=9, color=[BLUE, AQUA, PURPLE, ORANGE][i], va="center")
ax.scatter(0, 0, marker="x", s=160, color=INK, linewidths=2.4, zorder=5)
ax.text(0.5, 0.6, "output cell", fontsize=8.5)
ax.set_xlim(-9, 14); ax.set_ylim(-12, 12)
ax.set_xlabel("Doppler bins from centre"); ax.set_ylabel("Range bins from centre")
ax.set_title("Receptive field growth, layer by layer", loc="left", fontsize=11)
ax.grid(False); ax.set_aspect("equal")

ax = axes[1]
hw_r, hw_d = CFAR_CFG.half_width_r, CFAR_CFG.half_width_d
g_r, g_d = CFAR_CFG.guard_r, CFAR_CFG.guard_d
for rr in range(-hw_r, hw_r + 1):
    for dd in range(-hw_d, hw_d + 1):
        if abs(rr) <= g_r and abs(dd) <= g_d:
            c = ORANGE if (rr == 0 and dd == 0) else GOLD
        else:
            c = BLUE
        ax.add_patch(Rectangle((dd - 0.5, rr - 0.5), 1, 1, facecolor=c, alpha=0.35,
                               edgecolor="white", linewidth=0.4))
ax.add_patch(Rectangle((-hw_d - 0.5, -hw_r - 0.5), 2 * hw_d + 1, 2 * hw_r + 1,
                       facecolor="none", edgecolor=PURPLE, linewidth=2.6, linestyle="--"))
ax.text(hw_d + 1.0, hw_r - 0.5, "TinyCNN receptive field\\n(the whole 21x9)", fontsize=8.6,
        color=PURPLE, va="center")
ax.text(hw_d + 1.0, 0, "CFAR:\\n  orange = CUT\\n  gold = guard (EXCLUDED)\\n  blue = reference ring",
        fontsize=8.6, color=MUTED, va="center")
ax.set_xlim(-6, 16); ax.set_ylim(-12, 12)
ax.set_xlabel("Doppler bins from centre")
ax.set_title("Same 21x9 extent, different use of it", loc="left", fontsize=11)
ax.grid(False); ax.set_aspect("equal")
plt.tight_layout(); plt.show()

n_ref = CFAR_CFG.num_ref_cells
n_tot = (2 * hw_r + 1) * (2 * hw_d + 1)
print(f"cells in the 21x9 window: {n_tot}")
print(f"  CFAR reference cells (used for the noise estimate): {n_ref}")
print(f"  CFAR guard + CUT (excluded from the estimate):      {n_tot - n_ref}")
print(f"  TinyCNN: all {n_tot} are inside its receptive field")''')

cap("Left: the receptive field doubles outward on the range axis as the dilation increases, while "
    "Doppler grows by a constant 2 per layer. Right: the same 21 × 9 footprint, but CFAR spends it "
    "on a structured estimate that deliberately throws away the 25 guard and CUT cells, whereas "
    "every one of the 189 cells is inside TinyCNN's receptive field.")

md("""### Why match the CFAR window at all?

Because the alternative makes the comparison unanswerable. If the network saw more of the map than
CFAR did, any margin could be explained by context alone, and the interesting question — is a
*learned* decision rule better than a *derived* one? — would be unreachable. Matching the extent
removes the cheapest explanation for a difference.

The window itself was not chosen for the network: it is the 21 × 9 geometry selected for CFAR on a
dedicated tuning split, with guard = 2 measured from the −30 dB mainlobe half-width. The network was
built to it afterwards.

**Why the two axes differ.** Range has 256 bins and no large-scale structure; Doppler has 64 and
carries the zero-Doppler clutter ridge. A window sized for range would span a third of the Doppler
axis and drag ridge energy across a wide band of thresholds. The range dilations (1, 2, 3, 4) also
share no common factor, which avoids the "gridding" artefact that dilations like (2, 4, 8) produce by
never sampling some input positions together.

### ⚠️ Matched extent ≠ matched information

Worth repeating in the section where it is most tempting to overclaim: **the same 21 × 9 footprint
does not mean the two detectors have access to the same information.** CFAR's guard cells are
excluded *by design*, because target energy leaking into the noise estimate would raise the threshold
and mask the target. The network has no such restriction and may use the mainlobe energy in those
cells directly.

So part of the DL margin could be a better decision rule, and part could be access to energy CFAR
deliberately discards. **The current experiments cannot separate these.** That is Priority 2 in
`PLAN.md`, and it is the most important open question about the headline result.""")

plain("""Each output cell of the network depends on a 21×9 patch of the input — the same size as the
window CFAR uses. We did that on purpose, so that neither detector wins just by looking at more of
the map. But "same size patch" is not "same information": CFAR throws away the cells immediately
around the target on purpose, and the network is free to use them. We do not yet know how much of
DL's advantage comes from that.""")

# ============================================================================== 5
md("""---
## 5 · How the network learns

**The question:** what is the training setup, and which parts of it were chosen by measurement rather
than by default?""")

code('''setup = [
    ("task", "dense per-cell binary classification (one logit per RD cell)"),
    ("loss", f"{TRAIN_CFG['loss']}"),
    ("pos_weight", f"{TRAIN_CFG['pos_weight']:g}   (200 -> 974 -> 100; see below)"),
    ("optimizer", f"{TRAIN_CFG['optimizer']}, lr {TRAIN_CFG['lr']}, weight decay {TRAIN_CFG['weight_decay']}"),
    ("batch size", TRAIN_CFG["batch_size"]),
    ("epochs", f"{TRAIN_CFG['epochs']}   (chosen by rule, see section 8)"),
    ("early stopping", "OFF, deliberately - val loss plateaus later than Pd"),
    ("seeds", f"{len(TRAIN_CFG['seeds'])}  {TRAIN_CFG['seeds']}"),
    ("threads", f"{TRAIN_CFG['threads']}   (LOAD-BEARING - see section 6)"),
    ("device", TRAIN_CFG["device"]),
    ("train split", f"{FROZEN['data']['train_split']} (`unresolved` excluded from training)"),
]
display(pd.DataFrame(setup, columns=["", "value"]).style.hide(axis="index"))''')

md("""### `pos_weight`, and why it needs to exist

Targets occupy a tiny fraction of the map, so a network that predicts "background" everywhere is
right about 99.9% of the time. Plain binary cross-entropy would reward exactly that.
`BCEWithLogitsLoss(pos_weight=w)` multiplies the loss contribution of **positive** cells by `w`, so
missing a target costs `w` times more than a false alarm.

**What it does not do:** set the operating point. The ROC is produced by *sweeping* a threshold over
the score map, so `pos_weight` governs trainability and where the untuned scores sit — not the
reported Pd, which is always read at a matched false-alarm rate.

**The history matters here**, because the middle value was chosen by a principle that turned out to
be wrong.

| value | why it was used | outcome |
|---|---|---|
| **200** | inherited from the old sensor's class ratio (1:1563) | stale after the map halved |
| **974** | the *measured* inverse class frequency — "principled" exact class balancing | **retracted**: it manufactured a `clutter` deficit that did not survive |
| **100** | won the three-way ablation below | the frozen value |

The lesson is worth keeping: *exact class balancing looked like the principled default, and the
measurement disagreed with it.*""")

code('''import re
H = pd.read_csv(DLR / "checkpoint_history.csv")
h = H[H.pd.notna()].copy()


def pw_of(c):
    m = re.match(r"tinycnn_pw(\\d+)_s\\d+_ep050", str(c))
    if m:
        return int(m.group(1))
    return 974 if re.match(r"tinycnn_n\\d+_ep050", str(c)) else None


h["pw"] = h.checkpoint.map(pw_of)
h = h[h.pw.notna()]

ABL = h.pivot_table(index="condition", columns="pw", values="pd", aggfunc="mean")
ABL_SE = h.pivot_table(index="condition", columns="pw", values="pd",
                       aggfunc=lambda v: v.std(ddof=1) / np.sqrt(len(v)))
ABL = ABL.reindex([c for c in ORDER if c in ABL.index])
ABL.columns = [f"pw{int(c)}" for c in ABL.columns]
ABL["pw100 - pw974"] = ABL.pw100 - ABL.pw974
display(ABL.round(4))
print(f"seeds per arm: {h.groupby('pw').seed.nunique().to_dict()}   "
      f"(bench2_val, epoch 50, matched measured Pfa)")
d = ABL["pw100 - pw974"]
print(f"pw100 vs pw974 over {len(ABL)} conditions: "
      f"{int((d > 1e-9).sum())} better, {int((d.abs() <= 1e-9).sum())} tied, "
      f"{int((d < -1e-9).sum())} worse")
print(f"  the only loss is {list(ABL.index[d < -1e-9])} ({d.min():+.4f}) -- at the physics "
      f"ceiling, where all three arms sit")
print(f"  the largest gains are on the clutter-bearing conditions: "
      f"{', '.join(f'{c} {d[c]:+.4f}' for c in d.nlargest(3).index)}")''')

code('''fig, ax = plt.subplots(figsize=(11.0, 4.6))
xs = np.arange(len(ABL))
for j, (col, c) in enumerate([("pw100", AQUA), ("pw200", BLUE), ("pw974", RED)]):
    se = [ABL_SE.loc[i, int(col[2:])] for i in ABL.index]
    ax.bar(xs + (j - 1) * 0.27, ABL[col], width=0.27, color=c, label=col,
           yerr=se, capsize=2, error_kw=dict(linewidth=0.9, ecolor=MUTED))
ax.set_xticks(xs); ax.set_xticklabels(ABL.index, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Pd at matched measured Pfa")
ax.set_title("pos_weight ablation - 8 seeds per arm, bench2_val, epoch 50", loc="left",
             fontsize=11)
ax.legend(frameon=False, fontsize=9)
tidy(ax)
plt.tight_layout(); plt.show()''')

cap("`pw974` — exact class balancing — is worst or tied on every condition except `unresolved`, "
    "where all three sit at the physics ceiling. Its `clutter` deficit is the one that produced a "
    "retracted claim: the apparent 'DL loses in clutter' result was a property of the chosen "
    "`pos_weight`, not of the detector.")

plain("""Targets are rare, so we tell the loss function to care about them roughly a hundred times
more than about background cells. We first tried a "principled" value that exactly balanced the two
classes; measuring it showed that value was actually worse almost everywhere, and it had created a
fake weakness in cluttered scenes that we had to retract. The value we use, 100, was picked because
it won a controlled comparison, not because it is theoretically neat.""")

# ============================================================================== 6
md("""---
## 6 · Sanity checks before trusting anything

**The question:** what would have to be true before any number from this model means something?

These are cheap, they run against the real implementation, and each one closes a specific failure
mode that would otherwise be invisible in a Pd table.""")

code('''checks = []


def chk(name, ok, detail=""):
    checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=detail))


out = MODEL(torch.zeros(1, 1, *SHAPE))
chk("output is one logit per input cell", tuple(out.shape) == (1, 1, *SHAPE),
    f"{tuple(out.shape)} for input (1, 1, {SHAPE[0]}, {SHAPE[1]})")

xr = torch.randn(2, 1, *SHAPE)
chk("outputs are finite on random input", bool(torch.isfinite(MODEL(xr)).all()), "no NaN/Inf")

m2 = TinyCNN()
loss = torch.nn.BCEWithLogitsLoss()(m2(xr), torch.zeros(2, 1, *SHAPE))
loss.backward()
grads = [p.grad.abs().sum().item() for p in m2.parameters() if p.grad is not None]
chk("gradients reach every parameter tensor",
    len(grads) == len([p for p in m2.parameters()]) and all(np.isfinite(grads)),
    f"{len(grads)} tensors, total |grad| {sum(grads):.3e}")

chk("parameter count matches the frozen contract", N_PARAMS == MODEL_CFG["params"],
    f"{N_PARAMS:,} vs configs/frozen_dl.yaml {MODEL_CFG['params']:,}")

chk("analytic RF == measured RF", RF_ANALYTIC == RF_MEASURED,
    f"{RF_ANALYTIC} both ways")
chk("RF matches the CFAR window", list(RF_MEASURED) == list(MODEL_CFG["receptive_field"])
    == [2 * CFAR_CFG.half_width_r + 1, 2 * CFAR_CFG.half_width_d + 1],
    f"{RF_MEASURED} == CFAR "
    f"{(2 * CFAR_CFG.half_width_r + 1, 2 * CFAR_CFG.half_width_d + 1)}")

chk("spatial size preserved by every layer",
    all(r.feature_map.split("x")[1:] == [str(SHAPE[0]), str(SHAPE[1])]
        for _, r in ARCH.iterrows()),
    "no layer changes the RD grid size")

_ck = torch.load(DLR / f"tinycnn_frozen_s0_ep{FROZEN['training']['epochs']:03d}.pt",
                 weights_only=False)
_m = TinyCNN(width=_ck["config"].get("width", 16))
_m.load_state_dict(_ck["state_dict"])
_m.eval()
_lg = predict_heatmap(_m, MAG)
chk("a trained checkpoint loads and produces a finite score map",
    _lg.shape == SHAPE and np.isfinite(_lg).all(),
    f"logits {_lg.min():.2f} to {_lg.max():.2f}")

chk("frozen checkpoint config matches the frozen contract",
    _ck["config"]["pos_weight"] == FROZEN["training"]["pos_weight"]
    and _ck["config"]["threads"] == FROZEN["training"]["threads"],
    f"pos_weight {_ck['config']['pos_weight']:g}, threads {_ck['config']['threads']}")

CHK = pd.DataFrame(checks)
display(CHK.style.hide(axis="index"))
assert (CHK.result == "PASS").all(), "a model sanity check FAILED"
print(f"{len(CHK)} checks, all PASS")''')

md("""**What each one protects against.**

| check | the failure it would catch |
|---|---|
| output shape | a silent transpose or a resolution change; the evaluator would still run and give nonsense |
| finite outputs | exploding activations from an input scaling mistake |
| gradient flow | a disconnected branch that trains to nothing |
| parameter count | the model drifting from the frozen reproducibility contract |
| analytic == measured RF | an arithmetic slip in the dilation reasoning — the whole fairness argument rests on this number |
| RF == CFAR window | the comparison quietly ceasing to be controlled |
| spatial size preserved | an accidental pooling or padding change displacing every peak by a bin |
| checkpoint loads and scores | a checkpoint/architecture mismatch that would otherwise surface as a bad Pd |
| checkpoint matches contract | scoring the wrong arm under the right-looking filename |

*The `overfit-one-batch` gate (`src/dl/train.py`, `overfit_one_batch`) is run at training time
rather than here: it drives a handful of frames to near-zero loss and asserts the predicted peaks
land within a bin of the labels. It is a training-time gate, so re-running it here would train a
model, which this notebook deliberately does not do.*""")

plain("""Before trusting any performance number, we check the boring things: the model outputs the
right shape, the numbers are finite, gradients actually reach every layer, the parameter count and
field of view match what the frozen configuration says, and a saved model loads and produces sensible
scores. Each of these would otherwise fail silently and just look like poor detection performance.""")

# ============================================================================== 7
md("""---
## 7 · Why one training run proves nothing

**The question:** if two configurations differ by 0.02 in Pd, is that the configuration or the coin
toss?

A network's weights start from random numbers, and the data order is shuffled randomly. Two runs of
*identical* code with different seeds end at different weights and different scores. So the size of
that seed-to-seed spread sets the smallest difference any experiment in this project can honestly
claim.""")

code('''SW = pd.concat([pd.read_csv(DLR / f"epoch_sweep_epsweep_s{s}.csv").assign(seed_id=s)
                for s in range(4)])
sw = SW[SW.group == "ALL"]

fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))
for s, g in sw.groupby("seed_id"):
    g = g.sort_values("epoch")
    axes[0].plot(g.epoch, g.pd, color=[BLUE, ORANGE, AQUA, PURPLE][s], lw=1.5, label=f"seed {s}")
    axes[1].plot(g.epoch, g.dv_bins, color=[BLUE, ORANGE, AQUA, PURPLE][s], lw=1.5)
axes[0].set_ylabel("Pd (pooled, matched Pfa)"); axes[0].legend(frameon=False, fontsize=8.5)
axes[1].set_ylabel("Doppler localization error (bins)")
for ax, t in zip(axes, ["Pd vs epoch, one line per seed",
                        "Doppler error vs epoch, one line per seed"]):
    ax.set_xlabel("epoch"); ax.set_title(t, loc="left", fontsize=10.5); tidy(ax)
plt.tight_layout(); plt.show()

rows = []
for ep in sorted(sw.epoch.unique()):
    g = sw[sw.epoch == ep]
    rows.append(dict(epoch=ep, n_seeds=len(g), pd_spread=g.pd.max() - g.pd.min(),
                     dv_spread=g.dv_bins.max() - g.dv_bins.min()))
SP = pd.DataFrame(rows)
print(f"across epochs 1-{SP.epoch.max()}, {int(SP.n_seeds.median())} seeds:")
print(f"  Pd spread between seeds at a FIXED epoch: "
      f"{SP.pd_spread.min():.4f} to {SP.pd_spread.max():.4f} (median {SP.pd_spread.median():.4f})")
print(f"  Doppler-error spread at a FIXED epoch:    "
      f"{SP.dv_spread.min():.3f} to {SP.dv_spread.max():.3f} bins "
      f"(median {SP.dv_spread.median():.3f})")''')

cap("Same code, same data, same epoch — four different random seeds. The Pd curves are close "
    "together; the localization curves are not. Any claim about localization from a single run is "
    "therefore much weaker than the same claim about Pd.")

md("""**This is not a hypothetical concern — it produced a retraction.** An apparent trend in
`clutter_edge` ("still climbing past the plateau") was read from three correlated points on one
trajectory with a spread of 0.047, against a measured seed spread of 0.054 at fixed epoch. The trend
was inside the noise. It is logged as Correction C1.

**The standing rule that came out of it:** every reported configuration is run with **all eight
seeds**, and a single-seed result is not trusted for anything. Where a figure in this notebook shows
four seeds, that is the epoch-sweep experiment — deliberately cheaper because it scores 25 epochs per
seed — and its conclusions are stated with that in mind.""")

plain("""Neural networks start from random numbers, so training the same thing twice gives slightly
different results. We measured how big that random variation is, and it turned out to be about the
same size as some effects we had been getting excited about. So now everything is run eight times
with different random seeds, and we only believe a difference if it is bigger than the spread between
those runs.""")

# ============================================================================== 8
md("""---
## 8 · Choosing when to stop training

**The question:** more training keeps reducing the loss. Does it keep making the detector better?

**No — and the two halves of "better" move in opposite directions.**""")

code('''SA = pd.read_csv(DLR / "epoch_sweep_epsweep_seedavg.csv")
SEL = pd.read_csv(DLR / "epoch_sweep_epsweep_selection.csv").iloc[0]
a = SA[SA.group == "ALL"].sort_values("epoch")
CHOSEN = int(SEL.chosen_epoch)

fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4))
axes[0].plot(a.epoch, a.pd_mean, color=BLUE, lw=1.9)
axes[0].fill_between(a.epoch, a.pd_mean - a.pd_se, a.pd_mean + a.pd_se, color=BLUE, alpha=0.20)
axes[0].axhline(SEL.plateau_pd, color=MUTED, ls=":", lw=1.2)
axes[0].axhline(SEL.plateau_pd - SEL.tolerance_1se, color=MUTED, ls="--", lw=1.0)
axes[0].set_ylabel("Pd (pooled)")
axes[0].set_title("Detection: rises, then plateaus", loc="left", fontsize=10.5)

axes[1].plot(a.epoch, a.dr_mean, color=PURPLE, lw=1.9)
axes[1].fill_between(a.epoch, a.dr_mean - a.dr_se, a.dr_mean + a.dr_se, color=PURPLE, alpha=0.20)
axes[1].set_ylabel("range error (bins)")
axes[1].set_title("Range localization: gets WORSE", loc="left", fontsize=10.5)

axes[2].plot(a.epoch, a.dv_mean, color=ORANGE, lw=1.9)
axes[2].fill_between(a.epoch, a.dv_mean - a.dv_se, a.dv_mean + a.dv_se, color=ORANGE, alpha=0.20)
axes[2].set_ylabel("Doppler error (bins)")
axes[2].set_title("Doppler localization: gets WORSE", loc="left", fontsize=10.5)

for ax in axes:
    ax.axvline(CHOSEN, color=RED, lw=1.4, ls="-", alpha=0.8)
    ax.text(CHOSEN + 0.4, ax.get_ylim()[0], f" epoch {CHOSEN}", color=RED, fontsize=8.5,
            va="bottom")
    ax.set_xlabel("epoch"); tidy(ax)
plt.tight_layout(); plt.show()

e1, eN = a.iloc[0], a.iloc[-1]
print(f"epoch 1 -> {int(eN.epoch)}   ({int(a.n_seeds.iloc[0])} seeds, bench2_val, pooled)")
print(f"  Pd            {e1.pd_mean:.4f} -> {eN.pd_mean:.4f}   ({eN.pd_mean - e1.pd_mean:+.4f})")
print(f"  range error   {e1.dr_mean:.4f} -> {eN.dr_mean:.4f} bins "
      f"({eN.dr_mean - e1.dr_mean:+.4f}, {eN.dr_mean / e1.dr_mean:.2f}x)")
print(f"  Doppler error {e1.dv_mean:.4f} -> {eN.dv_mean:.4f} bins "
      f"({eN.dv_mean - e1.dv_mean:+.4f}, {eN.dv_mean / e1.dv_mean:.2f}x)")''')

md("""### The selection rule, stated before it was applied

Epoch 21 is not an eyeballed choice. The rule was written down first:

> **the earliest epoch whose pooled seed-mean Pd is within 1 standard error of the plateau.**

The rule exists because "pick the best epoch" would be selection on noise, and because the obvious
alternative — early stopping on validation loss — picks the wrong epoch here: **val loss plateaus
later than Pd does**, so it would train past the point where detection stops improving while
localization continues to degrade.""")

code('''sel = pd.DataFrame([
    ("plateau epoch", int(SEL.plateau_epoch)),
    ("plateau Pd", round(float(SEL.plateau_pd), 4)),
    ("1 SE tolerance", round(float(SEL.tolerance_1se), 4)),
    ("-> earliest epoch within tolerance", int(SEL.chosen_epoch)),
    ("Pd at the chosen epoch", round(float(SEL.chosen_pd), 4)),
    ("range error there", f"{SEL.chosen_dr:.4f} +/- {SEL.chosen_dr_se:.4f} bins"),
    ("Doppler error there", f"{SEL.chosen_dv:.4f} +/- {SEL.chosen_dv_se:.4f} bins"),
], columns=["", "value"])
display(sel.style.hide(axis="index"))
print(f"Pd given up by stopping at {int(SEL.chosen_epoch)} instead of "
      f"{int(SEL.plateau_epoch)}: {SEL.plateau_pd - SEL.chosen_pd:+.4f} "
      f"(tolerance was {SEL.tolerance_1se:.4f})")''')

cap("Stopping two epochs before the plateau costs a Pd difference smaller than one standard error, "
    "and buys back localization that continued to degrade. That is the trade the rule was written "
    "to make.")

md("""**A correction this experiment forced.** An earlier, cheaper study sampled only four epochs
(15 / 25 / 34 / 50) and concluded that "Pd is flat below 15, so the epoch does not matter". Sweeping
*every* epoch from 1 to 25 showed that is false: Pd is still climbing well below 15. The earlier
reading was an artefact of where the samples happened to land. Logged as Correction C5 — the sparse
study is kept as a historical record, not as a competing result.

⚠️ **Scope.** This sweep uses **4 seeds**, not the 8 used for the frozen arm, because it scores 25
epochs per seed. The frozen configuration was subsequently re-scored at epoch 21 with all 8.""")

plain("""Training longer keeps making the *detection* slightly better until it flattens out, but it
steadily makes the *position estimate worse* — the network becomes more confident about whether a
target is there and less precise about exactly where. We wrote down a stopping rule in advance
("stop as soon as detection is statistically indistinguishable from its best"), applied it, and got
epoch 21. An earlier study that only looked at four epochs had suggested the epoch did not matter;
looking at every epoch showed it does.""")

# ============================================================================== 9
md("""---
## 9 · The localization problem

**The question:** the network finds targets CFAR misses, yet places them less precisely. What does
that actually look like on a map?""")

code('''VALID = cfar_valid_mask(SHAPE, CFAR_CFG)
RATIO, _, _ = cfar_ratio_map(MAG ** 2, CFAR_CFG)
DL_MAP = predict_heatmap(_m, MAG)

tr, tv = TRUTHS[0].astype(int)
win = 9
sub = slice(tr - win, tr + win + 1), slice(tv - win, tv + win + 1)
ext = [tv - win - 0.5, tv + win + 0.5, tr - win - 0.5, tr + win + 0.5]

phys_r, phys_v = np.unravel_index(np.argmax(MAG[sub]), MAG[sub].shape)
phys_r += tr - win; phys_v += tv - win

dl_pk = all_peaks(DL_MAP, VALID, nms_size=NMS)
near = dl_pk[(np.abs(dl_pk[:, 0] - tr) <= win) & (np.abs(dl_pk[:, 1] - tv) <= win)]
dl_r, dl_v = (near[np.argmax(near[:, 2]), :2].astype(int) if len(near) else (np.nan, np.nan))

fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
for ax, (IMG, title, cmap) in zip(axes, [(MAG[sub], "RD magnitude (the physics)", "magma"),
                                       (RATIO[sub], "CFAR ratio map", "magma"),
                                       (DL_MAP[sub], "TinyCNN logit map", "magma")]):
    ax.imshow(IMG, aspect="auto", origin="lower", cmap=cmap, extent=ext)
    ax.add_patch(Rectangle((tv - 1.5, tr - 1.5), 3, 3, facecolor="none",
                           edgecolor="#00b4d8", linewidth=1.6, linestyle="--"))
    ax.scatter(tv, tr, marker="x", s=170, color="#00b4d8", linewidths=2.4, zorder=5,
               label="labelled bin")
    ax.scatter(phys_v, phys_r, marker="s", s=70, facecolor="none", edgecolor=GOLD,
               linewidths=1.8, zorder=5, label="physical peak")
    if np.isfinite(dl_r):
        ax.scatter(dl_v, dl_r, marker="o", s=80, facecolor=ORANGE, edgecolor=INK,
                   linewidths=0.8, zorder=5, label="TinyCNN peak")
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xlabel("Doppler bin"); ax.grid(False)
axes[0].set_ylabel("Range bin")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Dashed box = the 3x3 region the network was trained to call 'target'",
             x=0.005, ha="left", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.show()

print(f"labelled bin      ({tr}, {tv})")
print(f"physical RD peak  ({phys_r}, {phys_v})   offset ({phys_r - tr:+d}, {phys_v - tv:+d})")
if np.isfinite(dl_r):
    print(f"TinyCNN peak      ({dl_r}, {dl_v})   offset ({dl_r - tr:+d}, {dl_v - tv:+d})")
print()
print("Every offset above is inside the 3x3 block the network was trained to call target,")
print("so none of them is penalised by the training loss at all.")''')

md("""### Why detection can succeed while placement drifts

The network is trained that **all nine cells** in the dashed box are equally correct. A model that
puts its maximum on any of them incurs no training penalty. The evaluator's ±1 gate then accepts any
of those nine as a hit — so the target is *detected* — but the localization metric measures the
offset, and there the drift shows up in full.

CFAR has no such freedom: its ratio responds to the physical peak, which is sharper.""")

code('''L = pd.read_csv(DLR / "localization_frozen_ep021_test.csv")
d1 = L[(L.gate == 1) & (L.group == "ALL")].set_index("method")
best_r, best_v = d1.drop("DL").dr_bins.idxmin(), d1.drop("DL").dv_bins.idxmin()
tab = d1[["n_seeds", "dr_bins", "dr_bins_se", "dv_bins", "dv_bins_se"]].copy()
tab["range (m)"] = d1.range_m
tab["Doppler (m/s)"] = d1.vel_mps
display(tab.round(4))
print(f"pooled, bench2_test, gate +/-1 (the pre-registered headline)")
print(f"  DL Doppler error {d1.loc['DL', 'dv_bins']:.4f} vs best CFAR "
      f"({best_v}) {d1.loc[best_v, 'dv_bins']:.4f}  -> "
      f"{d1.loc['DL', 'dv_bins'] / d1.loc[best_v, 'dv_bins']:.2f}x")
print(f"  DL range   error {d1.loc['DL', 'dr_bins']:.4f} vs best CFAR "
      f"({best_r}) {d1.loc[best_r, 'dr_bins']:.4f}  -> "
      f"{d1.loc['DL', 'dr_bins'] / d1.loc[best_r, 'dr_bins']:.2f}x")
print()
print("Full analysis, including the common-detected-subset check that rules out a selection")
print("effect, is in evaluation_metrics_overview.ipynb section 6.")''')

md("""### The hypothesis, and what would test it

**Working hypothesis:** the flat 3 × 3 supervision is a contributing cause of the localization
deficit.

**What supports it:** the deficit grows with training (§8) while Pd plateaus, which is what you would
expect if the network is progressively learning the *label shape* rather than the target position;
and the offsets sit inside the labelled block, where no gradient opposes them.

**What has not been shown:** that changing the label fixes it. No experiment in this project varies
the label representation, so causality is not established. Two other explanations remain live — the
network may simply lack the capacity to localise precisely, and the ±1 gate itself bounds what the
metric can see.

Priority 1 in `PLAN.md` is the experiment that would settle it: retrain identically with 1 × 1 labels
and with soft/Gaussian labels, and see whether localization improves without costing Pd.

⚠️ There is also an unexplained **asymmetry**: range and Doppler errors degrade differently, which a
symmetric 3 × 3 label does not obviously predict. Logged as an open question (Open/O4) and not
chased.""")

plain("""The network is told that a 3×3 patch of cells all count as "the target", so it is never
punished for putting its answer one cell off. It therefore finds targets reliably but points at them
less precisely than CFAR, whose response follows the actual physical peak. We think the label shape
causes this, but we have not yet run the experiment that would prove it.""")

# ============================================================================== 10
md("""---
## 10 · What changed during development

The values in the frozen configuration are endpoints of paths, and several of those paths involved
retracting a published claim. Keeping the path is deliberate: it is the difference between a
configuration that was chosen and one that was inherited.""")

code('''hist = [
    ("receptive field", "chosen for the network", "21x9, matched to the CFAR window",
     "a mismatched field makes any margin attributable to context instead of the decision rule"),
    ("pos_weight", "200 (old sensor ratio), then 974 (exact class balancing)", "100",
     "974 was 'principled' and lost the 3-way ablation on 6 of 7 conditions; it manufactured a "
     "`clutter` deficit that was retracted (C3)"),
    ("epochs", "12 (arbitrary), then 50, then 15", "21",
     "12 stopped with val loss still falling; 50 assumed 'Pd is flat'; 15 came from 4 sampled "
     "epochs; 21 came from sweeping every epoch 1-25 under a pre-registered rule (C5)"),
    ("stopping signal", "validation loss", "Pd at matched measured Pfa",
     "val loss plateaus ~5 epochs later than Pd, so early stopping trains past the useful point"),
    ("seeds", "1", "8 (all reported)",
     "a 3-point 'trend' in `clutter_edge` had spread 0.047 against a 0.054 seed spread (C1)"),
    ("threads", "incidental", "2, frozen",
     "CPU reduction order changes the trained weights at fixed seed: max |dW| 0.38, ~1 SE on "
     "localization. Arms at different thread counts may not be pooled"),
    ("localization", "not extracted at all", "reported at both gates",
     "aggregate() always computed it; every script read only `pd`. Extracting it changed the "
     "headline from 'DL wins' to 'DL wins detection, CFAR wins placement'"),
    ("`isolated` result", "DL win, p = 6.9e-07", "re-examined with error bars on both sides",
     "CFAR's Pd had been treated as an exact constant (C4)"),
]
DH = pd.DataFrame(hist, columns=["decision", "earlier value / interpretation", "final value",
                                 "why it changed"])
display(DH.style.hide(axis="index"))''')

md("""**The pattern worth naming.** Four of the eight rows moved because a *measurement* disagreed
with something that sounded principled: exact class balancing, early stopping on loss, a visible
trend, and treating a baseline as exact. None of them was a coding error. They are the reason the
project's standing rules exist — all eight seeds, decisions on Pd rather than loss, error bars on
both arms, and a stopping rule written before it is applied.

The full dated record of each, with the reasoning at the time, is in `PROGRESS.md` and
`cnn_development_log.ipynb`; the retractions are numbered C1–C8 there.""")

# ============================================================================== 11
md("""---
## 11 · The frozen model

Generated directly from `configs/frozen_dl.yaml` and the live model — if the file and the code ever
disagree, the checks in §6 fail.""")

code('''E, D = FROZEN["evaluation"], FROZEN["data"]
card = [
    ("**model**", ""),
    ("architecture", f"{MODEL_CFG['name']} - 4 dilated 3x3 convs, no pooling, full RD resolution"),
    ("parameters", f"{N_PARAMS:,} (verified live)"),
    ("width", MODEL_CFG["width"]),
    ("dilations (range)", str(MODEL_CFG["dilations_range"])),
    ("dilations (Doppler)", str(MODEL_CFG["dilations_doppler"])),
    ("receptive field", f"{RF_MEASURED[0]} x {RF_MEASURED[1]} - measured, equals the CFAR window"),
    ("output", "one logit per RD cell, scored as logits (never probabilities)"),
    ("**data & supervision**", ""),
    ("input", D["input"]),
    ("label", TRAIN_CFG["label"]),
    ("train / val / test", f"{D['train_split']} / {D['val_split']} / {D['test_split']}"),
    ("**training**", ""),
    ("loss", f"{TRAIN_CFG['loss']}, pos_weight {TRAIN_CFG['pos_weight']:g}"),
    ("optimizer", f"{TRAIN_CFG['optimizer']}, lr {TRAIN_CFG['lr']}, batch {TRAIN_CFG['batch_size']}"),
    ("epochs", f"{TRAIN_CFG['epochs']} (pre-registered rule; early stopping off)"),
    ("seeds", f"{len(TRAIN_CFG['seeds'])} - all reported"),
    ("threads", f"{TRAIN_CFG['threads']} - load-bearing, not a speed setting"),
    ("**evaluation**", ""),
    ("operating point", f"{E['operating_point']}, target {E['target_pfa']:g}"),
    ("gate", f"+/-{E['gate']} pre-registered, +/-{E['gate_robustness']} reported beside it"),
    ("NMS", f"radius {E['nms_size']}, shared with CFAR"),
    ("uncertainty", f"{E['n_bootstrap']} paired frame bootstrap; {E['multiplicity']}"),
    ("**provenance**", ""),
    ("frozen on", FROZEN["meta"]["frozen_on"]),
    ("checkpoints", f"results/dl/tinycnn_{FROZEN['meta']['arm_tag']}_s{{0..7}}"
                    f"_ep{TRAIN_CFG['epochs']:03d}.pt"),
]
display(pd.DataFrame(card, columns=["", ""]).style.hide(axis="index"))''')

md("""**This is the authoritative controlled synthetic baseline.** Any run differing in *any* field
above is a different experiment and its numbers may not be pooled with these — including `threads`,
which changes the trained weights at fixed seed.

It is a *baseline*, not a best effort. It is the smallest model that makes the comparison
interpretable, trained under a configuration whose every value has a measurement behind it.""")

# ============================================================================== 12
md("""---
## 12 · What the DL work has established

**Supported by the completed experiments.**

* **Detection.** A receptive-field-matched learned rule finds more targets than the best classical
  variant in interference and clutter-edge conditions, at a matched measured false-alarm rate, on the
  pre-registered test split. Six of seven conditions, significant after multiplicity correction.
  *(Numbers and statistics: `evaluation_metrics_overview.ipynb` and
  `cfar_vs_dl_comparison.ipynb`.)*
* **Localization.** It places those targets substantially less precisely than CFAR — a real,
  measured trade, not an artefact of which targets each arm detected (the common-subset check rules
  that out).
* **A controlled comparison.** The two arms share peak extraction, valid region, threshold
  procedure, matching and metrics; only the per-cell scoring rule differs, and the spatial extent of
  each decision is matched by construction.
* **A configuration whose values are measured.** `pos_weight`, the stopping epoch, the seed count and
  the thread count were each fixed by an experiment, and three earlier answers were retracted along
  the way.

**Not supported, and it is important not to overstate these.**

* **Not that DL is better in general.** It wins detection and loses localization. Which matters more
  depends on the application, and this project does not decide that.
* **Attribution is unresolved.** Matched *extent* is not matched *information*: CFAR excludes its
  guard region by design and the network need not. How much of the margin is a better decision rule
  versus access to mainlobe energy is **not answered here** (Priority 2, `PLAN.md`).
* **No evidence of CFAR-like behaviour.** Every threshold is chosen per method per condition from
  labelled data. Whether a single fixed TinyCNN threshold holds a constant false-alarm rate across
  changing clutter is untested (Priority 3).
* **Not real-world generalization.** Every frame comes from one simulator, and train/val/test share a
  generator. This establishes sampling generalization only. The radar was configured to reproduce
  CARRADA's *published RD grid geometry* — not its waveform, scenes, clutter or sensor — so grid
  similarity is not domain similarity.

### Questions that remain open

* Does the 3 × 3 label cause the localization deficit, and can a different label fix it without
  costing Pd?
* How much of the detection margin survives when information access is equalised?
* Does the network have any constant-false-alarm property at a fixed threshold?
* Why do range and Doppler localization degrade *asymmetrically* under a symmetric label?
* Does any of this transfer to real measurements?

*These are recorded, in priority order and with proposed designs, in `PLAN.md`. Nothing in this
notebook attempts to answer them.*""")

md("""---
*Rebuild with `python scripts/build_dl_development_nb.py`. Every number is computed at run time from
`src/dl/`, `configs/frozen_dl.yaml`, or a cached result CSV; nothing is pasted, trained or re-scored.
The dated research log this notebook was distilled from is `cnn_development_log.ipynb`, which remains
unchanged.*""")

for _i, _c in enumerate(CELLS):
    if _c["cell_type"] == "code":
        try:
            compile(_c["source"], f"<cell {_i}>", "exec")
        except SyntaxError as _e:
            raise SystemExit(f"cell {_i} does not parse: {_e}\n---\n{_c['source']}\n---")

nb = {
    "cells": [dict(c, source=c["source"].splitlines(keepends=True)) for c in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "dl_development.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
