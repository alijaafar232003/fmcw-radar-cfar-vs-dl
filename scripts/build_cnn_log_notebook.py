"""Build notebooks/cnn_development_log.ipynb — the CNN development log.

Append a new experiment by copying the SECTION TEMPLATE block at the bottom of this file and
filling it in, then re-running this script. Keep the five headings (question / setup / results /
what it showed / what it changed) so the log stays uniform and skimmable.
"""
import json
import pathlib

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    CELLS.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": t})


# ============================================================================== header
md("""# CNN Detector — Development Log

**What this is.** The story of the deep-learning detector: what we tried, what we found, and what
we ruled out. Written so that in six months it explains not just *what the numbers were* but *why
each experiment was shaped the way it was* — including the readings we got wrong and retracted.

**How to read it.** Chronological, one section per experiment, each with the same five headings:
the question · the setup · the results · what it showed · what it changed.

**How to extend it.** Add a new section in the same format at the end of the experiment list, using
the template in `scripts/build_cnn_log_notebook.py`, then re-run that script. Results are computed
from saved artefacts in `results/`, not pasted in, so re-running refreshes them.

**Scope.** This file covers the *network*. Milestone-numbered background — the radar re-spec, the
CFAR window selection, the data-quality audit, the clutter-edge investigation — lives in
`PROGRESS.md` and is cross-referenced rather than repeated.""")

code('''import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

DL = ROOT / "results" / "dl"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
TRAIN, VAL, DLC, CFARC = "#2a78d6", "#eb6834", "#2a78d6", "#898781"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "axes.grid": True, "grid.alpha": 0.6, "figure.dpi": 120})

def history(tag):
    p = DL / f"history_{tag}.json"
    return pd.DataFrame(json.load(open(p))) if p.exists() else None

# Cumulative archive of every checkpoint ever scored (see scripts/compare_checkpoints.py).
HIST = pd.read_csv(DL / "checkpoint_history.csv") if (DL / "checkpoint_history.csv").exists() else None

# CFAR on the SAME split, for every comparison below.
CFAR_VAL = pd.read_csv(ROOT / "results" / "cfar_val" / "summary.csv")
CFAR_BEST = (CFAR_VAL.dropna(subset=["pd"]).loc[CFAR_VAL.dropna(subset=["pd"])
             .groupby("condition").pd.idxmax()][["condition", "method", "pd"]]
             .set_index("condition"))
print("CFAR best per condition on bench2_val:")
print(CFAR_BEST.round(3).to_string())''')

# ============================================================================== standing decisions
md("""---
## Standing decisions, and why

These constrain every experiment below. They were each forced by a specific finding, so the reason
is recorded with the rule.

**1 — `bench2_val` for every comparison while choices are still open. `bench2_test` stays
untouched.** The test split was measured once for the pre-registered CFAR baseline (`PROGRESS.md`
Milestone 21) and is not read again until the configuration is locked. The cost is a wider noise
floor — 200 frames per condition against test's 400 — and that is the right price for keeping the
reported numbers uncontaminated by the choices we are still making.

**2 — Validation loss is not a stopping signal here.** It plateaued from ~epoch 26 while Pd on the
clutter-bearing conditions *appeared* to still be moving. Loss is a proxy; the reported metric is Pd
at matched measured Pfa, and when the two disagreed we followed Pd. (The apparent movement later
turned out to be seed noise — see Corrections — but the principle stands: the proxy and the metric
disagreed, so the proxy lost.) **§16 sharpens this rather than reversing it:** swept at every epoch,
Pd genuinely does keep improving up to ~21 — the movement that was seed noise was the movement
*above* ~21 (Corrections C1), not the real rise below 15 that nobody had looked at. Val loss
plateaus at ~26, later than Pd does, so it would not have picked the right epoch either way.

**3 — Single-seed results are not trusted.** The noise floor must be known before any comparison is
believed. One trajectory cannot separate "the model changed" from "the evaluation is noisy"; only
varying the seed can.

**4 — Loss values are not comparable across `pos_weight` values.** The loss scale is set by
`pos_weight × positive_rate`. Only Pd at matched measured Pfa is comparable across configurations.

**5 — Operating point is always matched *measured* Pfa**, swept per condition, never design Pfa.
Rationale in `PROGRESS.md` step 5: clutter drives achieved rates 60–125× above design, by different
factors per method.""")

# ============================================================================== 1
md("""---
## 1 · Initial runs — 12 epochs (old sensor) and 50 epochs (current)

### Question
Does the pipeline work end to end, and where does training stop improving?

### Setup
| | 12-epoch run | 50-epoch run |
|---|---|---|
| when | 2026-07-31 | 2026-08-11 |
| sensor | old 256×128 | current 256×64 (CARRADA-matched) |
| data | `bench_train` 9,600 | `bench2_train` **8,400 (all)** |
| val | 800 of 1,600 | **1,600 (all)** |
| epochs | 12, fixed | 50, no early stopping |
| `pos_weight` | 200 | 974 |
| command | notebook, since deleted (`14b38bd`) | `python scripts/train_tinycnn.py --epochs 50 --tag bench` |

The 12-epoch run's numbers do not carry over — different sensor, different class ratio, and it had
three defects fixed since: it stopped at a fixed epoch count with val loss *still falling*, kept the
last epoch rather than the best, and used half the validation set.

The 50-epoch run also rebuilt the infrastructure: a script instead of a notebook (a run is now a
command with a recorded config), a pre-normalised cache (reading 8,400 `.npy` files cost 325 s/epoch
against 89 s of compute), and seeded determinism — later **verified** when a 30-epoch re-run
reproduced epochs 27–30 of the original to all five decimals.""")

code('''h = history("bench")
if h is not None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(h.epoch, h.train_loss, color=TRAIN, lw=1.8, label="train")
    ax.plot(h.epoch, h.val_loss, color=VAL, lw=1.8, label="val")
    b = int(h.val_loss.idxmin()) + 1
    ax.axvline(b, color=MUTED, ls=":", lw=1)
    ax.text(b + 0.6, h.val_loss.max(), f"best val (ep {b})", fontsize=8, color=MUTED)
    ax.axvspan(26, 50, color=MUTED, alpha=0.08)
    ax.text(27, h.train_loss.max(), "plateau", fontsize=8, color=MUTED)
    ax.set_xlabel("epoch"); ax.set_ylabel(f"BCE loss (pos_weight=974)")
    ax.set_title("50-epoch run, seed 0 — train vs validation")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.show()

    print(f"best val {h.val_loss.min():.5f} at epoch {b};  last {h.val_loss.iloc[-1]:.5f} at 50")
    print(f"train {h.train_loss.iloc[0]:.5f} -> {h.train_loss.iloc[-1]:.5f}")''')

md("""### What it showed
Gates all passed: receptive field measured **21×9** (matching the CFAR window, which is what makes
this a controlled experiment — see `PROGRESS.md` Milestone 13), overfit-one-batch collapsed 1370×
with 17/17 peaks within 1 bin, cache byte-identical to reading frames individually.

Best val 0.51529 at epoch 43, last 0.51971 at 50. Train kept falling throughout.

### What it changed
Nothing yet — the stopping question moved to the trend analysis below.""")

# ============================================================================== 2
md("""---
## 2 · Trend analysis on validation loss

### Question
Is validation loss still improving after ~epoch 26, or has it plateaued? Should we run to 100?

### Setup
Linear trend fits on the 50-epoch history above, seed 0. No new compute.""")

code('''h = history("bench")
rows = []
for lo in (20, 26, 35):
    m = h.epoch >= lo
    s = stats.linregress(h.epoch[m], h.val_loss[m])
    rows.append(dict(window=f"{lo}-50", slope=s.slope, p=s.pvalue,
                     ci_lo=s.slope - 1.96 * s.stderr, ci_hi=s.slope + 1.96 * s.stderr))
st = stats.linregress(h.epoch[h.epoch >= 26], h.train_loss[h.epoch >= 26])
rows.append(dict(window="train 26-50", slope=st.slope, p=st.pvalue,
                 ci_lo=st.slope - 1.96 * st.stderr, ci_hi=st.slope + 1.96 * st.stderr))
display(pd.DataFrame(rows).set_index("window").round(6))

noise = h.val_loss.tail(25).std()
proj = stats.linregress(h.epoch[h.epoch >= 26], h.val_loss[h.epoch >= 26]).slope * 50
print(f"epoch-to-epoch val noise (std, last 25): {noise:.4f}")
print(f"projected val change over 50 MORE epochs: {proj:+.4f}  ({abs(proj)/noise:.1f}x the noise)")
print(f"train-val gap: {h.val_loss[25]-h.train_loss[25]:+.4f} @26 -> "
      f"{h.val_loss[49]-h.train_loss[49]:+.4f} @50")''')

md("""### What it showed
Validation loss is **statistically flat**: slope +0.0002/epoch over epochs 26–50 with **p = 0.17**,
and p = 0.85 over 20–50. Training loss over the same span falls at −0.0007/epoch with **p = 1.3e-21**
— unmistakably real. The train–val gap widens from +0.009 to +0.030: mild overfitting, no upward
turn in val.

Projecting the val trend over another 50 epochs predicts a change of 0.011 against a 0.0055
epoch-to-epoch noise floor — about two noise widths. With 5,041 parameters against 8,400 frames, the
classic U-shaped validation curve should not be expected at all; flat-forever is the likely shape.

### What it changed
**Cancelled a proposed 100-epoch run** — it would have cost 90 minutes to observe a signal the same
size as the noise. And it demoted loss as the deciding quantity: the stopping question moved to Pd.

> ⚠️ At this point we drew a *further* conclusion that was wrong — that the plateau meant training
> was effectively done. See Corrections.""")

# ============================================================================== 3
md("""---
## 3 · Measured class ratio, and `pos_weight` 200 → 974

### Question
Is `pos_weight=200` justified by the actual class balance?

### Setup
Exact count over **every cell** of the cached labels for both splits — not sampled.""")

code('''from dl.dataset import cache_paths
rows = []
for split in ("bench2_train", "bench2_val"):
    y = np.load(cache_paths(split)[1], mmap_mode="r")
    pos = tot = 0
    for i in range(0, len(y), 500):
        blk = np.asarray(y[i:i + 500]); pos += int(blk.sum()); tot += blk.size
    rows.append(dict(split=split, positive=pos, total=tot, rate=pos / tot, ratio=f"1:{tot/pos:.0f}"))
display(pd.DataFrame(rows).set_index("split"))

print("loss scale is set by pos_weight x positive_rate:")
print(f"   old sensor: 200 / 1563 = {200/1563:.3f}")
print(f"   now:        974 /  974 = {974/974:.3f}   -> {(974/974)/(200/1563):.1f}x larger")
print("\\nVerified directly: the SAME trained model on the SAME val set scores")
print("   0.27555 at pos_weight=200   and   0.51529 at pos_weight=974")''')

md("""### What it showed
The imbalance is real — 144,996 positive cells in 137.6 million, per-frame counts landing on exact
multiples of 9 (the 3×3 label blob per target).

**`pos_weight=200` was never justified by anything.** On the old sensor the ratio was 1:1563, so 200
sat ~8× below the inverse frequency; it then survived a sensor change that moved the ratio to
~1:950. A guess that outlived its premise.

### What it changed
Set `pos_weight` to the measured inverse frequency, **974** — recording that this is *also* an
assumption, since exact class balancing is a choice rather than a derivation. An ablation over
{100, 200, 974} is queued (see Open).

It also made **standing decision 4** necessary: the loss scale moved 7.8×, so no loss value is
comparable across runs with different `pos_weight`.""")

# ============================================================================== 4
md("""---
## 4 · Checkpoints 30 / 43 / 50, single seed

### Question
Do checkpoints from across the loss plateau differ in the metric we actually report? If they
coincide, the stopping-epoch question is closed; if they scatter, the evaluation is noisier than the
loss curve suggests.

### Setup
`python scripts/compare_checkpoints.py --checkpoints ep030 best last` · seed 0 only ·
`bench2_val`, 1,600 frames · per-condition threshold sweep to matched measured Pfa 1e-4.

Epoch 30 was recovered by re-running 30 epochs at the same seed — deterministic, and its epochs
27–30 matched the original exactly. `--save-every` / `--save-at` were added so this cannot recur.""")

code('''single = HIST[HIST.checkpoint.isin(["tinycnn_ep30_ep030", "tinycnn_bench_best",
                                    "tinycnn_bench_last"])] if HIST is not None else None
if single is not None and len(single):
    piv = single.pivot_table(index="condition", columns="epoch", values="pd").dropna()
    piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
    display(piv.round(3).sort_values("spread"))
else:
    print("run: python scripts/compare_checkpoints.py --checkpoints ep030 best last")''')

md("""### What it showed — ambiguous, and split by condition
Clean and interference conditions were settled (spread 0.000–0.010): `unresolved` 0.488 flat,
`masking` 0.702–0.705, `isolated` and `multi` within 0.010. For those the stopping epoch does not
matter.

Clutter-bearing conditions were not, and two moved **monotonically upward**: `clutter_edge`
0.664 → 0.705 → 0.711 and `mixed` 0.489 → 0.497 → 0.515.

**The flaw, stated at the time:** these three checkpoints are the *same training trajectory*, so
their spread confounds genuine model change with evaluation noise and cannot separate them. Three
correlated points cannot establish a trend.

### What it changed
Deferred the `pos_weight` ablation until the noise floor was known, and made a like-for-like CFAR
run on `bench2_val` necessary — the CFAR baseline was on `bench2_test` and the two cannot be read
across.

> ⚠️ We nonetheless read the monotone climb as real improvement. It was not. See Corrections.""")

# ============================================================================== 5
md("""---
## 5 · CFAR on `bench2_val` — how much of the clutter gap was a split artefact?

### Question
The DL numbers are on `bench2_val`; the CFAR baseline is on `bench2_test`. Does the apparent DL
clutter deficit survive on the *same* split?

### Setup
`python scripts/cfar_test_baseline.py --split bench2_val` · same protocol as the DL evaluation:
21×9 window, per-condition sweep to matched measured Pfa 1e-4 · `bench2_test` untouched.""")

code('''cmp = CFAR_VAL.pivot_table(index="condition", columns="method", values="pd")[["CA", "GO", "OS"]]
cmp["best CFAR"] = cmp.max(axis=1)
cmp["best method"] = cmp[["CA", "GO", "OS"]].idxmax(axis=1)
display(cmp.round(3))

test = pd.read_csv(ROOT / "results" / "cfar_test" / "summary.csv")
side = pd.DataFrame({
    "CFAR val": CFAR_VAL.pivot_table(index="condition", columns="method", values="pd").max(axis=1),
    "CFAR test": test.pivot_table(index="condition", columns="method", values="pd").max(axis=1),
}).dropna()
side["val - test"] = side["CFAR val"] - side["CFAR test"]
print("\\nCFAR scores differently on the two splits — which is why cross-split reads mislead:")
display(side.round(3))''')

md("""### What it showed
CFAR scores **0.149** on `clutter` on val against **0.171** on test. So the earlier "DL 0.127 vs
CFAR 0.171" was comparing two different splits and **overstated the gap roughly twofold**. On the
same split the deficit is 0.127 vs 0.149 — about 0.022, and at that point the same size as the
0.020 drift already seen across epochs on that condition.

### What it changed
Established the like-for-like baseline every later comparison uses. Also flagged `clutter_edge`
(+0.268 over the best classical method) as needing attribution before being reported — see Open.""")

# ============================================================================== 6
md("""---
## 6 · The noise floor — 3 seeds, epochs 15 and 30

### Question
(a) What is the seed-to-seed spread at a fixed epoch — our noise floor? (b) Does `clutter_edge` keep
climbing past epoch 50?

### Setup
3 seeds × 100 epochs planned, `pos_weight=974`, checkpoints at 15/30/50/75/100.
**The laptop slept at epoch 31 and all three died** — epochs 15 and 30 survived.

The run could not be resumed because only model weights were being checkpointed; optimizer moments
and RNG state were not. Training now writes a full resume state every 5 epochs, atomically renamed,
and `--keep-awake` asks the OS not to sleep for the duration.""")

code('''ns = HIST[HIST.checkpoint.str.match(r"tinycnn_s[012]_ep0(15|30)")] if HIST is not None else None
if ns is not None and len(ns):
    for ep in sorted(ns.epoch.unique()):
        sub = ns[ns.epoch == ep].pivot_table(index="condition", columns="seed", values="pd").dropna()
        sub["spread"] = sub.max(axis=1) - sub.min(axis=1)
        print(f"EPOCH {ep} — Pd by seed (spread = the noise floor)")
        display(sub.round(3))
    ep15 = ns[ns.epoch == 15].groupby("condition").pd.mean()
    ep30 = ns[ns.epoch == 30].groupby("condition").pd.mean()
    chg = (ep30 - ep15).rename("epoch 15->30 change").to_frame()
    chg["seed spread @30"] = (ns[ns.epoch == 30].groupby("condition").pd.max()
                              - ns[ns.epoch == 30].groupby("condition").pd.min())
    chg["noise / signal"] = chg["seed spread @30"] / chg["epoch 15->30 change"].abs()
    display(chg.round(3).sort_values("seed spread @30", ascending=False))
else:
    print("run: python scripts/compare_checkpoints.py --checkpoints 's*_ep015' 's*_ep030'")''')

md("""### What it showed — the result that reshaped the plan
**The noise floor is larger than the effect.** Changing the seed moved `clutter_edge` by up to
**0.054** at one fixed epoch. Training from epoch 15 to 30 moved it by **0.006** — nine times
smaller. At epoch 30, seeds 0 and 1 both gave 0.664 while seed 2 gave 0.718.

### What it changed
1. **Killed the 100-epoch plan.** Question (b) presumes a climb the data does not support; another
   six hours would have produced another number inside the same ±0.054 band.
2. **Closed the stopping-epoch question.** Nothing moves more than 0.010 between epochs 15 and 30,
   well under the noise, so anything from ~15 onward is equivalent.
3. **Buy seeds, not epochs** — the missing quantity was an error bar, not a longer curve.
4. Set the bar for the `pos_weight` ablation: single-seed it would have to beat 0.054 to mean
   anything.

> **⚠️ Point 2 is superseded by §16; points 1, 3 and 4 stand.** The noise-floor result — the finding
> this section exists for — is unaffected. But "closed the stopping-epoch question" was doing more
> work than the data supported, twice over: it was measured on the discarded `pw974` arm, and its
> two checkpoints (15 and 30) sat *inside* the plateau, so it could not have seen the rise below
> epoch 15 that §16 measures. "Anything from ~15 onward is equivalent" is also false on
> **localization**, which §14–§16 show moves substantially across that range. §16 is the final
> answer on the stopping epoch.""")

# ============================================================================== 7
md("""---
## 7 · 8 seeds × 50 epochs — the headline comparison

### Question
What are the per-condition mean ± standard error, and does the DL clutter deficit against CFAR
survive once error bars are attached?

### Setup
```
for S in 0..7: python scripts/train_tinycnn.py --epochs 50 --seed $S --tag n$S \\
    --pos-weight 974 --save-at 15,30,50 --threads 2 --resume --keep-awake --skip-gates
```
Identical to the 50-epoch baseline except the seed. `pos_weight=974` **fixed**, not re-measured per
run, so it cannot drift between seeds. All 8,400 train / 1,600 val frames. ~525 s/epoch, 8 processes
at 2 threads. Evaluated at epoch 50 on `bench2_val`, per-condition sweep to matched measured
Pfa 1e-4.""")

code('''d50 = HIST[(HIST.epoch == 50) & HIST.checkpoint.str.startswith("tinycnn_n")] if HIST is not None else None
d50 = d50[d50.pd.notna()]

rows = []
for c, g in d50.groupby("condition"):
    if c not in CFAR_BEST.index:
        continue
    v = g.pd.values
    m, sd = v.mean(), v.std(ddof=1)
    se = sd / np.sqrt(len(v))
    cf = float(CFAR_BEST.loc[c, "pd"])
    t, p = stats.ttest_1samp(v, cf)
    rows.append(dict(condition=c, n=len(v), DL_mean=m, std=sd, SE=se,
                     ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se,
                     CFAR=cf, best=CFAR_BEST.loc[c, "method"], delta=m - cf, p=p))
res = pd.DataFrame(rows).set_index("condition").sort_values("delta")
display(res.round(4))''')

code('''fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(res))
ax.errorbar(x, res.DL_mean, yerr=1.96 * res.SE, fmt="o", color=DLC, capsize=4, ms=7,
            lw=1.6, label="DL (8 seeds, mean ± 95% CI)")
ax.plot(x, res.CFAR, "_", color=CFARC, ms=26, mew=2.5, label="best CFAR (same split)")
for i, (c, r) in enumerate(res.iterrows()):
    ax.annotate(f"{r.delta:+.3f}", (i, max(r.DL_mean, r.CFAR) + 0.03),
                ha="center", fontsize=8,
                color="#c0392b" if r.delta < 0 else MUTED)
ax.set_xticks(x); ax.set_xticklabels(res.index, rotation=20, ha="right")
ax.set_ylabel("Pd at matched measured Pfa 1e-4")
ax.set_title("DL vs CFAR on bench2_val — 8 seeds, epoch 50")
ax.legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.show()

print("per-seed Pd on `clutter` (every seed below CFAR's 0.149):")
print(np.round(np.sort(d50[d50.condition == "clutter"].pd.values), 3))''')

md("""### What it showed
**The clutter deficit survives, decisively.** DL 0.125 ± 0.0032, 95% CI [0.118, 0.131]; CFAR's GO
sits at 0.149, well outside it, p = 0.00014. **All eight seeds land below GO** (0.109–0.139). CFAR
genuinely beats the network on targets inside the clutter ridge.

That matters beyond the number. `PROGRESS.md` Milestone 17 noted that a credible learned detector
normally loses *somewhere*, and that ours previously lost nowhere. It now loses somewhere, on a
condition with a physical explanation.

Everywhere else DL wins with p < 1e-5: `mixed` +0.086, `masking` +0.049, `multi` +0.031,
`isolated` +0.022.

**Seed variance shrinks as training converges.** SE on `clutter_edge` at epoch 50 is **0.0040**
(std 0.0114, range 0.033 over 8 seeds) against the 0.054 spread measured over 3 seeds at epoch 30.
Since range normally *grows* with sample size, that is not sampling luck — the seeds genuinely
converge. So the epoch-30 noise floor overstated the floor at epoch 50.

### What it changed
Gives the first DL-vs-CFAR statement with uncertainty attached, and identifies `clutter` as the
network's genuine weakness — the anchor for the attribution work in `PROGRESS.md` Milestone 18.""")

# ============================================================================== 8
md("""---
## 8 · `pos_weight` ablation — 100 vs 200 vs 974

### Question
Does a lower `pos_weight` recover the `clutter` deficit (DL 0.125 vs GO 0.149), and **what does it
cost on the other six conditions?** The trade matters, not just the clutter number.

### Setup
```
for W in 100 200; for S in 0..7:
    python scripts/train_tinycnn.py --epochs 50 --seed $S --pos-weight $W --tag pw${W}_s$S \\
        --save-at 50 --threads 2 --resume --keep-awake --skip-gates
```
8 seeds x 50 epochs at each of `pos_weight` = 100, 200, 974 — 24 runs, everything else identical.
The 974 arm is the existing 8-seed run from §7, reused rather than repeated. Evaluated at epoch 50
on `bench2_val`, per-condition sweep to matched measured Pfa 1e-4.

Seeds are shared across arms, so the same initialisation appears at each `pos_weight` and the
comparisons below are **paired**.

> **Pd only.** Final validation losses were ~0.118 / 0.191 / 0.520 for 100 / 200 / 974 — almost
> exactly the ratio the weights predict. That is the loss scale moving, not performance, and is why
> standing decision 4 forbids comparing them.""")

code('''import re
h = HIST[HIST.pd.notna()].copy()

def pw_of(c):
    m = re.match(r"tinycnn_pw(\\d+)_s\\d+_ep050", c)
    if m:
        return int(m.group(1))
    return 974 if re.match(r"tinycnn_n\\d+_ep050", c) else None

h["pw"] = h.checkpoint.map(pw_of)
h = h[h.pw.notna()]

tab = pd.DataFrame({"CFAR best": CFAR_BEST.pd, "best method": CFAR_BEST.method})
for p in (100, 200, 974):
    g = h[h.pw == p].groupby("condition").pd
    tab[f"pw{p}"] = g.mean()
    tab[f"pw{p} SE"] = g.std(ddof=1) / np.sqrt(g.count())
tab = tab.dropna(subset=["pw100"])
tab["pw100 - CFAR"] = tab.pw100 - tab["CFAR best"]
tab["pw974 - CFAR"] = tab.pw974 - tab["CFAR best"]
display(tab.round(3))''')

code('''# Paired by seed: the same initialisation at each pos_weight, so this is a within-seed contrast.
rows = []
for c in sorted(h.condition.unique()):
    a = h[(h.condition == c) & (h.pw == 974)].sort_values("seed").pd.values
    b = h[(h.condition == c) & (h.pw == 100)].sort_values("seed").pd.values
    d = h[(h.condition == c) & (h.pw == 200)].sort_values("seed").pd.values
    rows.append(dict(condition=c, pw974=a.mean(), pw100=b.mean(),
                     **{"100-974": b.mean() - a.mean(), "p (paired)": stats.ttest_rel(b, a).pvalue,
                        "100-200": b.mean() - d.mean(), "p 100v200": stats.ttest_rel(b, d).pvalue}))
paired = pd.DataFrame(rows).set_index("condition")
display(paired.round(4))

fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(tab))
for p, col, mk in ((100, "#2a78d6", "o"), (200, "#7a5cd0", "^"), (974, "#eb6834", "s")):
    ax.errorbar(x + (p == 200) * 0.12 - (p == 100) * 0.12, tab[f"pw{p}"],
                yerr=1.96 * tab[f"pw{p} SE"], fmt=mk, color=col, capsize=3, ms=6,
                lw=1.4, label=f"pos_weight {p}")
ax.plot(x, tab["CFAR best"], "_", color=CFARC, ms=26, mew=2.5, label="best CFAR")
ax.set_xticks(x); ax.set_xticklabels(tab.index, rotation=20, ha="right")
ax.set_ylabel("Pd at matched measured Pfa 1e-4")
ax.set_title("pos_weight ablation — 8 seeds each, bench2_val")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()

cf = float(CFAR_BEST.loc["clutter", "pd"])
print("clutter — the question:")
for p in (100, 200, 974):
    v = h[(h.condition == "clutter") & (h.pw == p)].pd.values
    t, pv = stats.ttest_1samp(v, cf)
    print(f"   pw{p:<4} {v.mean():.3f} +/- {v.std(ddof=1)/np.sqrt(len(v)):.4f}   "
          f"vs GO {cf:.3f}   gap {v.mean()-cf:+.3f}   p={pv:.2g}")''')

md("""### What it showed — **not a trade, a strict improvement**

**The `clutter` deficit essentially vanishes.** At `pos_weight` 100 the gap against GO is **−0.006
(p = 0.27)** and at 200 it is **−0.005 (p = 0.15)** — neither distinguishable from zero. At 974 it
was **−0.025 (p = 0.00013)**. Lower weighting recovers ~0.019 of the 0.025, and what remains is
inside the noise.

**Nothing pays for it.** Paired by seed, 100 against 974:

| condition | Δ Pd | p |
|---|---|---|
| mixed | **+0.029** | 3.2e-05 |
| clutter_edge | **+0.026** | 0.0016 |
| clutter | **+0.019** | 5.7e-06 |
| multi | +0.008 | 0.019 |
| isolated | +0.001 | 0.76 |
| masking | +0.000 | 1.00 |
| unresolved | −0.001 | 0.35 |

Four conditions improve significantly; three are unchanged; **none degrades**. There is no trade to
report, which is itself the finding — the expected tension between detecting rare positives and
suppressing false alarms does not appear, because the operating point is chosen by sweeping the
threshold afterwards and `pos_weight` only has to make the *ranking* learnable.

**100 and 200 are effectively equivalent** — the only condition separating them is `mixed`
(+0.011, p = 0.0015 in favour of 100). Everything else has p ≥ 0.2.

**`pos_weight = 974` was actively wrong.** It was set in §3 as a "principled default" — the measured
inverse class frequency, i.e. exact class balancing — and it is the worst of the three values tested,
never better on any condition. Exact balancing was an assumption, flagged as such at the time, and
the measurement rejects it.

### What it changed
1. **`pos_weight` moves to 100** (or 200 — pick 100 on `mixed`, the only significant separation).
2. **The §7 headline is retracted**: "CFAR genuinely beats the network on `clutter`" was a statement
   about a badly-chosen hyperparameter, not about the detectors. See Corrections C3.
3. All §7 numbers are superseded and must be re-run at the chosen `pos_weight` before anything is
   reported.""")

# ============================================================================== 9
md("""---
## 9 · Error bars on both sides — and a multiplicity correction

### Question
Every DL-vs-CFAR p-value so far used `ttest_1samp(dl_seeds, cfar_value)`, which treats CFAR's Pd as
an **exact constant**. It is not — it is one draw over 200 frames per condition, carrying sampling
error of the same order as the effects being claimed. What survives when CFAR gets error bars too,
and when the 7 simultaneous comparisons are corrected for?

### Setup
`python scripts/bootstrap_comparison.py --pos-weight 100` · `bench2_val` · cached CFAR peaks, 8 DL
checkpoints, **no retraining**.

**Paired frame bootstrap.** Resample frames with replacement; within each resample recompute Pd for
CFAR and for every DL seed *on the same frames*, and take the difference — so frame difficulty
cancels instead of adding noise. 2,000 resamples, percentile CI, bootstrap p-value.
**Bonferroni** across the 7 scored conditions: α = 0.05 → **0.0071**.

*Stated limitation:* detections are taken at each method's matched-Pfa threshold computed once on
the full condition, so the interval covers sampling error in Pd at a fixed operating point but not
the variability of *choosing* that threshold. Redoing the sweep per resample would cost ~200× the
compute. The intervals are therefore mildly optimistic.""")

code('''bs = DL / "bootstrap_pw100.csv"
if bs.exists():
    b = pd.read_csv(bs).set_index("condition").sort_values("diff", ascending=False)
    display(b.round(4))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(b))
    # One errorbar call per row: `ecolor` takes a single colour, not a per-point list.
    for i, (c, r) in enumerate(b.iterrows()):
        col = "#2a78d6" if r["sig (Bonf)"] else "#b9b7b0"
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]],
                    fmt="o", ms=8, capsize=4, lw=1.6, color=col, ecolor=col)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(b.index)
    ax.set_xlabel("Pd difference, DL - best CFAR  (95% bootstrap CI)")
    ax.set_title("Blue = survives Bonferroni correction; grey = does not")
    plt.tight_layout(); plt.show()

    print(f"survives correction: {int(b['sig (Bonf)'].sum())} of {len(b)}")
    print("ties (CI crosses zero):", list(b.index[b['crosses 0']]))
else:
    print("run: python scripts/bootstrap_comparison.py --pos-weight 100")''')

md("""### What it showed — the headline shrinks, and that is good news

**Four of seven survive** correction: clutter_edge (+0.288), mixed (+0.116), masking (+0.049),
multi (+0.038). **Three are ties:** clutter (−0.006, p = 0.69), unresolved (+0.001, p = 0.91), and —
the one that changed — **`isolated` (+0.023, p = 0.031)**, which fails the corrected threshold.

`isolated` had been reported at **p = 6.9e-07** from the seed-only t-test. Giving CFAR its own
sampling error and correcting for 7 comparisons removed the claim entirely. The nominal 95% CI
[0.002, 0.046] still excludes zero, so the honest phrasing is *positive but not significant after
correction* — not *"DL beats CFAR on clean noise"*.

**Why this strengthens the project.** `PROGRESS.md` Milestone 17 records that a well-built CNN is
expected to **tie** CFAR in homogeneous noise (RadCNN S4, Gusland), and flagged our win there as
*"the strongest single indication that something still favours the network."* With correct
statistics it **is** a tie. The anomaly was a statistical artefact, not an unfair advantage.

The result now reads: **DL wins decisively where interference or clutter is present, and ties CFAR
on clean noise, on unresolvable pairs, and inside the clutter ridge.** That shape matches the
literature; "wins everywhere" did not.

### What it changed
1. Every previously reported DL-vs-CFAR p-value is superseded by this table.
2. The `isolated` win is withdrawn (Corrections C4).
3. Same defect applies to `notebooks/cfar_baseline.ipynb`, whose CFAR method-to-method comparisons
   are bare point estimates judged against a rule of thumb — corrected there too.""")

# ============================================================================== 10
md("""---
## 10 · SO-CFAR — completing the classical baseline

### Question
`masking` is one of the surviving DL wins, and **SO-CFAR — the multiple-target specialist — was
missing from the baseline.** Where GO takes the *greatest* half-ring to survive a clutter edge, SO
takes the *smallest*, so an interfering target in one half cannot inflate the noise estimate and
mask its neighbour. If SO beats OS on `masking`, that claim narrows.

### Setup
Added to `src/cfar/cfar.py` as `np.minimum` of the half-rings (GO's mirror), sharing GO's half-ring
α. `python scripts/cfar_test_baseline.py --split bench2_val`, four classical methods.""")

code('''cv = pd.read_csv(ROOT / "results" / "cfar_val" / "summary.csv").dropna(subset=["pd"])
piv = cv.pivot_table(index="condition", columns="method", values="pd")
order = [m for m in ("CA", "GO", "OS", "SO") if m in piv.columns]
piv = piv[order]
piv["best"] = piv.idxmax(axis=1)
display(piv.round(3))
print("SO is best on:", list(piv.index[piv.best == "SO"]) or "no condition")''')

md("""### What it showed — a clean negative result
**SO does not beat OS on `masking`** (0.638 vs 0.645) — a hair behind, so OS was already the right
champion and the DL margin there is unchanged.

SO is severely worse wherever clutter appears: 0.065 on `clutter` against GO's 0.149, 0.205 on
`clutter_edge`, 0.213 on `mixed`. That is the exact mirror of GO — taking the *smallest* half-ring
**under**estimates noise at an edge, so matching the measured false-alarm rate forces SO's threshold
up and its Pd collapses. GO 0.149 vs SO 0.065 on `clutter` is the sharpest statement of GO/SO
complementarity in the project, and worth a sentence in the report.

### What it changed
Nothing in the conclusions — **SO is never the best classical method on any condition**, so
"best CFAR" per condition is unchanged and every DL comparison stands. The baseline is now four
variants including both half-ring designs, which is what makes it defensible rather than convenient.""")

# ============================================================================== 11
md("""---
## 11 · Tolerance sweep — is the ±1 gate doing the work?

### Question
The evaluator credits a detection within ±1 bin, and the label mask is a **3×3 block** — so the CNN
is trained to fire on exactly the region the evaluator rewards. CFAR gets no such instruction; it
responds wherever the physical peak lands. That is `PROGRESS.md` Milestone 16's open asymmetry #5,
recorded and never tested: *"±1-bin gate may measure localisation rather than detection."*

### Setup
`python scripts/tolerance_sweep.py --pos-weight 100` · gates of ±1, ±2, ±3 · `bench2_val` · 8 seeds.
Each gate gets its **own** threshold sweep, because the gate also sets the guard region excluded
from the false-alarm denominator. Absolute Pd therefore rises for everyone as the gate widens; only
the DL-minus-CFAR *difference* is comparable across gates.""")

code('''ts = DL / "tolerance_sweep_pw100.csv"
if ts.exists():
    t = pd.read_csv(ts)
    d = t.pivot_table(index="condition", columns="tol", values="diff")
    d["change 1->3"] = d[3] - d[1]
    display(d.round(3).sort_values("change 1->3"))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for c in d.index:
        style = dict(lw=2.4, color="#c0392b") if c == "clutter" else dict(lw=1.4, color=MUTED)
        ax.plot([1, 2, 3], d.loc[c, [1, 2, 3]], "o-", **style)
        ax.annotate(c, (3.04, d.loc[c, 3]), fontsize=8, va="center",
                    color="#c0392b" if c == "clutter" else MUTED)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks([1, 2, 3]); ax.set_xlim(0.9, 3.9)
    ax.set_xlabel("matching gate (bins)"); ax.set_ylabel("Pd difference, DL - best CFAR")
    ax.set_title("Margins hold as the gate widens — except clutter, which moves against DL")
    plt.tight_layout(); plt.show()
else:
    print("run: python scripts/tolerance_sweep.py --pos-weight 100")''')

md("""### What it showed — falsified, except on `clutter`
**Six of seven margins hold**, every change ≤0.013 and inside the noise. Widening the gate removes
the localisation requirement altogether, and DL's advantage survives it. The objection is
**falsified** for every condition where we claim a win: this is detection, not peak placement.

⚠️ **`clutter` is the exception, and it moves against DL** — the deficit triples, −0.006 → −0.038.
The mechanism is asymmetry #5 pointing the *other* way: on `clutter` the RD peak lies within 1 bin
of the label only **77%** of the time (Milestone 22), so CFAR — which detects the physical peak —
was being denied credit at ±1, while DL, trained on label positions, already sat inside the tight
gate. Widening the gate hands CFAR back detections it was already making.

### What it changed
**At ±1 the `clutter` comparison is biased in DL's favour**, so "clutter is a tie" (§9) may
understate CFAR. The ±1 CI was [−0.030, +0.017] and −0.038 sits at its edge, so this needs its own
bootstrap before being claimed either way — run at ±3 in §12.""")

# ============================================================================== 12
md("""---
## 12 · NMS sweep — is `unresolved` capped by physics or by suppression?

### Question
`unresolved` sits at Pd ≈ 0.5 and we attribute it to physics: two targets inside one mainlobe, one
detection can match only one truth. But NMS suppresses local maxima within (2·nms+1) bins, and
`nms_size=5` is an 11×11 window — 2.15 m in range, far wider than the ~2-bin mainlobe. **If NMS is
what merges the pair, the cap is a post-processing artefact, not a property of the radar.**
`masking` is included because step 6d documents the same setting capping it near Pd 0.69.

### Setup
`python scripts/nms_sweep.py` · nms_size ∈ {1, 2, 3, 5} · `bench2_val`, 4 DL seeds. Score maps are
computed once per frame and NMS applied afterwards at each width, so only the suppression changes.""")

code('''ns = DL / "nms_sweep.csv"
if ns.exists():
    n = pd.read_csv(ns)
    for cond in n.condition.unique():
        d = n[n.condition == cond]
        print(f"{cond} — Pd at matched measured Pfa 1e-4")
        display(d.pivot_table(index="method", columns="nms", values="pd").round(3))
    print("detections per target (dominated by false alarms at ~1.3/frame, so only a rise")
    print("ABOVE the other methods is interpretable as duplication):")
    display(n[n.condition == "unresolved"].pivot_table(index="method", columns="nms",
                                                       values="det_per_target").round(2))
else:
    print("run: python scripts/nms_sweep.py")''')

md("""### What it showed — physics on `unresolved`, and a conservative choice on `masking`
**`unresolved` is flat from nms 2 to 5 for every method** (0.482–0.485). Shrinking the window from
11×11 to 5×5 changes nothing, because the pairs are 0.26–1.79 range bins apart — below the measured
0.39 m equal-strength limit, so no suppression width can separate them. Only the disqualified
`nms_size=1` lifts anything, and it lifts **DL most** (0.554) while raising DL's
detections-per-target to 1.22 against 1.16 for the others: the free-attempts regime step 6d rejected,
where the evaluator forgives duplicates inside a truth's guard region. **The 0.5 cap is physics.**

**On `masking` the width matters, and our choice costs us.** DL−OS is **+0.052 at nms 5** against
**+0.079 at nms 1** — `nms_size=5` suppresses genuinely resolvable neighbours and **shrinks** the
claimed DL advantage rather than manufacturing it. CA and GO are flat throughout because they cannot
resolve these pairs at any width.

*Caveat on our own metric:* detections-per-target is dominated by false alarms (~1.3/frame at
Pfa 1e-4), so only the nms=1 rise above the other methods reads as duplication.

### What it changed
Both objections resolved in the project's favour, and neither claim moves. `nms_size=5` stays.""")

# ============================================================================== 13
md("""---
## 13 · The `clutter` claim at ±3 — CFAR does win there

### Question
§11 showed `clutter` is the one condition whose margin depends on the matching gate, moving
**against** DL as the gate widens. §9's "clutter is a tie" was measured at ±1, the gate that
penalises CFAR for detecting the physical peak rather than the labelled bin. What is the claim at
±3, where that penalty is removed?

### Setup
`python scripts/bootstrap_comparison.py --pos-weight 100 --tolerance 3` · same paired frame
bootstrap, 2,000 resamples, Bonferroni α = 0.0071 · DL peaks reused from the cache, since peaks do
not depend on the gate.""")

code('''rows = []
for tol in (1, 3):
    f = DL / f"bootstrap_pw100_tol{tol}.csv"
    f = f if f.exists() else (DL / "bootstrap_pw100.csv" if tol == 1 else None)
    if f is None or not f.exists():
        continue
    d = pd.read_csv(f).set_index("condition")
    d["tol"] = tol
    rows.append(d.reset_index())

if rows:
    b = pd.concat(rows)
    view = b.pivot_table(index="condition", columns="tol",
                         values=["cfar_pd", "dl_pd", "diff", "p_boot"])
    display(view.round(4))

    c = b[b.condition == "clutter"].set_index("tol")
    print("clutter, the condition in question:")
    for tol in c.index:
        r = c.loc[tol]
        print(f"   gate +/-{tol}:  CFAR {r.cfar_pd:.4f} ({r.cfar_method})   DL {r.dl_pd:.4f}   "
              f"diff {r['diff']:+.4f}   CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]   p={r.p_boot:.3f}")
    print("\\nCFAR gains far more from widening the gate than DL does -- which is the")
    print("77% label-to-peak alignment on this condition (Milestone 22) being paid back.")
else:
    print("run: python scripts/bootstrap_comparison.py --pos-weight 100 --tolerance 3")''')

md("""### What it showed — DL does lose somewhere
At ±3, **CFAR beats DL on `clutter` by 0.038**, CI [−0.069, −0.007] **excluding zero**, p = 0.013.
GO rises 0.149 → **0.192** while DL moves only 0.144 → 0.153: CFAR gains three times as much from
the wider gate, which is exactly the 77% alignment on this condition being paid back.

**Honest verdict:** nominally a CFAR win, but p = 0.013 does **not** survive the Bonferroni
threshold of 0.0071. Report it as *"CFAR is better on targets inside the clutter ridge; the effect
is at the edge of significance once corrected for seven comparisons."*

This partly restores what `PROGRESS.md` Milestone 17 says a credible learned detector should show —
a condition where it loses. Once CFAR is not penalised for detecting the physical peak rather than
the labelled bin, the clutter ridge is that condition.

### ⚠️ `isolated` is borderline, and no claim should rest on it
p = 0.031 at ±1 and p = 0.001 at ±3, while the estimate barely moves (+0.022 → +0.026). A 30× swing
in p from a 0.004 shift in the difference is **tail sensitivity, not evidence** — the estimate sits
right at the significance boundary at both gates. State it as *"small positive, borderline"*, and do
not let a conclusion turn on which side of 0.0071 a bootstrap tail probability happens to land.

### What it changed
The pre-registered ±1 gate stays the headline, with ±3 reported as the robustness check beside it.
Two claims are now gate-qualified rather than flat: `clutter` (CFAR better at ±3) and `isolated`
(borderline at both).""")

# ============================================================================== 14
md("""---
## 14 · Localization error — the one metric where CFAR clearly wins

### Question
`aggregate()` has always computed localization (mean |Δrange|, |Δdoppler|, RMSE, and the same in
metres and m/s), and `cfar_baseline.ipynb` reports it for CFAR. **It was never extracted for the DL
side** — every script computed it and then read only `pd`. It is also a proposal-required metric.
So: how precisely does each detector place the targets it finds?

### Setup
`python scripts/localization_table.py --per-condition` · `bench2_val` · CFAR from cached peaks,
DL averaged over 8 seeds at `pos_weight=100` · matched measured Pfa 1e-4 · reported at the
pre-registered ±1 gate and at ±3.

The metric is **bounded by the gate**: a detection displaced further than the gate is a miss plus a
false alarm, not a large error, so severe displacement shows up as lost Pd instead. Both gates are
reported for that reason.""")

code('''lz = DL / "localization_pw100.csv"
if lz.exists():
    L = pd.read_csv(lz)
    for tol in sorted(L.gate.unique()):
        d = L[(L.gate == tol) & (L.group == "ALL")].set_index("method")
        print(f"all conditions pooled, gate +/-{tol}")
        display(d[["pd", "dr_bins", "dv_bins", "rmse_bins", "range_m", "vel_mps"]].round(4))

    d1 = L[(L.gate == 1) & (L.group == "ALL")].set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    order = [m for m in ("CA", "GO", "OS", "SO", "DL") if m in d1.index]
    col = ["#2a78d6" if m == "DL" else MUTED for m in order]
    axes[0].bar(order, d1.loc[order, "dv_bins"], color=col)
    axes[0].set_ylabel("mean |Doppler error| (bins)"); axes[0].set_title("Localization — Doppler")
    axes[1].scatter(d1.loc[order, "dv_bins"], d1.loc[order, "pd"], c=col, s=90)
    for m in order:
        axes[1].annotate(m, (d1.loc[m, "dv_bins"], d1.loc[m, "pd"]),
                         xytext=(6, 4), textcoords="offset points", fontsize=9)
    axes[1].set_xlabel("mean |Doppler error| (bins)"); axes[1].set_ylabel("Pd")
    axes[1].set_title("The trade: DL detects more, CFAR places better")
    plt.tight_layout(); plt.show()

    best = d1.drop("DL").dv_bins.idxmin()
    print(f"DL Doppler error is {d1.loc['DL','dv_bins']/d1.loc[best,'dv_bins']:.1f}x the best "
          f"CFAR ({best}); range error {d1.loc['DL','dr_bins']/d1.drop('DL').dr_bins.min():.1f}x.")

    print("\\nPer condition, gate +/-1 (Doppler error in bins):")
    display(L[(L.gate == 1) & (L.group != "ALL")]
            .pivot_table(index="group", columns="method", values="dv_bins").round(3))
else:
    print("run: python scripts/localization_table.py --per-condition")''')

md("""### What it showed — the trade the project was missing

| gate ±1 | Pd | \\|Δr\\| bins | \\|Δv\\| bins | range (m) | vel (m/s) |
|---|---|---|---|---|---|
| CA | 0.482 | 0.216 | 0.101 | 0.042 | 0.042 |
| GO | 0.497 | 0.218 | 0.119 | 0.043 | 0.050 |
| **OS** | 0.481 | 0.218 | **0.086** | 0.043 | **0.036** |
| SO | 0.237 | **0.179** | 0.142 | **0.035** | 0.060 |
| **DL** | **0.587** | 0.498 | 0.485 | 0.097 | 0.204 |

**DL's Doppler error is 5.6× the best CFAR and its range error 2.8×** — 0.204 m/s against OS's
0.036 m/s. The ordering is identical at ±3, so it is not a gate artefact.

**The likely cause is the 3×3 label blob.** The network is trained that all nine cells around a
target are "target", so its peak may land anywhere inside that block; CFAR responds to the physical
peak, which is sharper. This is Milestone 16's asymmetry #5 — *"the ±1 gate may measure localisation
rather than detection"* — measured directly rather than argued.

Two details worth carrying into the report: **SO has the best range localization** (0.179 bins)
despite by far the worst Pd, consistent with a low threshold that only survives on sharp peaks; and
DL's error barely moves between gates (0.485 → 0.492 Doppler), meaning its displaced detections were
already inside the tight gate.

### What it changed
The headline is no longer one-sided. **DL detects more (Pd 0.587 vs 0.497); CFAR localizes far
better.** That is a genuine trade, it is proposal-required, and it is the strongest answer available
to *"does the network actually beat CFAR?"* — the honest answer being *"at finding targets, in
interference; not at placing them."*

It also suggests a concrete improvement to try later: a **smaller or Gaussian-weighted label**
instead of a flat 3×3 block would stop rewarding the network for firing anywhere in the blob.""")

# ============================================================================== 15
md("""---
## 15 · Epoch flatness at the reported configuration — and localization degrades with training

### Question
Two gaps. **The flatness evidence was measured on the wrong arm** — "nothing moves between epochs 15
and 30" came from the `pos_weight=974` runs, the value later shown to be the wrong choice, and had
never been demonstrated at `pos_weight=100`. And **localization had only ever been measured at
epoch 50**, so we did not know whether it depends on training length at all.

### Setup
4 seeds × `--save-at 15,25,34,50` at `pos_weight=100` (same seeds as the reported config, so the
trajectories are identical — only the intermediate checkpoints are new).
`python scripts/epoch_flatness.py` · `bench2_val` · matched measured Pfa 1e-4 · gate ±1.""")

code('''ef = DL / "epoch_flatness_epflat.csv"
if ef.exists():
    E = pd.read_csv(ef)
    a = E[E.group == "ALL"]
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    summ = a.groupby("epoch").agg(pd_mean=("pd", "mean"), pd_se=("pd", se),
                                  dr=("dr_bins", "mean"), dr_se=("dr_bins", se),
                                  dv=("dv_bins", "mean"), dv_se=("dv_bins", se))
    display(summ.round(4))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ep = summ.index.to_numpy()
    axes[0].errorbar(ep, summ.pd_mean, yerr=1.96 * summ.pd_se, fmt="o-", color=DLC,
                     capsize=4, lw=1.8, ms=7)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Pd at matched measured Pfa 1e-4")
    axes[0].set_title(f"Pd is flat (range {summ.pd_mean.max()-summ.pd_mean.min():.4f})")

    axes[1].errorbar(ep, summ.dv, yerr=1.96 * summ.dv_se, fmt="o-", color="#c0392b",
                     capsize=4, lw=1.8, ms=7, label="Doppler")
    axes[1].errorbar(ep, summ.dr, yerr=1.96 * summ.dr_se, fmt="s-", color=MUTED,
                     capsize=4, lw=1.8, ms=6, label="range")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("mean |error| (bins)")
    axes[1].set_title("Localization gets WORSE with training"); axes[1].legend(frameon=False)
    plt.tight_layout(); plt.show()

    piv = E[E.group != "ALL"].pivot_table(index="group", columns="epoch", values="dv_bins")
    piv["15 -> 50"] = piv[50] - piv[15]
    print("Doppler localization error by condition (bins):")
    display(piv.round(3))
    print(f"conditions degrading monotonically: "
          f"{int((piv[[15,25,34,50]].diff(axis=1).iloc[:,1:] >= -0.005).all(axis=1).sum())} of {len(piv)}")
else:
    print("run: python scripts/epoch_flatness.py")''')

md("""### What it showed — Pd flat, localization worse

| epoch | Pd | range err (bins) | Doppler err (bins) |
|---|---|---|---|
| **15** | 0.5825 | **0.467** | **0.385** |
| 25 | 0.5846 | 0.457 | 0.421 |
| 34 | 0.5875 | 0.468 | 0.460 |
| 50 | 0.5858 | 0.517 | 0.454 |

**Pd is flat** — total range 0.0050 against a mean SE of 0.0022 — so the flatness claim now holds at
the configuration we actually report, not only at the discarded `pw974` arm.

**But localization degrades with training**: Doppler error +18% (0.385 → 0.454), range +11%. It is
not a pooling artefact of the aggregate — **every one of the seven conditions degrades
monotonically**, from `unresolved` (0.278 → 0.364) to `clutter` (0.667 → 0.748). Seven of seven in
the same direction is far stronger evidence than the ~2 SE of the pooled figure suggests.

**The mechanism is the 3×3 label again.** With longer training the network grows confident across
the whole nine-cell block, so its peak within that block becomes progressively less sharp. Training
longer teaches it the *label shape*, not the target position — the same root cause as §14.

### What it changed
**The frozen epoch count moves from 50 to 15.** The earlier argument ("50 is fine, Pd is flat") only
looked at Pd; on the full picture epoch 15 costs **0.003 Pd (~1 SE, marginal)** and buys **15%
better localization**. That makes the epoch count a measured choice rather than an arbitrary one.

*Stated limitation:* nothing below epoch 15 was tested, so the optimum may be earlier still. Recorded
as a limitation rather than another sweep — the real fix is a better label (smaller or
Gaussian-weighted), not a shorter run.

> **⚠️ Superseded by §16.** The limitation stated in the last paragraph — *nothing below epoch 15
> was tested* — was then tested. §16 sweeps every epoch from 1 to 25 under a pre-registered
> selection rule and is the final answer on the stopping point; the epoch-15 recommendation above
> is **not**. Read this section for the localization mechanism, §16 for the epoch count.""")

# ============================================================================== 16
md("""---
## 16 · The full epoch sweep — every epoch from 1 to 25, under a pre-registered rule

### Question
§15 chose epoch 15 from four sampled points {15, 25, 34, 50} and flagged its own limitation:
**nothing below epoch 15 had ever been evaluated.** Four points cannot distinguish a curve that is
flat from one that was merely sampled where it happens to be flat. This measures *every* epoch, so
the stopping point is read off a curve instead of interpolated between four samples.

Two secondary questions it is sized to answer:

1. **Does localization degrade monotonically from epoch 1, or start flat and then turn?** Monotone
   from the very start supports the 3×3-label explanation — the network is learning the label
   *shape* from the first epoch. A flat-then-turn shape would point at something else, e.g. late
   overfitting, and would weaken §14/§15's mechanism.
2. **How far back does Pd stay flat?** Never tested below 15.

### Setup
4 seeds × 25 epochs at `pos_weight=100`, checkpointing **every** epoch → 100 checkpoints, all
evaluated on `bench2_val` at matched measured Pfa 1e-4, gate ±1. Same seeds and config as the
reported arm, so the trajectories are identical to §15's `epflat` runs — verified: seed 0 epoch 1
reproduces train 0.20474 / val 0.14189 exactly.

    for S in 0..3: python scripts/train_tinycnn.py --epochs 25 --pos-weight 100 --seed $S \\
        --tag epsweep_s$S --save-at 1,2,...,25 --threads 4 --keep-awake --skip-gates --resume
    python scripts/epoch_sweep.py --tag epsweep --seeds $S    # 4 procs, then --summarise

### ⚠️ The selection rule, fixed before any result existed
Written into `PROGRESS.md` **while the sweep was still running**, precisely so it could not be
re-negotiated once the curve was on screen:

> **plateau** = the maximum over epochs of the pooled seed-mean Pd.
> **tolerance** = the SE of the pooled seed-mean Pd *at the plateau epoch*.
> **chosen epoch** = the **earliest** epoch whose pooled seed-mean Pd ≥ plateau − tolerance.

"Earliest within 1 SE of the plateau" encodes the actual preference: among epochs that detect
equally well, take the least-trained one, because localization degrades with training and
localization is where CFAR wins (§14). Ties in Pd are broken by the metric we are losing on —
and that is a stated prior, not a post-hoc choice.

`scripts/epoch_sweep.py` reproduces `epoch_flatness.py`'s published numbers **exactly** (all 32
seed-0 rows, all three metrics, zero difference) before being used for anything new.

**Artefacts.** `results/dl/epoch_sweep_epsweep.csv` (100 checkpoints × 8 groups, the raw scores) ·
`_seedavg.csv` · `_selection.csv` (the rule's output) · `results/epoch_sweep_epsweep.png` (the same
four-panel figure the cell below draws) · per-checkpoint score caches in
`results/dl/epoch_sweep_cache/`, so a re-run costs nothing.""")

code('''es = DL / "epoch_sweep_epsweep.csv"
if es.exists():
    S = pd.read_csv(es)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    avg = (S.groupby(["group", "epoch"])
             .agg(pd_mean=("pd", "mean"), pd_se=("pd", se), dr=("dr_bins", "mean"),
                  dr_se=("dr_bins", se), dv=("dv_bins", "mean"), dv_se=("dv_bins", se))
             .reset_index())
    pooled = avg[avg.group == "ALL"].sort_values("epoch").reset_index(drop=True)

    # THE PRE-REGISTERED RULE, applied verbatim.
    k = int(pooled.pd_mean.idxmax())
    plateau, tol = float(pooled.pd_mean[k]), float(pooled.pd_se[k])
    chosen = pooled[pooled.pd_mean >= plateau - tol].iloc[0]
    print(f"plateau  Pd {plateau:.4f} at epoch {int(pooled.epoch[k])};  1 SE = {tol:.4f}"
          f"  -> floor {plateau - tol:.4f}")
    print(f"CHOSEN   epoch {int(chosen.epoch)}:  Pd {chosen.pd_mean:.4f}   "
          f"range {chosen.dr:.3f}+/-{chosen.dr_se:.3f}   Doppler {chosen.dv:.3f}+/-{chosen.dv_se:.3f} bins")
    display(pooled[["epoch", "pd_mean", "pd_se", "dr", "dr_se", "dv", "dv_se"]].round(4))

    # Four panels, not three: on a shared scale with the per-condition curves (clutter 0.14 to
    # isolated 0.73) the entire +0.059 rise this experiment exists to measure is an invisible
    # wiggle, so the pooled Pd gets its own zoomed axis. Same figure as
    # results/epoch_sweep_epsweep.png, produced by scripts/epoch_sweep.py.
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1, 1, 1]})
    conds = sorted(c for c in avg.group.unique() if c != "ALL")
    cmap = plt.get_cmap("tab10")

    ax = axes[0]
    ax.axhspan(plateau - tol, plateau, color="crimson", alpha=0.13, zorder=0,
               label="within 1 SE of the plateau")
    ax.errorbar(pooled.epoch, pooled.pd_mean, yerr=pooled.pd_se, color="k", lw=2.2,
                marker="o", ms=4, capsize=2, zorder=5)
    ax.set_ylabel("pooled Pd (zoomed)"); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Pre-registered rule selects epoch {int(chosen.epoch)} "
                 f"(plateau {plateau:.4f} at epoch {int(pooled.epoch[k])}, 1 SE = {tol:.4f})")

    for ax, (m, s, lab) in zip(axes[1:], [("pd_mean", "pd_se", "Pd by condition"),
                                          ("dr", "dr_se", "range error (bins)"),
                                          ("dv", "dv_se", "Doppler error (bins)")]):
        for j, c in enumerate(conds):
            g = avg[avg.group == c].sort_values("epoch")
            ax.plot(g.epoch, g[m], lw=1.0, alpha=0.5, color=cmap(j % 10), label=c)
        ax.errorbar(pooled.epoch, pooled[m], yerr=pooled[s], color="k", lw=2.2, marker="o",
                    ms=3.5, capsize=2, zorder=5, label="pooled")
        ax.set_ylabel(lab); ax.grid(alpha=0.25)
    for ax in axes:
        ax.axvline(int(chosen.epoch), color="crimson", ls="--", lw=1.4, zorder=4)
    axes[1].legend(fontsize=7, ncol=4, loc="lower right")
    axes[-1].set_xlabel("epoch")
    plt.tight_layout(); plt.show()

    # Q1 - monotone from epoch 1, or flat then turning? Segment slopes, since the whole question
    # is WHERE the curve turns, which a single correlation over 1-25 cannot answer.
    print("Q1  localization shape (pooled)")
    for col, name in (("dr", "range  "), ("dv", "Doppler")):
        y, e = pooled[col].to_numpy(), pooled.epoch.to_numpy()
        print(f"  {name}  min at epoch {int(e[np.argmin(y)]):>2} | "
              f"rho(all) {stats.spearmanr(e, y).statistic:+.3f}")
        for lo, hi in [(1, 8), (8, 15), (15, 25)]:
            m_ = (e >= lo) & (e <= hi)
            sl = stats.linregress(e[m_], y[m_])
            print(f"      epochs {lo:>2}-{hi:<2}  slope {sl.slope:+.5f}/epoch  p={sl.pvalue:.4f}"
                  f"   total {y[m_][-1]-y[m_][0]:+.3f}")
    # ... and is that shape the same in every condition, or an artefact of pooling?
    mins = {c: (int(avg[avg.group == c].sort_values("epoch").iloc[
                    np.argmin(avg[avg.group == c].sort_values("epoch").dr.to_numpy())].epoch),
                int(avg[avg.group == c].sort_values("epoch").iloc[
                    np.argmin(avg[avg.group == c].sort_values("epoch").dv.to_numpy())].epoch))
            for c in conds}
    print(f"  per condition: range min at epoch 1 in "
          f"{sum(1 for r, _ in mins.values() if r == 1)}/{len(conds)}; "
          f"Doppler min at epoch 2-4 in {sum(1 for _, d in mins.values() if 2 <= d <= 4)}/{len(conds)}")

    # Q2 - how far back does Pd stay flat?
    q2 = pooled[["epoch", "pd_mean", "pd_se"]].copy()
    q2["deficit"] = plateau - q2.pd_mean
    q2["deficit_in_SE"] = q2.deficit / tol
    q2["within_1SE"] = q2.deficit <= tol
    print("\\nQ2  deficit from the plateau, in units of the 1-SE tolerance")
    display(q2.round(4))

    # The chosen epoch against the two values previously argued for, PAIRED BY SEED.
    W = S[S.group == "ALL"].pivot(index="seed", columns="epoch")
    ef = pd.read_csv(DL / "epoch_flatness_epflat.csv")
    e50 = ef[(ef.group == "ALL") & (ef.epoch == 50)].set_index("seed")
    ch = int(chosen.epoch)
    rows = []
    for m, name in [("pd", "Pd"), ("dr_bins", "range (bins)"), ("dv_bins", "Doppler (bins)")]:
        a15, a_ch = W[m][15], W[m][ch]
        b50 = e50[m].reindex(a_ch.index).dropna()
        rows.append({"metric": name,
                     f"{ch} vs 15": f"{a_ch.mean() - a15.mean():+.4f} "
                                    f"(p={stats.ttest_rel(a_ch, a15).pvalue:.3f})",
                     f"{ch} vs 50": f"{a_ch.loc[b50.index].mean() - b50.mean():+.4f} "
                                    f"(p={stats.ttest_rel(a_ch.loc[b50.index], b50).pvalue:.3f})"})
    print(f"Chosen epoch vs the two previously-argued values, paired by seed "
          f"({len(W)} seeds; epoch 50 from the epflat run)")
    display(pd.DataFrame(rows).set_index("metric"))

    # Winner's curse: the plateau is a max over 25 noisy estimates, so it is biased upward.
    tail = pooled[pooled.epoch >= 20].pd_mean.mean()
    print(f"winner's-curse check: plateau {plateau:.4f} (max of {len(pooled)}) vs "
          f"{tail:.4f} averaged over epochs 20-25 -> gap {plateau - tail:.4f} "
          f"= {(plateau - tail) / tol:.2f} SE, which biases the rule LATER")

    print("\\nPd by condition (seed mean)")
    display(avg[avg.group != "ALL"].pivot_table(index="group", columns="epoch",
                                                values="pd_mean").round(3))
    print("Doppler error by condition (bins)")
    display(avg[avg.group != "ALL"].pivot_table(index="group", columns="epoch",
                                                values="dv").round(3))
else:
    print("run: python scripts/epoch_sweep.py --tag epsweep  then  --summarise")''')

md("""### What it showed — the rule picks epoch 21, and "Pd is flat" does not survive

| | |
|---|---|
| plateau | Pd **0.5862** at epoch 23 |
| tolerance (1 SE at the plateau epoch) | **0.0023** → floor 0.5839 |
| **chosen epoch** | **21** |
| Pd there | **0.5844 ± 0.0019** |
| range error there | **0.442 ± 0.022** bins |
| Doppler error there | **0.409 ± 0.036** bins |

**The flatness claim was an artefact of where the four points were sampled.** Pooled Pd rises
**+0.059** from epoch 1 (0.5272) to the plateau — roughly **26 SE**. §15 sampled {15, 25, 34, 50},
**all four of which sit in the flat top**, and the whole rise happens below 15 where nothing had
ever been measured. The runs agree *exactly* where they overlap (epoch 15 → Pd 0.5825, range 0.4667,
Doppler 0.3848, identical to §15; epoch 25 likewise), so this is not two measurements disagreeing —
it is a blind spot in the sampling.

**Q2 — how far back does Pd stay flat?** Deficit from the plateau, in units of the 1-SE tolerance:

| epoch | 1 | 5 | 10 | 14 | 15 | 18 | 20 | 21 |
|---|---|---|---|---|---|---|---|---|
| deficit (SE) | 25.7 | 9.8 | 3.9 | 2.4 | **1.6** | 1.8 | 1.1 | **0.8 ✓** |

Epoch 15 sits **1.6 SE below the plateau** — outside the pre-registered band. Strict 1-SE flatness
begins at **21**; a looser 2-SE reading reaches back to ~14; below 10 the curve is unambiguously
still climbing.

**Q1 — monotone from epoch 1? Not quite, and the two axes disagree.**

| axis | min at | epochs 1–8 | 8–15 | 15–25 | total 1→25 |
|---|---|---|---|---|---|
| range | epoch **1** | **+0.141** (p = 0.0014) | +0.016 (p = 0.025) | −0.010 (p = **0.51**) | +0.147 |
| Doppler | epoch **4** | +0.038 | **+0.145** (p < 1e-4) | +0.036 (p = 0.0005) | +0.220 |

**Range degrades from epoch 1 and then saturates** — flat from 15 on, slope indistinguishable from
zero. **Doppler improves slightly to epoch ~4, then turns and never saturates** — still climbing
significantly through 15–25. Both shapes hold across **all seven conditions** (range minimum at
epoch 1 in 7/7; Doppler minimum at epoch 2–4 in 7/7), so this is structure rather than seed noise.

**For the 3×3-label explanation this is partial support, plus one thing it does not predict.** The
"monotone from the very start" signature holds cleanly on **range**. It fails on Doppler, which has
a genuine early *improving* phase. And a 3×3 label is **symmetric**, so it predicts the same timing
on both axes — instead Doppler degrades 1.5× as much (+0.220 vs +0.147) and is still going when
range has stopped. The label mechanism survives but is no longer sufficient alone. Untested
candidates for the asymmetry: the receptive field is **21×9**, itself asymmetric; Doppler has 64
bins against range's 256, so the label block spans a larger fraction of the axis; and the clutter
ridge is a Doppler-axis structure — `clutter` has by far the worst Doppler error (0.711 bins at
epoch 21) and the steepest climb. Logged in **Open** as O4, not chased: it changes no published
number.

**Epoch 21 against the two previously-argued values, paired by seed:**

| comparison | Pd | range (bins) | Doppler (bins) |
|---|---|---|---|
| 21 vs **15** | +0.0019 (p = 0.14) | −0.025 (p = 0.37) | +0.024 (p = 0.20) |
| 21 vs **50** | −0.0014 (p = 0.77) | **−0.075 (p = 0.004)** | **−0.045 (p = 0.036)** |

**21 vs 50 is a real improvement** — identical Pd, significantly better localization on both axes —
so §15's *direction* was right. **21 vs 15 is inside the noise on every metric**, so the practical
cost of §15's choice was small. What was wrong was its stated *reason*, not badly wrong its number.

*Caveat on the rule itself, not on the data:* the plateau is the max over 25 noisy estimates, so it
carries a winner's-curse bias upward — 0.5862 against 0.5845 averaged over epochs 20–25, about
0.7 SE. That bias pushes the chosen epoch **later**, i.e. conservative on Pd and against our
localization preference. Noted rather than patched: the rule was pre-registered, and re-deriving it
after seeing the curve is precisely what pre-registration exists to prevent.

### What it changed
**The frozen epoch count moves from 15 to 21**, and §15's flatness argument is retracted as
**C5** — the first correction in this log where the *stopping-epoch* conclusion itself moved rather
than merely sharpened. The epoch count is now selected by a rule fixed in advance, applied to a
curve measured at every point, rather than read off four samples.""")

# ============================================================================== 17
md("""---
## 17 · `--threads` is a reproducibility parameter, not a speed knob

### Question
Incidental, and worth more than the question that prompted it. §16 trained fresh seeds 0–3 to get
epoch-21 checkpoints. The 8 reported seeds (`pw100_s0..s7`) have **only epoch-50** checkpoints, so
before re-running all eight the obvious shortcut was: *can the §16 runs stand in for the reported
seeds 0–3, since they share seed and config?*

### Setup
Compare `pw100_s0` against `epflat_s0` — same seed, same `pos_weight`, same optimizer, same epoch
50. Their `config.json` files differ in exactly one field: **`threads`, 2 vs 4**. And the training
code is byte-identical between the two commits:

    git diff 27a1f46 b1db7ee -- src/dl/ src/eval/     # empty""")

code('''import subprocess
a, b = "config_pw100_s0.json", "config_epflat_s0.json"
ca, cb = (json.load(open(DL / f)) for f in (a, b))
diff = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb)
        if k not in ("tag", "epochs", "started", "git_commit", "gates") and ca.get(k) != cb.get(k)}
print("pw100_s0 vs epflat_s0, config differences:", diff or "NONE")
print(f"  commits: {ca['git_commit']} vs {cb['git_commit']}")

# Same seed, same code -> do the WEIGHTS differ at all?
try:
    import torch
    wa = torch.load(DL / "tinycnn_pw100_s0_ep050.pt", weights_only=False)["state_dict"]
    wb = torch.load(DL / "tinycnn_epflat_s0_ep050.pt", weights_only=False)["state_dict"]
    md_ = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
    print(f"  max |weight difference| across all 5,041 params: {md_:.3e}")
except Exception as e:
    print("  (weights unavailable:", e, ")")

# And what does that cost in the REPORTED metrics?
p2 = DL / "epoch_sweep_pw100_s0.csv"          # threads=2 arm, scored by scripts/epoch_sweep.py
p4 = DL / "epoch_flatness_epflat.csv"         # threads=4 arm
if p2.exists() and p4.exists():
    A = pd.read_csv(p2)
    B = pd.read_csv(p4); B = B[(B.seed == 0) & (B.epoch == 50)]
    M = A.merge(B, on=["seed", "group"], suffixes=("_2thr", "_4thr"))
    M["d_pd"] = (M.pd_2thr - M.pd_4thr).abs()
    M["d_dr"] = (M.dr_bins_2thr - M.dr_bins_4thr).abs()
    M["d_dv"] = (M.dv_bins_2thr - M.dv_bins_4thr).abs()
    print("\\nSame seed, same epoch, threads 2 vs 4 — |difference| in the reported metrics:")
    display(M.set_index("group")[["d_pd", "d_dr", "d_dv"]].round(4))
    # For scale: the 4-seed SE at the chosen epoch.
    es = DL / "epoch_sweep_epsweep.csv"
    if es.exists():
        S_ = pd.read_csv(es); S_ = S_[(S_.group == "ALL") & (S_.epoch == 21)]
        sef = lambda x: x.std(ddof=1) / np.sqrt(len(x))
        print(f"for scale, the 4-seed SE at epoch 21:  Pd {sef(S_.pd):.4f}   "
              f"range {sef(S_.dr_bins):.4f}   Doppler {sef(S_.dv_bins):.4f}")
else:
    print("run: python scripts/epoch_sweep.py --tag pw100 --seeds 0")''')

md("""### What it showed — same seed, same code, different weights

Torch's CPU kernels reduce in a different order at a different thread count, so the floating-point
sums differ and the difference compounds over tens of epochs of Adam. `seed_everything()` cannot
prevent this: **it is not an RNG effect.**

**And the divergence is not cosmetic.** Max |weight difference| across the 5,041 parameters is
**0.38** — these are not the same model with rounding noise on it, they are genuinely different
points in parameter space that happen to detect equally well. A last-bit perturbation in epoch 1 is
amplified by 50 epochs of a momentum optimizer into a different local basin. That is the honest
reason the two arms cannot be interchanged, and it is a stronger statement than the metric table
below, which only shows they *perform* alike.

| pooled metric | threads 2 vs 4 | for scale |
|---|---|---|
| Pd | 0.0013 | well under the 0.0023 seed SE — negligible |
| range error | **0.0224** | ≈ 1 seed SE (0.0217) |
| Doppler error | **0.0284** | ≈ 1 seed SE (0.0363) |

Per condition it reaches 0.103 bins (range, `isolated`) and 0.079 (Doppler, `clutter`).

**Why it lands on localization and not Pd.** The matched-*measured*-Pfa threshold shifts slightly,
changing **which marginal detections are included**. Pd moves by one or two borderline targets out
of ~1,700 — invisible. Localization is averaged over exactly those newly-included marginal
detections, which are the badly-localised ones — so the same perturbation is ~1 SE there.

**This is one draw, not a variance estimate** — one seed, one epoch. It shows the effect is of order
one seed's noise; it does not establish a distribution, and nothing suggests either thread count is
*better*. It is a nuisance parameter to hold fixed, not to optimise.

### What it changed
1. **§16 is unaffected** — every `epsweep` run and the epoch-50 arm it is compared against used
   `threads=4`, so its within-experiment conclusions hold throughout.
2. **Arms may not be mixed.** The §16 seeds 0–3 are a *different numerical realization* from the
   reported `pw100` seeds 0–3 and cannot stand in for them, so the epoch-21 re-freeze re-runs **all
   eight** seeds at one fixed thread count rather than reusing four.
3. **`threads` joins the frozen config**, beside seed, lr and `pos_weight`. It was already written
   to `config_*.json` — which is how this was caught; what was missing was treating it as
   load-bearing.
4. Retro-qualifies §15's *"same seeds as the reported config, so the trajectories are identical"* —
   true at fixed threads, false across the 2-vs-4 boundary that actually separated those runs. It
   changes none of §15's numbers, which were internally consistent at 4 threads.""")

# ============================================================================== 18
md("""---
## 18 · The headline re-scored at epoch 21 — and the configuration frozen

### Question
Every DL-vs-CFAR number in §9–§14 was measured on epoch-50 checkpoints. §16 moved the frozen
stopping point to **21**, so all of them describe a model we no longer report. What survives, what
moves, and can the configuration finally be locked?

### Setup
8 seeds × 21 epochs at `pos_weight=100`, **`--threads 2` throughout** (§17), from commit `50755c9`.
`bench2_val`, matched measured Pfa 1e-4, paired frame bootstrap 2,000 resamples, Bonferroni across
the 7 conditions.

    for S in 0..7: python scripts/train_tinycnn.py --epochs 21 --pos-weight 100 --seed $S \\
        --tag frozen_s$S --save-at 21 --threads 2 --keep-awake --skip-gates --resume
    python scripts/bootstrap_comparison.py --tag frozen --epoch 21 --tolerance 1   # and 3
    python scripts/localization_table.py  --tag frozen --epoch 21 --per-condition

**The frozen arm IS the reported arm, stopped earlier.** At `threads=2` these runs reproduce the
original `pw100` trajectories **bit-for-bit** — epoch-21 val loss identical to 9 decimals on all 8
seeds — so the 21-vs-50 comparison is paired by construction, and §17's diagnosis is confirmed.""")

code('''b1 = DL / "bootstrap_frozen_ep021_tol1.csv"
b3 = DL / "bootstrap_frozen_ep021_tol3.csv"
lz = DL / "localization_frozen_ep021.csv"

# The epoch-50 tables are KEPT, and read here as the superseded comparison.
old1, old3 = DL / "bootstrap_pw100.csv", DL / "bootstrap_pw100_tol3.csv"
oldl = DL / "localization_pw100.csv"

if b1.exists() and b3.exists():
    for gate, newp, oldp in ((1, b1, old1), (3, b3, old3)):
        n = pd.read_csv(newp).set_index("condition")
        cmp_ = pd.DataFrame({"ep21_diff": n["diff"], "ep21_p": n.p_boot, "ep21_sig": n["sig (Bonf)"]})
        if oldp.exists():
            o = pd.read_csv(oldp).set_index("condition")
            cmp_.insert(0, "ep50_sig", o["sig (Bonf)"])
            cmp_.insert(0, "ep50_diff", o["diff"])
            cmp_["verdict"] = ["same" if a == b else ("NOW SIG" if b else "NO LONGER SIG")
                               for a, b in zip(cmp_.ep50_sig, cmp_.ep21_sig)]
        print(f"gate +/-{gate}:  epoch 50 -> epoch 21")
        display(cmp_.round(4))

if lz.exists():
    L = pd.read_csv(lz)
    p1 = L[(L.gate == 1) & (L.group == "ALL")].set_index("method")
    display(p1[["n_seeds", "pd", "pd_se", "dr_bins", "dr_bins_se", "dv_bins", "dv_bins_se"]].round(4))

    # THE headline ratio, re-measured. Never carry the epoch-50 value over.
    ratio = p1.loc["DL", "dv_bins"] / p1.drop("DL").dv_bins.min()
    best = p1.drop("DL").dv_bins.idxmin()
    print(f"DL Doppler error = {ratio:.2f}x the best CFAR ({best}) at epoch 21")
    if oldl.exists():
        O = pd.read_csv(oldl)
        o1 = O[(O.gate == 1) & (O.group == "ALL")].set_index("method")
        print(f"  ... against {o1.loc['DL','dv_bins'] / o1.drop('DL').dv_bins.min():.2f}x at epoch 50 (SUPERSEDED)")

        # Did it improve everywhere, or only on average?
        a = O[(O.gate == 1) & (O.method == "DL")].set_index("group")
        b = L[(L.gate == 1) & (L.method == "DL")].set_index("group")
        t = pd.DataFrame({"dr_50": a.dr_bins, "dr_21": b.dr_bins, "dr_d": b.dr_bins - a.dr_bins,
                          "dv_50": a.dv_bins, "dv_21": b.dv_bins, "dv_d": b.dv_bins - a.dv_bins})
        print(f"improved on range in {(t.dr_d < 0).sum()}/{len(t)} groups, "
              f"Doppler in {(t.dv_d < 0).sum()}/{len(t)}")
        display(t.round(4))
else:
    print("run the three commands in the Setup block above")''')

md("""### What it showed — Pd holds, `clutter` hardens into a real loss, localization improves

**Pd at the ±1 gate is structurally unchanged** — the same 4 of 7 significant, the same 3 ties, all
deltas within a few thousandths. Exactly what §16 predicted (21 vs 50 Pd is a tie, p = 0.77).

**⚠️ One conclusion changed, at the ±3 gate: `clutter` is now a *significant* loss.**
−0.0429, **p = 0.005**, clearing the Bonferroni threshold of 0.0071 — where at epoch 50 it was
−0.038, p = 0.013, reported as "edge of significance" (§13). **This strengthens our position.**
Milestone 17 records that a credible learned detector normally loses *somewhere*; ours now does so
demonstrably rather than marginally. It is the one condition where the classical method wins on
detection, and it is no longer arguable.

**Localization improves, and the headline ratio moves:**

| pooled, gate ±1 | epoch 50 | epoch 21 |
|---|---|---|
| DL range error | 0.498 | **0.473** bins |
| DL Doppler error | 0.485 | **0.393 ± 0.025** bins |
| **Doppler vs best CFAR (OS)** | **5.6×** | **4.6×** |
| range vs best CFAR (SO) | 2.79× | 2.65× |

Improved in **8 of 8** groups on **both** axes — not an averaging artefact.

**The "5.6× worse Doppler" figure is retired.** It must be quoted as **4.6× at epoch 21** and never
carried over from 50. The ~5× estimate written into `NEXT_SESSION.md` on 16 Aug is also superseded:
it extrapolated a 4-thread paired shift onto a 2-thread table, which §17 shows is exactly the kind
of mixing that costs ~1 SE on localization. The measured value is 4.6×.

*Caveat carried forward:* this ratio's denominator is a single CFAR run with its own frame-sampling
error, which the ratio does not propagate. `bootstrap_comparison.py` is the source for intervals
that do — the ratio is a headline, not an inference.

### What it changed
**🔒 The configuration is frozen** → **`configs/frozen_dl.yaml`**, validated field-by-field against
the `config_frozen_s0.json` the run actually produced. It records model, training, data, evaluation,
the CFAR baseline, environment and the exact commands — including `threads: 2` as a load-bearing
field, which is the concrete outcome of §17.

The epoch-50 tables are **kept unmodified** as the superseded record, per the project's rule that
the path is part of the result.

**`bench2_test` is now unblocked.** It is the next thing to touch — once, with the table
pre-registered before anything is read (N4).""")

# ============================================================================== 19
md("""---
## 19 · `bench2_test` — the one-shot, and three predictions that failed

### Question
The reportable numbers. Run once, against `docs/test_preregistration.md`, which fixed the tables,
the protocol, the predictions and the reading rules on 17 Aug **before anything was seen**.

### Setup
Frozen config (`configs/frozen_dl.yaml`): 8 seeds, epoch 21, `pos_weight` 100, `threads` 2,
`nms_size` 5, matched **measured** Pfa 1e-4 swept per condition. CFAR baseline CA/GO/OS/SO already
on disk (SO added 17 Aug, before the DL read). Gates ±1 and ±3.

    python scripts/bootstrap_comparison.py --tag frozen --epoch 21 --split bench2_test --tolerance 1
    python scripts/bootstrap_comparison.py --tag frozen --epoch 21 --split bench2_test --tolerance 3
    python scripts/localization_table.py   --tag frozen --epoch 21 --split bench2_test --per-condition
    python scripts/measured_pfa_table.py   --tag frozen --epoch 21 --split bench2_test
    python scripts/runtime_cost.py         --tag frozen --epoch 21 --split bench2_test --threads 2""")

code('''TEST = "frozen_ep021_test"
t1 = DL / f"bootstrap_{TEST}_tol1.csv"; t3 = DL / f"bootstrap_{TEST}_tol3.csv"
lz = DL / "localization_frozen_ep021_test.csv"
pf = DL / "measured_pfa_frozen_ep021_test_tol1.csv"
rt = DL / "runtime_cost_frozen_ep021_test.csv"

# T4 FIRST: every other table assumes the operating points matched. If this fails nothing else means anything.
if pf.exists():
    P = pd.read_csv(pf)
    r = P.pivot(index="condition", columns="method", values="ratio_to_target")
    print(f"T4  measured Pfa / 1e-4 target -- worst {r.to_numpy().max():.2f}x, best {r.to_numpy().min():.2f}x")
    display(r.round(3))

# T1 / T2, beside the val numbers they supersede.
for gate, newp, oldp in ((1, t1, DL / "bootstrap_frozen_ep021_tol1.csv"),
                         (3, t3, DL / "bootstrap_frozen_ep021_tol3.csv")):
    if not newp.exists():
        continue
    n = pd.read_csv(newp).set_index("condition")
    c = pd.DataFrame({"cfar": n.cfar_method, "cfar_pd": n.cfar_pd, "dl_pd": n.dl_pd,
                      "test_diff": n["diff"], "test_p": n.p_boot, "test_sig": n["sig (Bonf)"]})
    if oldp.exists():
        o = pd.read_csv(oldp).set_index("condition")
        c.insert(0, "val_diff", o["diff"]); c.insert(1, "val_sig", o["sig (Bonf)"])
        c["moved"] = ["-" if a == b else ("NOW SIG" if b else "NO LONGER SIG")
                      for a, b in zip(c.val_sig, c.test_sig)]
    print(f"gate +/-{gate}: {int(n['sig (Bonf)'].sum())} of {len(n)} significant")
    display(c.round(4))

# T3 localization + the headline ratio, re-measured on test.
if lz.exists():
    L = pd.read_csv(lz)
    for gate in (1, 3):
        d = L[(L.gate == gate) & (L.group == "ALL")].set_index("method")
        dl, cf = d.loc["DL"], d.drop("DL")
        print(f"T3  gate +/-{gate}:  DL Doppler {dl.dv_bins:.4f} = "
              f"{dl.dv_bins / cf.dv_bins.min():.2f}x best CFAR ({cf.dv_bins.idxmin()}); "
              f"range {dl.dr_bins / cf.dr_bins.min():.2f}x ({cf.dr_bins.idxmin()})")
        display(d[["n_seeds", "pd", "pd_se", "dr_bins", "dr_bins_se", "dv_bins", "dv_bins_se"]].round(4))

if rt.exists():
    display(pd.read_csv(rt).round(3))''')

md("""### What it showed — 6 of 7 wins, no losses, and three failed predictions

**T4 validates the protocol.** Measured Pfa is within **2%** of the 1e-4 target for every method on
every condition (worst 1.02×), so both arms really were placed at the same false-alarm rate. This is
the check everything else rests on, and it passed.

**T1 — Pd at ±1, 6 of 7 significant after Bonferroni:** `clutter_edge` **+0.302**, `mixed` +0.103,
`isolated` +0.058, `masking` +0.051, `multi` +0.045, `unresolved` +0.012 — all p ≤ 0.001.
`clutter` is a tie (−0.002, p = 0.79). **T2 at ±3: the same 6 of 7.**

**⚠️ Three pre-registered predictions failed.**

1. **`clutter` at ±3 was predicted a LOSS; it is a TIE** (val −0.043 p = 0.005 → test −0.019
   p = 0.068). §18 called that loss "a strengthening", since Milestone 17 says a credible learned
   detector should lose somewhere. **It does not replicate** → Correction **C6**.
2. **`unresolved` was predicted a TIE; it is a significant win** (+0.012, p = 0.001, both gates).
   **Do not report it as a win.** O2's reasoning is unchanged — both arms sit at the ~0.5
   two-targets-in-one-mainlobe physics cap, and 800 targets simply makes a negligible gap
   detectable. It is a significant non-difference, and O2 should now be cited *because* of this.
3. **SO was predicted to stay the best classical method on range.** Val: best on 5 of 7 conditions.
   Test: **2 of 7**, with **OS** taking both pooled minima. Running SO was still right — it is best
   on `clutter` and `multi`, so omitting it would still have biased T3 — but the claim was wrong.

`isolated` crossing into significance was flagged uncertain in advance, so it is not a failure —
though the delta **tripled** (+0.019 → +0.058), more than tighter error bars explain.

**T3 — CFAR still wins localization, by less than predicted:**

| pooled, ±1 | DL (8 seeds) | best CFAR | ratio |
|---|---|---|---|
| range | 0.476 ± 0.021 | 0.198 (OS) | **2.40×** |
| Doppler | 0.382 ± 0.026 | 0.100 (OS) | **3.83×** |

Predicted 2.65× and 4.57×. DL's own errors moved ≤0.012 bins from val, so the shift is on the
**CFAR** side. **The headline is 3.83× on test**, superseding 4.6× (val ep21) and 5.6× (val ep50) —
all three kept on record.

**T5 — cost.** DL **6.04 ms/frame** against CA 0.86, GO 7.79, SO 7.82, OS 69.95: **~7× slower than
CA, ~11.6× faster than OS**, and *within noise of GO/SO* — DL was slower than both on val and faster
on test, so that ordering is machine-load dependent and is not claimed. The §4.2 premise correction
holds: our CFAR is vectorised, so DL is not beating a per-cell loop; it beats OS because
rank-selection vectorises badly.

### What it changed
**The headline:** *at finding targets in interference DL wins; at placing them CFAR does, by ~4× in
Doppler; and it runs at roughly GO/SO cost, far cheaper than OS.*

**But DL now loses nowhere.** `clutter` was the one place a classical method beat it, and that did
not survive the test split. This revives Milestone 17's concern directly and is a **weaker**
position to report from than §18's. It is the honest state of the evidence.

**Scope, fixed before the read:** `bench2_test` shares a generator with `bench2_val` — different
frames and seed, no shared content, but not a different world. **Sampling generalization only.**

**`bench2_test` is spent.** Any further read is a second read and must be recorded as one.""")

# ============================================================================== corrections
md("""---
## Corrections""")

md("""> **Nothing in this log is deleted when it is superseded.** Every retracted claim keeps its
> number, its date and the evidence that displaced it, because the reasoning error is the part worth
> reading. The configuration's whole path — epochs 12 → 50 → 15 → 21, `pos_weight` 200 → 974 → 100 —
> is part of the result, not scaffolding to be tidied away before the report.

| parameter | path | why it moved |
|---|---|---|
| **epochs** | 12 → 50 → **15** → **21** | 12 arbitrary with val loss still falling · 50 on "Pd is flat so it does not matter" · 15 from four sampled points once localization was found to degrade · **21** from the full 1–25 sweep under a pre-registered rule (**C5**) |
| **`pos_weight`** | 200 → 974 → **100** | 200 tuned for the old 1:1563 ratio · 974 the measured inverse class frequency · **100** won the 3-way ablation and retracted the `clutter` loss 974 had manufactured (**C3**) |
| **`clutter_edge` trend** | "still climbing" → **seed noise** | three correlated points, spread 0.047, against a 0.054 seed spread at fixed epoch (**C1**) |
| **`isolated`** | "win, p = 6.9e-07" → **tie** | CFAR's Pd had been treated as exact; with error bars both sides + Bonferroni, p = 0.031 (**C4**) |
| **`threads`** | incidental → **frozen** | changes weights at fixed seed; ~1 SE on localization (§17) |""")

md("""---
### The retractions themselves

Claims made and later retracted, with the evidence in each case. Kept because the reasoning error
matters more than the number.

### C1 — "`clutter_edge` is still climbing past the loss plateau"
**Claimed** (§4): the monotone rise 0.664 → 0.705 → 0.711 across epochs 30/43/50 was *"the shape of
continued improvement, not scatter"*, and that stopping at epoch 26 would have cost ~0.04 Pd.

**Retracted by** §6. The three points came from one trajectory, and seed-to-seed spread at a *fixed*
epoch is 0.054 — nine times the 0.006 that training from 15 to 30 actually produces. At epoch 30,
seeds 0 and 1 both give 0.664 while seed 2 gives 0.718. The 50-epoch run was seed 0, so its 0.711 is
as well explained by *which seed it was* as by *how long it trained*.

**The reasoning error:** §4 explicitly flagged that three correlated points could not separate model
change from evaluation noise — and then read a trend into them anyway. A stated caveat is not a
substitute for the experiment that resolves it.

### C2 — "8 seeds will give SE ≈ 0.011 on `clutter_edge`"
**Claimed** when proposing the 8-seed run, extrapolating from the 3-seed epoch-30 spread of 0.054.

**Actual:** SE **0.0040** at epoch 50 — nearly 3× tighter.

**Why:** the estimate assumed seed variance is constant across training. It is not; it shrinks as
the run converges (§7). A 3-seed spread is also a poor variance estimator in its own right. The
prediction was conservative in the useful direction — the experiment was better powered than
promised — but it was wrong about the mechanism.

### C3 — "CFAR genuinely beats the network on `clutter`"
**Claimed** (§7): DL 0.125 ± 0.0032 against GO 0.149, all 8 seeds below GO, p = 0.00014 — reported
as the network's genuine weakness, and welcomed because `PROGRESS.md` Milestone 17 notes that a
credible learned detector normally loses *somewhere*.

**Retracted by** §8. At `pos_weight` 100 the same comparison is **0.144 vs 0.149, gap −0.006,
p = 0.27** — no detectable difference. The deficit was a property of the *hyperparameter*, not of
the detectors.

**The reasoning error:** §3 set `pos_weight = 974` from the measured inverse class frequency and
called it a principled default, while explicitly noting that exact class balancing is a choice
rather than a derivation. §7 then drew a conclusion about the detectors from a configuration whose
one free parameter had never been measured — and the caveat recorded in §3 is exactly the one that
should have blocked it.

**Consequence, stated plainly:** the network now ties or beats CFAR on all seven conditions, which
revives the Milestone 17 concern that ours loses nowhere. That is a weaker position to report from,
not a stronger one, and it is the honest state of the evidence.

*(Partly resolved by §9: with error bars on both sides, three of the seven are ties rather than
wins, which is the shape the literature predicts. `clutter` remains a tie, not a loss.)*

### C4 — "DL beats CFAR on `isolated`" (p = 6.9e-07)
**Claimed** (§7, and repeated in §8): DL 0.732 ± 0.001 against CA 0.710, reported as a significant
win on the clean, homogeneous-noise condition.

**Retracted by** §9. The p-value came from `ttest_1samp(dl_seeds, cfar_value)`, which treats CFAR's
Pd as exact. It is one draw over 200 frames. With a paired frame bootstrap giving CFAR its own
sampling error, and Bonferroni across 7 conditions, the same comparison is **+0.023, p = 0.031** —
short of the corrected threshold of 0.0071.

**The reasoning error:** the DL side had 8 seeds and visible scatter, so it *looked* like the
uncertainty had been accounted for. All the uncertainty on the *other* side of the comparison was
silently set to zero. Eight seeds of precision on one arm does not license a point estimate on the
other.

**Consequence:** the `isolated` claim is withdrawn, and — usefully — its withdrawal removes the
Milestone 17 anomaly, since a tie in homogeneous noise is exactly what the corpus predicts.

*(Update, §19: on `bench2_test` `isolated` is a significant win, +0.058 at p < 0.001. C4's
methodological point stands — the original p-value was computed wrongly — but the "tie" conclusion
it reached is superseded by the larger split.)*

### C5 — "Pd is flat with training, so stop at epoch 15"
**Claimed** (§6, then §15): that Pd does not depend on training length, and therefore that the
epoch count could be chosen on localization alone — §6 concluding *"anything from ~15 onward is
equivalent"*, §15 moving the frozen count 50 → 15 on the grounds that it *"costs 0.003 Pd (~1 SE,
marginal)"*.

**Retracted by** §16. Swept at **every** epoch, pooled Pd rises **+0.059 from epoch 1 to the
plateau — about 26 SE**. It is one of the least flat curves in this log. Epoch 15 sits **1.6 SE
below the plateau**, outside the pre-registered 1-SE band, and the rule selects **epoch 21**.

**The reasoning error — and it is a different one from C1.** Nothing here was measured wrongly:
§15's four points reproduce *exactly* in the sweep. The error was concluding *"the curve is flat"*
from four samples that all happened to lie **inside the flat region**, when the entire rise sits
below the lowest of them. §15 even recorded the gap — *"nothing below epoch 15 was tested, so the
optimum may be earlier"* — and then treated it as a limitation to note rather than the experiment
that would decide the question. That is C1's failure mode wearing different clothes: C1 read a
trend into noise, C5 read flatness into a range that was never sampled. **A stated caveat is still
not a substitute for the measurement that resolves it** — twice now.

**Consequence, stated plainly:** smaller than it looks. Epoch 21 vs 15 is inside the noise on all
three metrics (Pd p = 0.14, range p = 0.37, Doppler p = 0.20), so the *number* was nearly right; it
was the *justification* that was unsound. §15's real contribution — that training to 50 costs
localization for no Pd (21 vs 50: range −0.075, p = 0.004) — is confirmed and strengthened.

### C6 — "`clutter` is a significant loss, so DL does lose somewhere"
**Claimed** (§18, 17 Aug): at ±3 DL loses `clutter` by −0.043 at p = 0.005, clearing the Bonferroni
threshold where the epoch-50 run had left it at p = 0.013, "edge of significance". It was written up
as *strengthening* the position, since Milestone 17 records that a credible learned detector
normally loses somewhere.

**Retracted by** §19. On `bench2_test` the same comparison is **−0.019, p = 0.068** — a tie, on
**twice** the frames per condition, i.e. with *more* power to detect the effect, not less.

**The reasoning error:** a result that had just moved from p = 0.013 to p = 0.005 on a single split
was treated as established. It sat near the correction threshold from both sides, and the sensible
reading was "unresolved", not "now demonstrated". The prediction was pre-registered, which is the
only reason this is a clean retraction rather than an argument.

**Consequence, stated plainly:** on the test split **DL ties or beats CFAR on all seven
conditions** — it loses nowhere. That is a weaker position to report from, and the same one C3 left
us in before §18 appeared to resolve it. The detection comparison shows no condition where the
classical method wins; the real DL weakness is **localization** (§19 T3), which is large, measured,
and reproduces on both splits.""")

# ============================================================================== open
md("""---
## Open / not yet reported

Findings that exist but should **not** go in the report in their current state.

### O1 — `clutter_edge` +0.261: ATTRIBUTED (2026-08-13), and it survives

**Question.** Is the margin an edge effect, or an artefact of the stratum's 34% dilution
(`PROGRESS.md` Milestone 22)?

**Check 1 — against OS specifically, not just "best classical".** OS-CFAR is the method designed for
edge contamination, so if the margin shrank against it the finding would be narrower than it looks.
It does not: **OS *is* the best classical method on this condition** (0.443, against CA 0.419 and
GO 0.426), so +0.261 was always measured against OS. The margin against CA is +0.285 and against GO
+0.278 — OS's edge design buys it only 0.017–0.024 over its siblings. This echoes Milestone 22: at
matched *measured* Pfa, an edge-robust design shows up as fewer false alarms, not higher Pd, and
matching the rate cancels precisely that.

**Check 2 — where the margin falls relative to the clutter boundary.** Split by **target clearance
from the ridge edge in Doppler bins** — fixed by the data, so no window or detector can shift it.
The ridge spans ±4 bins; the 21×9 training ring sits at offsets ±3–4 from the target, so clearance
2–4 means the ring genuinely overlaps the ridge and clearance ≥5 means it never does (**28%** of
frames on val, consistent with Milestone 22's 24.5% frame-level figure on test).

The two hypotheses predict different shapes: **H-window** (reference-window contamination) says the
margin concentrates where the ring is contaminated; **H-dilution** says it is uniform, i.e. not an
edge effect at all.

`python scripts/attribute_clutter_edge.py` · `bench2_val` · CFAR from cached peaks, DL averaged over
all 8 seeds · matched measured Pfa 1e-4 swept **within each band**.""")

code('''att = DL / "clutter_edge_attribution.csv"
if att.exists():
    a = pd.read_csv(att)
    order = ["2 (closest)", "3", "4", "5-6 (farthest)"]
    dl = a[a.method == "DL"].groupby("band").pd.agg(["mean", "std", "count"]).reindex(order)
    dl["se"] = dl["std"] / np.sqrt(dl["count"])
    cf = a[a.method != "DL"].pivot_table(index="band", columns="method", values="pd").reindex(order)

    t = pd.DataFrame({"n frames": a[a.method == "CA"].set_index("band").n.reindex(order),
                      "true edge": [True, True, True, False],
                      "CA": cf.CA, "GO": cf.GO, "OS": cf.OS,
                      "best CFAR": cf.max(axis=1), "DL": dl["mean"], "DL SE": dl["se"]})
    t["DL - OS"] = t.DL - t.OS
    t["DL - best"] = t.DL - t["best CFAR"]
    display(t.round(3))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(order))
    ax.plot(x, t.OS, "s-", color=CFARC, lw=1.8, ms=7, label="OS-CFAR (best classical)")
    ax.errorbar(x, t.DL, yerr=1.96 * t["DL SE"], fmt="o-", color=DLC, lw=1.8, ms=7,
                capsize=4, label="DL (8 seeds)")
    ax.fill_between(x, t.OS, t.DL, color=DLC, alpha=0.10)
    ax.axvspan(-0.4, 2.5, color=MUTED, alpha=0.08)
    ax.text(0.1, 0.30, "training ring overlaps the ridge", fontsize=8, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_xlabel("target clearance beyond the ridge edge (Doppler bins)")
    ax.set_ylabel("Pd at matched measured Pfa 1e-4")
    ax.set_title("Where the clutter_edge margin lives")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.show()

    print(f"DL - best CFAR:  closest {t['DL - best'].iloc[0]:+.3f}   farthest "
          f"{t['DL - best'].iloc[-1]:+.3f}   difference {t['DL - best'].iloc[0]-t['DL - best'].iloc[-1]:+.3f}")
    print(f"OS degrades {t.OS.iloc[-1]-t.OS.iloc[0]:+.3f} as the target approaches the ridge; "
          f"DL degrades {t.DL.iloc[-1]-t.DL.iloc[0]:+.3f}")
else:
    print("run: python scripts/attribute_clutter_edge.py")''')

md("""**What it showed — H-window, supported.**

CFAR falls apart as the target approaches the ridge: OS goes 0.545 → 0.512 → 0.400 → **0.292**, a
drop of 0.253. DL is nearly flat over the same span: 0.724 → 0.724 → 0.736 → 0.619, a drop of 0.105,
and only in the closest band. That asymmetry *is* the reference-window signature — CFAR is
handicapped exactly where its training ring overlaps the ridge, and the network, having no reference
window, is not.

The margin follows: **+0.327 / +0.336 / +0.212** on the three true-edge bands against **+0.114** on
the non-edge band. That +0.114 is the same order as DL's general advantage elsewhere (isolated
+0.022, multi +0.031, masking +0.049, mixed +0.086) — so on frames that are not really edges, the
network wins by roughly its usual amount, and on genuine edges by two to three times that.

**Dilution cannot be the explanation — it works the other way.** The 28% non-edge frames carry the
*smallest* margin, so they drag the pooled figure **down**. The pooled +0.261 therefore
*understates* the true-edge effect, which is nearer +0.28 when frame-weighted.

**Strength of evidence, honestly.** The near-vs-far contrast is ~2.7σ allowing for CFAR's own
sampling error (~0.055 per band at 40–57 frames), supported by a near-monotone gradient across four
bands. Not overwhelming on the endpoint contrast alone; the gradient and the mechanism agreeing is
what makes it convincing. The two middle bands invert slightly (+0.327 vs +0.336), which is well
within that error.

**Reportable**, with the mechanism stated and the caveat that per-band counts are 40–57 frames.""")

md("""### O2 — `unresolved` +0.002 is not a win
p = 0.048, but both methods sit at the ~0.5 physics cap (two targets inside one mainlobe; one
detection can match only one truth). The significance is an artefact of a very small SE (0.0008),
not a detection advantage. **Do not report as a win.**

### O3 — `pos_weight` ablation: DONE (2026-08-14), see §8
Ran at {100, 200, 974} x 8 seeds. `pos_weight` moves to **100**; 974 was the worst of the three.
The §7 clutter finding is retracted (Corrections C3).

**What this opens instead:** every §7 number was measured at 974 and is now superseded. The
headline DL-vs-CFAR table must be re-run at the chosen `pos_weight` before anything is reported, and
the `clutter_edge` attribution (§Open/O1) should be re-checked there too — its +0.261 was measured
at 974, and at 100 the same condition scores 0.731 rather than 0.704.

### O4 — localization degrades *asymmetrically* between the two axes (2026-08-16, from §16)
Over epochs 1→25 Doppler error grows **+0.220 bins and is still climbing** at 25, while range grows
**+0.147 and saturates by ~15** (slope p = 0.51 over 15–25). Both shapes hold in **7/7 conditions**,
so it is structure, not noise. A flat **3×3 label is symmetric** and predicts neither the differing
timing nor the 1.5× differing magnitude, so §14–§15's label mechanism is incomplete as stated.

**Untested candidates:** the receptive field is **21×9**, itself asymmetric · Doppler has 64 bins
against range's 256, so the 3-cell label block spans 4× the fraction of the axis · the clutter ridge
is a Doppler-axis structure, and `clutter` shows both the worst Doppler error (0.711 bins at epoch
21) and the steepest climb.

**Not chased**, under the 15 Aug rule: it changes no published number. It is, however, the sharpest
available test of the label hypothesis — a Gaussian or 1×1 label should collapse the asymmetry — so
it belongs with the label ablation whenever that runs, not as a standalone experiment.""")

# ============================================================================== next
md("""---
## Next experiments

Running to-do in the same per-experiment format, results blank. Fill in and move up into the
chronological list once run.

### N1 — re-run the headline comparison at `pos_weight = 100`
- **Question:** what are the per-condition mean ± SE and the DL-vs-CFAR deltas at the chosen
  hyperparameter? (§7's table is superseded; it was measured at 974.)
- **Setup:** the 8 `pw100` checkpoints already exist — this is an evaluation, not a training run.
  Re-run the `clutter_edge` attribution (§O1) at 100 as well.
- **Results / showed / changed:** *(pending)*

### N2 — `clutter_edge` attribution: true-edge vs diluted frames
- **Question:** does the +0.261 margin hold on frames where the window genuinely straddles the ridge,
  or does it come from the 34% that do not?
- **Setup:** re-run the Milestone-22 split on the 8 DL checkpoints; compare against the CFAR
  true-edge numbers already measured.
- **Results / showed / changed:** *(pending)*

### N3 — why DL loses on `clutter`
- **Question:** what does CFAR's local statistic capture inside the ridge that the network does not?
- **Setup:** per-frame comparison at matched Pfa on `clutter`; inspect frames where CFAR detects and
  DL misses; check whether the deficit concentrates at particular CNR or Weibull shape (both swept
  and recorded per frame).
- **Results / showed / changed:** *(pending)*

### N4 — final test-split evaluation
- **Question:** the reportable numbers.
- **Setup:** **only once the configuration is locked.** Seed-averaged DL vs CFAR on `bench2_test`,
  same protocol. Until then `bench2_test` stays untouched.
- **Results / showed / changed:** *(pending)*""")

# ---------------------------------------------------------------------------------------------
# SECTION TEMPLATE — copy, fill in, paste above "Corrections", re-run this script.
#
# md('''---
# ## N · <title>
#
# ### Question
# ### Setup
# | | |  (command, config, split, seeds, epochs)
# ### Results        <- add a code() cell that reads from results/, do not paste numbers
# ### What it showed  <- include negative and ambiguous outcomes
# ### What it changed
# ''')
# ---------------------------------------------------------------------------------------------

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "cnn_development_log.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
