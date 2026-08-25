# PLAN — future work

> **The question this programme exists to answer:**
>
> **Why does TinyCNN outperform CFAR — better use of target/mainlobe energy, a better learned
> decision rule, or both?**

**Final framing.** *A controlled FMCW Range–Doppler study explaining **where** the learned detector's
advantage over CFAR comes from, rather than simply showing that DL beats CFAR.*

That reframing is the contribution. The published literature repeatedly demonstrates learned
detectors beating CFAR and very rarely decomposes *why* — and, in fairness, most of those
comparisons are not controlled tightly enough to permit it (differing context sizes, differing
post-processing, and in some cases a different task entirely). This project's arms already share peak
extraction, valid region, threshold procedure, matching and metrics, and their spatial extents are
matched by construction. That is what makes attribution tractable here.

**Scope: this file records work that has NOT been done.** Nothing below has been executed, measured,
or scripted. Completed work lives in `PROGRESS.md`; the immediate queue lives in `NEXT_SESSION.md`.

> **Note on file history.** An earlier `Plan.md` was deleted in the repository cleanup and
> `PROGRESS.md` was made "the single roadmap record". This file reintroduces a separate roadmap
> deliberately, with a narrower job: **future experiments only**, so completed and planned work
> cannot be confused. `PROGRESS.md` remains the authority on everything already done.

**Current position.** All synthetic experiments are complete; the configuration is frozen in
`configs/frozen_dl.yaml`; `bench2_test` has been read once against a pre-registration. The critical
path is **writing**. Everything below is deliberately deferred behind it.

---

## ⚠️ Three things to settle before any of this runs

These are not details. Each one silently invalidates results if left unresolved.

**1. The threshold protocol changes, and the new numbers will not be poolable with the old ones.**
Every published number so far uses a threshold chosen **per method per condition from the ROC of the
split being scored** — an evaluation oracle, and a stated limitation (`evaluation_metrics_overview`
§7.2). The plan below instead **tunes on validation and freezes for test**, which is the better
protocol and removes that oracle. But it is a *different* operating-point procedure, so:

* new results must be reported as a separate protocol, not merged into the existing tables;
* the existing headline (DL wins 6 of 7, 3.83× localization) stays as-is under its own protocol;
* the honest framing is "the attribution study uses a stricter operating-point rule", and the
  difference between the two is itself worth one line of reporting.

**2. `bench2_test` is spent.** It was read once, on 19 Aug, against `docs/test_preregistration.md`.
Every experiment below should be developed and decided on **`bench2_val`**. Any read of `bench2_test`
is a *second read* and must be recorded in `PROGRESS.md` with its reason before it happens. A
defensible option: pre-register the whole attribution study, then take one final test read for all
arms at once rather than one per experiment.

**3. There is no `multipath` stratum.** It was folded into `mixed` during the Milestone-19 rebuild,
because its own row spanned only 0.818–0.840 across all methods — it did not discriminate. The
scenario breakdown (E6) must therefore either report `mixed` (which contains multipath), or
regenerate a standalone multipath stratum from the config that still defines it
(`configs/bench2_debug.yaml`, `multipath: {enabled, delay_m, attenuation_db}`). Regenerating is a
dataset change and needs its own justification.

---

# Part A — the attribution programme

The whole of Part A is one experiment in four moves: measure the gap (E1), give the classical arm the
energy it discards (E2), take that energy away from the learned arm (E4), and read the difference
(E5). E0 fixes the ruler; E3, E6 and E7 support and qualify the result.

---

### E0 — Freeze the evaluation protocol

| | |
|---|---|
| **Purpose** | Fix the measuring instrument before any arm is built, so no later result can be shaped by a choice made after seeing it. |
| **Hypothesis** | None — this is a protocol definition, not a test. |
| **Controlled variables** | Peak extraction (`all_peaks`, NMS radius 5), valid mask, one-to-one matching, ±1 headline gate with ±3 beside it, Bonferroni across the scored conditions. All unchanged from the existing protocol. |
| **Metrics** | **Primary: Pd at matched Pfa.** Secondary: localization error (range and Doppler, in bins and physical units), ROC / FROC, false alarms per frame, runtime per frame. |
| **What changes** | **Thresholds are tuned on `bench2_val` and frozen for any test read** — not re-derived per condition on the scored split. See ⚠️ 1 above. |
| **Expected interpretation** | A stricter, deployment-honest operating point. Absolute Pd will likely fall slightly for every arm versus the oracle protocol; what matters is that all arms are penalised identically. |
| **Retraining** | None. |
| **Priority** | **Must precede everything else in Part A.** |

*Write the protocol down and pre-register it before E1 produces a number.*

---

### E1 — Baseline under the frozen protocol

| | |
|---|---|
| **Purpose** | Re-establish the DL-vs-CFAR gap under E0's protocol, giving every later contrast a like-for-like reference point. |
| **Hypothesis** | The direction of the existing result survives the stricter protocol: DL ahead on Pd in interference and clutter-edge conditions, behind on localization. Absolute margins may shrink. |
| **Controlled variables** | Current dense TinyCNN, frozen config, all 8 seeds, `threads=2`; CA / OS / GO / SO-CFAR at the current 21×9 window; **receptive field kept at 21×9** so the local-context comparison stays fair. |
| **Metrics** | Full E0 set, per condition. |
| **Expected interpretation** | If the gap largely survives, attribution is worth pursuing. If the gap mostly disappears under a non-oracle threshold, **that is itself the finding** and it changes the thesis — report it rather than reverting the protocol. |
| **Retraining** | None — existing checkpoints, re-scored. |
| **Priority** | **High.** Cheap, and it gates everything after it. |

---

### E2 — Integrated CFAR: give the classical arm the energy it discards

| | |
|---|---|
| **Purpose** | Test whether the CNN's advantage comes mainly from using target energy spread across neighbouring cells — energy CFAR deliberately throws away in its guard region. |
| **Hypothesis** | Integrating the mainlobe before the CFAR decision recovers a substantial part of the gap. Directional prediction: **largest gain on `masking` and `unresolved`**, where a neighbour's energy currently only harms the estimate; smaller gain on `isolated`. |
| **Controlled variables** | Guard and reference geometry unchanged; same 21×9 window; same NMS; same evaluator. **Only the CUT statistic changes.** |
| **Method** | (a) Measure the actual target PSF / mainlobe from **isolated** targets — `scripts/measure_psf.py` and `results/psf_profile.csv` already do this; (b) replace the single-cell CUT with an integration over that measured mainlobe; (c) **recalibrate alpha for the new statistic** — `configs/cfar_alpha_table.json` does *not* apply to a different CUT statistic; (d) score at matched Pfa. |
| **Metrics** | Full E0 set. The contrast that matters is **E2 − E1 (CFAR)**. |
| **Expected interpretation** | Large gain → the margin was energy integration, not decision-rule quality, and the headline claim weakens substantially. Small gain → CFAR was not simply discarding usable energy, and E4 becomes the important experiment. |
| **Retraining** | **None at all.** Runs on cached data. |
| **Priority** | **Highest of the experimental arms — run this first.** It is the cheapest and it can render E4 unnecessary. |

---

### E3 — Label ablation: what should supervision look like?

| | |
|---|---|
| **Purpose** | Determine whether the localization deficit (≈3.8× CFAR's Doppler error) comes from the supervision, and find the best supervision strategy. |
| **Hypothesis** | The flat 3×3 mask teaches the network that nine adjacent cells are equally correct, broadening its response. Point labels should sharpen localization but may cost Pd through sparser gradients; a **Gaussian keeps a unique maximum while retaining dense gradients**, and should be the best of the three. |
| **Arms** | **point** (1×1, precedent: RadCNN / Yavuz 2021) · **current** (3×3 flat, the control) · **Gaussian**, `y = min(1, Σ_k exp(−(r−r_k)²/2σ_r² − (d−d_k)²/2σ_d²))` with BCE-with-logits, **σ set from the measured PSF spread of E2(a)** (precedent: CFARNet / Liang 2025). |
| **Controlled variables** | Identical architecture, data, splits, seeds (all 8), `threads=2`, and evaluation. **Only the target labels change.** |
| **⚠️ Confounds that must be handled** | **Class weighting must be recalculated per arm** — the 1×1 arm carries ~1/9 the positive mass, so a fixed `pos_weight` silently makes it a different experiment. **The stopping epoch is arm-specific** — re-apply the pre-registered rule ("earliest epoch whose pooled seed-mean Pd is within 1 SE of the plateau") independently per arm; do not transplant epoch 21. |
| **Metrics** | Full E0 set, with **localization error as the primary read** and Pd as the constraint that must not degrade. |
| **Expected interpretation** | Localization improves without a Pd cost → the label was the cause and the fix is free. Localization improves but Pd falls → a genuine trade to report and choose between. Nothing changes → **label shape is not the cause**, and suspicion moves to capacity or to the ±1 gate itself. All three outcomes are worth reporting. |
| **Retraining** | **Yes — 3 arms × 8 seeds.** The largest compute item in Part A. |
| **Priority** | **High**, and independent of E2/E4 — it answers the localization half of the story while they answer the detection half. |

*Note: a Gaussian label is not a claim that nine cells are targets. It is a smoother optimisation
target whose maximum still marks the true bin; inference still reduces to peaks.*

---

### E4 — CUT-centred attribution CNN

| | |
|---|---|
| **Purpose** | Build a learned detector that can be **denied the guard/mainlobe cells exactly as CFAR is**, so the learned arm can be placed in both rows of the E5 table. |
| **Hypothesis** | If the network's advantage is a better decision rule, the guard-masked model stays well ahead of CFAR. If the advantage is mainlobe access, masking collapses it toward CFAR. |
| **Why a patch model at all** | "Hide the mainlobe from the input" is **not well defined for a fully-convolutional network**: the guard must be hidden *relative to each cell under test*, but dense TinyCNN emits all 16,384 outputs in one pass. Blanking around the true target is invalid — it uses labels at inference. Masking conv kernels does not yield a clean hole once four dilated layers compose. The correct control is a **CUT-centred 21×9 patch classifier** emitting one logit for the centre cell — RadCNN's formulation, so there is precedent. |
| **Arms** | **Normal Patch CNN** — CUT + guard/mainlobe + reference cells. **Guard-Masked Patch CNN** — CUT + reference only, with the guard/mainlobe neighbours replaced by a **neutral background estimate** (not zeros: a zeroed hole is itself a detectable feature, so use a local background level and state the choice). |
| **⚠️ Mandatory first step** | **Verify the Normal Patch CNN performs comparably to dense TinyCNN under E0.** If it does not, the patch reformulation is itself a confound and every E5 contrast involving it is uninterpretable. This is a gate, not a formality. |
| **Controlled variables** | Same 21×9 extent, same labels (whichever E3 selects), same seeds, same evaluator. Between the two patch arms, **only the masking changes**. |
| **Metrics** | Full E0 set, plus the patch-vs-dense agreement check. |
| **Expected interpretation** | See E5 — this experiment exists to populate that table. |
| **Retraining** | **Yes — two models** (masked and unmasked), each × 8 seeds. The unmasked one is the architecture control, not an optional extra. |
| **Priority** | **Medium-high**, and **conditional on E2**: if integrating the mainlobe already closes most of the gap, much of this question is answered more cheaply. |

---

### E5 — The core four-way comparison

| | |
|---|---|
| **Purpose** | Decompose the total DL-vs-CFAR margin into named, separately-estimated components. **This is the central result of the programme.** |
| **Hypothesis** | The margin is a mixture, with mainlobe access accounting for a meaningful share — which would make the honest claim narrower than the current headline. |
| **Design** | A 2 × 2 over *information available* × *decision rule*: |

| information available | classical rule | learned rule |
|---|---|---|
| CUT + reference | **CFAR** (E1) | **Guard-Masked Patch CNN** (E4) |
| CUT + mainlobe + reference | **Integrated CFAR** (E2) | **Normal Patch CNN** (E4) |

**The three contrasts to estimate**, each with a paired bootstrap CI and the Bonferroni correction:

* **learned-rule gain** = Masked CNN − CFAR *(same information, different rule)*
* **mainlobe gain (classical)** = Integrated CFAR − CFAR *(same rule, more information)*
* **mainlobe gain (learned)** = Normal CNN − Masked CNN *(same rule, more information)*

| | |
|---|---|
| **Controlled variables** | All four arms scored under E0 on the same frames, at matched Pfa, through the unmodified evaluator. |
| **Metrics** | Pd at matched Pfa primarily; localization and FA/frame reported beside it. |
| **Expected interpretation** | Learned-rule gain ≈ 0 and mainlobe gains large → *the advantage was information access, not learning*. Learned-rule gain large and mainlobe gains small → *the advantage is the decision rule and survives equal information*. Both large → a mixture, and the two contrasts give its composition. **If the two mainlobe gains differ substantially, that is itself a finding**: it would mean the network exploits mainlobe energy better than an explicit integrator does. |
| **Retraining** | None beyond E2/E4. |
| **Priority** | **Highest in scientific value; last in sequence** — it is the readout of the others. |

---

### E6 — Scenario breakdown

| | |
|---|---|
| **Purpose** | Establish *where* each component of the margin lives, since a pooled decomposition can hide opposite effects in different regimes. |
| **Hypothesis** | The mainlobe gain concentrates in close-target and masking conditions; the learned-rule gain concentrates in clutter and clutter-edge conditions, where the classical noise estimate is biased by structure rather than by a neighbour's energy. |
| **Conditions** | SNR sweep (`bench2_snr_sweep`) · clutter edge (`clutter_edge`) · close targets (`unresolved` sub-resolution, `masking` resolvable-but-interfering) · multi-target (`multi`) · isolated (`isolated`) · **multipath — ⚠️ no standalone stratum exists**; report `mixed`, which contains it, or regenerate one (see ⚠️ 3). |
| **Controlled variables** | Same detectors, same protocol; only the evaluated subset changes. |
| **Metrics** | Full E0 set per condition, plus Pd-vs-SNR curves for the sweep. |
| **Expected interpretation** | A mechanism-level story: which component of the advantage operates in which physical regime. This is what turns a single decomposition number into an explanation. |
| **Retraining** | None. |
| **Priority** | **Medium** — runs on whatever arms exist, and is the material that makes the result explanatory rather than numerical. |

---

### E7 — Receptive-field ablation *(optional)*

| | |
|---|---|
| **Purpose** | Check whether 21×9 is near an optimum for the learned arm, or merely the CFAR-matched choice. |
| **Hypothesis** | A larger receptive field improves Pd — but by giving the network context CFAR does not have, which **breaks the controlled comparison** rather than strengthening it. |
| **Controlled variables** | Everything except the dilation schedule. |
| **Metrics** | Pd at matched Pfa, localization, runtime. |
| **Expected interpretation** | Reported as a **sensitivity analysis, never as an improved headline.** Any arm with a receptive field larger than the CFAR window is no longer a controlled comparison and must not be presented as one. |
| **Retraining** | Yes — one arm per RF size × 8 seeds. |
| **Priority** | **Low. Only if time permits**, and only after E5. |

---

### Also deferred, unchanged in priority

* **Fixed-threshold false-alarm stability** — one frozen TinyCNN threshold applied unchanged across
  all conditions, measuring how far achieved Pfa wanders, against each CFAR variant as reference.
  Purpose: whether the network has anything resembling constant-false-alarm behaviour. Needs no
  retraining. ⚠️ May require DL operating points well above Pfa 1e-4, where the `score > 0` floor in
  `all_peaks` binds (the DL ROC cannot currently be swept past sigmoid 0.5 — see
  `evaluation_metrics_overview` A.2). **Note: E0's val-frozen thresholds already move the protocol
  part-way toward this, so the two should be designed together.**
* **Secondary ablations** — pooling (`TinyCNNPool` exists, never trained; measured RF 13×13, so it is
  a control for peak displacement and *not* a valid CFAR-comparison arm), input normalization, and
  quantitative occlusion sensitivity.

**Explicitly not first: do not start with a U-Net or a larger CNN.** A high-capacity model gains from
context, pooling and capacity simultaneously, making any margin uninterpretable — the exact failure
this programme exists to avoid.

---

# Part B — CARRADA / real-data roadmap

Comes **after** Part A. Recorded so the order is not lost.

> **Scope reminder, to be repeated wherever this is written up.** The simulator was configured to
> reproduce CARRADA's **published Range–Doppler grid geometry** — map dimensions and bin spacings —
> so a target occupies a comparable number of cells. It does **not** reproduce CARRADA's waveform,
> scenes, clutter statistics, or sensor. Matching a grid is not matching a radar.

### R0 — Synthetic vs CARRADA domain audit *(before any detector runs)*

Compare simulated and real RD characteristics first, so any later performance gap can be attributed:
RD geometry and physical axes · target/mainlobe footprint and peak shape · background amplitude
distribution · clutter structure including the zero-Doppler ridge · spatial correlation · dynamic
range · the normalization statistics the network depends on (notably per-frame median power).

Also audit **what CARRADA's annotations actually support** before deciding which of Pd, Pfa and
localization can be reproduced. If the annotations do not give a per-cell truth bin, the localization
metric used throughout this project does not transfer unchanged.

### R1 — Zero-shot synthetic → CARRADA

The first real-data experiment uses the **frozen synthetic TinyCNN with no fine-tuning**, to
determine whether the learned rule transfers at all. Keep the fully-frozen zero-shot result
**distinct** from any later threshold calibration.

### R2 — Real-data failure analysis

Where do TinyCNN and CFAR succeed and fail on CARRADA, and do those failures match the mechanisms
identified in E5/E6?

### R3 — Limited real-data adaptation

Only after the zero-shot result is recorded: synthetic-only vs synthetic pre-training + limited real
fine-tuning vs real-only, if the data justifies it. **Split by sequence, never by frame** — ~12,666
frames but only 30 sequences and 78 instances.

### R4 — Mechanism replication on real data

Whether the findings that matter survive on real measurements: the detection/localization trade, the
information-access decomposition, fixed-threshold behaviour, and the learned spatial patterns.

---

## Deliberately cut

Logged so they are not silently rediscovered; none changes a published number: N2/N3 noise studies,
the Open/O4 localization asymmetry, Milestone-18 attribution, and the remaining proposal ablations
(RD resolution, DL model size, magnitude vs I/Q input).

## Standing rule (adopted 15 Aug 2026)

*A new finding goes on the limitations list, not the work queue, unless it changes a number already
published in `PROGRESS.md`.* Adopted because the project was drifting into open-ended fixing.
