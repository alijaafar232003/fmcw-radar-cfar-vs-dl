# Literature Corpus Review — 19 papers in `papers/`

Full-text analysis, 31 July 2026. Extends `literature_dl_vs_cfar.md` (which was built from web
sources); RadCNN and the *Electronics* 2022 review — listed there as "could not fully obtain" — are
now read in full, plus 15 more.

**Reading order if short on time:** §2 (mechanisms) → §4 (what this changes about our audit) →
§5 (ideas, ranked).

> Housekeeping: `Range-Doppler Detection in Automotive Radar with.pdf` and
> `Range-Doppler Detection in Automotive Radar with DL.pdf` are byte-identical (md5 `9a3dc9d6`).
> Delete one.

---

# 1. The corpus at a glance

Sorted by relevance to *our* question — "does a learned detector beat CFAR, and is the comparison
fair?"

| # | Paper | Mechanism | Baseline | Metric | Data | Headline |
|---|---|---|---|---|---|---|
| **A1** | **Yavuz, RadCNN**, EUSIPCO 2021 | 1 conv layer (4 filters 5×3) on 25×5 RD patches, stride 1 | CA/SOCA/GOCA/OS | Pd vs SNR at Pfa 1e-3 | sim, pulse-Doppler, **Weibull + K clutter** | +0.5 dB avg, ~1 dB in clutter, **−0.8 dB in one scenario, TIE in pure noise** |
| **A2** | **Diskin et al., CFARnet**, 2023 | MMD penalty forcing H0 output distribution invariance | GLRT, Kelly, plain NN | ROC / **CFAR** / **NP** / speed | sim, generic detection | Plain NN has best ROC but **is not CFAR**; CFARnet trades a little ROC for CFAR + best worst-case |
| **A3** | **Gusland et al.**, FFI, RadarConf 2020 | Fully-conv CNN, **receptive field restricted to match CA-CFAR**, no guard cells; + temporal 3-frame stacking | CA-CFAR | full ROC, Pd vs range | **real** X-band FMCW, UAV GPS truth | Parity below 450 m; +0.25–0.35 Pd in clutter transitions; +40 % detection range |
| **A4** | **Banerjee et al., RCTD**, IEEE Sensors J 2025 | Recursive quadrant search over 8×8 tiles + light CNN; **proves CA-CFAR ≡ conv + ReLU** | CA/GO/SO | full ROC (Pfa swept 1e-5…1e-1) | sim + on-air IWR1642; FPGA | Comparable Pd, consistently lower Pfa; real-time on ZCU102 |
| **A5** | **Lallouet et al.**, Thales | U-Net 2.1 M params; **Dice / Tversky / Lagrangian-constrained** losses | CA-CFAR | Pd vs SNR at fixed Pfa | sim, exoclutter | All NNs beat CFAR; constrained loss best. Attributes win to **CFAR loss from finite reference window** |
| **B1** | **Yu & Wu, RF-LEGO**, MobiCom '26 | **Deep unrolling**: CFAR structure kept, noise estimator → differentiable state-space model | CA/OS/GO/SO + DL-CFAR | DR at matched FAR 1e-3 | real UWB/mmWave/Wi-Fi | Beats classical CFAR, competitive with black-box DL, **keeps explicit operating-point control** |
| **B2** | **Tran Vu Hop et al.**, Viettel, 2025 | **Two-stage CFAR**: global CA-CFAR → HDBSCAN clustering → per-region PDF estimation → local CFAR | single-stage CFAR | false-plot reduction | **real** 3D S-band surveillance radar | A *classical* adaptive detector — no DL needed |
| **B3** | **Liang et al., CFARNet**, SJTU 2025 | CNN predicts peak subcarrier indices; **Gaussian label smoothing**; top-K | CA-CFAR + MUSIC | 95th-pct parameter error | sim, mmWave OFDM rainbow beam | 10–20× lower error with 1° separation and 5 targets |
| **B4** | **Ng et al.**, IJCNN 2020 | U-Net segmentation; augments complex (16 ch) with **log-magnitude** (24 ch) | **none — DL vs DL only** | F1, FAR | sim, MATLAB PAST, extended targets | +5 F1 points from the magnitude channel |
| **B5** | **Schuessler et al.**, FAU 2023 | Attention U-Net; **digital-twin input + enhanced-virtual-sensor ground truth** | MUSIC, DaS, CNN-DoA | Pd / Pfa / precision | sim (ray-traced CARLA) → **real measurements** | Sim-only training transfers to real hardware; regression beat classification formulation |
| **C1** | Decourt et al., **DAROD**, IV 2022 | Light Faster R-CNN on RD maps | *no CFAR* — vs other DL | mAP@0.3/0.5 | real (CARRADA, RADDet) | 3.4 M params beats 41 M ResNet-FPN; "no need for very deep CNNs on radar data" |
| **C2** | Rao et al., Amazon 2024 | LeNet-style CNN, 4-radar network | conventional NLS + range calibration | position error | **real** robot + touchscreen truth | **Sub-resolution *localisation*** (0.84 cm vs 3.08 cm bin) |
| **C3** | Major et al., Qualcomm ICCVW 2019 | RAD-tensor object detection | — | AP | real | "Peak detection collapses the image-like signal into a sparse point cloud" |
| **C4** | Jeong & Lee, IEEE Access 2023 | U-Net GAN, low-res RD → super-res RD | — | MSE/PSNR | sim + real | Resource saving, not detection |
| **C5** | Pemasiri et al., QUT 2024 | 1-D U-Net on IQ; **compound seg + FFT loss** | energy detector | F1 | sim EW | Auxiliary loss improves the *main* task |
| **C6** | Gu et al., OJSP 2024 | DL pulse detection on raw IQ | — | — | sim EW | Peripheral (EW domain) |
| **D1** | **Jiang et al.**, *Electronics* 11(1):156, 2022 | Review of ANN/DL for radar target detection | — | — | — | "Comparing Pd at the **same Pfa** with DNN and CFAR is **the common metric**" |
| **D2** | Geng et al., IEEE Access 2021 | Survey: waveform/ATR/jamming | — | — | — | Background only — **does not cover detection-vs-CFAR** |
| **D3** | Wu et al., 2023 | Survey (same group as D2) | — | — | — | Background only |
| **D4** | Li, MCEE 2024 | Short survey | — | — | — | Background only |

---

# 2. Differences in mechanism — five families

The corpus is not one approach with variations. It splits into five families that answer *different*
questions, and conflating them is how the "DL beats CFAR by 40×" claims get made.

### Family 1 — Per-cell learned threshold (our family)
**A1 RadCNN, A3 Gusland, A5 Lattouet, and our TinyCNN.**
A network produces a score per RD cell; threshold + NMS turns it into points. Structurally identical
to CFAR, so the *only* thing that differs is the decision rule. This is the only family in which a
head-to-head Pd-at-matched-Pfa comparison is meaningful.

Within the family the sub-choices matter enormously:

| | RadCNN | Gusland | Lallouet | **Ours** |
|---|---|---|---|---|
| Context | 25×5 patch | **restricted to match CA-CFAR** | full U-Net (global) | **21×21, measured, matches CA-CFAR** |
| Guard region | none | **explicitly none** ("not necessary") | none | none |
| Input scaling | 0–1 per patch | random value scaling (augmentation) | — | **dB above per-frame noise-floor median** |
| Loss | cross-entropy | weighted softmax CE | Dice / Tversky / Lagrangian | BCE, pos_weight 200 |
| Threshold set by | Monte-Carlo to hit Pfa 1e-3 | swept for ROC | **Monte-Carlo on target-free noise** | **Monte-Carlo on pure noise** |

Two things stand out. **Gusland independently arrived at our exact design principle** — restrict the
CNN's field of view to the CFAR window so the comparison is controlled. And **both Gusland and
Lallouet calibrate the network threshold on target-free noise, exactly as we do.** Our protocol is
mainstream; that is reassuring.

### Family 2 — Learned detector with an explicit statistical constraint
**A2 CFARnet, A5's NN-Lagrange.**
These keep the network but *add a constraint* that recovers the property CFAR was invented to
guarantee. CFARnet penalises the MMD between H0 output distributions under different nuisance
parameters; NN-Lagrange penalises the false-alarm count directly. This family exists because the
authors recognised that **a network with a better ROC can still be a worse detector operationally.**

### Family 3 — Structure-preserving hybrids (deep unrolling)
**B1 RF-LEGO, and Banerjee's conv+ReLU formalism (A4).**
Keep the CFAR pipeline; make its parameters learnable. RF-LEGO writes CFAR as
`s_n = C·g₁(r_n) + g₂(r_n)` (g₁ = noise-level selection, g₂ = testing) and replaces the
non-differentiable g₁ with a differentiable state-space model. Banerjee proves CA-CFAR is exactly a
convolution + ReLU with weights `+1` on the CUT and `−α/N` on each training cell, bias 0 — with a
learned kernel as the general case.

### Family 4 — Object detection on RD/RAD (computer-vision transplant)
**C1 DAROD, C3 Major, B4 Ng.**
Bounding boxes or segmentation masks, scored with mAP or F1. **None of these compare against CFAR
quantitatively.** Different task, different metric, not commensurable with ours.

### Family 5 — Learned super-resolution / enhancement
**B5 Schuessler, C2 Rao, C4 Jeong.**
The network improves the *representation*, and detection happens afterwards.

---

# 3. Differences in results — the pattern

## 3.1 What happens in homogeneous noise (where CA-CFAR is near-optimal)

This is the diagnostic condition, and the corpus disagrees:

| Study | Result in pure/homogeneous noise | Stated reason |
|---|---|---|
| **RadCNN (A1)** | **TIE** — "performs on par with the CFAR algorithms" (S4) | — |
| **Gusland (A3)** | **TIE** below 450 m; explicitly "similar" | — |
| **Lallouet (A5)** | **DL wins** | "**CFAR loss resulting from a potentially imprecise noise variance estimation due to the limited number of noise samples**" |
| **Ours** | DL wins, 0.880 vs 0.867 | unexplained ⚠️ |

So there *is* a legitimate published mechanism by which a learned detector beats CFAR even in
homogeneous noise — **CFAR loss**. A CFAR detector estimates noise power from a finite window, so its
threshold is itself a random variable; the network learns a lower-variance statistic. This is a real
effect, it is not cheating, and it is a *competing hypothesis* to our matched-filter explanation.

**The two hypotheses are separable by experiment** (see §5, idea 1). Currently we have asserted the
matched-filter explanation without testing it, and the literature offers an equally plausible
alternative.

## 3.2 Does a good learned detector ever lose?

| Study | A condition where DL loses |
|---|---|
| RadCNN | **S3 — target in noise with the reference window overlapping clutter, −0.8 dB at high CNR.** Named as "the main challenge for future work" |
| CFARnet | Plain NN (Bnet) loses badly on **worst-case NP** despite winning the ROC |
| Tosi et al. (in prior review) | CNN **deteriorates at high SNR** — thin training coverage |
| Schuessler | Classification formulation lost to regression; all methods had "rather low detection rate" against high-res ground truth |
| **Ours** | **none** ⚠️ |

A detector that wins in every condition is not what the careful papers report. The two clean
precedents (RadCNN S3, CFARnet worst-case NP) both fail in ways we have not tested for.

## 3.3 The size of the win tracks the rigour of the protocol

| Protocol | Reported DL advantage |
|---|---|
| Matched pre-processing, matched context, matched Pfa, simulated with known truth (**RadCNN**) | +0.5 dB avg; tie in noise; **loses somewhere** |
| Restricted receptive field, real data, GPS truth, full ROC (**Gusland**) | Tie in homogeneous; large win in clutter (a genuine non-homogeneity the CFAR model does not cover) |
| Full U-Net, global context, no context matching (**Lallouet**) | Wins everywhere |
| Cross-sensor ground truth (lidar/camera; prior review) | 10–38× — authors attribute to protocol artifacts |

**Our protocol is rigorous, but our architecture-vs-baseline pairing sits closer to the Lallouet
end than the RadCNN end** on one specific axis: pre-integration. That remains the leading suspect.

## 3.4 Where DL genuinely and repeatably wins

Worth stating plainly, because the audit should not talk us out of a real result:

1. **Non-homogeneous clutter.** RadCNN >1 dB inside clutter; Gusland +0.25–0.35 Pd at clutter
   transitions on *real data*; ours 2.47 vs 4.83 FA/frame with higher Pd. Three independent studies,
   two of them on measured data. **This result is safe.**
2. **Closely-spaced / masked targets.** RadCNN, CFARNet (SJTU), Tosi. Ours: weak-target Pd 0.500 vs
   OS 0.333. Consistent with the field.
3. **Lower false alarms at equal Pd.** Universal across the corpus.

## 3.5 A distinction we must get right — "sub-resolution"

Rao et al. (C2) achieve position error of 0.84 cm with a 3.08 cm range bin — genuine sub-resolution
**localisation** of a single target. This is *not* the same as resolving two targets closer than a
bin, which no paper in the corpus claims. **Our retraction of the "19 % sub-resolution recovery"
claim was correct**, and this citation lets us make the distinction precisely in the report rather
than just dropping the claim.

---

# 4. What this changes about our fairness audit

Mapping the corpus onto the six open asymmetries in `NEXT_SESSION.md`:

| # | Our asymmetry | What the corpus says |
|---|---|---|
| 1 | CNN gets implicit matched filter, CFAR does not | **Confirmed as a real design axis.** RadCNN matched-filters *before both*. Tosi's CNN win traced to multi-window preprocessing. **But Lallouet supplies a competing explanation (CFAR loss) that also predicts a win in homogeneous noise.** Must distinguish them. |
| 2 | CFAR geometry tuned on the wrong condition | Banerjee uses guard 2 / train 10 per side (24×24) *globally*; RadCNN 20×2 / 4×2. Nobody tunes per condition — **so B2 would exceed the field's standard, not just meet it.** |
| 3 | Pfa matched by design, not measurement | **RadCNN reports exactly this as a finding**: measured 1.2–1.7e-3 against a design 1e-3, "the CFAR algorithms cannot maintain their preset Pfa". Our 4× spread is worse than their 1.4× but the phenomenon is expected and publishable. Report the table. |
| 4 | DL retrained per distribution, CFAR not | Universal in the corpus; nobody re-tunes CFAR. Fixing it is a contribution. |
| 5 | ±1-bin gate may measure localisation | **Ng et al. explicitly criticise** Brodeski's 3×3 (= ±1 bin) criterion as "not realistic". Schuessler tests **two** tolerances (2 and 4 pixels) and reports both. **Reporting a tolerance sweep is standard practice we are missing.** |
| 6 | Train and test share one generator | Schuessler (B5) is the strongest counter-model: train purely in simulation, validate on **real hardware measurements**. Gusland trains and tests at **different physical locations**. Liang trains on the hardest condition and tests on easier ones. |

**Net effect on the audit: Step A stays as planned. Step B1 gains a rival hypothesis that must be
tested alongside it. Two new cheap diagnostics appear (CFAR-property test; tolerance sweep).**

---

# 5. Ideas — ranked

Split by what they buy. Goal stated by you: *first* a correct project with defensible numbers, *then*
a paper if possible.

## Tier A — required for the numbers to be trustworthy

**A-1. Bracket the mainlobe-integration effect from both sides.** ⭐ *highest value*
Two experiments, both cheap, that together settle asymmetry #1 and distinguish it from Lallouet's
CFAR-loss explanation:

- **Guard-constrained CNN.** Mask the centre 5×5 of the CNN's effective receptive field (an annulus
  kernel — zero the central weights of the first layer, or apply a fixed hole mask). The network
  then sees *exactly* what CFAR sees: the training ring but not the CUT's own mainlobe.
  Retrain, re-evaluate.
- **Matched-filter + CFAR** (the planned B1): convolve the power map with the measured PSF before
  CFAR, so CFAR gets the integration the CNN was getting for free.

Interpretation is unambiguous:
| Outcome | Conclusion |
|---|---|
| Guard-constrained CNN loses its isolated-target win, MF+CFAR closes the gap | Mainlobe integration was the cause (our hypothesis) |
| Guard-constrained CNN *keeps* the win | **CFAR loss** is the cause (Lallouet's explanation) — a legitimate, citable result |
| Both effects partial | Decompose the margin into the two contributions — the most interesting outcome |

Gusland's remark that "guard cells are not necessary for the CNN-CFAR" is the citation that makes
this framing legible to a reviewer.

**A-2. Measure whether our detector is actually CFAR.** ⭐
Diskin's Definition 1: a detector is CFAR iff its false-alarm rate is invariant to the nuisance
parameters under H0, **at any fixed threshold**. This is directly measurable on our TinyCNN: hold the
threshold fixed, sweep noise variance and clutter strength, plot FPR. Then add Diskin's **NP
criterion** — Pd subject to Pfa ≤ α *for every* nuisance value (i.e. worst case, not average).

This converts our vague issue #3 into a rigorous, quantified property. Two possible outcomes, both
good:
- Our per-frame noise-floor-median normalisation makes the network approximately scale-invariant →
  **we can claim an approximately-CFAR learned detector**, which is what CFARnet needed a special
  loss to achieve. Strong result.
- It does not → we have precisely localised why the Pfa spread is 4×, and the fix (Lagrangian or MMD
  penalty) is off-the-shelf.

**A-3. Matching-tolerance sweep.** Report Pd at ±1, ±2, ±3 bins for all four detectors, not one
number. Schuessler reports two tolerances; Ng criticises ±1 as too lenient. One line of code
(`MatchConfig`), removes asymmetry #5 as an objection. Already Step A3 — just widen it to a sweep and
put it in the report as a table.

**A-4. Publish the measured-Pfa table.** RadCNN's Table I is the template. Not a flaw to hide — a
documented property of CFAR under clutter.

**A-5. Add an "S3" stratum.** Target in the noise region with the CFAR reference window *overlapping*
the clutter edge. This is the one condition where RadCNN's CNN loses. We have `clutter` (target in
clutter) but not this. If our DL also loses there → our study gains the credibility of a detector
with a known weakness. If it doesn't → we need to explain why, which is itself informative.

## Tier B — substantially strengthens the project

**B-1. Heavy-tailed clutter.** Our documented gap, now with an exact recipe to copy: RadCNN uses
Weibull (shape 1, scale 1) and (shape 2, scale 1), plus K-distributed (shape 3) and (shape 15),
arranged as *walls and transitions* — noise→clutter and clutter→clutter — specifically so the
transitions fall inside the CFAR reference window. Matching those settings makes our results
directly comparable to a published benchmark.

**B-2. A stronger classical baseline.** Right now we compare against textbook CA/OS/GO. Two upgrades
from the corpus:
- **Two-stage CFAR** (Viettel, B2): global CA-CFAR → cluster the plots (HDBSCAN/DBSCAN) → estimate
  the interference PDF per region → local CFAR with a local Pfa. Adaptive, data-driven, *no deep
  learning*. If our TinyCNN still wins against this, the result is far more defensible.
- **Per-condition tuned CFAR** (planned B2) — exceeds the corpus's standard.

**B-3. Loss-function ablation.** Lallouet's core contribution, transplantable in a day: our
`BCEWithLogits(pos_weight=200)` vs Dice vs **Tversky** (α controls the FP/FN balance) vs
**Lagrangian-constrained** (penalise deviation from a target false-alarm count). Their finding —
Tversky > Dice, Lagrangian > both, because it internalises the calibration step rather than doing it
post-hoc — is worth reproducing on RD maps. Pairs naturally with A-2.

**B-4. Gaussian label smoothing.** Liang et al. replace one-hot peak labels with
`y_m = min(1, Σ_k exp(−(m−m_k)²/2σ²))`. Directly relevant to asymmetry #5: it stops the network being
trained to hit the exact label bin and makes it robust to "slight localisation ambiguities". Cheap
ablation, and it interacts with A-3 in a way worth reporting.

**B-5. Mutual radar interference.** Banerjee models interference at −3, 0, +1 dB relative to the
target peak. A genuine automotive problem, absent from our generator, and a condition where CFAR is
known to be weak (interference raises the noise floor and creates false targets). Would add a
stratum with real physical motivation.

**B-6. Generalisation protocol.** Two options, both from the corpus:
- Liang: train on the *hardest* stratum only, test on all → tests whether difficulty transfers.
- Gusland/Schuessler: hold out an entire condition, or perturb the generator at test time (our
  Step C).

## Tier C — the path to a paper

The corpus has a specific, well-defined hole. Nobody in it does the following, and we are unusually
well-positioned because we already have the shared evaluator, the stratified generator, and the
frozen splits.

**C-1. The CFAR→CNN ablation ladder.** ⭐ *the paper*
Every paper in the corpus jumps from "CFAR" to "a neural network" and reports the difference. Nobody
decomposes it. Build the intermediate rungs and report Pd at matched *measured* Pfa for each:

| Rung | Detector | What is relaxed |
|---|---|---|
| L0 | CA-CFAR, fixed α, fixed geometry | — |
| L1 | CFAR, α and geometry tuned per condition | hand-tuning → per-condition optimum |
| L2 | **Learned-kernel CFAR** — Banerjee's conv+ReLU with the CUT weight pinned at +1 and the ring weights trainable | fixed uniform ring → learned ring |
| L3 | **Guard-constrained CNN** (annulus receptive field, from A-1) | linear statistic → nonlinear, still no mainlobe |
| L4 | TinyCNN (Model 0) | + mainlobe integration |
| L5 | U-Net (Model 1) | + global context |

This attributes the total DL margin to specific, named relaxations of the CFAR constraint. It is
directly enabled by Banerjee's proof that CA-CFAR *is* a convolution + ReLU, and it is exactly the
"structure-aligned interpretability" that RF-LEGO argues the field needs. **It also converts our
fairness problem from a weakness into the contribution.**

**C-2. Report CFAR-property and NP alongside ROC.** Diskin defines these for generic detection;
nobody in the automotive RD-map literature measures them. Adding them to a stratified RD benchmark is
novel and cheap once A-2 exists.

**C-3. Frame the whole thing as a benchmark.** Stratified generator with per-frame condition tags +
identical NMS and matcher for all detectors + matched *measured* Pfa + tolerance sweep + measured-Pfa
reporting + open code. Judged against this corpus, that protocol is stricter than the norm. The
honest, defensible contribution is not "our CNN is better" — it is **"here is what the comparison
looks like when you control it properly, and here is how much of the reported advantage survives."**

**Realistic scope note.** C-1 alone is a workshop/conference paper. C-1 + C-2 + C-3 is a solid
short-paper. Do not attempt a new architecture — the corpus (DAROD, RCTD, RF-LEGO) is crowded with
those, and our comparative advantage is protocol rigour, not model design.

---

# 6. Gaps in the corpus = our openings

| Gap | Who has it | Us |
|---|---|---|
| Matching at **measured** rather than design Pfa | nobody | Step A1 |
| **Per-condition** stratified evaluation with guaranteed sample counts | nobody | done (Milestone 14) |
| Measuring the **CFAR property** of an RD-map detector | Diskin (generic only) | A-2 |
| Decomposing the DL margin into named relaxations | nobody | C-1 |
| Separating **mainlobe integration** from **decision rule** | nobody (RadCNN sidesteps it, doesn't measure it) | A-1 |
| Reporting a **tolerance sweep** | Schuessler (2 values) | A-3 |
| Identical NMS + matcher across all detectors | nobody states it | done (Milestone 13) |
| Reporting a condition where the learned detector **loses** | RadCNN only | A-5 |

---

# 7. Citation-ready framings

> "Constant false-alarm-rate behaviour is a property of the *detector*, not of a particular operating
> point: a detector is CFAR if its false-alarm probability is invariant to the nuisance parameters
> under the null hypothesis, for any threshold [Diskin et al., 2023]. Learned detectors do not
> possess this property by construction, and a network may achieve a superior ROC while being
> markedly worse under a strict per-environment false-alarm constraint. We therefore report, in
> addition to the ROC, the variation of the measured false-alarm rate across noise and clutter
> conditions at a fixed threshold."

> "Cell-averaging CFAR can be written exactly as a single convolution followed by a rectifying
> nonlinearity, with the kernel fixed at +1 on the cell under test and −α/N on each training cell
> [Banerjee et al., 2025]. A convolutional detector with a matched receptive field is therefore the
> learned generalisation of CA-CFAR, and the difference between them is attributable to specific,
> enumerable relaxations of that fixed structure."

> "A cell-averaging detector estimates the interference level from a finite reference window, so its
> threshold is itself a random variable; the resulting CFAR loss is a recognised explanation for a
> learned detector outperforming CFAR even in homogeneous noise, where the cell-averaging rule is
> otherwise near-optimal [Lallouet et al.]. This is distinct from, and must be separated
> experimentally from, the advantage a convolutional detector obtains by integrating energy across
> the target mainlobe, which the single-cell CFAR test statistic discards."

> "Where the comparison is controlled — matched pre-processing, a receptive field restricted to the
> CFAR reference window, and a matched false-alarm rate — the reported advantage of a learned
> detector is modest and condition-dependent: approximately 0.5 dB on average, in excess of 1 dB
> within clutter, on par with CFAR in homogeneous noise, and approximately 0.8 dB *worse* where the
> reference window straddles a clutter edge [Yavuz, 2021]. Detection performance on measured data
> shows the same structure: parity at short range in homogeneous background, with the advantage
> emerging at clutter transitions [Gusland et al., 2020]."

> "Sub-resolution *localisation* of a single distributed target is achievable with a learned
> estimator, with reported position error well below the range-bin width [Rao et al., 2024]. This is
> distinct from resolving two targets separated by less than one resolution cell, which remains
> governed by the waveform bandwidth; we make no claim of the latter."

---

# 8. Corrections to the earlier review

`literature_dl_vs_cfar.md` remains accurate. Two updates from full text:

1. RadCNN's operation counts are now exact: CA 57.6×10⁵, SOCA/GOCA 58.1×10⁵, OS 39.8×10⁶,
   RadCNN 49.1×10⁷, Wang et al. [12] 54.2×10¹¹. The "≈85× CA-CFAR" figure is confirmed.
2. Item 2 on its "could not obtain" list (*Electronics* 11(1):156) is now in `papers/`. Its useful
   contribution is the confirmation that **"comparing the Pd under the same Pfa with the DNN-based
   detector and CFAR detector respectively is the common metric of detection performance"** — i.e.
   our protocol is the field standard — plus its list of open challenges (dataset deficiency,
   simulation model fidelity, and the observation that SP+DL *integration* rather than replacement is
   the trend, which supports C-1).
