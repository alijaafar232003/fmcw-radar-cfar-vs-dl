# Deep-Learning Detector — Design Document

Complete design record for the deep-learning stage of the FMCW CFAR-vs-DL project: the problem
formulation, why each architecture was chosen or rejected, every data/loss/training decision with
its justification, the fairness requirements, and the verification plan.

Written to be reusable in the report and defensible in a viva.

> ## ⚠️ Map size and receptive field change (4 Aug 2026)
> This document was written for the previous sensor, whose RD map was **256 × 128**. The radar is
> now matched to CARRADA and the map is **256 × 64** — see `docs/radar_config_carrada.md`.
>
> **Every design argument below still holds** — the heatmap formulation, no downsampling, the
> normalization choice, the loss and class-imbalance reasoning, and above all the principle that
> the network's receptive field must equal the CFAR window so the two detectors differ *only* in
> the decision rule.
>
> **What changed is the number — now resolved.** The CFAR window was re-selected in step 6 on a
> dedicated tuning split and is **21 × 9** (guard 2/2, train 8/2), rectangular because the Doppler
> axis is only 64 bins and carries the clutter ridge while range has 256 and no comparable
> structure. TinyCNN's dilations were re-derived to match: **range (1,2,3,4) → RF 21**, **Doppler
> (1,1,1,1) → RF 9**. Verified by backprop from a single output pixel: measured 21 × 9. Parameter
> count is unchanged at 5,041.
>
> Read "21×21" below as **21 × 9**, and "256 × 128" as **256 × 64**.

> ## ⚠️ Two things that substitution rule does NOT fix (4 Sep 2026)
> Found while writing report Chapter 2. Both are scheduled for correction when Chapter 4 is
> written; noted here so nobody uses them in the meantime.
>
> 1. **§3.5(b) "it sees all 441 cells" is wrong.** 441 = 21×21. Under the real 21 × 9 window the
>    footprint is **189 cells**, the guard region is 5 × 5 = **25**, so CFAR estimates from
>    **164 reference cells**. The *argument* of §3.5(b) is unaffected — the network is not forced
>    to discard the guard cells and CFAR is — but the numbers are.
> 2. **§3.2 "both detectors see the same information" overstates the match**, and §3.5's
>    "Report phrasing" repeats it. Matching the receptive field matches the **spatial footprint**,
>    which gives comparable local context; it does not make the two decision rules use that context
>    identically. CFAR takes the CUT as its test statistic, excludes the guard ring, and estimates
>    the background from the reference cells only. Report Chapter 1 §1.3 (O1) carries the corrected
>    wording; use that, not the phrasing suggested in §3.5.
>
> Also note §9 line "params ≈ 4,945" counts convolution weights only. With BatchNorm the total is
> **5,041**, which is the figure the frozen configuration and the report use.

---

# 1. Problem formulation — what kind of ML problem is this?

The input is a 256 × 128 Range-Doppler map; the required output is "targets are at these
positions". Three formulations are possible, and the choice determines everything downstream.

| Formulation | Output | Verdict |
|---|---|---|
| **Classification** | "target present?" per frame | ✘ discards location — cannot produce Pd or localization error |
| **Object detection** (YOLO / Faster-RCNN) | bounding boxes | ✘ over-parameterised: targets are *points*, so box width/height are meaningless |
| **Dense prediction / heatmap regression** | per-cell score map | ✔ **chosen** |

## Why heatmap regression is the right choice

It is **structurally identical to what CFAR does**. CFAR emits a per-cell decision statistic; the
network emits a per-cell score. Both then pass through the *same* post-processing
(threshold → non-maximum suppression → point list) and the *same* evaluator. That symmetry is
what makes the comparison airtight.

## The governing mindset

> The network is **not** "an AI that finds cars". It is a **learned replacement for the CFAR
> thresholding step**. Same input (RD map), same output type (per-cell detection score), same
> post-processing. **Only the decision rule changes** — from a hand-derived statistic to a
> learned function.

Everything in this document follows from that sentence.

---

# 2. Model progression — four models, four questions

Each model exists to answer one specific question, not merely to be "bigger".

| # | Model | Params | Question it answers | Priority |
|---|---|---|---|---|
| 0 | **Tiny CNN** (RF matched to CFAR) | ~5 k | Given *identical* context, can a **learned** rule beat a **hand-derived** statistical rule? | must |
| 1 | **U-Net** | ~1–2 M | Does **more context** (global receptive field) help? | must |
| 2 | **Attention U-Net / transformer bottleneck** | ~2 M | Does **global attention** beat a fixed receptive field? | should |
| 3 | **U-Net with I/Q input** | ~1–2 M | Does **phase** add information? (proposal-mandated ablation) | nice to have |

**Free ablation:** models 0 → 1 → 2 differ by ~300× in parameter count, which *is* the proposal's
required "DL model size" ablation. No extra work needed.

**Escalation discipline:** finish and validate each model before starting the next. If schedule
tightens, stop after Model 1 — the core comparison is already complete.

---

# 3. Model 0 — the Tiny CNN, in full

## 3.1 Its two jobs

**Job A — prove the plumbing.** Loader → model → loss → training loop → heatmap → NMS → point
list → evaluator → ROC. Seven components, each able to fail silently. Building them behind a
*deliberately simple* model means a failure has exactly one candidate cause. Debugging a complex
architecture and an unverified pipeline simultaneously is how DL projects fail.

**Job B — a genuine scientific baseline** (below). This is the answer to "why bother with a tiny
model?"

## 3.2 The key design decision: match the receptive field to CFAR

CA-CFAR's decision for a cell is a function of a **21 × 21 window**:

```
window half-width = guard + train = 2 + 8 = 10   ->   2 x 10 + 1 = 21
```

Within that window it applies a hand-derived rule (mean of the training ring × α, compare).

If the CNN's receptive field is **also 21 × 21**, both detectors see the **same information** and
differ *only* in the decision rule. That isolates one variable:

> **Model 0 asks:** given identical context, can a learned rule beat a hand-derived one?

Model 1 (U-Net) then deliberately *breaks* that constraint with a far larger receptive field:

> **Model 1 asks:** does more context help?

Two models, two clean questions — an experimental design rather than a model zoo.

## 3.3 Receptive-field arithmetic

For stacked 3 × 3 convolutions at stride 1 with dilation `d_i`:

```
RF = 1 + Σ 2·d_i
```

| Dilations | Σd | RF | Assessment |
|---|---|---|---|
| [1, 1, 1, 1] | 4 | 9 × 9 | ✘ **less** context than CFAR — a loss would be confounded |
| [1, 2, 4, 2] | 9 | 19 × 19 | ≈ matched, but not exact |
| **[1, 2, 3, 4]** | **10** | **21 × 21** | ✔ **exact match — chosen** |
| [1, 2, 4, 8] | 15 | 31 × 31 | more context than CFAR |

**Why dilation at all?** It expands the receptive field **without adding parameters and without
downsampling** — exactly what is needed when the output must stay at full 256 × 128 resolution.

**Why [1,2,3,4] and not [1,2,4,3] or [1,2,4,8]?** Stacked dilated convolutions can suffer
**gridding artifacts**: when dilations share common factors, some input positions are never
sampled in combination, so the output is computed from a sparse lattice rather than the full
window. Standard guidance (Hybrid Dilated Convolution) is **smoothly increasing dilations with no
common factor**. `[1,2,3,4]` also gives a sensible functional progression — small dilations
capture fine local detail (the mainlobe), larger ones aggregate context (clutter, nearby strong
targets). Order does not change RF *size* (the sum is unchanged) but does affect coverage quality.

## 3.4 Architecture

```
input   (1, 256, 128)                       normalized RD map
  -> Conv 3x3, 16ch, dilation 1, pad 1  -> BatchNorm -> ReLU
  -> Conv 3x3, 16ch, dilation 2, pad 2  -> BatchNorm -> ReLU
  -> Conv 3x3, 16ch, dilation 3, pad 3  -> BatchNorm -> ReLU
  -> Conv 3x3,  1ch, dilation 4, pad 4      (no activation: raw logits)
output  (1, 256, 128)                       per-cell logit
```

| Choice | Justification |
|---|---|
| **No downsampling** | output must be the same size (dense prediction); down/up-sampling risks losing the exact peak bin, and a 1-bin error degrades the localization metric |
| **padding = dilation** | preserves spatial size exactly through every layer |
| **16 channels** | ≈ 4,945 parameters — tiny by design; if it cannot learn, that is itself informative |
| **BatchNorm** | stabilises training; adds robustness to imperfect input scaling |
| **ReLU** | cheap, standard, no vanishing-gradient issue |
| **No final activation** | emit **logits**; the loss applies sigmoid internally (`BCEWithLogitsLoss` is numerically stable, unlike separate sigmoid + BCE) |

Parameter breakdown: 160 + 2,320 + 2,320 + 145 = **4,945**.

## 3.5 Two honest caveats about "matched" receptive field

Both should be stated rather than hidden — an examiner can push on either.

**(a) Nominal RF ≠ effective RF.** The *theoretical* receptive field is 21 × 21, but the
**effective** receptive field is smaller and centre-weighted: influence decays roughly Gaussian
from the middle, because central pixels contribute through many more paths than edge pixels. CFAR
weights all training cells **uniformly**. So the CNN's *usable* context is somewhat tighter than
CFAR's at the same nominal RF.

*Report phrasing:* "the CNN's receptive field was matched to the CFAR reference window (21 × 21)
so the comparison isolates the decision rule rather than the extent of available context, noting
that a CNN's effective receptive field is centre-weighted rather than uniform."

**(b) The CNN sees slightly *more* information.** CFAR deliberately **excludes the guard region**
(5 × 5 around the cell) from its noise estimate — it must, or a target's own leaked mainlobe
energy would inflate its threshold and mask it. The CNN has no such exclusion: it sees all 441
cells.

This is **not unfairness — it is one of the findings.** CFAR excludes those cells out of
*necessity*, because a simple mean cannot distinguish "this neighbour is bright because of my own
mainlobe" from "this neighbour is bright because the noise is high". A learned function *can*
potentially make that distinction. So: **the network can exploit information the classical rule is
forced to discard.** The correct framing is "same window, but CFAR is constrained to a
hand-derived statistic that requires discarding the inner cells".

---

# 4. Input pipeline — normalization is the highest-leverage decision

## 4.1 The measured problem

Measured on 200–400 frames of `frozen_train`:

| Issue | Measurement |
|---|---|
| Raw power unusable | spans **~7 orders of magnitude** → exploding gradients |
| Absolute scale varies frame-to-frame | `noise_var` ranges **554 → 31,830 (57×)**, because scene SNR is randomised per frame |
| Per-frame max normalization (the stored `rd_map_db`) | ✘ **destroys absolute SNR**: a weak-target frame and a strong-target frame become identical after dividing by their own max |

## 4.2 The chosen transform — dB above the frame's noise floor

```
x = 10 · log10( power / median(power) )        then clip to [-10, +60] dB, scale by /20
```

**Four reasons:**

1. **It is stable.** Measured: after the transform the median is `-0.00 dB` with **std 0.000
   across frames**. Every frame arrives on an identical scale, so no capacity is wasted learning
   to renormalise.
2. **It preserves the true detection cue.** What matters physically is "how far above the noise is
   this cell", not absolute power. A 15 dB target reads +15 dB regardless of the noise level.
3. **It mirrors CFAR.** CFAR compares each cell to a locally-estimated noise level; this gives the
   network the same *kind* of reference. It is information CFAR also possesses → no unfair
   advantage.
4. **The median is robust.** Targets occupy 0.064 % of cells and the clutter ridge ~3 %, so
   neither meaningfully shifts the median.

Measured post-transform statistics: median 0 dB, 99.9th percentile +19.5 dB, max ≈ 32 dB
(range 13–64 dB).

## 4.3 Dataset sufficiency for alternative input types

| Input representation | Requires | Status |
|---|---|---|
| Magnitude (linear) | `rd_map_mag.npy` | ✔ saved |
| Magnitude in dB, any normalization | `rd_map_mag` | ✔ derive (better — we control the reference) |
| Power | `rd_map_mag` | ✔ derive |
| **I/Q (real + imaginary)** | `rd_complex` | ✘ **not saved** |
| Phase only | `rd_complex` | ✘ not saved |
| CFAR ratio map (hybrid input) | `rd_map_mag` | ✔ compute on the fly via `src/cfar` |
| Local noise estimate as a channel | `rd_map_mag` | ✔ compute on the fly |
| Label: hard 3 × 3 mask | `label_mask.npy` | ✔ saved |
| Label: soft Gaussian heatmap | `labels.csv` bins | ✔ derive |
| Label: point list (evaluation) | `labels.csv` | ✔ saved |
| Per-target SNR (analysis/weighting) | `labels.csv → cell_snr_db` | ✔ saved |

**The I/Q gap.** `compute_rd_map()` computes `rd_complex` but never saves it. Adding it costs
256 KB/frame (frames 166 KB → ~420 KB; the 14 k-frame set 2.4 GB → ~5.9 GB) and ~1.5 h to
regenerate.

**It is fully recoverable — not a now-or-never decision.** Every frame's `seed` is recorded
(10,000 unique seeds in `metadata.csv`, plus per-frame `noise_var`), and generation is
deterministic given the dataset `random_seed`. Re-running with an added save call yields
**bit-identical frames plus I/Q**; adding a save does not perturb the RNG sequence.

**Scientific prediction (state this *before* running the ablation).** Detection theory says
**magnitude is very nearly a sufficient statistic** here: for a point target with unknown/random
initial phase (the standard assumption), the optimal detector is the square-law/envelope detector,
i.e. magnitude only. The phase of a target's peak cell is essentially its random initial phase and
carries no detection information. **Therefore I/Q is predicted to give little or no improvement for
pure detection.** Reporting a *predicted* null result demonstrates understanding; fishing for an
improvement and finding none looks like failure. Phase *could* help for super-resolution,
discriminating coherent multipath from real targets, or extended-target structure — none of which
is the primary detection task.

---

# 5. Labels

**Start with the existing 3 × 3 hard mask** (`label_mask.npy`): on disk, binary, simple.

Note a deliberate mismatch: the target's *energy* footprint is ~5 × 5 (mainlobe ±2 bins: −20 dB at
±2, < −90 dB by ±3), while the *label* is 3 × 3. This is acceptable — the label defines the desired
output, and asking for a tighter blob than the input energy is a sharpening task CNNs handle well.
Because evaluation uses the **peak after NMS**, label width affects training dynamics more than the
final metric.

**Planned improvement (Model 1+):** soft Gaussian heatmaps centred on each target. A graded target
("closer is better") trains more stably than a binary cliff and is standard in keypoint detection
(e.g. human pose estimation). Derivable from `labels.csv` — no regeneration required.

---

# 6. Loss function — the measured imbalance is 1 : 1563

Measured on `frozen_train`: **8,388 positive cells out of 13,107,200 = 0.0640 %.**

Consequence: a model predicting "no target anywhere" achieves **99.94 % accuracy** and is useless.
Plain BCE collapses to exactly that. Accuracy must never be used as a metric here.

**Chosen: `BCEWithLogitsLoss(pos_weight = w)`**, starting `w ≈ 100–300`.

**Why the exact value of `w` matters less than intuition suggests.** The ROC is produced by
**sweeping the detection threshold**, so the network's output does not need to be a *calibrated
probability* — it needs to produce a good **ranking** of cells (targets above noise). `pos_weight`
mainly affects *trainability* (avoiding collapse) and where the untuned operating point sits, but
ranking quality determines the ROC. This is also a fairness parallel: **the DL operating point is
selected by threshold sweep, exactly as CFAR's α is.**

**Planned upgrades:** focal loss (α = 0.25, γ = 2), invented precisely for dense detection with
extreme foreground/background imbalance; or Dice / soft-F1, which optimises overlap directly.

---

# 7. Training setup

| Setting | Value | Reason |
|---|---|---|
| Optimizer | Adam, lr = 1e-3 | robust default; no schedule needed for a first pass |
| Batch size | 16 | fits comfortably in CPU RAM |
| Data (first run) | 1,000-frame subset | ~5 s/epoch → fast iteration to prove the pipeline |
| Data (real run) | full 10,000 | ~60 s/epoch |
| Epochs (first pass) | 5–10 | enough to confirm the loss decreases |
| Validation | `frozen_val` | monitored every epoch |

**Critical monitoring point.** Track train loss, val loss, **and val Pd at a fixed Pfa** computed
through the real evaluator. Decreasing loss does **not** guarantee better detection — the two are
correlated but not identical. The detection metric is the meaningful early-stopping signal and the
number actually reported.

---

# 8. Inference → points, and the fairness requirements

```
logits -> sigmoid -> probability map -> threshold -> NMS (local maxima) -> point list (r, v, score)
```

**Fairness requirement 1 — identical post-processing.** DL **reuses the same function** as CFAR:
`all_peaks(score_map, valid, nms_size)` from `src/cfar/detect.py`, with the **same `nms_size = 5`**.
Calling literally the same code makes divergent post-processing structurally impossible. A
different merge radius would resolve close targets differently and invalidate the comparison.

**Fairness requirement 2 — identical evaluation region.** CFAR cannot score border cells (no full
21 × 21 window fits), so it is evaluated only on the interior and its Pfa denominator counts only
those cells. A padded CNN *can* emit border predictions, but from zero-padded (fabricated) context.
**Decision: apply the same `valid` mask to DL evaluation** — same cells scored, same denominator,
same target population. Training uses the full map (no reason to waste data); evaluation uses the
identical valid region.

**Fairness requirement 3 — identical test set and metrics.** Both methods are scored on the frozen
test set through the same `src/eval` module (one-to-one Hungarian matching within a ±1-bin gate,
guard region excluded from the false-alarm denominator, Pd / per-cell Pfa / FA-per-frame /
localization error).

---

# 9. Verification plan — a gate at every step

| Step | Deliverable | Gate |
|---|---|---|
| 0 | PyTorch installed | `import torch` succeeds |
| 1 | `dataset.py` | correct shapes/dtypes; all finite; input median ≈ 0 dB; positive fraction ≈ 0.064 %; **orientation check** — input local max coincides with the label bin (proves no transpose in the loader) |
| 2 | `models.py` | output size == input size; params ≈ 4,945; **empirical receptive field = 21 × 21**, measured by backpropagating from a single output pixel and finding which input pixels receive nonzero gradient (verifies the design claim rather than trusting arithmetic) |
| 3 | loss + **overfit-one-batch** | **the critical gate** — on a single batch of 8 frames, loss → ~0 **and** predicted peaks land on labelled bins. If this fails, **stop and debug**: a model that cannot memorise 8 frames has a broken pipeline and no hyperparameter will fix it |
| 4 | training loop | loss decreases; no NaN; val loss behaves sensibly |
| 5 | `infer.py` | visual check — heatmap and detections vs ground truth on several val frames |
| 6 | evaluation hookup | a real ROC curve, plotted on the same axes as CA/OS/GO on identical frames |
| 7 | metric monitoring | val Pd improves across epochs (loss alone is insufficient evidence) |

Each step depends only on already-verified steps, so a failure has exactly one new candidate cause.

---

# 10. Expected results (stated in advance, so outcomes are interpretable)

- **Model 0 ≈ CA-CFAR**, since context is matched. Slightly better is plausible (a learned rule can
  exploit sidelobe/clutter *shape*, which a mean discards); slightly worse is also plausible
  (5 k parameters, finite data).
- **Either outcome is a valid result.** "A tiny CNN with CFAR-matched context performs comparably
  to CA-CFAR" is meaningful and honest, and it sets up U-Net's larger context.
- **Where DL is expected to pull ahead:** clutter and masking, because those are where the *shape*
  of the surroundings carries information a local mean cannot represent.
- **If DL loses, that is reported.** A fair comparison in which the classical method wins is a
  legitimate finding; a rigged DL victory is worthless.

---

# 11. How everything affects everything (interaction map)

| Change this | Consequence |
|---|---|
| **Normalization** | gradient magnitudes → whether the LR is stable → convergence. Wrong normalization means no learning, and it *looks* like a model problem |
| **Receptive field** | how much context the decision uses → clutter and masking performance. Too small and the model cannot beat CFAR **by construction** |
| **Loss / `pos_weight`** | recall-vs-precision balance → where the untuned operating point sits. Mostly moves *along* the ROC rather than changing it |
| **Label blob size** | sharpness of the predicted heatmap → how many peaks NMS finds → false-alarm rate |
| **NMS window** | whether nearby detections merge → close-target Pd → **must match CFAR for fairness** |
| **Threshold** | not a single choice — it is the ROC sweep variable |
| **Model size** | capacity *and* runtime → both a Pd axis and the runtime metric |
| **Batch size / epochs / subset** | wall-clock time → schedule risk (the reason for starting at 1,000 frames) |

**The chain worth memorising:** normalization decides whether it learns → receptive field decides
its ceiling → loss decides whether it trains at all → NMS decides fairness → threshold sweep
decides the reported curve.

---

# 12. Failure modes and diagnosis

| Symptom | Likely cause | Action |
|---|---|---|
| Loss drops fast then flatlines; output all zeros | class imbalance won; `pos_weight` too low | raise `pos_weight` or switch to focal loss |
| Loss = NaN | bad normalization, or LR too high | check input finiteness; lower LR |
| Cannot overfit a single batch | genuine pipeline bug | isolate: shapes, label alignment, gradient flow |
| Val loss rises while train loss falls | overfitting | more data, weight decay, early stopping |
| Peaks systematically offset by one bin | coordinate/transpose bug | the orientation check (gate 1) catches this |
| Heatmap blurry, far too many detections | label too broad, or NMS window too small | sharpen labels; revisit NMS |

---

# 13. Advanced architectures — assessed honestly

The user asked about transformers, YOLO, AFA, DenseCLIP and SCMRKD. Assessment:

## YOLO — real, but structurally mismatched
A single-shot detector: divides the image into a grid and per cell predicts boxes (x, y, w, h),
confidence and class in one forward pass. Rejected because (1) targets are **points**, so
width/height are meaningless and capacity is wasted regressing them; (2) it outputs a **box list**,
not a per-cell score, which **breaks the "identical post-processing" fairness argument** — the
cleanest part of the methodology; (3) its grid/anchor design assumes objects spanning many pixels,
while targets span 2–3 bins. *In fairness:* YOLO has been applied to RD maps in the radar
literature, so it is defensible — just not the right fit for this formulation.

## Transformers — genuinely relevant; the strongest "advanced" option
Self-attention lets every position attend to **every other position**, giving a **global receptive
field in a single layer**. Dense-prediction variants: **SegFormer** (hierarchical transformer
encoder + lightweight all-MLP decoder), **Swin-UNet** (shifted-window attention in a U-Net shape),
**TransUNet** (CNN encoder → transformer bottleneck → CNN decoder).

*Why it fits the thesis:* it is the sharpest possible contrast with CFAR. CFAR sees a 21 × 21
window, full stop; attention sees the whole map. The project's thesis is "local statistics vs
learned global context", and a transformer is that argument at its logical extreme.

*Real risks:* **data hunger** (ViTs lack CNNs' locality/translation-equivariance inductive biases,
so they typically need very large datasets or pretraining — 10 k frames of a simple task may be
insufficient); **pretraining does not transfer** (ImageNet weights assume natural-image statistics;
RD maps are physics-derived spectra); **compute** (attention is O(n²) in sequence length — painful
on CPU).

*Pragmatic version (recommended as Model 2):* a **hybrid** — keep the U-Net CNN encoder/decoder for
its inductive biases and data efficiency, and add either **attention gates on the skip connections**
(Attention U-Net) or a **self-attention block at the bottleneck** (TransUNet-lite). Gains the
global-context capability without betting the deadline on training a full ViT from scratch.

## AFA — solves a problem this project does not have
Understood as **"Learning Affinity from Attention"** (CVPR 2022): derives pixel affinity from
transformer attention maps to refine pseudo-labels for **weakly-supervised** semantic segmentation,
i.e. when only *image-level* labels exist. **Rejected:** its entire purpose is coping with *missing*
pixel labels; this project has pixel-perfect ground truth by construction. (Caveat: "AFA" is an
overloaded acronym — also Attentional Feature Fusion and others.)

## DenseCLIP — solves a problem this project does not have
**"DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting"** (CVPR 2022): adapts
CLIP (pretrained on hundreds of millions of **image–text pairs**) from image-level to **pixel-text**
matching. **Rejected for two independent reasons:** (1) there is **no language component** — one
class ("target"), no text, nothing for prompts to describe; (2) **wrong image domain** — CLIP's
features encode natural photographs, and an RD map is a physics-derived 2-D spectrum. It also
targets open-vocabulary/zero-shot settings, whereas this task has one class and full supervision.

## SCMRKD — not recognised
This acronym was not recognised and no explanation was invented. The trailing "KD" suggests
**Knowledge Distillation** (training a small "student" to mimic a large "teacher"). If that is what
it is, it *is* relevant in one specific way: distillation is a legitimate route to a small, fast
model that performs like a larger one — directly useful for the **runtime** metric. Source needed
before assessing further.

## The meta-lesson (transferable)

| Method | Problem it solves | Present here? |
|---|---|---|
| AFA | missing pixel labels | ✘ labels are perfect |
| DenseCLIP | no labels / open vocabulary; natural images + text | ✘ one class, no text, non-natural images |
| YOLO | localising extended objects with boxes | ✘ targets are points |
| **Transformers** | limited receptive field / need for global context | ✔ **a real limitation of both CFAR and small CNNs** |

**Choose methods by the problem they solve, not by how recent they sound.** Three of those four are
excellent papers that would add nothing here.

**The actual bottlenecks in this project, and what fixes each:**
1. **Extreme class imbalance** (0.064 % positive) → **loss function** (focal / weighted / Dice), not architecture
2. **Sub-bin localization accuracy** → **label representation** (Gaussian heatmaps) and skip connections
3. **Distinguishing target from clutter and sidelobes** → **receptive field / context** ← where attention genuinely helps

So the highest-value work is loss + label representation + receptive field, not chasing SOTA names.

---

# 14. Guiding principles

1. **Fair comparison over impressive numbers.** Same test set, same post-processing, same metrics.
   A rigged win is worthless.
2. **Prove the pipeline before improving the model.** Get end-to-end working badly, then improve.
   Never debug two things at once.
3. **Data and loss matter more than architecture.** Normalization and class imbalance will make or
   break this; swapping U-Net for something fancier will not.
4. **Escalate deliberately.** Each model answers a question.
5. **Be honest about what cannot be claimed.** No guaranteed Pfa (unlike CFAR, which guarantees it
   by design), possible domain shift, synthetic-only training. Naming limitations is a strength.

---

# 15. Scope for the deadline

| Tier | Contents |
|---|---|
| **Must have** | Tiny CNN (pipeline) → U-Net → fair CFAR-vs-DL comparison on the frozen test set |
| **Should have** | Pd-vs-SNR and per-condition breakdown; runtime + model size |
| **Nice to have** | I/Q ablation, model-size ablation, Gaussian-heatmap labels |
| **Stretch** | Hybrid CFAR + DL input (feed the network both the RD map and the CFAR ratio map — tests whether DL adds value *on top of* classical processing rather than replacing it; this is what the modern CFAR-CNN cascade literature actually does) |
