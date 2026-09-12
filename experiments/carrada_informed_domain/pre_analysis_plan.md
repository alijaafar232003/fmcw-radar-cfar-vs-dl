# Pre-analysis plan — CARRADA-informed synthetic training

**Written and frozen before the new models touched CARRADA test.** Its hash is recorded in
`FROZEN_CONFIG.yaml`, and `results/freeze.json` records the hash of every file it covers.

## What this experiment is, and what it is not

A **post-hoc domain-intervention experiment using synthetic-only model optimisation and
target-domain-informed simulator calibration.** Short name: *CARRADA-informed synthetic training*.

It is **not** a new pristine zero-shot external validation. The CARRADA test split was already read
once, for the Chapter 7 result. This is a second, deliberately-designed read, and it is reported as
such throughout.

What remains true, and is enforced rather than asserted:

* **No real radar frame enters neural-network optimisation.** No gradient, no fine-tuning, no
  initialisation, no early stopping, no epoch selection, no seed selection, no checkpoint selection.
* **No real radar frame enters threshold calibration.** Thresholds come from the new *synthetic*
  validation split alone.
* **CARRADA test did not influence the design.** No test sequence was opened by any file in this
  experiment before the frozen evaluation.
* **CARRADA validation was not used at all**, for anything.

What *is* deliberately used: the 18 official **CARRADA train** sequences, to measure the target
domain's background statistics and calibrate the synthetic generator. That is the intervention.

## Primary hypothesis

**H1.** Training TinyCNN on a synthetic domain whose background/clutter statistics are calibrated
from CARRADA train improves transfer to CARRADA test relative to the original simulator-trained
TinyCNN.

## Primary metric

**Exact dense-mask object recall** on the official CARRADA test split, under the frozen Chapter 7
protocol: 2,130 frames, detector-valid interior, NMS radius 2, dense segmentation support,
one-to-one matching, exact support.

## Primary statistical comparison

**CARRADA-informed TinyCNN − original TinyCNN**, paired over the **six** test sequences.

* per-sequence recall for each arm, seed-averaged within arm;
* sequence-balanced mean difference;
* paired bootstrap over the six sequences, **10,000** replicates, 95 % CI;
* the CI is the inferential quantity. Seed variability is reported **separately** and never pooled
  into it.

**8 seeds × 1,627 objects is not thousands of independent samples.** The sequence is the resampling
unit, exactly as in Chapter 7.

This is the comparison the experiment exists to make. New-versus-CFAR is secondary.

## Secondary outcomes

Reported for all three arms (original TinyCNN, CARRADA-informed TinyCNN, CA-CFAR):

1. **±1 and ±3 support-dilation recall.** Reported as tolerance, never as localisation error —
   CARRADA provides extended truth, not point truth.
2. **Unmatched detections per frame.** An **output-burden** metric. Never called false alarms and
   never called $P_{fa}$: CARRADA cannot support a false-alarm measurement.
3. **Range strata**, the frozen Chapter 7 blocks `(10,60) (60,110) (110,160) (160,210) (210,246)`.
   The pre-specified question: does adding train-derived range statistics reduce the original
   arm's near-range deficit in **10–60**, where Chapter 7 measured TinyCNN 0.8455 against CA 0.9475?
4. **Zero-Doppler strata**, the frozen band `(28, 37)`: `near_zero` if an object's valid support
   meets it, else `away_zero`. Chapter 7 found similar CNN-versus-CA differences on both sides;
   the question is whether the intervention changes that pattern. **The ridge is not assumed to be
   the cause of anything.**
5. **Class strata** (car, pedestrian, cyclist).
6. **Seed variability**: per-seed recall, mean, SD, min, max, and how many seeds beat CA-CFAR.
7. **CARRADA-informed TinyCNN versus CA-CFAR**, same sequence-balanced method.

**Strata are frozen here and will not be redesigned after seeing the result.**

## What is held fixed

Architecture (TinyCNN, width 16, **5,041 parameters**, dilations (1,2,3,4)/(1,1,1,1), measured
receptive field 21×9, no downsampling); optimizer Adam; lr 1e-3; weight decay 0; batch size 32;
epochs **21**; `pos_weight` **100**; `BCEWithLogitsLoss`; the **flat 3×3 label**; `threads: 2`
(load-bearing — torch CPU kernels reduce differently at other thread counts); seeds **0–7**;
the freeze/checkpoint methodology; the evaluation code; NMS radius 2; the detector-valid interior;
one-to-one matching; CARRADA dense-mask truth; exact/±1/±3; the output-burden calculation;
CA-CFAR and its calibrated alphas.

**Target-side generation is unchanged and asserted equal.** `make_configs.py` derives the dataset
configs from `configs/bench2_train.yaml` and `configs/bench2_val.yaml` and asserts that every
top-level key except `dataset` and `save` compares equal, and that the scenario list is identical.
Target counts, placement, separations, amplitude ranges, the SNR sweep, propagation and the strata
proportions are therefore the originals.

**Not used, deliberately:** Gaussian or point supervision (Chapter 6 showed supervision changes
localisation; varying it here would confound domain matching with label design); extended
CARRADA-like target morphology (would confound background statistics with target morphology); any
larger network, U-Net, augmentation pipeline or fine-tuning.

## The intervention, in full

Three background properties change, all set from CARRADA train via
`domain_audit/train_derived_targets.json`:

| component | setting | train-derived source |
|---|---|---|
| multi-look smoothness | $L \sim \mathrm{Unif}\{7,8\}$, incoherent power average | detrended $L = 7.42$, per-sequence 7.08–7.88 |
| range dependence | measured 256-bin unit-mean-power profile, background only | mean background power per range bin, 9-bin smoothed |
| ridge structure | `width_bins=0`, `cnr_db ~ Unif(19.4, 26.4)`, prevalence 0.990 | peak-over-floor p5–p95 and the measured >10 dB frame fraction |

`generate_dechirped_cpi`, the Blackman range/Doppler windows and `compute_rd_map` are the frozen
originals, so the target model, the PSF, the 21×9 receptive-field equality and the guard geometry
are untouched.

### Declared residuals, fixed before the result existed

1. **Lag-1 Doppler speckle correlation is not matched** (≈0.58 against CARRADA's ≈0.02
   within-block). It is set by the slow-time window, which also defines the target PSF, and the PSF
   is frozen.
2. **Lag-1 range correlation regresses** (frame-level error 0.016 → 0.131). The range profile's
   common-mode raises it, on top of a window-set baseline already above CARRADA's.
3. **The ridge skirt is wider than CARRADA's** (4.9 against 1.8 bins above 3 dB) although the core
   matches (half-power 1.31 against 1.01 bins). Same frozen-window constraint.
4. **Gamma $k$ overshoots** (5.005 against 3.975) while dB spread matches to 0.023. CARRADA's
   moment-$k$ is dragged down by fixed range-localised bright structure this generator does not
   model.
5. **Fixed range-localised bright structure and extended target morphology are not modelled.**

All five were recorded in `domain_audit/pilot_validation.md` **before** any model was trained.

## Threshold rule

One global threshold per seed, from the **new synthetic validation split only**: the lowest score
threshold whose pooled measured per-cell $P_{fa} \le 10^{-4}$, pooled over all 8 synthetic
conditions, at the ±1 gate, with the frozen valid region.

### A defect in the Chapter 7 implementation, found before running

`dl.infer.predict_points` declares `nms_size: int = NMS_SIZE`, and a Python default binds at
function-definition time. `dl.infer.NMS_SIZE` is 5, and `collect_split` calls `predict_points`
without passing `nms_size`. So `collect_split` always extracts at radius 5. The NMS audit's patched
copy sets `det.NMS_SIZE = 2`, which reaches `det.tinycnn_peaks` but **not** `collect_split` — so the
thresholds published as radius-2 were in fact derived at **radius 5**. That is why they are
byte-identical to the radius-5 set.

Consequence: in the existing Chapter 7 radius-2 result, thresholds derived at radius 5 are applied
to candidates extracted at radius 2. Radius 2 keeps a superset of radius-5 candidates, so measured
$P_{fa}$ at a given threshold is higher at radius 2, and the rule-correct radius-2 threshold is
therefore **higher** than the one used. The arm operates slightly above its intended false-alarm
rate.

**Pre-specified handling, decided before any result:**

* **Primary comparison uses `r=5`-derived thresholds for the new arm**, reproducing the Chapter 7
  code path exactly, so that new-versus-original differs in **training domain only** and not in
  threshold protocol. A protocol difference between arms would confound H1 directly.
* **`r=2`-derived thresholds are also produced**, and the primary comparison is repeated under them
  as a **pre-registered sensitivity check**. Both threshold sets and both results are reported.
* If the two disagree on the sign or the significance of the primary difference, **both are reported
  and no single headline is claimed.**
* Chapter 7's artefacts are not modified. Whether to correct the thesis is a separate decision.

## Interpretation rules, fixed in advance

H1 is not to be forced. The outcome is classified as:

* **A — improvement without added burden.** Recall improves meaningfully with similar or lower
  unmatched output. Strongest permitted statement: *"reducing the measured synthetic-to-real
  statistical domain gap improved transfer of the learned detector."*
* **B — improvement with burden increase.** *"Domain-informed training improved recall, but the
  improvement was accompanied by denser output."* Not an unconditional detector improvement.
* **C — no meaningful improvement.** *"Matching the measured background/clutter statistics alone
  was insufficient to recover the synthetic learned advantage."*
* **D — degradation.** *"Matching selected real-domain summary statistics did not produce a
  synthetic training distribution that transferred better."* Reported in full.

### Causal limit

Several simulator properties changed jointly. The experiment can support a causal statement about
the **combined domain intervention** only:

> Changing the synthetic training domain toward the measured CARRADA-train statistics changed
> transfer performance.

It **cannot** support "Gamma shape caused the domain shift", "Doppler correlation caused the
transfer failure", or "the ridge caused the result". Isolating individual causes needs separate
ablations — background distribution only, correlation only, ridge only, range non-stationarity
only, extended-target morphology — which are **noted as future work and not run here.**

## Stopping rule

One evaluation on CARRADA test, after this plan and the configuration are frozen. No re-tuning
after seeing the test result; no reselection of arms, strata, seeds, epochs or thresholds. The
sensitivity check above is part of the plan, not a response to the outcome.
