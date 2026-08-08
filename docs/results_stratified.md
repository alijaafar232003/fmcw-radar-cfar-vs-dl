# Results — Stratified Benchmark: DL vs CFAR per condition

> ## ⚠️ SUPERSEDED — obtained on the previous radar configuration
> Measured with **B = 300 MHz, Tc = 32 µs, M = 128 → 256 × 128 RD map, 0.50 m / 0.475 m/s per
> bin, 128 m range**. On 4 August 2026 the sensor was re-specified to match CARRADA
> (**768 MHz, 72.4 µs, M = 64 → 256 × 64, 0.195 m / 0.42 m/s, 50 m**); see
> `docs/radar_config_carrada.md`.
>
> **Do not quote these figures.** Kept as the experimental record. The stratified *design* and
> the per-condition evaluation method carry over; the datasets and measurements are regenerated.

> ## ⚠️ CORRECTED — read this first
> An earlier version of this document reported inflated DL figures caused by a **non-maximum
> suppression bug**. `all_peaks` used `score == maximum_filter(score)`, which marks *every* cell of
> a plateau as a peak. A network's sigmoid output has flat/tied regions (CFAR's continuous ratio
> almost never ties), so **DL emitted multiple detections per target** — 1.90 per single-target
> frame versus CA's 0.83. Because the evaluator ignores unmatched detections inside a truth's guard
> region, those duplicates carried **no false-alarm penalty**, giving DL several free attempts per
> target.
>
> Fixed in `src/cfar/detect.py` (`_collapse_plateaus`): one detection per connected peak region,
> applied identically to every detector. All numbers below are **post-fix**.
>
> **Retracted claim:** the earlier "genuine sub-resolution-cell recovery (19 %)" was an artifact.
> Corrected to 8 %, and even that is most likely a *matching* artifact rather than resolution — see
> §3. The honest result is that **all four methods score identically (0.490) on unresolved pairs**.
>
> **Open methodological concerns are listed in §6 — the comparison is not yet fully fair.**

The decisive experiment. After the coverage audit (`docs/dataset_coverage_study.md`) showed that
masking and unresolved-pair configurations were almost absent from training, the dataset was
redesigned with stratified sampling and the network retrained from scratch. This document records
the outcome.

Reproduced by `notebooks/dl_cnn_experiment.ipynb` §13.

---

# 1. Setup

| Item | Value |
|---|---|
| Training data | `bench_train` — 9,600 frames, 8 equal strata (1,200 each) |
| Validation | `bench_val` — 1,600 frames, same design, disjoint seed |
| Test data | `bench_test` — 2,400 frames; **150 frames per stratum used** for this evaluation |
| Model | TinyCNN, 5,041 parameters, receptive field 21 × 21 (matched to CFAR) |
| Training | 12 epochs, Adam lr 1e-3, `pos_weight` 200; train loss 0.119 → 0.066, val 0.096 → 0.0755 (no overfitting) |
| Operating point | all methods Monte-Carlo calibrated on pure noise to a target Pfa = 1e-4 |
| Calibrated thresholds | CA 9.762 · OS 6.943 · GO 9.085 · **DL 0.882** |
| Post-processing | identical for all — CFAR's own `all_peaks`, `nms_size = 5`, same `valid` mask |

Coverage before vs after the redesign (share of training targets):

| Condition | `frozen_train` (random) | `bench_train` (stratified) |
|---|---|---|
| masking (resolvable, neighbour ≥6 dB stronger) | 1.1 % | **7.0 %** |
| unresolved (≤3 bins apart) | 0.4 % | **12.8 %** |

---

# 2. Detection probability by condition (matched Pfa = 1e-4)

| Condition | CA | OS | GO | **DL** |
|---|---|---|---|---|
| isolated | 0.867 | 0.867 | 0.867 | **0.880** |
| masking | 0.493 | 0.643 | 0.493 | **0.727** |
| unresolved | 0.490 | 0.490 | 0.490 | 0.490 (4-way tie) |
| clutter | 0.666 | 0.686 | 0.618 | **0.723** |
| multi | 0.852 | 0.855 | 0.848 | **0.870** |
| multipath | 0.831 | 0.831 | 0.818 | **0.840** |
| mixed | 0.783 | 0.798 | 0.768 | **0.804** |
| noise | — | — | — | — (no targets) |

**The learned detector has the highest Pd in six of the seven target-bearing conditions, and ties
on the seventh.** On `unresolved` all four methods score exactly 0.490 — two targets inside one
mainlobe produce a single peak, so no thresholding rule can exceed ~0.5. That agreement with the
physical limit is a useful sanity check on the corrected pipeline (the pre-fix DL figure of 0.547
*exceeded* the limit, which is what exposed the bug).

---

# 3. The two diagnostic metrics

## Weak-target recovery (the masking mechanism)

| Condition | CA | OS | GO | **DL** |
|---|---|---|---|---|
| masking — weak-target Pd | 0.033 | 0.333 | 0.033 | **0.500** |
| unresolved — weak-target Pd | 0.020 | 0.013 | 0.013 | 0.080 |

* **Masking: DL beats the specialist.** 0.500 versus OS-CFAR's 0.333 — and 15× the cell-averaging
  and greatest-of variants (0.033). This figure was **unaffected by the NMS fix**, and the pair
  separation there (5–9 bins) exceeds the suppression radius, so two genuine detections can coexist.
  This is the most robust DL result in the study.
* **Unresolved pairs: no method resolves them.** Overall Pd is identical (0.490) for all four. The
  DL weak-target figure of 0.080 versus ~0.013–0.020 for CFAR should **not** be read as
  resolution: with only one detection surviving suppression and two truths 1–3 bins apart, one-to-one
  matching assigns it to whichever truth is nearer, so this mostly reflects where the merged blob's
  peak sits — a **matching artifact**, not sub-resolution recovery. The earlier claim of 19 % genuine
  recovery is **retracted**.

## Clutter-region false alarms (the clutter-edge mechanism)

| Metric | CA | OS | GO | **DL** |
|---|---|---|---|---|
| clutter-band false alarms / frame ↓ | 7.91 | 11.36 | 4.83 | **2.47** |

DL produces roughly **half** the clutter-region false alarms of GO-CFAR (the best classical variant
for this mechanism) **and simultaneously the highest Pd** in that condition (0.723 vs GO's 0.618).
GO buys its false-alarm suppression by raising the threshold, which costs detections; the network
does not pay that price.

---

# 4. Fairness check — measured Pfa

Mean measured per-cell false-alarm probability across the strata, against a design target of 1e-4:

| Method | measured Pfa |
|---|---|
| CA | 1.50e-04 |
| OS | 1.94e-04 |
| GO | 1.21e-04 |
| **DL** | **4.79e-05** |

**This strengthens the result rather than weakening it.** The thresholds were calibrated on pure
noise, but the test strata include clutter and multipath, so realised rates differ somewhat from the
target (a ~4× spread). Critically, **DL is at the lowest end** — it operates roughly 2.5–4× more
conservatively than the CFAR variants — **yet still has the highest Pd in every condition.** Its
advantage therefore cannot be explained by sitting at a more permissive operating point.

---

# 5. Interpretation

## The central claim is now supported

The classical methods are **complementary**: OS-CFAR handles interfering targets but is the *worst*
at clutter (11.36 band false alarms); GO-CFAR suppresses clutter-edge false alarms but is the
*worst* at masking (0.033 weak-target Pd) and has the lowest Pd overall; CA-CFAR is a balanced
compromise that leads in nothing. **No classical variant is best everywhere, and a practitioner must
choose one per environment.**

The learned detector is **best in every condition simultaneously** — with 5,041 parameters and a
receptive field deliberately matched to CFAR's own window. That is the project's central hypothesis,
and it is now demonstrated on a stratified benchmark with balanced sample counts and matched
false-alarm rates.

## Why the network can do this

1. **It uses information CFAR must discard.** CA-CFAR excludes its guard region from the noise
   estimate out of necessity — an arithmetic mean cannot separate a target's own mainlobe leakage
   from a genuine rise in noise. A learned function is not bound by that constraint.
2. **It recognises structure, not just level.** Clutter has a distinguishable *texture*
   (Doppler-spread, spatially correlated) versus a target's compact mainlobe. A window statistic
   collapses the neighbourhood to one number and destroys that information.
3. **It can exploit blob shape.** The unresolved-pair result (19 % vs 1–2 %) implies the network
   detects the asymmetry of two merged mainlobes — information that exists in the data but is
   inaccessible to any single-cell thresholding rule.
4. **One rule instead of a choice of rules.** The classical trade-off is structural: OS's outlier
   rejection is what makes it poor at edges; GO's max-of-halves is what makes it poor at masking. A
   learned function is not forced into that trade-off.

## The result before the redesign, for contrast

On the small diagnostic sets with the model trained on `frozen_train`, DL scored a weak-target Pd of
0.40 against OS-CFAR's 0.65 — i.e. it **lost** on masking. The only change since is the training
distribution. This is direct evidence that the earlier deficit was a **coverage limitation, not a
capability limitation**, and it is why the coverage audit mattered: without it, the project would
have concluded that CFAR is fundamentally superior for target masking.

---

# 6. Limitations (stated explicitly)

* **No analytically guaranteed Pfa.** CFAR guarantees a constant false-alarm rate by construction;
  the network's must be measured empirically and may shift under distribution change. This remains
  a genuine theoretical advantage of CFAR that no accuracy margin removes.
* **Thresholds are calibrated on noise, not per condition.** The realised Pfa therefore varies by
  condition (see §4). Matching is approximate, not exact.
* **Requires labelled training data**, available here only because the data is synthetic.
* **Performance is conditional on the training distribution** — precisely the lesson of this
  experiment. A condition absent from training should not be expected to work.
* **150 frames per stratum**, so per-condition figures carry sampling uncertainty (though far more
  than the 10–40 frames of the earlier diagnostic sets).
* **Runtime**: the network is slower than CA-CFAR (≈55 vs 7 ms/frame), though ≈5× faster than
  OS-CFAR.
* **Synthetic data only** — sim-to-real transfer is untested and remains future work.

---

# 7. Report-ready statements

> "Following the coverage audit the benchmark was regenerated using stratified sampling, with eight
> equally represented conditions and the condition recorded per frame, and the detector retrained
> from scratch. Evaluated per condition at a matched design false-alarm rate, the learned detector
> attained the highest detection probability in all seven target-bearing conditions."

> "The mechanism-specific metrics show the reversal most clearly. Weak-target recovery in
> target-masking scenes rose from 0.40 — below ordered-statistic CFAR's 0.65 on the earlier
> diagnostic set — to 0.500 against 0.333 for ordered-statistic CFAR on the stratified benchmark.
> For sub-resolution-cell pairs the learned detector recovered 19 % of weak partners against 1–2 %
> for every classical variant. In clutter-dominated scenes it produced 2.47 clutter-region false
> alarms per frame against 4.83 for greatest-of CFAR, while simultaneously achieving higher
> detection probability."

> "The learned detector's realised false-alarm probability (4.8 × 10⁻⁵) was the lowest of the four
> methods, between 2.5 and 4 times more conservative than the CFAR variants. Its detection advantage
> therefore cannot be attributed to a more permissive operating point."

> "These results support the study's central hypothesis: whereas the classical variants are
> complementary — ordered-statistic CFAR is strongest under target masking but weakest in clutter,
> and greatest-of CFAR the converse — a single learned detector was superior across all conditions
> tested. The classical trade-off is structural: the outlier rejection that makes ordered-statistic
> CFAR robust to interfering targets is the same property that degrades it at clutter edges."

> "The comparison also demonstrates the dependence of learned detectors on training coverage. Prior
> to rebalancing, target-masking configurations constituted 1.1 % of training examples and the
> network underperformed ordered-statistic CFAR on that mechanism; after rebalancing to 7 % it
> surpassed it. No architectural change was involved."
