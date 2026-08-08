# Results — Model 0 (Tiny CNN) vs CFAR

> ## ⚠️ SUPERSEDED — obtained on the previous radar configuration
> Every number in this document was measured with **B = 300 MHz, Tc = 32 µs, M = 128
> → 256 × 128 RD map, 0.50 m / 0.475 m/s per bin, 128 m range**.
>
> On 4 August 2026 the sensor was re-specified to match the CARRADA dataset
> (**768 MHz, 72.4 µs, M = 64 → 256 × 64, 0.195 m / 0.42 m/s, 50 m**) so that simulated and real
> maps are directly comparable — see `docs/radar_config_carrada.md`.
>
> **Do not quote these figures in the report.** They are kept as the experimental record of how
> the method was developed and verified. The *method* — matched receptive field, shared NMS and
> matcher, matched-Pfa comparison — carries over unchanged; only the measurements are re-run.

Detailed results record for the first CFAR-vs-DL comparison: setup, verification, measurements,
per-method pros and cons, interpretation, and an explicit statement of what can and cannot be
claimed. Reproduced by `notebooks/dl_cnn_experiment.ipynb`; design record in `docs/dl_design.md`.

---

# 1. Experimental setup

| Item | Value |
|---|---|
| Test data | `frozen_test`, 500 frames (of 2,000), never used for training or tuning |
| Data character | neutral mixture: 0–5 targets, cell SNR spanning the detection transition, clutter + multipath enabled, radar-equation range law |
| Radar | 77 GHz, B = 300 MHz, Tc = 32 µs, N = 512, M = 128 → 256 × 128 RD map |
| CFAR geometry | guard 2, train 8 (N_ref = 416), window 21 × 21 |
| CFAR thresholds | Monte-Carlo calibrated per method (CA 9.74, GO 9.09, OS 6.98 at Pfa = 1e-4) |
| DL model | TinyCNN, 5,041 parameters, dilations (1,2,3,4) → RF 21 × 21 |
| DL training | 3,000 frames of `frozen_train`, 8 epochs, Adam lr 1e-3, `pos_weight` 200 |
| Matching | one-to-one (Hungarian) within a ±1-bin gate |
| Post-processing | **identical for all methods** — CFAR's own `all_peaks`, `nms_size = 5` |
| Evaluation region | **identical** — CFAR's border-excluded `valid` mask applied to DL too |

**Fairness enforcement (three mechanisms).** (1) DL *imports* CFAR's peak-extraction function, so
post-processing cannot diverge. (2) The same `valid` mask gives both an identical Pfa denominator and
target population. (3) Both are scored by the same `src/eval` module with the same matching rule.
Additionally, the DL receptive field was **matched to CFAR's window (21 × 21)**, so the comparison
isolates the *decision rule* rather than the amount of context available.

---

# 2. Verification results (all gates passed)

| Gate | Measurement | Verdict |
|---|---|---|
| Dataset shapes / dtypes | (16, 1, 256, 128) float32 | PASS |
| Finiteness | no NaN/Inf | PASS |
| Normalization stability | input median **−0.0000** (std 0.000 across frames) | PASS |
| Positive fraction | 0.070 % (matches the 0.064 % dataset measurement) | PASS |
| **Orientation** | **72/72** visible targets peak at the label bin | PASS — no transpose bug |
| Output resolution | 256 × 128 preserved | PASS |
| Parameter count | 5,041 | as designed |
| **Receptive field (measured)** | **21 × 21** by backprop from one output pixel | PASS — matches CFAR exactly |
| **Overfit one batch** | loss 0.798 → 0.00023 (**3514×**); **25/25** peaks within 1 bin | PASS — pipeline sound |

The overfit test is the critical one: a model that cannot memorise 8 frames has a broken pipeline,
and no hyperparameter would fix it. It checks **geometry as well as loss**, because a low loss alone
does not prove the peak is in the right place.

---

# 3. Training

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.20325 | 0.14054 |
| 4 | 0.11191 | 0.11561 |
| 8 | **0.10246** | **0.10657** |

Validation loss tracks training loss closely → **no significant overfitting**. The model was
**deliberately under-trained** (3,000 of 10,000 available frames, 8 epochs) because the objective at
this stage was to prove the pipeline, not to maximise performance. **All DL numbers below are
therefore a floor, not a ceiling.**

---

# 4. Main result — Pd at matched false-alarm rate

`frozen_test`, 500 frames, identical post-processing and evaluator:

| Method | Params | Pd @ Pfa=1e-4 | Pd @ Pfa=1e-3 | Pd @ Pfa=1e-2 | ms/frame |
|---|---|---|---|---|---|
| **DL (TinyCNN)** | 5,041 | **0.784** | **0.832** | 0.856 | 55.5 |
| CA-CFAR | — | 0.752 | 0.814 | 0.859 | **6.9** |
| OS-CFAR | — | 0.742 | 0.820 | **0.862** | 296.7 |
| GO-CFAR | — | 0.752 | 0.806 | 0.855 | 38.4 |

**Reading it:**
* **DL leads in the low-Pfa regime**: +3.2 points over the best CFAR at Pfa = 1e-4, +1.2 at 1e-3.
  This is the operationally important region — a practical radar must keep false alarms rare.
* **At high Pfa (1e-2) the methods converge**, with OS marginally ahead. Once the threshold is
  permissive, nearly all detectable targets are found by any rule, so the decision rule stops
  mattering.
* **The margin is modest — and that is the expected, healthy outcome.** With context deliberately
  matched, a large DL win would be a red flag suggesting an unfair comparison.

## Why the network can win at all with matched context

CFAR must **exclude its guard region** from the noise estimate. It has no choice: a simple mean
cannot distinguish *"this neighbour is bright because of my own mainlobe"* from *"the noise here is
high"*, so including those cells would let a target raise its own threshold and mask itself.

A learned function faces no such constraint — it can use the guard-region values *intelligently*,
because it can learn the characteristic shape of a mainlobe versus elevated noise. **The network
exploits information the classical rule is structurally forced to discard.** That is the mechanism
behind the observed gain, and it should be stated explicitly rather than presented as unexplained
DL superiority.

---

# 5. Per-method pros and cons

## CA-CFAR (cell-averaging)
**Pros**
* **Theoretically optimal** in homogeneous noise — lowest CFAR loss of any variant.
* **Cheapest by a wide margin**: 6.9 ms/frame (8× faster than the network, 43× faster than OS).
* **Analytically guaranteed Pfa** via the closed form `α = N(Pfa^(−1/N) − 1)`. A design guarantee,
  not an empirical observation — decisive for safety-critical use.
* Fully interpretable; no training data; no domain-shift risk.

**Cons**
* **Severe target masking**: weak-target Pd of only **0.15** in the masking scenario — an
  interfering target inflates the mean and the real target is hidden by its own raised threshold.
* **Clutter-edge false alarms**: 14.2 per frame in the clutter band.
* Assumes a homogeneous exponential background; degrades whenever that fails.

## OS-CFAR (ordered statistic)
**Pros**
* **Best classical method under masking**: weak-target Pd **0.65** — **4.3× CA**. Excluding high
  outliers via a rank statistic is exactly the right defence against interfering targets.
* Tolerates up to `N − k` interferers before degrading.
* Retains a guaranteed Pfa once calibrated.

**Cons**
* **By far the most expensive**: 296.7 ms/frame (43× CA) — the rank filter must sort a 416-cell
  window per cell. A serious obstacle to real-time use.
* **Worst at clutter edges**: 17.1 false alarms/frame, worse than CA.
* Requires choosing the rank `k` (robustness vs loss trade-off).
* ~0.5 dB extra CFAR loss in homogeneous noise.

## GO-CFAR (greatest-of)
**Pros**
* **Best clutter-edge control**: 8.4 false alarms/frame — **41 % fewer than CA**. Taking the greater
  half-window keeps the threshold high on the clutter side and suppresses the edge burst.
* Moderate cost: 38.4 ms/frame.
* Retains a guaranteed Pfa once calibrated.

**Cons**
* **Worst under masking**: weak-target Pd **0.12** — the max-of-halves is dominated by the strong
  interferer, so it raises the threshold even further than CA.
* Slightly higher homogeneous loss than CA.
* In 2-D the leading/lagging split axis must be chosen (Doppler here, since the clutter ridge edges
  are traversed in velocity); a wrong choice silently reduces it to CA-like behaviour.

## DL (TinyCNN)
**Pros**
* **Best Pd in the low-Pfa regime** (0.784 @ 1e-4) — the operationally relevant region.
* **No analytic clutter model required** — it learns the background from data, so it is not tied to
  an assumed distribution.
* **Uses information CFAR must discard** (the guard region), as analysed above.
* **Extremely small**: 5,041 parameters, and **not converged** — trained on 30 % of the data for
  8 epochs, so this is a lower bound on achievable performance.
* Single learned rule, rather than needing to pick a specialist per environment.

**Cons**
* **No analytically guaranteed Pfa.** This is the most important limitation. CFAR guarantees a
  constant false-alarm rate *by construction*; a learned detector's Pfa can only be **measured**
  empirically, and may shift if the input distribution changes. For safety-critical radar this is a
  genuine theoretical disadvantage that no accuracy margin removes.
* **Requires labelled training data** — available here only because the data is synthetic.
* **Domain-shift risk**: performance is conditional on the test distribution resembling training.
* **Slower than CA-CFAR** (55.5 vs 6.9 ms/frame), though ~5× faster than OS.
* **Less interpretable** — no closed-form explanation of a given decision.

---

# 6. The complementarity argument (the project's central claim)

From the diagnostic scenarios (`docs/cfar_methods_review.md`):

| Scenario | Metric | CA | OS | GO |
|---|---|---|---|---|
| Clean (homogeneous) | Pd | 0.53 | 0.53 | 0.53 (tie) |
| Masking | weak-target Pd | 0.15 | **0.65** | 0.12 |
| Clutter edge | false alarms/frame ↓ | 14.2 | 17.1 | **8.4** |

**Each classical specialist wins its own regime and loses the other's.** OS wins masking and is
worst at clutter; GO wins clutter and is worst at masking; CA is the balanced, cheapest baseline.
**No single classical method wins everywhere** — which is precisely the gap a learned detector could
fill, since one trained model might handle all regimes without the practitioner selecting a variant
per environment.

**That hypothesis is not yet tested — see §7.**

---

# 7. What can and cannot be claimed

## Established
* The DL pipeline is correct and verified (all gates, including a measured 21 × 21 receptive field
  and an orientation check).
* On a **neutral mixed test set**, a 5 k-parameter CNN with CFAR-matched context **outperforms all
  three CFAR variants at low Pfa**, under a demonstrably fair protocol.
* The classical variants are **complementary**, with no universal winner.
* Runtime ordering: CA (6.9) < GO (38.4) < DL (55.5) < OS (296.7) ms/frame.

## NOT yet established (open gap)
* **Whether DL beats the specialists on their home turf.** The DL comparison was run only on
  `frozen_test` (the neutral mixture). It has **not** been evaluated on `masking_debug` or the
  strengthened `clutter_debug`. So the central claim — *"one learned model handles all regimes,
  unlike the complementary classical methods"* — **remains untested**. This is the single most
  valuable next experiment.
* **DL's converged performance.** The model saw 30 % of the training data for 8 epochs.
* **Whether more context helps** (Model 1, U-Net).
* **Sim-to-real transfer** — out of scope; future work.

---

# 8. Report-ready statements

> "Detection performance was compared at matched false-alarm rate on a held-out test set of 500
> frames never used for training or threshold tuning. All detectors were reduced to a common output
> form — a list of detected cells — using identical non-maximum suppression, an identical valid
> evaluation region, and a common matching and scoring implementation. The learned detector's
> receptive field was matched to the CFAR reference window so that the comparison isolates the
> decision rule rather than the extent of available context."

> "At a per-cell false-alarm probability of 10⁻⁴ the learned detector achieved Pd = 0.784 against
> 0.752 for the best classical variant, a gain of 3.2 percentage points; the methods converged at
> less stringent false-alarm rates. The modest margin is consistent with the deliberately matched
> receptive field."

> "The gain is attributed to a structural asymmetry: cell-averaging CFAR must exclude its guard
> region from the background estimate, since an arithmetic mean cannot separate a target's own
> mainlobe leakage from a genuine rise in the noise level. A learned function is not subject to that
> constraint and can therefore exploit information the classical estimator is required to discard."

> "The principal limitation of the learned approach is the loss of an analytically guaranteed
> constant false-alarm rate. CFAR provides this by construction, whereas a learned detector's
> false-alarm behaviour can only be established empirically and may not persist under distribution
> shift."
