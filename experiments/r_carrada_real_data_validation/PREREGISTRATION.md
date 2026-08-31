# R1 PRE-REGISTRATION — zero-shot transfer of the frozen detectors to CARRADA

**Experiment:** CARRADA Real-Data External Validation
**Folder:** `experiments/r_carrada_real_data_validation/`
**Written:** 31 August 2026, Phase 3
**R0 baseline commit:** `a3a3b88e4f26dcffa14d97e2fc8f4c03e08eff04`
**Status: FINAL PRE-SCORING FREEZE** — final rulings applied 31 Aug 2026 (§14), before any scoring.

> ## Status at the moment of writing
>
> **No detector has been run on CARRADA. No CARRADA label — train, validation or test — has been
> used to choose any quantity in this document.** No Pd, Pfa, recall, threshold, or match has been
> computed on real data.
>
> This file fixes every consequential R1 choice **before** results exist. Once frozen (§13) it is
> not edited. Any later departure goes in `DEVIATIONS.md` with its date, its reason, and an explicit
> statement of whether results had been seen.

---

## 1. The question

> **Do the already-frozen TinyCNN and classical CFAR detectors transfer zero-shot from the
> synthetic development domain to real CARRADA Range–Doppler data?**

This is an **external validation** experiment. It does **not** replace or revise the synthetic
benchmark, and it cannot: the synthetic study answers controlled questions about detector
performance and mechanism, under a truth definition CARRADA does not supply. R1 asks the separate
question of whether the frozen behaviour survives the domain shift that R0 measured.

**A poor R1 result would not retract any synthetic finding.** It would bound its external validity.

---

## 2. Prior information from R0 — established, not to be re-derived or re-interpreted

Carried in from the Phase 2 audit (`PROGRESS.md`, 31 Aug; notebook §4–§14):

| fact | value |
|---|---|
| RD geometry | identical, `(256, 64)`, no resampling |
| primary representation | `range_doppler_numpy`, power in dB up to a constant |
| CFAR input | `P ∝ 10^(D/10)` — constant cancels in a ratio test |
| TinyCNN adapter | `D − median(D)`, exact because dB is monotone in power |
| simulator background | single-look: k = 0.990 ± 0.006, std 5.573 dB, range-stationary |
| CARRADA background | smoother: k = 4.771 ± 0.223, std 2.212 dB |
| Doppler correlation | **lower** in CARRADA: 0.227 vs 0.574 |
| range correlation | comparable: 0.564 vs 0.573 |
| zero-Doppler ridge | CARRADA 24.0 dB, half-width ±1; simulator 14.8 dB, ±4 |
| range non-stationarity | CARRADA k varies 2.9–7.5 across range blocks; simulator flat |
| unannotated reflectors | 5.85% of frames carry bright returns outside every annotation |

The last row is why §6 forbids the word *Pfa* anywhere in R1.

---

## 3. Data and split

**Official CARRADA sequence-level split only. No frame-level split, ever** — with 30 sequences and
77 tracks, a frame split would place the same physical object on both sides.

| split | sequences | frames | annotated | object-frames |
|---|---|---|---|---|
| Train | 18 | 8,088 | 4,319 | 5,318 |
| Validation | 6 | 2,448 | 1,483 | 1,804 |
| **Test — the R1 evaluation set** | **6** | **2,130** | **1,391** | **1,628** |

**Primary R1 evaluation uses the official Test sequences only.**

CARRADA **Train and Validation** sequences may be used **only** for the implementation sanity checks
enumerated in §11, all of which are defined here, before any test scoring.

**CARRADA test labels are never used to** design a metric, choose a threshold, choose NMS, alter
preprocessing, choose CFAR alpha, or choose the truth representation. Every one of those is fixed
below.

---

## 4. Ground truth — dense object support

**Primary truth is the CARRADA dense segmentation mask.**

R0 measured that CARRADA objects are extended RD structures (median car support 18 × 13 bins, 147
cells), and that the `box` field is **not** the bounding box of the dense mask — it is strictly
smaller in 8,750 of 8,750 instances, so it bounds the sparse annotation instead.

**No point target is invented.** The following are explicitly **rejected as primary truth**:
centroid, box centre, brightest cell in mask, sparse-point centroid. Each would be a construct of
this project rather than an annotation CARRADA supplies, and the choice among them would be
unfalsifiable.

The detection question is therefore:

> **Did the detector place a peak inside the annotated object support?**

Sparse and box representations may appear **only** as the secondary sensitivity analysis named in
§9.4. They may never replace the dense mask, and specifically may not replace it after results are
seen.

---

## 5. Eligible region, eligible objects, and matching

### 5.1 Common detector-valid region — unchanged from the frozen convention

Both arms are scored on exactly the region the frozen benchmark uses
(`configs/frozen_dl.yaml`: *"CFAR-testable interior only, margin 10 range / 4 Doppler"*):

```
valid_region = rows 10 … 245  ×  columns 4 … 59      (236 × 56 = 13,216 cells)
```

This is **not** modified to accommodate CARRADA.

### 5.2 Eligible objects

For each annotated object:

```
valid_support = dense_mask ∩ valid_region
eligible      = |valid_support| ≥ 1
```

**To be recorded and reported**: total objects, eligible objects, and objects excluded because no
support cell intersects the valid region.

A **secondary** stricter variant — object mask *entirely* inside the valid region — is registered
here as a sensitivity check. It is **not** primary.

### 5.3 One-to-one matching

One prediction may not detect two objects. The same assignment code runs for both arms; no
detector-specific rule exists.

Eligible edge: peak `p` → object `o` **iff** `p ∈ valid_support(o)`.

Objective, applied in strict order:

1. **maximise the number of matched objects** (maximum-cardinality bipartite matching);
2. tie-break: prefer the assignment with greater total detector score;
3. tie-break: deterministic ordering by `(range_bin, doppler_bin, object_id)`.

**Unit tests, to be written and passing before any test scoring** (§11): one object/one peak;
one object/many peaks; two objects sharing one candidate peak; two objects/two peaks; overlapping
masks; no peaks; empty frame.

---

## 6. False alarms — terminology is fixed here

**Synthetic-style Pfa is not reported on CARRADA, in any form.**

R0 established that 5.85% of frames contain bright returns belonging to no annotation, and only the
77 tracked objects are annotated at all. A peak outside every mask may be infrastructure, static
clutter, multipath, another physical reflector, an unannotated object — or a genuine false alarm.
**Detection outside a labelled mask is not a known false alarm.**

The burden metric is:

> **unmatched detections per frame**

Reported as mean, median, p90, and per sequence. It is **never** renamed `Pfa`, `false-positive
rate`, `false alarm rate`, or `FPR`, in the notebook, the tables, the figures, or the report.

**Secondary:** unmatched detections per **frame with no CARRADA tracked-object annotation**. These
frames are described that way in full. They are **not** called "true background-only frames",
because unannotated real reflectors may be present in them too.

---

## 7. Detectors — both frozen

### 7.1 TinyCNN

Frozen benchmark set, unchanged: `results/dl/tinycnn_frozen_s{0..7}_ep021.pt`, TinyCNN, 5,041
params, RF 21×9, epoch 21, `threads=2`.

**Prohibited:** retraining, fine-tuning, architecture change, label change, BatchNorm adaptation,
CARRADA-specific learned normalization, and any modification to `src/dl/`. All 8 seeds are evaluated
**independently**; seeds are **not** ensembled, because the frozen benchmark does not ensemble them.

**Input adapter** (experiment-local; global preprocessing source is not touched):

```
relative_db = D − median(D)                 # D = range_doppler_numpy, already power in dB
relative_db = clip(relative_db, −10, +60)
x           = relative_db / 20
```

This is algebraically identical to the frozen `10·log10(P / median P)` pipeline because dB is a
monotone transform of power, so `median(P) = 10^(median(D)/10)` exactly. **§11 requires this to be
verified numerically against `src/dl/dataset.normalize_rd` before any CARRADA inference.**

### 7.2 CFAR

Frozen: 21×9 window, guard 2/2, train 8/2, `os_rank_frac` 0.75, `go_axis` 1, and the calibrated
alphas in `configs/cfar_alpha_table.json`. Source, geometry, statistic and alpha are unchanged.

Input `P' = 10^((D − c)/10)` for any constant `c` is permitted, since CFAR is a ratio test and the
constant cancels. **§11 requires scale invariance to be unit-tested before scoring.**

**All four variants — CA, GO, OS, SO — are run and reported.** The frozen benchmark's "best CFAR
per condition" selection **cannot transfer**: it needs labels to pick the winner, which would be an
oracle on real data.

**Primary CFAR arm: CA.** *(Final ruling, 31 Aug 2026, before any scoring.)*

Rationale:

* CA is the **simplest and canonical** classical baseline;
* it **preserves continuity** with the main synthetic benchmark;
* and critically, it **avoids selecting the primary comparator because R0 hinted another variant
  might suit CARRADA better.** Choosing OS on the strength of an R0 observation would import a
  data-driven preference into what is meant to be a strict zero-shot transfer.

**GO, SO and OS are all run and reported as secondary.** **OS is highlighted as the
robust/heterogeneous-background secondary comparison** — R0's non-stationary background, narrow
24 dB ridge and bright unannotated reflectors are exactly what an order-statistic estimator exists
to tolerate — but **OS must not be promoted to primary after results are seen.** If OS outperforms
CA on CARRADA, that is reported as a secondary finding about estimator choice under heterogeneous
clutter, not as the headline comparison.

*(This supersedes an earlier draft that named OS primary. The change was made before any CARRADA
detector was run and before any R1 number existed.)*

### 7.3 NMS

Frozen `all_peaks`, `nms_size = 5` (11×11), shared by both arms — the same call, by construction.
**Not retuned on CARRADA; no CARRADA-specific suppression radius.**

> ### Registered limitation — extended objects under a point-target NMS
>
> The 11×11 suppression window was sized for a synthetic **point** target whose mainlobe is ~2 bins.
> A CARRADA car occupies a median 18 × 13 bins, which is **larger than the suppression window**, so
> a single real object can legitimately produce **several surviving post-NMS peaks inside its own
> annotated support**.
>
> Consequences, registered before scoring:
>
> * **Object recall remains valid.** One-to-one matching (§5.3) assigns exactly one peak per object,
>   so extra peaks cannot inflate recall.
> * **Unmatched-detection burden is inflated for both arms.** Under §5.3 the surplus peaks inside an
>   object's own mask become *unmatched*. Therefore **unmatched detection burden ≠ pure detector
>   noise** on CARRADA: part of it is a metric-transfer artefact of applying a point-target NMS to
>   extended truth.
> * It affects both arms through the same shared call, so it is **common-mode in the TinyCNN-vs-CFAR
>   difference**, but it is **not** common-mode in the absolute burden numbers.
>
> Phase 4 quantifies how often this actually happens, on **non-test** frames only, as a
> metric-behaviour check. **NMS is not altered in response to what that check finds.**

---

## 8. Operating points

### 8.1 TinyCNN threshold — a new protocol choice, declared as such

**The frozen project has no global TinyCNN threshold.** `configs/frozen_dl.yaml` specifies
*"matched MEASURED Pfa, swept per condition"* — a per-condition operating point derived from the
ROC of the split being scored. That is an evaluation oracle (already recorded as a limitation in
`PLAN.md` ⚠️1 and `evaluation_metrics_overview` §7.2), and it **cannot transfer**: it needs labels
on the scored split, and CARRADA cannot supply an equivalent Pfa at all (§6).

A single global threshold must therefore be defined. **This is a new protocol choice, not a
transcription of the frozen one**, and it is declared here rather than buried:

> **Rule.** One threshold **per seed**, derived on **`bench2_val` only** (synthetic validation),
> **pooled across all 8 scenario conditions**, chosen as the lowest threshold whose pooled
> **measured per-cell Pfa ≤ 1×10⁻⁴**, at the pre-registered ±1 gate, using the frozen `all_peaks`
> extraction and the frozen valid region.

Per-seed rather than one shared value, because each seed's logit scale differs slightly and the
frozen benchmark treats seeds as independent arms; this is the direct analogue of CFAR's fixed
design-Pfa alpha.

**No CARRADA data of any split enters this derivation.** The resulting eight thresholds, the script,
and the SHA-256 of its output are recorded in the implementation log **before** any CARRADA
inference is run.

**Consequence, stated in advance:** R1 numbers are produced under a *different operating-point
protocol* from the synthetic headline, so they are **not poolable** with the synthetic tables and are
reported as a separate protocol.

### 8.2 CFAR operating point — strict zero-shot

**Primary: frozen alpha, unchanged.** This tests whether the entire existing detector configuration
transfers without adaptation.

**Interpretation fixed in advance:** R0 established that these alphas were calibrated under
single-look exponential statistics while CARRADA's background sits near k ≈ 4.8. The same alpha will
therefore produce a different achieved operating point. **If CFAR recall is poor, that does not
establish that CFAR is intrinsically inferior** — it may show only that its operating point moved.
This sentence is registered now so it cannot be softened later.

**Real-domain recalibration is NOT performed in R1.** It would require a defensible calibration rule,
and no such rule exists while CARRADA's annotations cannot supply a true Pfa (§6). A calibrated-CFAR
arm may be added later **only** if such a rule can be stated without pretending incomplete
annotations provide false-alarm ground truth. **It is not invented merely to improve CFAR's numbers.**

---

## 9. Metrics

### 9.1 Primary — mask-based object recall

For each **eligible** labelled object:

* **detected** — at least one peak is *assigned to it* by §5.3 and lies inside its `valid_support`;
* **missed** — otherwise.

```
object recall = matched eligible objects / total eligible objects
```

Reported overall and per class. **Identical definition for both arms.**

> ⚠️ **Registered wording.** Synthetic Pd and CARRADA mask recall measure related detection
> capability under **different truth definitions**. The two numbers are **never** placed in the same
> column, differenced, or described as the same quantity.

### 9.2 Tolerance / robustness — dilated mask recall

Recall recomputed with the dense mask geometrically dilated, identically for both arms:

| variant | status |
|---|---|
| exact dense mask | **primary headline** |
| dense mask dilated ±1 bin | secondary |
| dense mask dilated ±3 bins | secondary |

This answers *"how often is a miss only just outside the annotated support?"* without inventing a
point centre. The undilated result stays the headline.

### 9.3 Support-distance — **deliberately excluded from confirmatory metrics**

Distance-to-nearest-mask-cell is **not** reported as a localization metric. For any successful
detection it is **identically zero by construction**, so it cannot discriminate among successes; and
a nearest-of-many rule would reward whichever detector emits more peaks. §9.2's dilation ladder
serves the same purpose deterministically. **This is decided now, not after results.**

**No localization metric comparable to the synthetic one is reported at all**, because CARRADA
supplies no point truth. That is a limitation of the transfer, not an omission.

### 9.4 Registered secondary analyses

| # | analysis | definition |
|---|---|---|
| S1 | per class | car / pedestrian / cyclist, dense-mask truth |
| S2 | range stratification | the **exact** R0 blocks, unchanged: 10–60, 60–110, 110–160, 160–210, 210–246. **An object spanning more than one block is assigned by the median range index of its `valid_support`** — an annotation-only rule; the detector's prediction never decides an object's stratum |
| S3 | zero-Doppler stratification | **near-zero**: `valid_support` intersects the R0 common band, Doppler bins 28–36 (±4). **away**: it does not. The ridge is **not** removed from the primary analysis — the frozen detectors see the real map unchanged |
| S4 | strict eligibility | object mask entirely inside the valid region |
| S5 | sparse / box truth | sensitivity only; never replaces dense |
| S6 | recall–unmatched-burden curve | object recall vs unmatched detections per frame, swept across detector score. **Full curve shown**; no test-set threshold is selected from it. The x-axis is **not** Pfa and the curve is **not** an ROC |
| S7 | empty-frame burden | unmatched detections per frame with no tracked-object annotation |

S6 is included only if the frozen score outputs support it cleanly; **no detector is modified to
enable it.** If they do not, it is dropped and the omission recorded.

> ⚠️ **Registered caveat on per-class recall.** Class masks differ greatly in size (median support:
> car 18×13, cyclist 11×11, pedestrian 9×9). A larger mask is a larger area for a peak to land
> inside, so **class differences must not be read as detector sensitivity alone.** The notebook must
> state this wherever per-class recall appears.

---

## 10. Statistical plan

**The test split has 6 sequences. That is the effective sample size**, and the analysis says so
plainly rather than borrowing power from correlated frames.

**Prohibited:** bootstrapping individual RD cells; treating the 1,628 test object-frames as
independent; any interval whose n is a frame or object count.

**Unit of analysis: the test sequence (n = 6).**

```
for each test sequence s:
    for each seed i in 0..7:  recall_TinyCNN(s, i)
    recall_TinyCNN(s) = mean over seeds
    recall_CFAR(s)    = recall of the primary CFAR arm
    delta(s)          = recall_TinyCNN(s) − recall_CFAR(s)
```

**Primary inference: paired bootstrap over the 6 test sequences**, resampling sequences with
replacement, 10,000 replicates, reporting the mean paired difference and a percentile CI.

**Seed variability is reported separately** — per-seed recall, and the between-seed spread — and is
**never combined with sequence variability into a single standard error.** They answer different
questions and pooling them would misstate both.

**With n = 6, emphasis is on effect sizes, confidence intervals and the per-sequence table shown
directly.** p-values are de-emphasised; no significance claim rests on n = 6 alone. Class-specific
results report object counts and contributing sequence counts, and are labelled **exploratory**
where n is small.

---

## 11. Implementation sanity checks — all before any test scoring

| # | check | pass condition |
|---|---|---|
| C1 | adapter equivalence | experiment adapter vs `normalize_rd` on synthetic frames: max abs diff < 1e-9 |
| C2 | CFAR scale invariance | CFAR output on `P` and on `P·10^k` identical for several `k` |
| C3 | frozen-model integrity | 8 checkpoints hash-match `isolation_manifest.json` |
| C4 | matching unit tests | all seven cases in §5.3 pass |
| C5 | dense-mask reconstruction | masks rebuilt from JSON match the R0 verification |
| C6 | valid-region cell count | exactly 13,216 |
| C7 | TinyCNN threshold derivation | runs on `bench2_val` only; eight thresholds recorded with the script's output hash |
| C8 | end-to-end smoke test | full pipeline on **CARRADA Train sequences only**, confirming it runs and produces plausible output |

**C8 uses CARRADA Train sequences, never Validation or Test.** Its numbers are diagnostic only and
are **not reported as results**.

---

## 12. Result tables — structure fixed now, values not populated in Phase 3

**Table 1 — primary**

| metric | CFAR (OS) | TinyCNN (8-seed mean) | difference [CI] |
|---|---|---|---|
| overall dense-mask recall | | | |
| car recall | | | |
| pedestrian recall | | | |
| cyclist recall | | | |
| mask ±1 recall | | | |
| mask ±3 recall | | | |
| unmatched detections / frame | | | |
| unmatched detections / unannotated frame | | | |

**Table 2** — per test sequence (6 rows). **Table 3** — TinyCNN per seed (8 rows).
**Table 4** — CA / GO / OS / SO side by side. **Table 5** — near-zero vs away-from-zero Doppler.
**Table 6** — range-stratified, R0 blocks.

### Qualitative figures — rules fixed now

Test frames for display are chosen by a **deterministic seeded draw**, and the **same frame** is
shown for both methods, with the dense mask overlaid together with TinyCNN and CFAR peaks, including
both successes and misses. Cherry-picking frames favourable to either arm is prohibited.

Any deliberately chosen failure case is labelled **"selected diagnostic example"** — never
"representative example".

---

## 13. Freeze record

SHA-256, timestamp, and the git commit are recorded in `results/preregistration_freeze.json`, in the
experiment `README.md`, and in the notebook. After the freeze this file is not edited; departures go
to `DEVIATIONS.md` with date, reason, whether results had been seen, and scientific consequence.

---

## 14. Final rulings — applied 31 August 2026, before any scoring

The two items flagged at Phase 3 review are now **decided and incorporated above**. Both decisions
were taken with **no CARRADA detector output in existence**.

| # | item | ruling |
|---|---|---|
| 1 | global TinyCNN threshold (§8.1) | **Approved as proposed** — one threshold per frozen seed, from `bench2_val` only, pooled over all 8 synthetic conditions, lowest threshold with pooled measured per-cell Pfa ≤ 1e-4, frozen extraction / NMS 5 / valid region. Declared a **new zero-shot operating protocol**, never mixed numerically with the per-condition matched-Pfa synthetic benchmark |
| 2 | primary CFAR arm (§7.2) | **Changed from OS to CA.** CA is the canonical baseline and preserves continuity with the main benchmark; it avoids picking the primary comparator because R0 suggested another variant might suit CARRADA. GO/SO/OS reported as secondary, **OS highlighted as the robust heterogeneous-background comparison but never promoted to primary after results** |
| 3 | NMS (§7.3) | **Frozen at radius 5**, not retuned. Extended-object multi-peak limitation registered in §7.3: **unmatched detection burden ≠ pure detector noise** |
| 4 | range-stratum assignment (§9.4 S2) | Objects spanning blocks are assigned by the **median range index of `valid_support`**. Recorded here as a pre-result methodological clarification — it was not explicit in the first draft, and it is annotation-only, so no detector output can influence it |

### FINAL PRE-SCORING FREEZE

This is the version frozen for scoring. Its SHA-256 is recorded in
`results/preregistration_freeze.json` and in the notebook. **At the moment of this freeze, no
CARRADA detector inference of any kind had been run and no R1 number existed.**

After this point the file is not edited. Departures go to `DEVIATIONS.md` with date, reason, whether
results had been seen, and scientific consequence.
