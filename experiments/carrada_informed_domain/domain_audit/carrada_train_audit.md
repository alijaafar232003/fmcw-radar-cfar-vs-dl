# CARRADA-train domain audit, and the gap to the original simulator

**Frozen audit snapshot — written before any change to the generator.** Phases 2 and 3 of the
CARRADA-informed domain intervention. Nothing in this document depends on CARRADA Validation or
CARRADA Test, neither of which was opened.

## What was measured, and how

One script, `scripts/audit_domain.py`, run three times with the domain as an argument. The
background-selection protocol is **imported**, not reimplemented, from
`experiments/r_carrada_real_data_validation/src/r0_background.py` — the module Chapter 7's R0 audit
used — so these numbers are commensurable with the published domain audit:

* annotated / labelled object support, **dilated by 3 bins**, excluded;
* zero-Doppler band of **half-width 4** excluded for the ordinary background statistics (R0's
  "common applied" band), and characterised separately from the un-excluded ridge profile;
* outer **10 range / 4 Doppler** bins excluded — the interior a 21×9 CFAR window can test;
* **no percentile trimming, ever.**

| domain | source | frames | groups |
|---|---|---|---|
| CARRADA train | 18 official Train sequences | 8,088 | 18 sequences |
| original simulator | `bench2_train` (the split the frozen TinyCNN trained on) | 8,400 | 7 conditions |
| original simulator | `bench2_val` (cross-check against published R0) | 1,600 | 8 conditions |

No detector ran: no TinyCNN, no CFAR, no threshold, no recall.

### Data-discipline guard

`--domain carrada_train` routes every CARRADA read through a path guard holding the 18 Train
sequence names. It raises `PermissionError` on any other sequence, and the sequences actually opened
are written into `carrada_train_stats.json` (`groups_opened`) so the claim is checkable rather than
asserted. The Validation and Test lists are read only to assert they do **not** intersect Train.

---

## 1. The headline gap

Medians over frames. The simulator distributions are needles, so median and mean agree there; for
CARRADA the median is the honest summary because a minority of frames carry bright structure that
drags the moment estimators (see §3).

| property | CARRADA train | original simulator | gap |
|---|---|---|---|
| Gamma shape $k$ (moment) | **3.97** | **0.99** | simulator 4.0× too spiky |
| coefficient of variation | **0.502** | **1.005** | simulator 2.0× too variable |
| background spread [dB] | **2.23** | **5.58** | simulator 2.5× too wide |
| lag-1 **range** correlation | 0.557 | 0.573 | **already close** |
| lag-1 **Doppler** correlation | **0.228** | **0.576** | simulator 2.5× too correlated |
| zero-Doppler ridge, peak over floor | **23.3 dB** | 17.2 dB (clutter strata) / 0.7 dB (rest) | see §4 |
| ridge half-power width | **1.0 bin** | 7.0 bins (clutter strata) | simulator 7× too wide |
| frames with a ridge > 10 dB | **99.0 %** | **39.2 %** | structural |
| range mean-power span | **2.5 dB** falling | 0.04 dB (flat) | simulator has none |

The three distributions in `plots/fig1_distributions.png` for $k$, CV and dB spread are **almost
completely non-overlapping**. This is not a small calibration offset.

## 2. The single mechanism behind the first three rows

Those three rows are one property, not three. The simulator's measured background spread is
**5.569 dB**; the closed-form value for single-look exponential power is **5.570 dB**. The original
simulator is, to three decimals, a textbook **one-look** detector: thermal AWGN through the
processing chain gives Rayleigh amplitude, exponential power, $k = 1$, CV $= 1$.

Converting the measured dB spread to an equivalent look number $L$ via
$\sigma_{\rm dB} = (10/\ln 10)\sqrt{\psi'(L)}$:

| range block | CARRADA dB spread | $L$ from dB spread | $k$ from moments | agree? |
|---|---|---|---|---|
| 10–60 | 1.655 | **7.37** | 7.13 | yes |
| 60–110 | 1.668 | **7.27** | 2.69 | **no — see §3** |
| 110–160 | 1.609 | **7.77** | 7.70 | yes |
| 160–210 | 1.610 | **7.77** | 7.62 | yes |
| 210–246 | 3.368 | **2.12** | 2.87 | yes |
| *whole frame* | 2.227 | 4.28 | 3.97 | — |
| **simulator, any block** | **5.569** | **1.00** | 0.99 | yes |

So: **CARRADA's local background behaves like ≈7.5 incoherent looks over most of the range and
≈2 looks in the far block**, while the simulator is 1 look everywhere. The frame-level figure of
≈4.3 looks is not a third value — it is what you measure when you pool blocks whose mean levels
differ by 2.5 dB, and it is the number to quote only for whole-frame comparisons.

Seven to eight looks is consistent with CARRADA's four receive antennas plus further non-coherent
integration in the published `range_doppler_numpy` product. This audit does not need that
explanation to be right; the measurement stands on its own.

## 3. One anomaly worth recording

Block **60–110** is the only place the two estimators disagree, and they disagree hard: the dB
spread implies 7.3 looks, the moment estimator says 2.69. dB spread is a log-domain quantity and is
insensitive to a handful of extreme cells; the moment estimator is not. The range profile
(`plots/fig2_profiles.png`, panel B) shows a **persistent narrow spike near range bin 100** in the
sequence-averaged background.

The reading is that block 60–110's speckle is as smooth as its neighbours', and its low moment-$k$
reflects **fixed bright structure at a particular range** — a stationary scatterer or a sensor
artefact — not spikier clutter. Two consequences:

* **dB spread is the robust descriptor** and is what the generator should be matched on;
  moment-$k$ should be reported beside it, not instead of it.
* Fixed range-localised structure is a real property of the target domain that a stationary
  background model cannot produce. It is **not** in scope for this intervention (see §7).

## 4. The zero-Doppler ridge is structurally different, not just mis-scaled

| | CARRADA train | original simulator |
|---|---|---|
| peak over floor | 23.3 dB (p5–p95: **19.4–26.4**) | 17.2 dB on clutter strata, 0.7 dB elsewhere |
| bins above 3 dB | 1.8 | 9.3 (clutter strata) |
| half-power width | **1.0 bin** | 7.0 bins (clutter strata) |
| present (>10 dB) in | **99.0 % of frames** | **39.2 % of frames** |

Two separate differences:

1. **Shape.** CARRADA's ridge is a sharp spike: 23 dB at zero Doppler, ~3 dB by ±1, under 1 dB by
   ±3. The simulator's is a **flat-topped plateau** spanning ±3–4 bins with steep shoulders — a
   direct consequence of `add_near_zero_doppler_clutter` band-limiting the slow-time spectrum to
   `±width_bins` and then renormalising per range bin.
2. **Prevalence.** Every CARRADA frame has a ridge. In the original training split only three of
   seven strata enable clutter at all (`clutter`, `clutter_edge`, `mixed`), so 61 % of training
   frames have **no ridge whatsoever**. The frozen TinyCNN therefore spent most of its training
   seeing a structure-free background that does not occur in the real domain.

Prevalence may matter more than shape here, and it is the cheaper of the two to fix.

## 5. Range non-stationarity

CARRADA falls **≈2.5 dB** from near range to ≈bin 220, then drops sharply at the outer edge, and its
far block (210–246) is a **different statistical regime**: 2.1 looks instead of 7.5, lag-1 range
correlation 0.75 instead of 0.45, lag-1 Doppler correlation 0.56 instead of 0.02.

The original simulator is **flat to 0.04 dB** and statistically identical in every block
(`plots/fig3_range_blocks.png`: four straight lines). It has no range non-stationarity of any kind
in its background — the radar equation is applied to *targets* only.

This matters for the follow-up because Chapter 7's largest TinyCNN deficit was in the **nearest**
block, 10–60 (TinyCNN 0.8455 vs CA 0.9475).

## 6. Sequence-to-sequence variability: match distributions, not constants

`plots/fig4_group_variability.png`. CARRADA sequences genuinely differ:

| statistic | CARRADA per-sequence range | simulator per-condition range |
|---|---|---|
| Gamma $k$ | 2.90 – 5.10 | 0.85 – 1.00 |
| dB spread | 2.18 – 2.46 | 5.568 – 5.590 |
| lag-1 Doppler | 0.232 – 0.388 | 0.543 – 0.580 |
| ridge peak | 19.4 – 26.4 dB (p5–p95 over frames) | — |

The new generator must therefore **sample** these parameters per frame from train-derived ranges.
Fixing them at the CARRADA means would produce a synthetic domain with the right centre and none of
the spread — brittle, and not what the pre-analysis plan asks for.

---

## 7. What I propose to change, and what the existing simulator can do

### Reproducible with existing components

| target property | mechanism | existing knob |
|---|---|---|
| ridge **strength** 19–26 dB | sample `cnr_db` per frame from a train-derived range | `ClutterConfig.cnr_db` — exists |
| ridge **prevalence** 99 % | enable clutter in **every** scenario, not 3 of 7 | scenario YAML — exists |
| ridge **width** ≈1 bin | reduce `width_bins` from 3 toward 0–1 | `ClutterConfig.width_bins` — exists |
| clutter tail shape | `weibull` / `k` texture, shape sampled | `ClutterConfig.distribution`, `.shape` — exists |

### Requires new generator code (none of it touches the original simulator)

| target property | proposed mechanism | why new |
|---|---|---|
| **≈7.5 looks** (the dominant gap) | non-coherent averaging of $L$ independent RD realisations per frame, $L$ sampled per frame | no look/averaging knob exists anywhere in `src/radar/` |
| **far-range ≈2 looks** | make $L$ range-dependent: high in blocks 10–210, low in 210–246 | same |
| **2.5 dB range slope + edge roll-off** | range-dependent background gain profile applied to the noise floor | the radar equation is applied to targets only; the noise floor is flat by construction |

The multi-look change is the one to get right, because it alone accounts for the $k$, CV and dB-spread
rows — three of the four largest discrepancies. I will verify in the pilot that it does **not**
disturb the cell-SNR definition: non-coherent averaging preserves mean target and mean noise power,
so cell SNR should be unchanged in expectation, but that is a claim to measure rather than assume,
since the target SNR sampling methodology must stay fixed.

### Cannot be represented cleanly — and I am not going to fake it

**Lag-1 Doppler correlation (0.576 → 0.228).** The simulator's Doppler correlation is set by the
slow-time window in `src/radar/rd_processing.py`. That same window defines the target point-spread
function, and the frozen specification requires the **target PSF model to stay fixed**. Averaging
independent looks reduces variance but leaves the correlation coefficient untouched, so there is no
way to reach CARRADA's Doppler decorrelation without changing the window — which would change the
PSF, the measured mainlobe width, and hence the guard-cell geometry that makes the CFAR comparison
fair.

I therefore propose to **leave Doppler correlation unmatched and declare it**, rather than change
the window. It becomes a named residual of the intervention: the experiment tests whether matching
the *amplitude statistics, ridge structure and range non-stationarity* improves transfer, with
Doppler correlation explicitly outside the intervention. Matching it would require re-deriving the
PSF, the guard geometry and the receptive-field equality, i.e. a different experiment.

**Fixed range-localised bright structure** (§3). A stationary stochastic background cannot produce a
persistent scatterer at a fixed range. Out of scope, and noted as future work alongside extended
target morphology.

**Absolute power scale.** CARRADA's `range_doppler_numpy` is power in dB up to an unknown constant.
Only shape, spread and correlation transfer; absolute levels are not comparable and are not matched.
The frozen input transform normalises by the frame median anyway, so this is harmless.

---

## Artefacts

```
domain_audit/
  carrada_train_stats.json        full summary: background, ridge, range blocks, per sequence
  carrada_train_stats.csv         compact one-row-per-statistic summary
  carrada_train_frames.csv        8,088 rows, one per Train frame
  carrada_train_ridge.csv         per-frame ridge metrics
  carrada_train_blocks.csv        per-frame per-range-block statistics
  sim_bench2_train_*.csv/json     same, original simulator training split
  sim_bench2_val_*.csv/json       same, original simulator validation split
  domain_gap.csv / domain_gap.md  side-by-side table with gaps and ratios
  plots/fig1_distributions.png    per-frame distributions, all five statistics
  plots/fig2_profiles.png         ridge profile and range profile
  plots/fig3_range_blocks.png     statistics by frozen range block
  plots/fig4_group_variability.png per-sequence / per-condition spread
```

## Cross-check against the published R0 audit

R0 reported, over all 30 sequences and all splits, CARRADA background CV mean 0.4734, Gamma $k$
(moment) mean 4.771, dB spread mean 2.2115, lag-1 range 0.5635, lag-1 Doppler 0.2265, and a ridge
peak over floor of 24.02 dB. This audit, restricted to the 18 Train sequences, gives frame-median
CV 0.502, $k$ 3.97, dB spread 2.227, lag-1 range 0.557, lag-1 Doppler 0.228, ridge 23.3 dB. The two
agree wherever they measure the same thing; the differences are the restriction to Train and the
use of medians. **Chapter 7's domain audit is not modified, re-run, or superseded by this document.**
