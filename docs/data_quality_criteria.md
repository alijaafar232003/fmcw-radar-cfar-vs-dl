# Is My Synthetic Data Actually Useful?

A framework for judging whether a simulated radar dataset supports conclusions that mean
something in the real world — rather than being a self-fulfilling toy. Written to be defended
in a viva and reused in the report.

> **The question an examiner will ask:** *"How do you know your synthetic data isn't just
> rigged to give you the answer you wanted?"*
> **The answer must be evidence, not assertion.** This document defines the evidence.

---

## Why this matters (and why synthetic is still legitimate)

Synthetic data is not a compromise here — it is a *requirement*. Fair Pd/Pfa scoring needs
exact per-target ground truth in `(range_bin, Doppler_bin)`, which real radar recordings almost
never provide (their labels come from camera/lidar fusion as boxes, not bin-level truth).

The precedent is established: **RadSimReal** (CVPR 2024) showed that detectors trained purely on
physically-simulated radar reach performance comparable to models trained on real data, and
*generalise better across datasets*. They quantified realism with FID: 6.54 (synthetic vs real
RADDet) versus 6.76 (real RADDet train vs its own test set) — i.e. their synthetic data was
about as close to the real training set as the real test set was.

So the standard is not *"is it real?"* but ***"is it faithful in the dimensions that determine
my conclusions?"***

---

## The nine criteria

### 1. Physical fidelity
Data must come from the actual signal-generation physics, not from drawing blobs on an image.

*Check:* every step traceable to radar theory (chirp → delay → dechirp → range/Doppler FFT);
derived quantities (resolution, max range/velocity) match the closed-form equations.

### 2. Label correctness
The ground truth must actually correspond to the signal. A subtle indexing/transpose bug makes
every downstream number meaningless while everything still "looks fine".

*Check:* verify the RD peak of each target coincides with its stored label bin and mask, across
all scenarios, within one resolution cell.

### 3. Statistical realism
The *distributions* — noise, clutter, target amplitude — should match what real sensors see.
This is the criterion most often skipped, and the one that most affects whether conclusions
transfer.

Known real-world facts to compare against:
- **Receiver noise**: complex Gaussian → exponential cell power after square-law. ✔ standard.
- **Ground clutter**: at low grazing angles it is **heavy-tailed** — Weibull, K, or log-normal,
  *not* Gaussian. High-resolution radar sees compound-Gaussian behaviour. A Gaussian clutter
  model **understates** the false-alarm problem, which is exactly the regime where CA-CFAR fails
  and OS/DL should win.
- **Target RCS**: fluctuates frame to frame (Swerling models), rather than being constant.
- **Amplitude vs range**: received power falls as ~1/R⁴ (radar equation), so far targets are
  systematically weaker.

### 4. Difficulty calibration
The data must span easy → hard. If every target is trivially detectable, all methods score ~1.0
and the benchmark is uninformative; if all are impossible, everything scores 0.

*Check:* the Pd-vs-SNR curve must actually transition from ≈0 to ≈1 inside the swept range.

### 5. Coverage and diversity
The dataset should span the operating conditions the conclusions claim to cover: target counts
(including zero), ranges, velocities, amplitudes, and the interference types of interest.

*Check:* histogram each factor; look for gaps, mode collapse, or accidental correlations.

### 6. No exploitable shortcuts
Synthetic pipelines often leak cues a model can exploit but that don't exist in reality — e.g.
targets always exactly on bin centres, a fixed number of targets, positions correlated with the
random seed, or per-frame normalisation that encodes the answer.

*Check:* continuous (off-grid) target placement; randomised counts/positions; nothing in the
input that trivially predicts the label.

### 7. Benchmark neutrality
Deliberately adversarial scenarios are good *science* (they isolate a mechanism) but must not
form the headline benchmark, or the comparison becomes rigged.

*Check:* separate **diagnostic** sets (built to expose a specific failure) from the **frozen
test set** (a neutral mixture used for the headline numbers). Report both, labelled.

### 8. External validity (the sim-to-real gap)
Would the ranking of methods still hold on real data? Known systematic gaps: point density,
noise statistics, extended (non-point) targets, sensor non-idealities, and multipath complexity.

*Check:* compare your dataset's parameters and statistics against published real datasets; state
the gaps explicitly as limitations rather than pretending they don't exist.

### 9. Reproducibility
Anyone must be able to regenerate the exact dataset.

*Check:* fixed seeds, versioned configs stored with the data, documented generation procedure.

---

## How practitioners actually verify these

| Method | What it tells you | Cost |
|---|---|---|
| **Physics/unit checks** | derived quantities match theory | trivial |
| **Label-alignment audit** | ground truth matches the signal | cheap |
| **Distribution fitting** (e.g. fit clutter to Rayleigh/Weibull/K, goodness-of-fit) | statistical realism | cheap |
| **Side-by-side property table vs real datasets** | external validity | cheap |
| **Ablation/difficulty sweeps** | difficulty is calibrated | cheap |
| **FID / distribution distance vs real data** | quantitative realism (RadSimReal used this) | needs real data |
| **Train-on-synthetic → test-on-real** | the gold standard for transfer | needs labelled real data |

The last two need real recordings; the first five do not. **A project that does the first five
and honestly documents the gaps is defensible.** One that does none is not.

---

## Applying this: the honest position for this project

Say this, in this order:

1. Synthetic data is **necessary** here (bin-level labels are required for Pd/Pfa; real datasets
   don't provide them).
2. It is **physically generated** and the labels are **verified**, not assumed.
3. Difficulty is **calibrated** so the benchmark is informative.
4. Its properties sit in the **same regime as published real datasets** (comparison table).
5. The **remaining gaps are named** (e.g. Gaussian rather than heavy-tailed clutter, point rather
   than extended targets) and their likely effect on conclusions is stated.
6. Conclusions are therefore claimed **for this operating regime**, with sim-to-real transfer as
   future work.

That is a defensible scientific position. Claiming "my data is realistic" without 4 and 5 is not.

---

## Audit outcome and decisions (this project)

**Re-audited 11 August 2026** on the frozen `bench2_*` splits at the CARRADA-matched sensor
(256 × 64). This supersedes the original audit, which was run on the old sensor and on datasets
that no longer exist. Evidence is reproducible from four scripts:

```bash
python scripts/audit_data_quality.py    # criteria 3, 4, 5, 6, 7, 9 — statistics of the frozen data
python scripts/verify_splits.py         # criterion 2 + leakage, on all 14,900 frames
python scripts/audit_datasets.py        # each stratum vs its config
python scripts/verify_clutter.py        # the clutter model itself
```

| Criterion | Result | Key evidence |
|---|---|---|
| 1 Physical fidelity | **PASS** | λ, slope, fs, ΔR, R_max, Δv, v_max all match closed form to 5 s.f. |
| 2 Label correctness | **PASS** (with a documented 7%) | 100% of `isolated`/`multi` peaks on the label bin; 93% overall — the shortfall is confined to the strata where interference *is* the condition |
| 3 Statistical realism | **PASS** (was PARTIAL) | noise KS 0.006 vs exponential; clutter now Weibull **c = 1.51–1.56** (was 2.03); 1/R⁴ exponent fits at **−1.003** |
| 4 Difficulty calibration | **PASS** | ~48–50% of targets sit in the theory Pd 0.1–0.9 band; 13–16% below, 35–39% above |
| 5 Coverage | **PASS** | 0–5 targets, 8–45 m, ±8 m/s, cell SNR −10…+50 dB, no empty range decile |
| 6 No shortcuts | **PASS** | off-grid placement uniform (std 0.290 vs 0.289 ideal, KS 0.007); no frame statistic predicts the label; 8,400/8,400 unique seeds |
| 7 Benchmark neutrality | **ACTION** | the frozen test set is 8 **equal** strata, 37.5% of it deliberately adversarial — report per-condition, never a bare overall Pd |
| 8 External validity | **PARTIAL** | bin geometry matches CARRADA exactly; point targets and single-channel RD remain gaps |
| 9 Reproducibility | **PASS** | config + radar config + axes stored with every split; master seeds 91001–95001, all frame seeds unique |

### What changed since the first audit

**Clutter is now heavy-tailed — the one previously-failing criterion.** The old model measured
Weibull `c = 2.03`, i.e. Rayleigh, which *understates* the false-alarm problem in exactly the
regime the project exists to study. The compound-Gaussian model (Weibull texture × Rayleigh
speckle) measures `c = 1.51` on `bench2_test` and `1.56` on `bench2_train`, with p99/p50 ≈ 3.8.
CNR calibration is accurate to +0.16…+0.27 dB above 10 dB (+1.27 dB at 5 dB, where the ridge is
barely above noise), and off-ridge cells sit at −0.01 dB — the clutter stays where it is supposed
to be.

*Caveat (already recorded in `PROGRESS.md` step 2b — repeated here because it belongs with the
measurement):* the config's `shape` is the **texture** parameter, not the shape of the resulting
amplitude distribution. Requesting `shape = 2.0` measures back as `c ≈ 1.68`, because the chain
always adds its own Rayleigh speckle. The swept range 1.1–2.0 delivers measured `c` roughly
1.4–1.7. **Quote the measured value in the report**; both are in each frame's metadata.

**The radar equation is verified, not just applied.** Regressing the stored post-propagation
`amplitude_db` on `40·log10(R/25)` gives slope **−1.003** (r = −0.92), and the residual spans
exactly **4.0–16.0 dB**, the config's drawn-amplitude window. So the loss is applied once, at the
right exponent, with nothing else contaminating it. Mean cell SNR falls **26.1 → 5.4 dB** from the
8–18 m band to the 38–46 m band, and corr(range, cell SNR) = **−0.77**.

### Two correlations that look like faults and are not

1. **corr(range, amplitude_db) = −0.90.** Criterion 5 asks for no accidental correlation between
   sampled factors; this one is not accidental, it *is* the radar equation. Range and the *drawn*
   amplitude are independent; the stored amplitude is post-loss.
2. **Velocity is not uniform.** The `clutter` stratum confines targets to |v| < 1.2 m/s and
   `clutter_edge` to 2.5–4.2 m/s, so the pooled Doppler histogram is peaked. That is the
   stratification doing its job, not a sampler defect.

### Open items from this audit

- **Criterion 7 is the live one.** `bench2_test` is 12.5% each of eight conditions, three of which
  (`masking`, `unresolved`, `clutter_edge`) are built to break a specific detector. An unweighted
  mean over that mixture is not a "typical scene" number and must not be presented as one. The
  per-condition table is the headline; any pooled figure needs its composition printed beside it.
- **Class balance measures 1:943–1:1027** here (200-frame sample per split) against 1:888 in
  step 6d. Both are samples of the same quantity, so this is agreement, not a correction — the
  point for `pos_weight` is simply that the ratio is now ~1:900–1000, not the 1:1563 that
  `dl/train.py`'s docstring still quotes from the old sensor.
- **Alignment is 93%, and 77% on `clutter`.** Already the documented position — a clutter spike can
  outrank the target within the ±2-bin search window. It is a property of the scenario, not a
  labelling error, and the ≥15 dB gate is what makes the number interpretable at all.

**Measured vs cited — keep the distinction in the write-up.** The shape `c = 1.51` is *measured
from this project's data*. The reference values that make `c < 2` "realistic" come from the
published radar clutter-statistics **literature**, not from distributions fitted to CARRADA or
RADDet (neither was available for fitting). The claim is therefore *our measurement sits in the
regime the literature reports*, not *our clutter matches dataset X*.

**Where real radar data fits.** Not usable for Pd/Pfa scoring (no bin-level ground truth, and no
truth for false alarms). Usable for statistical validation (fit real clutter, compare
distributions — needs no labels) and for a qualitative demonstration (run the detectors on real
frames). Full sim-to-real transfer (train synthetic → test real) is the gold standard and is
identified as future work.

## References

- Ouaknine et al., *CARRADA Dataset: Camera and Automotive Radar with Range–Angle–Doppler
  Annotations*, ICPR 2021. https://arxiv.org/abs/2005.01456
- Bansal et al., *RadSimReal: Bridging the Gap Between Synthetic and Real Data in Radar Object
  Detection with Simulation*, CVPR 2024. https://arxiv.org/abs/2404.18150
- Schumann et al., *RadarScenes: A Real-World Radar Point Cloud Data Set for Automotive
  Applications*, FUSION 2021.
- Ward, Tough & Watts, *Sea Clutter: Scattering, the K Distribution and Radar Performance*.
- Rohling, *Radar CFAR Thresholding in Clutter and Multiple Target Situations*, IEEE T-AES, 1983.
