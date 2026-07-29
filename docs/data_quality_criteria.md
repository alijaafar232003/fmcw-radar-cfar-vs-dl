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

Measured against the criteria (full evidence in `notebooks/data_quality_audit.ipynb`):

| Criterion | Result |
|---|---|
| Physical fidelity | PASS — full chirp→dechirp→FFT chain; derived quantities match theory |
| Label correctness | PASS — RD peak = label bin = mask, verified across all scenarios |
| Statistical realism | **PARTIAL** — noise exponential (KS 0.005) ✔; clutter Weibull c = 2.03 (Gaussian-like) ✘ |
| Difficulty calibration | PASS — cell-SNR calibrated, Pd sweeps 0→1 |
| Coverage | PASS — 0–5 targets, 9–100 m, ±15 m/s, cell SNR −1.5…38.5 dB |
| No shortcuts | PASS — targets off-grid (0.13 m of a 0.50 m bin) |
| Benchmark neutrality | ACTION — keep diagnostic sets separate from the frozen test set |
| External validity | PARTIAL — same 77 GHz regime as CARRADA/RADDet; gaps listed below |
| Reproducibility | PASS — fixed seeds, configs stored with the data |

**Measured vs cited — an important distinction.** The clutter shape `c = 2.03` is *measured from
this project's data*. The heavy-tailed reference values (`c < 2`) come from the published radar
clutter-statistics **literature**, not from distributions fitted to CARRADA/RADDet (which were
not available). The comparison is therefore *our measurement vs literature reference models*.

**Closed gap — range-dependent amplitude (the radar equation).** The audit found that target
amplitude and range were *independent*: a target at 100 m could be as strong as one at 10 m,
which is physically impossible. Received power obeys `P_r ∝ σ/R⁴`, so the frozen datasets now
apply a `40·log10(R/R_ref)` loss (`src/radar/propagation.py`), referenced to 50 m. Verified:
doubling the range costs **12.04 dB** (the 1/R⁴ law), and in generated data the correlation
between range and measured cell SNR is **−0.77** (mean cell SNR falls 42.5 → 10.2 dB from the
0–30 m to the 90–120 m band). This makes difficulty *physically* motivated — distant targets are
hard because they are distant — and matches the systematic range-dependence seen in real
recordings.

**Decision — Gaussian clutter retained; heavy-tailed clutter deferred to future work.** The
present CFAR results are valid for homogeneous Gaussian clutter, the classical regime CFAR was
derived for. Changing the model would require regenerating every dataset and repeating the CFAR
study. The gap is **conservative**: real, spikier clutter would make CA-CFAR perform *worse*, so
it does not threaten the conclusions — and adding K/Weibull clutter later would strengthen them,
since that is where OS-CFAR and a learned detector gain most.

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
