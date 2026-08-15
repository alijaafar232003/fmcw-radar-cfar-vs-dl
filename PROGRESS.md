# Progress Log

A backward-looking journal of what I did, what I decided, and *why* — so I can re-orient after
a break. (Forward-looking plan lives in `Plan.md`; deep notes in `docs/`.)

> Project: FMCW Radar Target Detection — CFAR vs Deep Learning on Range–Doppler maps.
> A simulation-based study comparing classical CFAR against a DL detector, fairly.

---

## Milestone 1 — Radar model & simulator  ✅
- Built `src/radar/`: `config.py` (radar params), `simulator.py` (chirp → echo → dechirp →
  AWGN → clutter → multipath), `rd_processing.py` (windows + range/Doppler FFT → RD map),
  `labels.py` (point labels + mask), `io.py` (saving).
- **Decision — radar parameters:** medium-range 77 GHz automotive FMCW
  (B=300 MHz, Tc=32 µs, N=512, M=128 → 256×128 RD map, 0.5 m / 0.48 m·s⁻¹ resolution).
  **Why:** designed top-down from requirements (band→fc, range-res→B, coverage→N,
  max-vel→Tc, vel-res→M); realistic, citable, and a good size for both CFAR and DL.
- **Decision — retired the old 24 GHz pipeline** under `notebooks/fmcw_python_streamlit_refactor/`
  (kept for presentation) because it used different, wastefully large parameters.

## Milestone 2 — Dataset generator & scenarios  ✅
- Built `src/data/` (`create_dataset.py`, `generate_single_frame.py`), YAML-driven, reproducible
  seeds, per-frame RD map + labels.csv + mask + metadata + preview.
- **7 scenarios** (`configs/`): clean, noise_only, snr_sweep, clutter, close_targets,
  multi_target, multipath. **Why each:** isolate one effect per set to find *where CFAR fails*.
- **Decision — synthetic data only** (no real data). **Why:** real RD data lacks the bin-level
  ground-truth labels needed for fair Pd/Pfa; real data reserved as a possible qualitative
  demo later.

## Milestone 3 — Validation  ✅
- Built `notebooks/validate_rd_alignment.ipynb` (replaced an old broken checker). Confirms, per
  frame, that **RD peak ≈ label bin ≈ mask cell**, with DataFrames + previews + saved reports
  (`reports/validation/`).
- **Result: 7/7 datasets pass** — every isolated target within ±1 bin, no NaN/Inf.
- **Why it mattered:** proves the labels are spatially correct *before* building detectors on
  them; without it every later Pd/Pfa could be silently wrong.
- **Finding:** close-target pairs correctly show the weak target *masked* by the strong one =
  the radar's resolution limit (not a bug).

## Milestone 4 — Notes / theory  ✅ (in progress on my side)
- Wrote reference docs: `docs/project_reference_notes.md` (radar physics + parameters),
  `docs/cfar_methods_review.md` (CA/GO/SO/OS/VI/DL-CFAR + papers),
  `docs/benchmarking_conventions.md` (SNR definition + scoring rulebook).
- Personal notebook notes taken: radar, chirps, dechirping, RD map, CFAR + techniques.
- **Still to note before coding:** detection metrics (Pd/Pfa/ROC) + fairness rules, post-CFAR
  clustering/NMS, and the cell-SNR idea. (DL theory deferred to Phase C.)

## Milestone 5 — SNR calibration  ✅
- **Problem:** SNR was set on the raw CPI samples, but the FFTs add ~43.4 dB processing gain,
  so even "−10 dB" frames were trivially detectable → Pd ≈ 1 everywhere → flat, useless ROC.
- **Fix:** SNR is now defined at the detector as **cell SNR** (target peak-cell power ÷ mean
  noise power per RD cell). New `src/radar/calibration.py` computes the processing gains
  analytically from the windows, converts a requested cell SNR into the injected noise
  variance, and measures each target's *actual* peak power from a clean RD map (capturing
  scalloping loss, up to ~1.9 dB).
- **Per-target `cell_snr_db` is now stored in `labels.csv`** — a single frame-level SNR is
  meaningless when a strong target sits beside a weak one.
- **Verified:** on-bin error 0.02 dB; noise model within 1.1%; gain = 43.38 dB.
- **Acceptance check PASSED:** Pd rises **0.00 → 1.00** across the sweep (transition ~9–12 dB).
  Before the fix every point would have read 1.00.
- Also fixed: clutter and thermal noise shared one seed (they drew the identical random field);
  they are now independent.
- `snr_sweep_debug` reworked: amplitude 0 dB (so requested == actual), sweep 0–21 dB,
  240 frames (30 per SNR point, for a low-variance Pd estimate).
- Validation gained an **SNR gate**: alignment is only strictly checked for targets ≥15 dB cell
  SNR — below that the target is invisible and the "peak" is just noise. Still 7/7 passing.

## Milestone 6 — Shared evaluation module  ✅
- `src/eval/detection.py` — the scoring rulebook in code, used **identically** by CFAR and DL
  (this is what makes the comparison fair).
- **Matching:** optimal one-to-one assignment (Hungarian) inside a ±1-bin tolerance gate.
- **Counting:** TP / FN / FP, with a **guard region** — cells inside a truth's gate are removed
  from the false-alarm denominator, and an unmatched detection sitting there is *ignored*
  (near-miss of a real target, not a spurious detection).
- **Metrics:** Pd, per-cell Pfa, false alarms per frame, localization error (bins and m / m·s⁻¹).
- **Curves:** `roc_curve()` (sweep a score threshold) and `pd_vs_snr()` (bin targets by their
  own `cell_snr_db`).
- **Verified:** 10/10 edge cases pass (duplicates on one target, one-detection-two-targets,
  outside tolerance, empty frames, detections-with-no-targets…). End-to-end on 240 real frames
  it produced a monotonic ROC and a clean Pd-vs-SNR S-curve (0.00 → 1.00 across 0–20 dB).

---

## Milestone 7 — CFAR detectors (Phase B)  ✅ (first version)
- `src/cfar/cfar.py` — CA / OS / GO on linear power |RD|², vectorised with scipy.ndimage
  (box filters for CA/GO means, rank filter for OS). Returns a **ratio map** (CUT / local
  noise) + **validity mask**; thresholding the ratio == sweeping design Pfa, so one pass gives
  the whole ROC. Border cells excluded (not padded).
- `src/cfar/detect.py` — ratio map → point list via threshold + non-maximum suppression
  (one detection per target blob), each with its ratio as a score.
- Guard=2, train=8 (N_ref=416) as starting geometry; all configurable for the later sweep.
  GO leading/lagging split defaults to the **Doppler axis** (where the zero-Doppler clutter
  edge lies).
- **Verified:** vectorised CA matches a brute-force reference to 3.5e-15; OS matches exactly.
- **First results (guard 2, train 8, design Pfa 1e-4):**
  - CA false-alarm calibration on noise-only frames: **measured 8.5e-5 vs design 1e-4** ✓
  - False-alarm ordering as theory predicts: CA ~2.5–3 FA/frame, OS ~0.1–0.2, GO ~0.
  - `close_targets`: **Pd capped at 0.50** for all methods — the resolution limit (weak
    partner unresolvable), exactly as predicted.
  - CA masking visible: two close targets scored 204/104 under CA vs 4025/3615 under OS.
  - Runtime: CA ~28 ms, GO ~6 ms, **OS ~266 ms/frame** (rank filter is costly) — real
    robustness-vs-speed trade-off for the runtime metric.
- Ported/cleaned from the `InteractivePipeline.ipynb` prototype, vectorised, wired to `src/eval`.

**Known follow-ups (for the sensitivity sweep / Phase D):** OS and GO thresholds are not yet
calibrated to an exact Pfa (OS currently over-conservative → 0 FA on noise-only); the fair
head-to-head uses the ROC (Pd at *matched measured* Pfa), not a single shared alpha.

## Milestone 8 — CFAR audit, GO fix, comparison notebook  ✅
- **Audit found a real bug in GO**: `_half_ring_means` divided the FULL-ring sum by HALF the
  cell count → noise estimate ~2.1x too high on homogeneous noise. Rewrote it with explicit
  half-ring kernels; now verified against brute force (2.7e-15) and gives 1.09x true (correct:
  max of two half-means).
- **OS was NOT a bug**: its 1.39x estimate is the 0.75-percentile of exponential noise
  (=ln4). It just needs its own alpha; the ROC (matched Pfa) sidesteps that.
- Added `valid_mask` to `evaluate_frame` so the Pfa denominator counts only CFAR-tested
  (non-border) cells — otherwise Pfa is understated.
- **Fair comparison (Pd at matched measured Pfa, multi_target):** CA 0.69, OS 0.65, GO 0.66 at
  Pfa=1e-3 — i.e. **close**, CA marginally ahead where targets are well separated (CA is
  optimal in homogeneous noise). The dramatic single-Pfa "differences" earlier were operating-
  point artifacts + the GO bug.
- `notebooks/cfar_comparison.ipynb` — RD/threshold/detection visual, noise-estimate + Pfa
  calibration, ROC/FROC, Pd-vs-SNR, runtime, per-scenario summary. Runs end-to-end.

**Honest open observation (for scenario design in Phase C/D):** the current debug sets don't
yet isolate a regime where OS/GO clearly beat CA — multi_target separations (>=5 bins) give
only mild masking, close_targets is capped at Pd 0.5 by resolution, and clutter targets are
high-SNR. To *demonstrate* OS's masking advantage and GO's clutter-edge advantage we need
targeted scenarios: two RESOLVABLE targets within one training window, and detectable targets
embedded in/next to the clutter ridge.

## Milestone 9 — CFAR threshold calibration + geometry tuning  ✅
- **Problem:** only CA has a closed-form `alpha(Pfa)`; OS/GO reused it and so were not at the
  design Pfa (over-conservative → ~0 false alarms on noise).
- **Fix:** `src/cfar/calibrate.py` — empirical Monte-Carlo calibration. Process many pure-noise
  CPIs → CFAR ratio maps → pool ratios over valid cells → `alpha` = (1−Pfa) quantile. Works for
  any CFAR statistic.
- **Validated:** empirical CA matches the analytic formula to 2.9% (1e-3) / 4.6% (1e-4) — the
  small bias is the window's inter-cell correlation, so the empirical value is the more accurate
  one for real data. **Verified** on independent noise: all three hit target (Pfa=1e-4 → CA
  9.4e-5, OS 9.8e-5, GO 9.8e-5).
- Calibrated alpha (guard 2, train 8): Pfa=1e-4 → CA 9.74, GO 9.09, OS 6.98; Pfa=1e-3 → 7.17 /
  6.69 / 5.17.
- **Geometry sweep:** guard 2 vs 3 changes Pd < 1% (insensitive, as predicted from the mainlobe);
  train Pd flat beyond ~8 → **guard 2, train 8** chosen (knee of CFAR-loss vs locality).
- `notebooks/cfar_comparison.ipynb` rebuilt: added a calibration section + geometry sweep, and
  the single-operating-point tables (Pd-vs-SNR, per-scenario) now use the CALIBRATED alpha so all
  three methods sit at the same measured Pfa. Documented in `docs/cfar_methods_review.md`.

## Milestone 10 — CFAR "home-turf" scenarios (OS + GO wins)  ✅
- Built `masking_debug` (close_pair, separation 5–9 range bins: resolvable but inside one
  training window, strong+weak amplitudes). **OS beats CA/GO on the weak target: Pd_weak
  OS 0.65 vs CA 0.15 vs GO 0.12.** The clean demonstration of *why OS exists*.
- Strengthened `clutter_debug` to a real edge. Note: `strength_db` is an ABSOLUTE amplitude
  scale, so +10 did nothing (buried under the ~688 calibrated noise var); **~45 forms a real
  edge**. **GO wins: clutter-edge FA GO 8.4 vs CA 14.2 vs OS 17.1** (GO suppresses ~41%).
- Complementarity result (the CFAR headline): OS wins masking / worst in clutter; GO wins
  clutter / worst in masking; CA balanced + cheapest, optimal on clean (clean is a 3-way tie).
  No universal winner -> motivates DL.
- Notebook gained a "home-turf comparison" section (table + masking figure). Documented in
  `docs/cfar_methods_review.md`.
- **Flagged for realism roadmap:** clutter should use a proper clutter-to-noise-ratio
  calibration (like cell-SNR) instead of the current absolute `strength_db`.

## Milestone 11 — Data quality audit + real-dataset comparison  ✅
- **Why:** to answer "how do you know your synthetic data isn't rigged?" with evidence.
- `docs/data_quality_criteria.md` — 9-criterion framework (physical fidelity, label correctness,
  statistical realism, difficulty calibration, coverage, no shortcuts, benchmark neutrality,
  external validity, reproducibility) + how practitioners verify each + references.
- `notebooks/data_quality_audit.ipynb` — measured audit, 6 figures, scorecard.
- **Measured results:** noise exponential (KS 0.005 ✔); targets **off-grid** (0.13 m of a 0.50 m
  bin → no bin-centre shortcut ✔); coverage 0–5 targets / 9–100 m / ±15 m/s / SNR −1.5…38.5 dB ✔;
  difficulty calibrated ✔. **Gap: clutter Weibull shape c = 2.03 → Gaussian-like**, whereas real
  ground clutter is heavy-tailed (c < 2).
- **Real-dataset comparison** (published specs): same 77 GHz automotive regime as CARRADA
  (4 GHz BW, 0.2 m, 256×64) and RADDet (0.5 GHz, 0.3 m, 256×64); ours 256×128, 0.5 m, 128 m —
  a medium-range profile. Precedent for synthetic-only: RadSimReal, CVPR 2024.
- **Care taken:** the heavy-tailed reference curves are **literature values**, NOT distributions
  fitted to CARRADA/RADDet (not available) — the notebook and docs label this explicitly.
- **Key argument:** every identified gap makes the simulation *easier* than reality (Gaussian
  clutter, point targets, no phase noise, constant RCS), so results are not flattered by an
  artificially hostile environment; CFAR's weaknesses would be **worse** on real data.
- **DECISION: keep Gaussian clutter now; heavy-tailed clutter = documented future work.**
  Changing it would mean regenerating everything and redoing CFAR, right on the DL critical path;
  and the gap is conservative. Logged on the realism roadmap.
- **DECISION: real data is for statistical validation / qualitative demo only** — never for
  Pd/Pfa scoring (no bin-level truth, no false-alarm truth). Sim-to-real transfer = future work.

## Milestone 12 — Radar equation + FROZEN benchmark datasets  ✅
- **Radar equation** (`src/radar/propagation.py`) — closes the audit's "amplitude independent of
  range" gap. Received power `∝ σ/R⁴`, applied as `40·log10(R/R_ref)` referenced to 50 m, folded
  in *before* the clean-map SNR measurement so `cell_snr_db` reflects true post-propagation
  strength. **Verified:** doubling range costs 12.04 dB; in generated data
  corr(range, cell_SNR) = **−0.79** (mean cell SNR 42.5 dB at 0–30 m → 10.2 dB at 90–120 m).
  Difficulty is now *physically* motivated.
- **Frozen splits generated** (the authoritative benchmark; `configs/frozen_{train,val,test}.yaml`):
  train 10,000 / val 2,000 / test 2,000 frames, same distribution, **different seeds**.
  A neutral MIXTURE (0–5 targets, SNR spanning the transition, clutter + multipath enabled) —
  deliberately not built from the masking/clutter diagnostic sets, which would favour OS/GO.
- **Compact storage** — `rd_map_db` and per-frame axes dropped (axes saved once as `axes.npz`):
  177 KB/frame vs 317 KB (−44%), no information lost.
- **Verification:** no leakage (0 shared frames between any split pair); distributions match
  (mean targets 2.53/2.53/2.56, median cell SNR ≈21 dB); no NaN/Inf.
- **Alignment:** rate rises with SNR and is **exactly 1.000 above 22 dB** (966 targets, 0 errors);
  0.97 at 15–18 dB. Investigated the ~0.5% marginal cases — they are **noise spikes beating a
  weak target near the detection threshold, i.e. correct physics, not a labelling bug** (errors
  vanish at high SNR, which a labelling bug would not do). The old 15 dB validation gate was
  slightly permissive; the reliably-visible threshold is ~22 dB.

## Milestone 13 — DL Model 0 (Tiny CNN): pipeline + first CFAR-vs-DL result  ✅
**Week 1 milestone met (30 Jul, target was 2 Aug).** Full design record in `docs/dl_design.md`.

- **Formulation:** dense per-cell heatmap regression — deliberately *structurally identical* to
  CFAR (per-cell score → threshold → NMS → points), so only the **decision rule** differs.
  The network is a *learned replacement for the CFAR thresholding step*, not a generic detector.
- **`src/dl/`**: `dataset.py` (normalization), `models.py` (TinyCNN), `train.py` (loss, gates,
  loop), `infer.py` (heatmap→points, evaluation).
- **Key design — receptive field matched to CFAR.** CA-CFAR uses a 21×21 window; TinyCNN uses
  dilated 3×3 convs with dilations (1,2,3,4) → RF = 21×21. This makes Model 0 a **controlled
  experiment**: identical context, learned vs hand-derived rule. Dilations increase smoothly with
  no common factor (Hybrid-Dilated-Convolution guidance) to avoid gridding artifacts.
- **Normalization (highest-leverage choice):** dB above the frame's own noise-floor median.
  Rejected raw power (7 orders of magnitude), fixed global reference (scale varies 57× between
  frames), and the stored `rd_map_db` (per-frame max normalization **destroys absolute SNR**).
  Measured: median std **0.000** across frames.
- **Loss:** `BCEWithLogitsLoss(pos_weight=200)`; measured imbalance **1:1563** (0.064% positive),
  so plain BCE would collapse to all-zeros at 99.94% "accuracy".
- **ALL GATES PASSED:** dataset shapes/finiteness/median≈0; **orientation 72/72** (no transpose
  bug); params 5,041; **receptive field MEASURED at 21×21** by backprop from one output pixel;
  **overfit-one-batch: loss 0.798→0.00023 (3514×), 25/25 peaks within 1 bin**; training
  converged (train 0.203→0.102, val 0.141→0.107, no overfitting).
- **Fairness enforced three ways:** DL imports **CFAR's own `all_peaks`** (same NMS, size 5);
  the **same border `valid` mask** is applied so the Pfa denominator matches; the same
  `src/eval` matching and metrics.

### FIRST CFAR-vs-DL RESULT (frozen_test, 500 frames, matched Pfa)
| Method | Pd@1e-4 | Pd@1e-3 | Pd@1e-2 | ms/frame |
|---|---|---|---|---|
| **DL (TinyCNN, 5k params)** | **0.784** | **0.832** | 0.856 | 55.5 |
| CA-CFAR | 0.752 | 0.814 | 0.859 | 6.9 |
| OS-CFAR | 0.742 | 0.820 | 0.862 | 296.7 |
| GO-CFAR | 0.752 | 0.806 | 0.855 | 38.4 |

**DL wins in the low-Pfa regime** (+3.2 pts at 1e-4, +1.8 at 1e-3) — the operationally important
one. Methods converge at high Pfa. The margin is **modest, as predicted for matched context** — a
dramatic win would signal an unfair comparison. **Why it can win at all:** CFAR must *exclude* its
guard region from the noise estimate (a mean cannot tell "bright from my own mainlobe" from "noise
is high"); a learned function can use those cells, i.e. it exploits information the classical rule
is forced to discard.

- `notebooks/dl_cnn_experiment.ipynb` — all gates, training curves, heatmap visualisation,
  ROC/FROC vs CFAR, Pd-vs-SNR, runtime-vs-accuracy, conclusions.
- **Honest limitations recorded:** trained on only 3,000/10,000 frames for 8 epochs (a *floor*,
  not a ceiling); no analytically guaranteed Pfa unlike CFAR; synthetic-only training.

## Milestone 14 — Coverage study + stratified dataset redesign  ✅
Full record: `docs/dataset_coverage_study.md`; notebook §10–12.

**The narrative: evidence → diagnosis → redesign** (deliberately run in that order, so the design
change is empirically justified rather than asserted).

**Evidence — home-turf test (all 4 methods, matched Pfa=1e-4).** The DL threshold was Monte-Carlo
calibrated on pure noise with the CFAR protocol (0.8638).

| Scenario (metric) | CA | OS | GO | **DL** |
|---|---|---|---|---|
| Masking — weak-target Pd ↑ | 0.15 | **0.65** | 0.12 | **0.40** |
| Clutter — band FA/frame ↓ | 14.15 | 17.10 | 8.35 | **0.95** |
| Unresolved — Pd (0.5 = physics cap) | 0.50 | 0.50 | 0.50 | 0.55 |

- **Clutter: DL dominates** — ~9× fewer clutter-band false alarms than GO (best classical) while
  holding Pd = 0.86. Local window statistics can't represent clutter; a learned function reads its
  *texture*.
- **Masking: DL mid-table** — beats CA (0.15) and GO (0.12) but loses to the specialist OS (0.65).
- **Unresolved: all capped ~0.5** (physics). DL's 0.55 = 1 frame of 10 → within noise, NOT
  super-resolution.
- Caveats recorded: sets are small (40/20/10 frames → preliminary), and they predate the radar
  equation → secondary distribution shift.

**Diagnosis — DL performance tracks training coverage almost monotonically:**

| Condition | Coverage in `frozen_train` | Outcome |
|---|---|---|
| Clutter | ~16 % | DL wins |
| Masking | ~1.1 % | DL loses to OS |
| Unresolved | ~0.4 % | no advantage |

→ The masking weakness is a **data-coverage problem, not a capability limit**. Without this
diagnosis the project would have wrongly concluded "CFAR beats DL at masking".

**Redesign — stratified sampling.** Generator gained a `scenarios:` block: per frame it cycles
through 8 named strata and deep-merges that stratum's overrides onto the defaults; **each frame
records its `condition` in metadata**, so per-condition evaluation has guaranteed sample counts.
Strata: `isolated, masking, unresolved, clutter, multi, multipath, noise, mixed`.
`bench_train` 9,600 / `bench_val` 1,600 / `bench_test` 2,400 (1200/200/300 per stratum), disjoint
seeds, **radar equation on in every stratum** (consistent physics). Verified on a 40-frame trial:
exactly balanced, tags recorded, per-stratum behaviour correct.
**Nothing deleted** — `frozen_*` kept as the neutral aggregate benchmark; `mixed` preserves the
realistic recipe.

**Defence of the redesign** (the "why not from the start?" question): simple random sampling
maximises *realism* but necessarily under-represents rare configurations; **stratified sampling
sizes strata for estimation reliability instead**. The correct proportions were not knowable before
measuring them — the audit produced the design. `random → audit → stratified` is standard
experimental method.

## Milestone 15 — Stratified benchmark: DL wins EVERY condition  ✅
Full record: `docs/results_stratified.md`; notebook §13.

**Datasets generated + verified:** `bench_train` 9,600 / `bench_val` 1,600 / `bench_test` 2,400 —
exactly 1200/200/300 per stratum, **no leakage**, per-stratum behaviour correct. Coverage fixed:
masking **1.1% → 7.0%**, unresolved **0.4% → 12.8%**.

**Retrained** TinyCNN from scratch on `bench_train` (12 epochs): train 0.119→0.066,
val 0.096→0.0755, no overfitting. Same 5,041 params, same 21×21 receptive field.

**Per-condition results (bench_test, matched Pfa=1e-4, 150 frames/stratum):**

| Condition | CA | OS | GO | **DL** |
|---|---|---|---|---|
| isolated | 0.867 | 0.867 | 0.867 | **0.880** |
| masking | 0.493 | 0.643 | 0.493 | **0.727** |
| unresolved | 0.490 | 0.490 | 0.490 | **0.547** |
| clutter | 0.666 | 0.686 | 0.618 | **0.723** |
| multi | 0.852 | 0.855 | 0.848 | **0.871** |
| multipath | 0.831 | 0.831 | 0.818 | **0.840** |
| mixed | 0.783 | 0.798 | 0.768 | **0.804** |

*(Table above is the CORRECTED post-NMS-fix version; `unresolved` DL was 0.547 pre-fix.)*

**DL has the highest Pd in 6 of 7 conditions and ties on the 7th.** Mechanism metrics:
- **masking weak-target Pd: DL 0.500 vs OS 0.333**, CA/GO 0.033. Unaffected by the NMS fix, and the
  5–9 bin separation exceeds the suppression radius → the most robust DL result in the study.
- **unresolved: 4-way tie at 0.490** — no method beats the resolution limit, as physics dictates.
- **clutter-band FA/frame: DL 2.47 vs GO 4.83** (best classical), CA 7.91, OS 11.36 — *and* higher
  Pd (0.723 vs GO 0.618).

## Milestone 16 — Fairness audit: NMS bug found & fixed; result NOT yet trustworthy  ⚠️
**See `NEXT_SESSION.md` for the full plan. Do not put these numbers in the report yet.**

- **BUG FOUND (and fixed).** `all_peaks` used `score == maximum_filter(score)`, marking *every* cell
  of a plateau as a peak. A sigmoid output has flat/tied regions (CFAR's continuous ratio does not),
  so **DL emitted multiple detections per target** — 1.90 per single-target frame vs CA's 0.83 — and
  because the evaluator ignores unmatched detections inside a truth's guard region, the duplicates
  cost nothing. DL was getting several free attempts per target.
  Fixed via `_collapse_plateaus()` in `src/cfar/detect.py` (one detection per connected peak region,
  applied identically to all detectors).
- **How it was caught:** DL scored 0.547 on `unresolved`, *exceeding* the physical 0.5 cap. A physics
  sanity check exposed it.
- **RETRACTED:** the "19 % genuine sub-resolution-cell recovery" claim. Corrected to 8 %, and that is
  most likely a matching artifact. All four methods now tie at 0.490.
- **Survived the fix unchanged:** masking (DL 0.500 vs OS 0.333), clutter (DL 2.47 vs GO 4.83),
  isolated, multi, multipath, mixed.

**Open asymmetries — the comparison is not yet fair:**
1. **The CNN gets an implicit matched filter; CFAR does not** (it tests a single cell while the CNN
   integrates the whole ~5×5 mainlobe). Leading explanation for DL winning even on isolated targets.
2. **CFAR geometry was tuned on `snr_sweep`** (clean, single target) then applied everywhere — likely
   penalising it badly in the clutter stratum, where a 21×21 window dwarfs the ~4-bin ridge.
3. **Pfa is matched by *design*, not by *measurement*** (4.8e-5 … 1.9e-4, a 4× spread).
4. **DL was retrained per distribution; CFAR was not re-tuned.**
5. **±1-bin gate may measure localisation rather than detection** (DL learns label bins; CFAR finds
   physical peaks).
6. **Train and test share one generator** → possible exploitation of generator regularities.

**Notebook fixed and verified:** `dl_cnn_experiment.ipynb` — 36 cells, **19/19 executed, 10 figures,
zero errors** (earlier run had been killed and left empty). Includes the before/after coverage plot.

## Milestone 17 — Literature corpus review (19 papers, full text)  ✅
Full record: `docs/literature_corpus_review.md`. Extends `docs/literature_dl_vs_cfar.md`, which was
built from web sources; RadCNN and the *Electronics* 2022 review, listed there as "could not obtain",
are now read in full, plus 15 more. PDFs live in `papers/` (gitignored — publisher copies).

**The corpus splits into five families that answer different questions.** Only one — per-cell learned
threshold (RadCNN, Gusland, Lallouet, *and our TinyCNN*) — makes a head-to-head Pd-at-matched-Pfa
comparison against CFAR meaningful. Conflating families is how "DL beats CFAR by 40×" claims arise.

**Four findings that bear directly on our result:**
- **In homogeneous noise, a well-built CNN ties CFAR** (RadCNN S4; Gusland below 450 m). We win there
  (0.880 vs 0.867) — the strongest single indication that something still favours the network.
- **A good learned detector normally loses somewhere.** RadCNN loses 0.8 dB in its S3 (target in
  noise, reference window overlapping clutter). We lose nowhere. That is atypical, not impressive.
- **Two rival explanations exist for a win in homogeneous noise**, and we had only found one:
  (a) mainlobe integration — the CNN convolves the whole ~5×5 mainlobe, CFAR tests one cell; and
  (b) **CFAR loss** — the finite reference window makes CFAR's threshold a random variable
  (Lallouet et al., Thales). They must be separated by experiment, not argued.
- **Banerjee et al. prove CA-CFAR *is* a convolution + ReLU** with fixed weights (+1 on the CUT,
  −α/N on each training cell, bias 0). Our matched-receptive-field TinyCNN is therefore the *learned
  instance of a proven generalisation of CA-CFAR* — not merely "a small CNN". This is the theoretical
  backbone the project was missing.

**Independent support for choices we made:** Gusland restricts the CNN's receptive field to match the
CA-CFAR patch (our Milestone 13 design, arrived at independently) and notes guard cells are
unnecessary for a CNN — the citation that makes our asymmetry #1 legible. RadCNN reports measured Pfa
drifting from design Pfa under clutter as a *finding*, not a flaw. The *Electronics* review confirms
Pd at matched Pfa is the field's standard metric — so matching at **measured** Pfa exceeds it.

## Milestone 18 — Phase renamed: attribution, not audit  🔄 IN PROGRESS
"Fairness audit" framed the work as fixing a defect. The work is actually **attribution**: *where
does the DL margin come from?* — a question whose answer is a result whichever way it falls. Renamed
because the outcome must not be presupposed, and because "hypothesis testing" would collide with the
binary hypothesis testing (H₀/H₁) that is the project's own subject matter.

Six named hypotheses, each with an experiment that can falsify it:

| | Hypothesis | Decisive experiment | Falsified if |
|---|---|---|---|
| H1 | CNN integrates the ~5×5 mainlobe; CFAR tests one cell | guard-constrained CNN (annulus RF) + matched-filter→CFAR | isolated-target margin survives the guard constraint |
| H2 | CFAR loss — finite reference window ⇒ noisy threshold | clairvoyant-threshold detector as upper bound | CFAR↔clairvoyant gap ≪ observed margin |
| H3 | Protocol — ±1-bin gate, design-vs-measured Pfa | tolerance sweep + matched *measured* Pfa | margin unchanged by both |
| H4 | Adaptation asymmetry — DL retrained, CFAR not | per-condition CFAR tuning | margin unchanged |
| H5 | Generator exploitation | perturb generator at test, no retraining | DL does not collapse |
| H6 | Clutter texture is unrepresentable by local window statistics | **already supported** — 2.47 vs 4.83 FA/frame, replicated by RadCNN S2 and Gusland | — |

**H2 is cheap to bound analytically and should go first.** With guard 2 / train 8 (N=416 reference
cells) at Pfa 1e-4, `α = N(Pfa^(−1/N) − 1)` ≈ 9.31 against the clairvoyant `−ln(Pfa)` ≈ 9.21 → CFAR
loss ≈ **0.05 dB**; using our empirically calibrated α = 9.74 (which absorbs inter-cell correlation)
≈ **0.24 dB**. Small in absolute terms — but near the knee of the Pd-vs-SNR curve that is the same
order as the 1.3-point isolated-target gap. So H2 is **live, not dismissible**, and needs the
clairvoyant bound rather than a hand-wave.

## Milestone 19 — Radar re-specified to match CARRADA  🔄 STEP 1 OF 7 DONE
Full derivation: `docs/radar_config_carrada.md`.

**Why:** access to CARRADA (real 77 GHz automotive RD maps with per-cell annotations, recorded on
a closed test track) makes an external-validity chapter possible — but only if a target *looks the
same* in simulation and in the real data. At the old settings it would not: 0.50 m vs 0.195 m range
bins means a target spreads over a different number of cells, so any transfer failure would be a
trivial scale artifact rather than a finding.

| | Old | **New** |
|---|---|---|
| Bandwidth B | 300 MHz | **768 MHz** |
| Chirp time Tc | 32 µs | **72.4 µs** |
| Chirps M | 128 | **64** |
| Samples N | 512 | 512 |
| RD map | 256 × 128 | **256 × 64** |
| Range/bin, span | 0.50 m, 128 m | **0.195 m, 50 m** |
| Velocity/bin, span | 0.475 m/s, ±30.4 m/s | **0.420 m/s, ±13.44 m/s** |

Verified against `derive_radar_params`: every derived quantity matches CARRADA to 3 decimal places.

**⚠️ Finding — CARRADA's published parameter table is internally inconsistent.** Its quoted 4 GHz
sweep bandwidth contradicts its own stated 0.20 m range resolution under the range-resolution
formula the same paper gives (`c/2B` → 0.0375 m), and would imply a 9.6 m maximum range against the
stated 50 m — while their own Fig. 1 shows a car at 33 m. The angle entry proves the pattern: the
quoted "FFT resolutions" are **bin spacings** (180°/256 = 0.70°, against ~14° physical for an
8-element array). B was therefore derived from the *bin geometry*: 768 MHz reproduces 0.195 m bins
across 50 m in 256 bins. The velocity side is fully self-consistent and cross-validates the rest.

**Consequences logged, not yet actioned:**
- **Processing gain drops 3.01 dB** (coherent 48.16 → 45.15 dB; post-window ≈ 40.4 dB expected).
  `calibration.py` derives it analytically so cell SNR self-corrects — but the old 43.38 dB figure
  must not be carried forward.
- **CFAR window must be re-tuned.** Doppler axis halves; a 21×21 window would cover 33% of it and
  leave only 44 of 64 Doppler bins testable. A non-square window may be needed, which `CfarConfig`
  does not yet support.
- **CNN dilations follow the window**, not the reverse — the 21×21 receptive-field match is the
  basis of the controlled experiment, so it is re-derived *after* the geometry sweep.
- **Scenarios must be re-specified in metres**, not bins: 5 bins was 2.5 m, it is now 0.98 m.

**Superseded, kept as record:** `docs/results_model0.md` and `docs/results_stratified.md` are
banner-marked — their measurements are from the old sensor. The *methods* carry over unchanged;
only the numbers are re-run.

**Planned order:** 1 ✅ radar params → 2a ✅ PSF → 2b clutter rewrite → 2c scenario configs →
3 small debug dataset → 4 CFAR debug → 5 CFAR calibration → 6 window/geometry decision →
7 full generation.

### Step 2a — point-spread function measured (`scripts/measure_psf.py`)
**Claim tested and confirmed: mainlobe width *in bins* is set by the Blackman window, not by the
radar's resolution.** Old and new sensors measure identically, so the geometry reasoning carries
over even though every physical number changed.

| Worst case (off-bin ½) | OLD | NEW |
|---|---|---|
| −3 dB width | 1.63 / 1.64 bins | 1.62 / 1.65 bins |
| −30 dB half-width | 2 bins | **2 bins** |
| −40 dB half-width | 3 bins | 3 bins |

→ **guard = 2 still correct.** It covers the −30 dB spread exactly. Guard 3 would buy the −40 dB
skirt but costs 2 more bins of the scarce Doppler axis; the old sweep already showed guard 2 vs 3
moves Pd < 1%. Confirm in step 6.

**Resolution limit measured directly** (Rayleigh-style: saddle ≥ 3 dB below the weaker peak, and
the weak response must be a genuine local maximum):

| Axis | Equal targets | Weak −6 dB | Weak −12 dB |
|---|---|---|---|
| Range | 2.00 bins (**0.39 m**) | 3.00 bins (**0.59 m**) | 3.00 bins (0.59 m) |
| Doppler | 2.75 bins (**1.16 m/s**) | 2.75 bins | 2.75 bins |

→ sets the scenario boundaries: `unresolved` below ~0.39 m, `masking` above ~0.59 m (where a weak
target genuinely has its own peak). Because separation will be *swept* rather than fixed, these are
the expected boundaries to verify from data, not assumptions baked into generation.

*Two measurement bugs found and fixed while writing the script: a null-to-null search that walked
to the map edge for off-bin targets (no exact nulls exist there), and a saddle test that was
degenerate below 2 bins separation. Both produced physically impossible answers — 1-bin resolution —
which is how they were caught.*

### Step 2b — clutter rewritten: heavy-tailed + CNR-calibrated (`scripts/verify_clutter.py`)
Closes two items that had been logged as known gaps since the data-quality audit.

**1. `strength_db` → `cnr_db`.** The old field was an *absolute* amplitude scale, which is why it
had to be raised to ~45 before forming a visible ridge — its meaning depended on the requested SNR.
It is now a clutter-to-noise ratio, calibrated against the measured noise floor by
`calibration.clutter_scale_for_cnr` (empirical, cached, same pattern as the cell-SNR calibration).
Old configs raise a `ValueError` explaining the change rather than silently mis-scaling.

Measured accuracy: **within 0.2–0.3 dB** for CNR ≥ 15 dB. The +1.27 dB at CNR 5 dB is not error —
the measurement includes thermal noise, and 10·log10(1 + 10^−0.5) = +1.19 dB accounts for it.

**2. Compound-Gaussian clutter.** Each RD cell is `texture × speckle`; speckle is Rayleigh for free
(coherent sum of random-phase contributions), and `distribution` sets the *texture*:

| Setting | Fitted Weibull c | p99/p50 |
|---|---|---|
| `rayleigh` (no texture) | **2.10** | 2.49 |
| `weibull` shape 2.0 | 1.68 | 3.24 |
| `weibull` shape 1.5 | 1.54 | 3.79 |
| `weibull` shape 1.2 | **1.40** | 4.42 |
| `k` ν = 4.0 | 1.99 | 2.67 |
| `k` ν = 1.5 | 1.82 | 3.09 |

`rayleigh` reproduces the old behaviour (audit measured 2.03), so the baseline is preserved and the
tail is now dialable down to 1.40. Off-ridge cells sit at −0.01 dB: clutter stays in the ridge.

**Three modelling errors found by the verification, each invisible without it:**
1. Heavy tail applied per fast-time *sample*. A range bin is a tone spanning all N samples, so the
   range FFT averaged 512 draws and Gaussianised it — every distribution fitted c ≈ 1.97. Fixed by
   synthesising the range profile in the range-bin domain and inverse-transforming.
2. Band-limiting leaves only 2·width+1 of M Doppler bins, so each column's surviving power was
   itself a high-variance draw acting as a second texture. Fixed by per-column normalisation.
3. Drawing a Rayleigh *texture* double-counted the speckle — the `rayleigh` baseline came out at
   c = 1.62, heavier-tailed than the baseline it was meant to define. Fixed by making texture
   constant for `rayleigh`, which is the correct compound-Gaussian formulation.

### 🔔 REMINDER FOR STEPS 4–5 (CFAR debug + calibration) — the sweep analyses
Two quantities are now **swept per frame and recorded in metadata**, deliberately, so that two
analyses come almost for free once CFAR runs. Do not forget to do them:

1. **Pd vs target separation** (`masking` stratum, separation swept 0.40–2.50 m).
   Gives the proposal-required *close-target separation* ablation, and lets the resolution limit
   and the masking boundary emerge from data rather than being asserted. It is also immune to the
   CFAR window changing in step 6 — the cut is made at analysis time, not baked into generation.

2. **Measured Pfa vs clutter shape** (`clutter`, `clutter_edge`, `mixed`; Weibull shape swept
   1.1–2.0, `cnr_db` swept too).
   CA-CFAR's α assumes exponentially distributed cell power. Heavy tails break that assumption, so
   this measures *how fast the false-alarm rate drifts away from design as clutter gets spikier* —
   and whether the learned detector holds its rate better. That is the mechanism behind the whole
   clutter result, and it doubles as the proposal's *clutter strength* ablation.

⚠️ **Requested ≠ measured shape.** The chain always adds its own Rayleigh speckle, so a requested
texture of 1.5 is observed as ≈1.54 and 1.2 as ≈1.40. Quote the **measured** value in the report;
both are in each frame's metadata.

### Step 2c — scenario configs re-specified in physical units
`configs/bench2_debug.yaml` (80 frames, 10 per stratum) is the trial config for steps 3–6.

- **Separations in metres, not bins**, and *swept*. `close_pair` no longer snaps targets to bin
  centres either — snapping removed scalloping variation and made these targets on-grid while
  `random` targets were off-grid, an inconsistency the data-quality audit would have flagged.
- **Targets confined to the CFAR-testable region** (5–45 m, ±8 m/s). Border exclusion at
  half-width 10 leaves only ~2–48 m and −9.2…+8.8 m/s; the old ±12 m/s would have placed targets
  in the dead border where nothing is tested.
- **`ref_range_m` 50 → 25.** At a 50 m maximum range, referencing the radar equation to 50 m made
  every target closer than reference and therefore amplified.
- **New `clutter_edge` stratum**; `multipath` folded into `mixed`; `unresolved` test-only.
- Parser now rejects bin-valued separations and `strength_db` with an explanatory error rather
  than silently mis-scaling.

### Steps 3–4 — debug dataset generated; CFAR verified on the new map
`configs/bench2_debug.yaml` → 80 frames, 10 per stratum. Scripts: `scripts/preview_strata.py`,
`scripts/cfar_sanity.py`.

**Step 3 verification.** Counts exact, no non-finite values, no targets in the untested border,
difficulty as predicted (median 13.7 dB cell SNR). Alignment 91.7% for targets ≥15 dB, and the
breakdown explains itself: **100% on isolated / multi / mixed**, lower only on masking, unresolved,
clutter — i.e. where interference is the point. Regenerating three failing frames with noise and
clutter off put every label exactly on the clean-map peak, proving the labels are correct.

⚠️ **The preview sheet caught what the numbers missed.** `clutter_edge` targets were sitting
*inside* the ridge, not beside it. The nominal ridge half-width is 1.26 m/s, but measured, the
Doppler window smears it to 13.6 dB at 1.68 m/s and 2.1 dB at 2.10, reaching the noise floor only
by 2.5 m/s. The 1.5 m/s lower bound therefore put ~⅓ of that stratum inside the clutter skirt,
duplicating `clutter`. Raised to **2.5–4.2 m/s**; that stratum's alignment went 72.7% → **100%**.

**Step 4 — CFAR sanity.** Valid region **236 × 44** as predicted (63.4% of the map; the Doppler
axis keeps 69%). Noise estimates match theory to under 1%:

| | measured | theory |
|---|---|---|
| CA | 1.001 | 1.000 (unbiased ring mean) |
| GO | 1.090 | 1.090 (max of two half-means) |
| OS | 1.396 | 1.386 (= ln 4, the 0.75-quantile of exponential noise) |

Runtime: CA 1.3 ms, GO 17.9 ms, OS 125.9 ms per frame.

⚠️ **INPUT TO STEP 6 (not a result) — square windows spread clutter contamination ~3× wider than
the clutter.** Stated carefully, because it is easy to over-read: this is a *proxy* measurement
taken at a geometry that is about to be replaced. Contamination is not itself the quantity that
matters — a raised threshold where there are no targets costs nothing. Pd and measured Pfa are
what matter, and step 6 optimises those directly on the tuning split.

What survives regardless of what step 6 picks is the *structural* point: the sweep covered train
4–10, and **no square window avoids this on a 64-bin Doppler axis**. The ridge occupies 9 Doppler
bins; measured against a noise-only reference, the fraction of the testable Doppler axis whose
threshold is raised >3 dB:

| train | half-width | usable bins | contaminated | **clean** |
|---|---|---|---|---|
| 4 | 6 | 52 | 21 | 31 (60%) |
| 6 | 8 | 48 | 23 | 25 (52%) |
| **8 (current)** | 10 | 44 | 27 | **17 (39%)** |
| 10 | 12 | 40 | 31 | 9 (22%) |

This did not arise on the previous sensor, whose 128-bin Doppler axis absorbed it.

**What this changes about step 6 — it sets what to sweep, not what to choose:**
- Include training rings **smaller than the old default**. The previous sweep bottomed out around
  8 and found Pd flat below it; that will not hold when the axis is half as long.
- Include **asymmetric** windows — wide in range, where the axis is long and unstructured, narrow
  in Doppler, where the ridge sits and the axis is short. This was never previously on the table.
  `CfarConfig` takes one guard/train pair for both axes and would need extending.

Non-square is a **hypothesis to test, not a conclusion**. If a small square window wins on Pd at
matched measured Pfa, it wins. Without this measurement the sweep would probably have covered
6/8/10 and never questioned squareness at all — that is its whole value.

**Prediction recorded before measuring** (step 6 / the CFAR-vs-CFAR check): because the ridge
contaminates a wide Doppler band, GO — which splits its window into leading and lagging halves
*along Doppler* and takes the max — can lean on whichever half is clean, and OS's rank statistic
ignores a ridge occupying a minority of the ring. CA averages everything and has no defence. So
**CA should suffer most in `clutter_edge`, with the GO/OS gap larger than on the old sensor**. If
GO does not win there, something is wrong and it needs explaining.

*Two script bugs found and fixed: the valid-region check sampled row/column 0, which is border and
therefore all-invalid, reporting 0×0 for a correct mask; and the first contamination measurement
used the profile's own median as baseline, so wider contamination dragged the baseline up and hid
itself, producing the impossible result that the widest window had zero contamination.*

### Step 5 — thresholds calibrated; clutter drift measured (`scripts/cfar_calibrate.py`)
Geometry guard 2 / train 8 (416 reference cells). These α values are recomputed for whatever
geometry step 6 selects — what is established here is that the *procedure* works at this sensor,
and the shape of the clutter-drift relationship.

**Calibration.** Empirical CA sits 3.6–3.7% above the closed form (9.647 vs 9.313 at Pfa 1e-4) —
the window's inter-cell correlation, which the formula does not model, so the empirical value is
the more accurate one here.

*Internal consistency check:* OS's noise estimate is 1.396× the true level (step 4), so its α
should be CA's ÷ 1.396 = 6.91. Calibrated: **6.977**. The two independent measurements agree.

**Achieved Pfa on independent noise** (calibrated on seed 0, verified on seed 9000):

| design | CA | GO | OS |
|---|---|---|---|
| 1e-3 | 1.00e-3 | 9.95e-4 | 1.00e-3 |
| 1e-4 | 1.13e-4 | 1.15e-4 | 1.19e-4 |

Exact at 1e-3; 13–19% high at 1e-4, which is finite-sample — 150 frames put only ~156 cells above
the 1e-4 quantile. Harmless for the ROC-based comparison, which matches at *measured* Pfa.

⚠️ **Clutter destroys the design false-alarm rate — by 60–125×.** Using the noise-calibrated α:

| clutter | CA | GO | OS | CA ÷ design |
|---|---|---|---|---|
| none (thermal only) | 1.12e-4 | 1.25e-4 | 1.17e-4 | 1.1× |
| rayleigh, 15 dB | 6.34e-3 | 3.57e-3 | 1.77e-2 | **63×** |
| weibull c=2.0, 15 dB | 9.75e-3 | 6.22e-3 | 2.65e-2 | 98× |
| weibull c=1.5, 15 dB | 1.10e-2 | 7.30e-3 | 3.06e-2 | 110× |
| weibull c=1.1, 15 dB | 1.25e-2 | 8.67e-3 | 3.59e-2 | **125×** |
| weibull c=1.5, 25 dB | 1.26e-2 | 8.20e-3 | 3.39e-2 | 126× |
| k ν=1.5, 15 dB | 8.94e-3 | 5.65e-3 | 2.38e-2 | 89× |

Two separable effects. **The clutter edge alone costs 63×**, before any tail — that is the window
straddling the ridge boundary, and it happens even for Rayleigh clutter. **Heavy tails then roughly
double it**, 63× → 125× as the Weibull texture goes 2.0 → 1.1. So spikiness matters, but the
geometry matters more.

**This is the measurement that justifies matching at measured rather than design Pfa** (attribution
Step A1). A comparison run at "design 1e-4" would in fact be comparing CA at 1.2e-2 against GO at
8.7e-3 against OS at 3.6e-2 — three different operating points, which is precisely the error that
produced the misleading Milestone 8 result.

**Prediction scored: half right.**
- ✅ **GO best**, as predicted — it takes the max of two half-rings split along Doppler, so it can
  lean on whichever half is clear of the ridge.
- ❌ **OS worst, not best.** I predicted a rank statistic would ignore a minority ridge. Wrong, and
  backwards: OS takes the 0.75-quantile precisely to be robust against interfering *targets*, but
  at a clutter edge that quantile lands in the *clean* part of the ring, so it **under**-estimates
  the noise on the clutter side and the threshold falls too low. Robust to point interferers,
  vulnerable to clutter edges — which is exactly the complementarity story, and the ordering
  (GO < CA < OS) reproduces the old sensor's result. The earlier data already said so; I should
  have checked it before predicting.

### Step 6 — CFAR window selected: **21 × 9** (guard 2/2, train 8/2)
Selected on `bench2_tune` (400 frames, seed 91001, disjoint from every evaluation split), by Pd at
matched **measured** Pfa = 1e-4, per condition. Scripts: `cfar_geometry_sweep.py`,
`cfar_geometry_percondition.py`. `CfarConfig` defaults updated.

**Why a tuning split at all:** it removes open asymmetry #4. The network is fitted to the data
distribution; CFAR previously inherited a geometry chosen for a different sensor on a different
scenario. Both sides now adapt, neither sees the test split. No paper in the 19-paper corpus
re-tunes CFAR at all.

**Pooled sweep, 10 geometries.** Pd rose monotonically as the Doppler window narrowed —
21×21 → 0.306, 21×11 → 0.429, 29×9 → 0.444 — while range width barely mattered (21×9 0.441 vs
29×9 0.444 vs 37×11 0.431). ⚠️ But Pd tracked *usable Doppler bins* almost perfectly, which is
what a measurement artefact looks like, so it was not taken at face value.

**Per-condition breakdown separated the two explanations.** A pooling artefact would shift every
condition uniformly; a real clutter mechanism would concentrate the gain in clutter-bearing
conditions. Result, 21×21 → 21×9:

| condition | 21×21 | **21×9** | gain |
|---|---|---|---|
| clutter_edge | 0.094 | **0.390** | **+0.296** |
| mixed | 0.196 | **0.381** | **+0.185** |
| multi | 0.586 | 0.592 | +0.006 |
| masking | 0.501 | 0.504 | +0.003 |
| unresolved | 0.470 | 0.471 | +0.001 |
| clutter | 0.170 | 0.153 | −0.016 |
| isolated | 0.654 | 0.629 | −0.025 |

Concentrated in the clutter-bearing conditions and **zero on isolated** → the mechanism is real,
and it is the one step 4 measured: a narrow Doppler ring stays clear of the ridge.

**Why 21×9 and not the pooled winner 29×9.** 29×9 scored marginally higher pooled but cost
**−0.041 on masking**. Diagnosed as its *range* half-width widening 10 → 14, pulling more
interferers into the ring — nothing to do with Doppler. Tested directly: 21×9 keeps the range
half-width at 10 and recovers masking (+0.003) while retaining 98% of the clutter gain. The pooled
metric would have chosen the worse window; the per-condition breakdown is what caught it.

**Costs, accepted:** isolated −0.025 and clutter −0.016. The isolated loss is textbook CFAR loss
from fewer reference cells (416 → 164; analytic loss 0.05 → 0.12 dB). The clutter loss is the
mirror of the clutter_edge gain — for a target *inside* the ridge, a narrow ring sits entirely in
clutter and over-estimates the noise, where a wide ring reaches clean cells.

**Complementarity at 21×9** (Pd at matched measured Pfa):

| condition | CA | GO | OS | best |
|---|---|---|---|---|
| masking | 0.484 | 0.494 | **0.534** | **OS** |
| mixed | 0.390 | **0.400** | 0.354 | GO |
| clutter | 0.145 | **0.167** | 0.147 | GO |
| isolated | **0.632** | 0.623 | **0.632** | CA / OS |
| multi | **0.595** | 0.590 | 0.591 | CA |
| clutter_edge | 0.390 | 0.390 | 0.390 | 3-way tie |

The ordering survives — **OS best at masking, GO best in clutter, CA best on clean** — but the
spreads are small (0.005–0.050), so only masking (+0.050 for OS) is clearly outside sampling noise
at 400 frames.

⚠️ **Prediction scored, and it was about the wrong metric.** I predicted GO would win
`clutter_edge`. At matched Pfa it is a **three-way tie** (spread 0.001). GO's clutter advantage is
real but expresses itself as a *lower false-alarm rate at a fixed threshold* (step 5: GO 8.7e-3 vs
CA 1.3e-2), and matching the measured rate neutralises exactly that by construction. **Which method
looks best depends on the matching criterion** — a methodological point worth reporting, and a
direct illustration of why the attribution phase exists.

### Step 6c — CNN receptive field re-matched to 21 × 9
The receptive-field match is the entire basis of the controlled experiment — identical context,
learned vs hand-derived decision rule — so it follows the window rather than the other way round.

`TinyCNN` now takes **per-axis dilations**: range `(1, 2, 3, 4)` → RF 21, Doppler `(1, 1, 1, 1)` →
RF 9, with `padding = dilation` per axis so the output stays at full resolution.

**Verified, not assumed:**

| | |
|---|---|
| CFAR window | 21 × 9 |
| Analytic RF | 21 × 9 |
| **Measured RF** (backprop from one output pixel) | **21 × 9** ✅ |
| Parameters | 5,041 — unchanged |
| Forward | (2,1,256,64) → (2,1,256,64), finite |
| Gradients | all 14 tensors receive gradient |

Parameter count is unchanged because the kernels and channels are unchanged — only the *spacing*
of the taps differs per axis. The range dilations still share no common factor (Hybrid-Dilated-
Convolution guidance, avoiding gridding); the Doppler dilations are all 1, which is dense by
construction and cannot grid.

Stale references updated in `docs/dl_design.md` and two notebook cells — one of which formatted
`receptive_field` as a scalar and would have printed incorrectly now that it returns a tuple.

### Step 6d — audit of steps 1–6c, and the NMS investigation
Everything re-verified end-to-end at the new sensor; one real limitation found and documented
rather than tuned away.

**Pipeline health (all pass):** dataset returns (1, 256, 64); per-frame normalisation exact
(median 0.0000, std-of-medians 0.0000); forward preserves shape; 5,041 params; all 14 gradient
tensors live; `dl/infer.py` still imports CFAR's own `all_peaks`, so both detectors share one NMS.

⚠️ **Class imbalance changed: 1:1563 → 1:888.** The map halved while target counts stayed similar.
`pos_weight=200` was tuned for the old ratio. Per `dl/train.py`'s own note the exact value matters
less than intuition suggests — the ROC sweeps the threshold, so `pos_weight` mainly governs
trainability and where the *untuned* operating point sits — but it should be revisited at retrain.

**The NMS finding.** `nms_size=5` gives an 11×11 window = **2.15 m in range**, far wider than the
~2-bin mainlobe, and it suppresses genuinely resolvable neighbours. On clean, noiseless maps only
**38%** of masking pairs keep both targets — **5.9%** for pairs under 1 m. That caps the masking
stratum near Pd 0.69.

**But shrinking it is worse, and I was wrong to call it a bug.** NMS is not only merging targets,
it is also *filtering noise*. At nms=2 there are ~4× more candidate peaks per noise frame (427 vs
113), which at matched measured Pfa forces the threshold up and costs Pd everywhere — including
`isolated`, where nothing can be merged. Swept at 21×9, Pd at matched Pfa on `isolated`:

| nms_size | 1 | 2 | 3 | 4 | **5** | 7 |
|---|---|---|---|---|---|---|
| window | 3×3 | 5×5 | 7×7 | 9×9 | **11×11** | 15×15 |
| isolated Pd | 0.462 | 0.291 | 0.293 | 0.372 | **0.521** | 0.638 |
| detections per target | **1.20** | 0.96 | 0.94 | 0.86 | 0.80 | 0.80 |

`nms_size=1` scores well but is **disqualified**: 1.20 detections per target means a target gets
several attempts and the evaluator forgives all but one inside its guard region — the same
free-attempts failure the plateau fix removed in Milestone 16. `nms_size=7` edges out 5 but spans
2.9 m, which would merge two adjacent vehicles.

**Kept at 5.** The masking cap is recorded as a limitation, and `nms_size` is added to the
attribution sweep because it interacts with the evaluator's guard-region forgiveness — exactly the
kind of protocol choice that phase exists to attribute.

*Two of my own errors caught here: a `replace_all` that matched only `nms_size: int = 5,` (with
comma) and so silently missed `all_peaks`, spotted because the re-run produced byte-identical
numbers; and calling the NMS width a bug before measuring what shrinking it cost.*

### Step 7 — full datasets generated and verified (`scripts/verify_splits.py`)
14,900 frames, 3.24 GB. **All gates pass.**

| split | frames | strata | seed | role |
|---|---|---|---|---|
| `bench2_train` | 8,400 | 7 × 1200 | 92001 | network training |
| `bench2_val` | 1,600 | 8 × 200 | 93001 | overfitting check |
| `bench2_test` | 3,200 | 8 × 400 | 94001 | **the reported numbers** |
| `bench2_tune` | 400 | 8 × 50 | 91001 | CFAR geometry (already used) |
| `bench2_snr_sweep` | 1,300 | 1 condition | 95001 | Pd-vs-SNR diagnostic |

**Leakage: zero shared frames across all ten split pairs.** Checked by **hashing frame contents**,
not by comparing seeds — trusting the seeds would mean trusting the thing under test.

**Alignment 92.3–93.6%**, and the per-condition breakdown is the same signature as before:

| isolated | multi | clutter_edge | mixed | masking | unresolved | clutter |
|---|---|---|---|---|---|---|
| **100%** | **100%** | 97% | 94% | 90% | 90% | 77% |

Perfect where targets are isolated; degraded only where interference is the point of the condition.
`clutter` at 77% is the lowest, which is expected now that clutter is heavy-tailed — spikes beat
weak targets. **`bench2_snr_sweep` is 100.0%**, which is the cleanest possible confirmation that
the labelling is correct: single target, no clutter, nothing to interfere.

**Difficulty after the +2 dB lift:** median 14.9–16.2 dB, targets below 5 dB now **13.3–16.2%**
(was 21.1%). The sweep sits at median 11.2 dB with 25% below 5 dB, which is by design — it starts
at 0 dB so the curve begins near Pd = 0.

**Swept parameters recorded** as intended: CNR 5.0–25.0 dB, Weibull shape 1.10–2.00.
**Zero targets in the untested border** in any split. **Zero non-finite frames.**

Per frame: `rd_map_mag` 65 KB + `rd_map_complex` 131 KB + `label_mask` 16 KB + labels/metadata.
The complex map is what makes the proposal's magnitude-vs-I/Q ablation possible without a third
regeneration.

*Note: the verifier reports `bench2_snr_sweep` as "0 strata UNEVEN" — an artefact of the check
assuming a `scenarios:` block. The sweep is deliberately single-condition; not a failure.*

### Step 7b — deeper dataset audit (`scripts/audit_datasets.py`)
`verify_splits.py` answers *"is anything broken?"*. This answers *"is it what we asked for?"* — the
failure mode that passes every structural check and still invalidates the study, because a
mis-specified stratum produces perfectly valid frames of the wrong scenario.

**Every stratum matches its config.** Sampled across all five splits:

| stratum | targets | range (m) | \|v\| (m/s) | separation (m) | clutter | spec |
|---|---|---|---|---|---|---|
| isolated | 1 | 8.1–44.6 | 0.01–7.99 | — | 0% | ✅ |
| masking | 2 | 8.0–42.3 | 0.01–7.32 | **0.41–2.49** | 0% | ✅ 0.40–2.50 |
| unresolved | 2 | 8.4–40.0 | 0.04–6.91 | **0.05–0.35** | 0% | ✅ exact |
| clutter | 1–3 | 8.1–45.0 | **0.01–1.20** | — | 100% | ✅ <1.2 |
| clutter_edge | 1–2 | 8.0–45.0 | **2.50–4.20** | — | 100% | ✅ exact |
| multi | 4–5 | 8.0–45.0 | 0.00–8.00 | — | 0% | ✅ |
| noise | 0 | — | — | — | 0% | ✅ |
| mixed | 0–5 | 8.1–44.9 | 0.01–8.00 | — | 100% | ✅ |

**Other checks, all pass:**
- `radar_config_used.json` in every split matches the code exactly.
- **1047/1047 sampled labels** have `range_bin`/`velocity_bin` consistent with `range_m`/
  `velocity_mps` — the bin indices are not drifting from the physical coordinates.
- Complex map orientation matches magnitude in every split. The residual |‖z‖ − mag| of ~3e-2
  looked large until checked in *relative* terms: **2.18e-07, about 2× float32 epsilon**, on values
  reaching 13,335. Both arrays are stored at float32 precision, so this is rounding, not error.
- All 16 metadata fields present in every split.

⚠️ **Class balance is now 1:936–1:973** (was 1:1563 on the old sensor), consistent across
train/val/test. `pos_weight=200` was tuned for the old ratio — revisit at retrain.

*Note: the audit's "separation" column is meaningless for `clutter`, `clutter_edge` and `mixed`,
which place targets randomly rather than as pairs; the figure there is just the distance between
two arbitrary targets.*

## Milestone 20 — Data-quality re-audit at the new sensor  ✅

Run 11 Aug 2026, before retraining, because the original audit (Milestone 11) was measured on the
old sensor and on datasets that have since been deleted. Nine criteria from
`docs/data_quality_criteria.md`, measured on the frozen `bench2_*` splits; new script
`scripts/audit_data_quality.py` covers the statistical criteria the existing scripts did not.

**Eight of nine PASS. The one that previously failed now passes.**

| Criterion | Then (M11) | Now | Evidence |
|---|---|---|---|
| 3 statistical realism | PARTIAL — clutter c = 2.03 | **PASS** | noise KS 0.006 vs exponential; clutter Weibull **c = 1.51 / 1.56** (test/train), p99/p50 ≈ 3.8 |
| 3 radar equation | applied | **verified** | `amplitude_db` on 40log10(R/25): slope **−1.003**, r −0.92; residual exactly 4.0–16.0 dB = the config window |
| 4 difficulty | PASS | **PASS** | 48–51% of targets in theory Pd 0.1–0.9; 13–16% below, 33–39% above |
| 6 shortcuts | PASS | **PASS** | off-grid std 0.290 (0.289 ideal), KS 0.007; corr(frame stat, num_targets) ≤ 0.21; 8400/8400 unique seeds |
| 2 labels + leakage | — | **PASS** | 0 shared frame hashes between any split pair; 0 non-finite frames; 0 targets in the untested border |

**One thing to carry forward:** **criterion 7 (benchmark neutrality) is the live action.**
`bench2_test` is eight *equal* strata, 37.5% of which are deliberately adversarial. Per-condition is
the headline; a pooled Pd needs its composition printed beside it.

Class balance measures 1:943–1:1027 (200-frame sample per split) against step 6d's 1:888 — two
samples of one quantity, i.e. agreement. What matters for `pos_weight` is that it is ~1:900–1000,
not the 1:1563 still quoted in `dl/train.py`'s docstring.

Two flagged correlations were checked and are **by design**: corr(range, amplitude_db) = −0.90 is
the radar equation, and the peaked Doppler histogram is the `clutter` / `clutter_edge` strata.
The `bench2_snr_sweep` "0 strata" report is the known step-7 checker artefact, not a new finding.
Full write-up in `docs/data_quality_criteria.md`.

### Evaluation-protocol review (same session, before retraining)
Read the whole scoring path — `src/eval/detection.py`, `src/cfar/detect.py`, `src/dl/infer.py`,
`src/dl/stratified_eval.py`, `docs/benchmarking_conventions.md`. The architecture holds: both
detectors go through the same `all_peaks`, the same `valid` mask, the same Hungarian matching in the
same ±1 gate. Three items, only one of them new:

1. **`stratified_eval.evaluate_stratified()` still matches Pfa by *design*, not by measurement** —
   it Monte-Carlo calibrates each method's threshold on pure noise and reports whatever measured Pfa
   results. This is Milestone 16's open asymmetry #3, still open in this one file. The CFAR side was
   fixed (`cfar_geometry_percondition.py` does per-condition ROC + interpolation at matched measured
   Pfa), so the DL numbers would otherwise be produced by a *different protocol* from the CFAR
   baseline they are placed beside. Step 5 measured why this matters: clutter moves the achieved rate
   60–125× off design, by different factors per method.
2. **`pd_vs_snr()` takes no score threshold** and drops the score column via `_as_points`. Fed
   `all_peaks` output — which is what every other call site produces — it scores at "detect
   everything". Pd-vs-SNR-against-theory is an earmarked report figure, so the threshold should be a
   required argument. *(New; no prior record of it.)*
3. **Sigmoid saturation** compresses the ROC's low-Pfa tail and leaves `_collapse_plateaus` picking
   an arbitrary cell from a tied region. Scoring on **logits** (a monotone transform — Pd and Pfa
   unchanged) removes the cause rather than the symptom. Deferred to the attribution phase, where
   `nms_size` is already slated for a sweep and the two interact. *Not urgent: the duplicate-detection
   failure this used to cause was closed in Milestone 16.*

`docs/benchmarking_conventions.md` is stale in three places and is the document earmarked for the
report: Part 1's processing-gain warning (now measured — frame metadata records 40.33 dB, against
the ~40.4 dB predicted in Milestone 19), Part 2's "fixed Pfa (e.g. 1e-3)" rule (contradicts the
matched-**measured**-Pfa practice established in step 5), and Part 3's review of the seven deleted
`*_debug` configs.

### Milestone 20b — both evaluator fixes applied and verified

**`pd_vs_snr()` now takes a keyword-only, required `threshold`**, applies it, and raises if handed
unscored `(N, 2)` detections. Verified: threshold above the target's score → Pd 0.00, below → 1.00.

**`evaluate_stratified()` now matches at measured Pfa**, per condition: run each detector once,
sweep its own score distribution on the condition being scored, interpolate Pd where the achieved
per-cell rate equals the target. Closes asymmetry #3. The noise-calibrated thresholds are still
returned under a `"design"` key, since design-vs-measured is itself a reported finding.

**Three bugs found while validating it, none of which a "does it run" check would have caught.**
Each was exposed by a physical cross-check, not by the code failing:

1. **Tied Pfa.** On clean strata a whole span of thresholds gives the same measured Pfa (often
   exactly 0), and `np.interp` on duplicated x picks an arbitrary partner. Collapse ties to their
   lowest threshold / highest Pd first.
2. **Grid too coarse in the tail.** 60 uniform quantiles put ~2 points in the top 1%, and the
   operating point lives there — the sweep stepped straight over Pfa 1e-4 and the recovered
   threshold landed in the gap, 4× too high on `isolated` (40.3 against a design 10.2). Pd
   tolerates that gap; a threshold does not, and the false-alarm columns are evaluated *at* it.
   Grid is now dense in the upper tail.
3. **Linear interpolation on an exponential curve.** Pfa falls roughly exponentially with the
   threshold, so linear bracketing overshot and parked the operating point at 3.5–8.7e-5 against a
   1e-4 target. Interpolate against log Pfa. Zero-Pfa rows then need the **half-count floor**
   (0 false alarms over N cells means "below 0.5/N", not "zero") — dropping them made all of
   `multi` unscoreable, because with 4–5 targets per frame every surviving peak sits inside a guard
   region and Pfa steps from 1.2e-4 straight to 0.

**Independent validation: the recovered threshold reproduces the calibration it never saw.** On the
`noise` stratum the swept threshold lands at **CA 10.173** against the Monte-Carlo design value
**10.163** — 0.1%, from a completely separate route. In clutter it is forced up to 22.4, which is
the 60–125× drift step 5 measured, seen from the other side.

**Reproduces the step-6 baseline to ≤0.017**, with every ordering intact — OS best at masking
(0.539), GO best in clutter (0.184), CA/OS best on isolated. The residual differences are the
grid and interpolation improvements above; `scripts/cfar_geometry_percondition.py` still uses the
coarse grid and linear interpolation. Its **window decision stands** — the 21×9 gains were +0.185
to +0.296, an order of magnitude above this discrepancy — but `evaluate_stratified` is now the
better implementation and should be the single path for 2.1.

## Milestone 21 — CFAR baseline on the FULL test split (task 2.1)  ✅
`scripts/cfar_test_baseline.py` → `results/cfar_test/`; presented in `notebooks/cfar_baseline.ipynb`.

**Pre-registered.** Measured before the network exists, so it cannot be unconsciously tuned
against. All 3,200 frames, 6,163 targets, 21×9 window from `CfarConfig` defaults, matched
**measured** Pfa 1e-4 per condition. Run once, not iterated.

| condition | CA | GO | OS | best | spread |
|---|---|---|---|---|---|
| multi | **0.687** | 0.680 | 0.682 | CA | 0.007 |
| isolated | **0.675** | 0.662 | 0.662 | CA | 0.013 |
| masking | 0.535 | 0.529 | **0.622** | **OS** | **0.093** |
| unresolved | 0.464 | 0.462 | 0.462 | — | 0.002 |
| clutter_edge | 0.397 | 0.415 | **0.449** | **OS** | **0.052** |
| mixed | 0.408 | **0.426** | 0.388 | GO | 0.038 |
| clutter | 0.166 | **0.171** | 0.163 | GO | 0.008 |

*(Final values, on the false-alarm-anchored threshold grid — see Milestone 21b. Measured Pfa is
9.81e-5–1.02e-4 on every row, so "matched at 1e-4" is literally true rather than approximately.)*

Runtime: CA 2.3 ms, GO 6.6 ms, **OS 57.8 ms** per frame. Localisation ~0.21 range bins / 0.10
Doppler bins for all three (bounded by the ±1 gate, so this says *where inside* the gate).

**Only two rows separate beyond sampling noise**, and both are OS: masking **+0.086** and
clutter_edge **+0.039**. isolated / multi / unresolved / clutter are ties at 400 frames each. GO
takes clutter and mixed by small margins, reproducing the step-6 ordering on the tuning split.

### Audited after the fact — three defects found, all now fixed

The `clutter_edge` result looked wrong (GO is *designed* for clutter edges), so the whole thing was
re-audited rather than accepted. The result stands, but the audit found three real defects:

1. **GO's threshold used the wrong cell count.** `alpha` was derived from `num_ref_cells // 2` = 82,
   but a half-ring holds **74** cells on the 21×9 window — the 16 cells of the centre Doppler
   column belong to neither half. Halving overstates the sample size and understates the threshold
   GO needs. Fixed with a counted `num_half_ring_cells` and an explicit `alpha_go()`. This only
   moves fixed-threshold results (matched-Pfa comparisons are immune by construction), but the
   fixed-threshold table is reported, and GO's false-alarm rate is the number most affected.
2. **The headline Pd was the interpolated value, not the measured one.** Interpolation's job is to
   locate the threshold; Pd should then be *measured* there. The two differed by up to 0.017,
   biased optimistic on the clutter strata where the ROC is steepest. Now measured at the threshold.
3. **Verified the operating point is real**, not just interpolated to: at each recovered threshold
   the achieved Pfa is 7.8e-5–1.06e-4 against the 1e-4 target, across all 21 rows.

### The `clutter_edge` question: GO does win it — on the metric GO exists for

Checked directly on a synthetic edge (unit noise, ridge 20 dB up). Three bins outside the ridge the
noise estimates are **CA 26.1, GO 56.7, OS 4.8**: GO is twice as conservative as CA, OS five times
less. That conservatism is GO's entire design goal — it stops the false-alarm rate exploding at an
edge — and it is paid for in detections.

False alarms per frame at the design threshold (**GO's metric**):

| condition | CA | GO | OS |
|---|---|---|---|
| clutter_edge | 11.25 | **5.26** | 7.00 |
| clutter | 10.52 | **4.88** | 6.41 |
| mixed | 10.30 | **4.95** | 6.75 |

**GO wins every clutter condition, by better than 2×.** And matching the measured Pfa erases exactly
that advantage by construction: forcing all methods to one false-alarm rate means GO's caution buys
nothing it is not already handed, leaving only its Pd cost. So OS leads the matched-Pfa table and GO
leads the fixed-threshold one — same detectors, same data, opposite rankings.

This restates step 6's finding on a full test split: **which method looks best depends on the
matching criterion**, which is a methodological result to report, not an inconsistency to resolve.
Both tables are now in the notebook, each labelled with the question it answers.

**The protocol validated itself.** The threshold each method needed, against its design α:

| | isolated / multi / noise / masking / unresolved | clutter / clutter_edge / mixed |
|---|---|---|
| CA | **1.05–1.07×** | **2.25–2.46×** |
| GO | 0.93–0.95× | 1.59–1.70× |
| OS | 0.76–0.78× | 1.97–2.26× |

On clean conditions CA lands within 7% of homogeneous-noise theory — theory works where its
assumptions hold. In clutter the same method needs **2.3× the threshold** to hit the same measured
rate. OS's steady 0.77× is the documented consequence of its α still being CA's first
approximation (its noise estimate is 1.396× the true level, so 1/1.396 = 0.72 — measured 0.77).

**Where the false alarms land.** At matched rate the *count* is equal by construction, so the
difference is location: in `clutter`, `clutter_edge` and `mixed`, **~100% of every method's false
alarms fall in the zero-Doppler ridge** (±8 bins ≈ 26% of the testable axis). On clean conditions it
is 28–31%, i.e. chance. That is the classical failure mode stated quantitatively, and it is the
specific thing a learned detector is expected to improve on.

### Milestone 21b — the ROC threshold grid was sampling the wrong place

Found by the user asking why the `multi` and `unresolved` ROC curves looked truncated.

**`unresolved` being flat is physics** — two targets in one mainlobe, one detection can match only
one truth, so Pd is capped near 0.5 whatever the threshold (0.45 → 0.49 across two decades). That
is the sanity check working.

**`multi` stopping at Pfa 7e-5 was real too**: with 4–5 strong targets per frame, above a ratio of
~175 *every* surviving peak is a genuine target — 455 detections, **zero** false alarms — and zero
cannot be drawn on a log axis.

**But the grid underneath was wrong.** It was built from quantiles of the whole score pool, dense in
the upper tail; on `multi` the top few percent of scores are all targets, so 40 of 79 grid points
landed where Pfa is identically 0 while the informative span went unsampled. The smallest non-zero
false-alarm count ever measured was **371** — nothing between 1 and 370 was evaluated, and the
operating point was interpolated across that hole.

`eval.detection.roc_thresholds()` now anchors the grid to the false alarms themselves: pool the
scores of peaks *not* within the gate of any truth, sort them, and place thresholds at geometrically
spaced ranks — one false alarm, two, three, five, … Using ground truth to choose *where to sample*
is not using it to score; every point is still measured by `evaluate_frame` in the ordinary way.

**Effect: Pd shifted by ≤0.010 and no conclusion changed**, which is itself reassuring about the
earlier numbers. What improved is the operating point — measured Pfa is now **9.81e-5–1.02e-4** on
all 24 rows, against a 7.9e-5–1.23e-4 spread before.

*Correction logged:* mid-investigation I claimed the top-scoring `multi` peaks were not all targets,
from a diagnostic that built a near-truth mask in regrouped frame order and indexed scores in the
original order. The two were misaligned; the claim was wrong, and a direct threshold test disproved
it.

## Milestone 22 — Why GO loses `clutter_edge`, and what it exposed about the window choice  ⚠️

Raised by the user: GO is *designed* for the clutter edge, so losing it at matched measured Pfa
looked wrong. Investigated on `bench2_tune` + `bench2_val` (never on test, so nothing here could
become a test-set fit). Scripts: `diagnose_go_clutter_edge.py`, `cfar_window_resweep.py`,
`cfar_true_edge_analysis.py`, `cfar_window_true_edge_sweep.py`.

**The answer: GO genuinely loses on Pd, and the margin GROWS once the data is cleaned up.**

### 1. The `clutter_edge` stratum is diluted, and the dilution depends on the window

The ridge measures **±4 Doppler bins** (>3 dB above floor: 20.5 dB at centre, 14.0 at offset 4,
2.4 at offset 5, gone by 6). `clutter_edge` targets sit at offsets **6–10 bins**, and the 21×9
window's training columns reach only ±3–4 bins from the cell under test. So a target at offset 6
is straddled and one at offset 10 is not:

| window | Doppler reach | targets genuinely at an edge |
|---|---|---|
| 21×7 | ±3 | **41.5%** |
| **21×9 (current)** | ±4 | **65.6%** |
| 21×11 | ±5 | 87.6% |

**A third of the condition is an ordinary noise frame wearing the stratum's label.** The 2.5–4.2 m/s
velocity band was set in step 2c when the window was 21×21 (reach ±10 bins = 4.2 m/s), where every
frame straddled. Step 6 narrowed the window and silently broke that design assumption.

### 2. Split by whether the window actually reaches the ridge, OS wins by more

Re-analysis of the already-run test experiment, sliced by a geometric covariate — no re-tuning:

| `clutter_edge` subset | frames | CA | GO | OS |
|---|---|---|---|---|
| **true edge** (window straddles) | 302 | 0.367 | 0.372 | **0.439** |
| no edge (mislabelled) | 98 | 0.504 | **0.555** | 0.487 |
| pooled — *what Milestone 21 reports* | 400 | 0.401 | 0.416 | 0.451 |

**GO wins only the frames that are not edges.** At a true edge OS leads by **+0.068**, nearly double
the pooled margin. The dilution was masking OS's advantage, not creating it.

### 3. The mechanism, measured at the target cell

Noise estimate relative to CA, by how many training columns fall in the ridge:

| ridge overlap | GO/CA | OS/CA |
|---|---|---|
| 0 columns | 1.15 | 1.39 |
| 1 column | 1.85 | 0.60 |
| 2 columns | 2.12 | 0.51 |

At zero overlap both sit at their homogeneous-noise values (1.09 and 1.40). At a real edge GO
becomes **2× more conservative** and OS **2× less**. GO takes the *greatest* half-ring, so it
deliberately picks the contaminated side; OS's 0.75-quantile, with only 25% of cells contaminated,
lands in the clean part. GO raises its threshold, OS lowers it, and matching the measured Pfa leaves
only the Pd consequence. **This is GO working exactly as designed** — its purpose is false-alarm
control at edges (2× fewer than CA at a fixed threshold, Milestone 21), and matched-Pfa scoring
neutralises that by construction.

### 4. The decisive control: stratify by target clearance from the ridge

Clearance — the target's distance in Doppler bins from the ridge *edge* — is fixed by the data and
cannot be changed by the window, so every geometry is scored on the same physical frames. Pd at
matched measured Pfa, mean over CA/GO/OS, on 250 frames (tune + val):

| clearance | 21×7 | **21×9** | 21×11 | 21×13 | 21×21 |
|---|---|---|---|---|---|
| **2 bins** (hardest) | 0.337 | 0.276 | 0.207 | 0.142 | 0.065 |
| 3 bins | 0.422 | 0.315 | 0.237 | 0.174 | 0.070 |
| 4 bins | 0.545 | 0.490 | 0.365 | 0.247 | 0.090 |
| 5–6 bins (easiest) | 0.574 | 0.557 | 0.479 | 0.365 | 0.156 |

**Two conclusions, and the first corrects what I wrote above before this ran.**

**(a) The narrow-window gain is REAL, not dilution.** At clearance 2 — where every geometry in the
table straddles the ridge, so dilution cannot be the explanation — Pd runs 0.065 → 0.276 from 21×21
to 21×9, a 4× improvement at fixed physical difficulty. Step 6's selection is therefore
**vindicated**: narrowing the Doppler window genuinely helps at a clutter edge. The earlier
suspicion that the +0.296 gain was largely an artefact of the condition changing meaning is **wrong**
and is retracted.

**(b) GO's win at 21×7 IS dilution.** Pooled, 21×7 showed GO leading `clutter_edge` (GO 0.455 vs CA
0.403, OS 0.429). Restricted to clearance 2, where the frames really are edges:

| clearance 2 | 21×7 | 21×9 | 21×11 | 21×21 |
|---|---|---|---|---|
| CA | 0.310 | 0.241 | 0.184 | 0.069 |
| GO | 0.310 | 0.264 | 0.184 | 0.057 |
| OS | **0.391** | **0.322** | **0.253** | 0.069 |

**OS wins at every window once difficulty is held fixed, and GO never beats CA by more than 0.023.**
GO's apparent 21×7 victory came from that window straddling only 41.5% of the stratum versus 21×9's
65.6% — it was winning the frames that had stopped being edges.

### 5. So why keep 21×9?

Not because 21×7 is artefactual — on `clutter_edge` at fixed difficulty it is genuinely better. But
the window has to serve all seven conditions, and across them 21×7 is a wash: `clutter_edge` +0.039
and `mixed` +0.010 against `clutter` −0.020, `masking` −0.010 and `multi` −0.010, for a mean of
**+0.004** — inside sampling noise at 400 tuning frames. With no overall gain, the incumbent stands
and the test baseline does not need re-running.

(`21×5` scores higher still on the pooled mean and is rejected on principle: `guard_v=1` is below the
measured 2-bin mainlobe half-width, so the target's own mainlobe leaks into the training ring.)

**Worth revisiting if `clutter_edge` is ever regenerated undiluted** — a clean stratum would give the
Doppler-width choice a much sharper test than a 34%-diluted one can.

### What is being reported

- **Keep 21×9.** No window change.
- **Report the true-edge / no-edge split**, not just the pooled `clutter_edge` row. It is a stronger
  result and it makes the mechanism visible.
- **GO's win stays where it is real**: false alarms at a fixed threshold, 2× better than CA on every
  clutter condition.
- **Limitation to record**: the `clutter_edge` velocity band should be tightened to ~2.5–3.4 m/s so
  every frame straddles the 21×9 window. Needs regeneration; not on the critical path.

## Milestone 23 — DL retrain: experiment log  🔄 IN PROGRESS

A dated entry per run from here on: the question it was meant to answer, what was run, the numbers,
what it showed **including negative and ambiguous results**, and what it changed about the plan.
Decisions are recorded with their reasons, not just their outcomes.

### Standing decisions

**All comparisons use `bench2_val` while choices are still open.** `bench2_test` was measured once
for the pre-registered CFAR baseline (Milestone 21) and is not touched again until the network is
final. Every checkpoint comparison, ablation and stopping-epoch decision is made on `bench2_val`
(1,600 frames, 200/condition). The cost is a noise floor ~1.4× wider than test's 400/condition, and
that is the right trade for keeping the reported numbers uncontaminated.

**Validation loss is no longer trusted as a stopping signal.** The 50-epoch run's val loss is
statistically flat from ~epoch 26 (trend p = 0.17) while Pd on the clutter-bearing conditions kept
climbing to epoch 50. Loss is a proxy; Pd at matched measured Pfa is what gets reported, and the two
disagreed. Stopping decisions are made on Pd.

**`pos_weight` ablation deferred until the noise floor is known.** The effect it would try to detect
is the same size as the drift already visible between checkpoints of a *single* run (0.02–0.05 Pd on
clutter conditions). Single-seed, it would produce a number indistinguishable from chance.

### 2026-07-30 — Model 0 baseline *(backfilled)*
**Question:** does the pipeline work end to end, and does a receptive-field-matched CNN beat CFAR at
all? **Run:** TinyCNN 5,041 params from `dl_cnn_experiment.ipynb` (deleted in `14b38bd`), old
256×128 sensor, 3,000 of 10,000 `frozen_train` frames, 8 epochs. **Result:** DL 0.784 vs CA 0.752 at
Pfa 1e-4. **Showed:** pipeline works, margin modest as predicted for matched context. **Limitation
recorded at the time:** 30% of the data, 8 epochs — a floor, not a ceiling.

### 2026-07-31 — Stratified retrain *(backfilled)*
**Question:** does DL still win once rare conditions are properly represented? **Run:** `bench_train`
9,600 frames, 12 epochs, old sensor, `pos_weight=200`; train 0.119→0.066, val 0.096→0.0755.
**Result:** DL highest Pd in 6 of 7 conditions, tied on the 7th. **What was wrong with it:** stopped
at a fixed 12 epochs with val loss *still falling*, kept the last epoch rather than the best, and
used only 800 of 1,600 val frames. All three fixed below. Old-sensor numbers do not carry over.

### 2026-08-11 — Class ratio remeasured; `pos_weight` 200 → 974
**Question:** is `pos_weight=200` justified by the actual class balance? **Run:** exact count over
every cell of the cached labels, not sampled.

| split | positive | total | rate | ratio |
|---|---|---|---|---|
| `bench2_train` | 144,996 | 137,625,600 | 0.001054 | **1:949** |
| `bench2_val` | 26,943 | 26,214,400 | 0.001028 | **1:973** |

Per-frame counts land on exact multiples of 9, confirming the 3×3 label blob.

**Showed:** the imbalance is real, and **`pos_weight=200` was never justified by anything** — on the
old sensor the ratio was 1:1563, so 200 sat ~8× below the inverse frequency, and it then survived a
sensor change that moved the ratio to ~1:950. A guess that outlived its premise.

**Decision:** set `pos_weight` to the measured inverse frequency (974) as a principled default —
recording that this is *also* an assumption, since exact class balancing is a choice, not a
derivation. Ablation over {100, 200, 974} queued but deferred.

**Consequence for every loss number:** the scale is set by `pos_weight × positive_rate` — old
200/1563 = 0.128, now 974/973 = 1.000, a **7.8×** difference. Verified directly: the *same* model on
the *same* val set scores 0.276 at `pos_weight=200` and 0.515 at 974. **Loss is not comparable across
runs with different `pos_weight` or class ratio.** Only the curve's shape is, and ultimately only Pd.

### 2026-08-11 — Training rebuilt: reproducible, all data, 50 epochs
**Question:** where does val loss stop improving, and what is the right stopping epoch?

    python scripts/train_tinycnn.py --epochs 50 --tag bench

**Config:** TinyCNN width 16, RF measured 21×9, Adam lr 1e-3, batch 32, `pos_weight=974`, seed 0, no
early stopping, **all 8,400 train and all 1,600 val frames** (asserted at startup — the previous run
silently used a third). 92 min.

**Infrastructure, and why:** no training entry point existed (it died with the deleted notebook), so
a run is now a command with a recorded config · reading 8,400 `.npy` files cost 325 s/epoch against
89 s of compute, so the split is pre-normalised once into a cache, with a gate proving the cache is
byte-identical to reading frames individually · one `--seed` fixes init and batch order, **verified**
by a later 30-epoch re-run reproducing epochs 27–30 to all five decimals · every run writes its own
config/log/history/checkpoints so experiments cannot overwrite each other.

**Gates:** shapes OK · cache == raw OK · orientation +11.7 dB · RF **(21, 9)** · overfit-one-batch
**1370×**, 17/17 peaks within 1 bin.

**Result:** best val 0.51529 @ 43, last 0.51971 @ 50.

| | |
|---|---|
| val trend, ep 26–50 | +0.0002/epoch, **p = 0.17** — indistinguishable from zero |
| train trend, ep 26–50 | −0.0007/epoch, **p = 1.3e-21** |
| val noise (std, last 25) | 0.0055 |
| train−val gap | +0.009 @26 → +0.021 @43 → +0.030 @50 |

**Showed:** val plateaus ~epoch 26 while train keeps falling — mild overfitting, no upward turn.
Extrapolated over another 50 epochs the val trend predicts 0.011 against a 0.0055 noise floor, so
**running to 100 would not have resolved the stopping question on loss**. With 5,041 params against
8,400 frames the classic U-shaped val curve should not be expected at all.

### 2026-08-11 — Checkpoints 30 / 43 / 50 on the reported metric
**Question:** do checkpoints across the loss plateau differ in Pd at matched measured Pfa? Coincide →
question closed; scatter → the evaluation is noisier than the loss curve suggests.

    python scripts/compare_checkpoints.py --checkpoints ep030 best last

Epoch 30 recovered by re-running 30 epochs at the same seed. `--save-every` added so this cannot
recur. **Split:** `bench2_val`, per-condition sweep to matched measured Pfa 1e-4.

| condition | ep 30 | ep 43 | ep 50 | spread |
|---|---|---|---|---|
| unresolved | 0.488 | 0.488 | 0.488 | **0.000** |
| masking | 0.705 | 0.702 | 0.702 | **0.002** |
| isolated | 0.720 | 0.730 | 0.725 | 0.010 |
| multi | 0.708 | 0.718 | 0.715 | 0.010 |
| clutter | 0.132 | 0.112 | 0.127 | 0.020 |
| mixed | 0.489 | 0.497 | **0.515** | 0.025 |
| clutter_edge | 0.664 | 0.705 | **0.711** | **0.047** |

**Showed — ambiguous, and split by condition.** Clean and interference conditions are settled
(0.000–0.010); the stopping epoch does not matter there. Clutter-bearing conditions are not, and two
move **monotonically upward**: clutter_edge 0.664 → 0.705 → 0.711, mixed 0.489 → 0.497 → 0.515. That
is the shape of continued improvement, in the epochs where val loss says nothing is happening.

**The flaw in this experiment, stated plainly:** the three checkpoints are the *same trajectory*, so
their spread confounds genuine model change with evaluation noise and cannot separate them. A
monotone trend hints at the former; three correlated points cannot establish it. **The noise floor
needs different seeds, not different epochs.**

**Changed the plan:** (1) early stopping on val loss is not used — `--patience 8` would have stopped
~epoch 34 and cost ~0.04 Pd on clutter_edge; (2) the `pos_weight` ablation waits for the noise floor;
(3) a like-for-like CFAR run on `bench2_val` is needed, since the existing baseline is on
`bench2_test` and the two cannot be read across.

### 2026-08-12 — STEP 1: CFAR on `bench2_val`, like for like
**Question:** the DL numbers are on `bench2_val` and the CFAR baseline is on `bench2_test`, so the
apparent gaps cannot be read across. Does the clutter deficit survive on the same split?

    python scripts/cfar_test_baseline.py --split bench2_val

Same protocol as the DL evaluation: 21×9 window, per-condition threshold sweep to matched **measured**
Pfa 1e-4. `cfar_test_baseline.py` gained a `--split` flag; `bench2_test` is untouched.

| condition | CA | GO | OS | best CFAR | **DL ep 50** | DL − CFAR |
|---|---|---|---|---|---|---|
| clutter | 0.134 | **0.149** | 0.137 | 0.149 | 0.127 | **−0.022** |
| unresolved | 0.485 | 0.482 | 0.485 | 0.485 | 0.488 | +0.003 |
| isolated | **0.710** | 0.700 | **0.710** | 0.710 | 0.725 | +0.015 |
| multi | **0.680** | 0.671 | 0.676 | 0.680 | 0.715 | +0.035 |
| masking | 0.515 | 0.522 | **0.645** | 0.645 | 0.702 | +0.057 |
| mixed | 0.400 | **0.426** | 0.373 | 0.426 | 0.515 | +0.089 |
| clutter_edge | 0.419 | 0.426 | **0.443** | 0.443 | **0.711** | **+0.268** |

**Showed — the clutter deficit is real but smaller, and not yet resolvable.** DL is worse on
`clutter` (0.127 vs GO 0.149). The cross-split read overstated it roughly twofold: CFAR scores
lower on val (0.149) than on test (0.171), so the earlier "0.127 vs 0.171" compared two different
things. **And the surviving 0.022 deficit is the same size as the 0.020 drift measured across
epochs 30/43/50 on that same condition** — on current evidence it cannot be separated from where
training happened to stop.

**Flagged for attribution, not reported as a result:** **+0.268 on `clutter_edge`** is enormous
against the best classical method. Milestone 17's corpus review notes that a credible learned
detector normally loses somewhere — this one does, on `clutter`, which is the right shape. But a
margin that large on the condition *adjacent* to its only loss needs explaining before it is
believed. Candidate explanations to test: the network exploits the ridge's structure as context in
a way a local window cannot; or the `clutter_edge` stratum's 34% dilution (Milestone 22) interacts
with a learned detector differently than with CFAR.

**Changed the plan:** nothing yet — the noise floor comes first (Step 2). Attribution of the
clutter_edge margin is queued behind it.

### 2026-08-12 — STEP 2, aborted at epoch 31: the noise floor killed the plan
**Question:** (a) what is the seed-to-seed spread at fixed epoch — our noise floor? (b) does
clutter_edge keep climbing past epoch 50, or flatten?

**Run.** 3 seeds × 100 epochs, `pos_weight=974`, checkpoints at 15/30/50/75/100, three processes in
parallel at 5 threads each. **The laptop slept at epoch 31** and all three died. Checkpoints at
epochs 15 and 30 survived; nothing after.

**Why it could not be resumed, and what changed.** Only model weights were being checkpointed —
optimizer moments and RNG state were not — so the run could be restarted but never *continued*.
Training now writes a full resume state (weights + optimizer + torch/numpy/generator RNG) every 5
epochs, to a temp file that is atomically renamed so a crash mid-write cannot corrupt it.
`--resume` picks up where it stopped and appends to the existing log.

**Result — question (a), evaluated on the 6 surviving checkpoints.** Pd at matched measured Pfa,
`bench2_val`, by seed:

| condition | seed spread @15 | **seed spread @30** | epoch 15→30 change |
|---|---|---|---|
| clutter_edge | 0.013 | **0.054** | +0.006 |
| mixed | 0.021 | 0.027 | +0.010 |
| clutter | 0.022 | 0.017 | +0.005 |
| isolated | 0.020 | 0.020 | +0.005 |
| multi | 0.008 | 0.009 | +0.010 |
| masking | 0.005 | 0.005 | +0.002 |
| unresolved | 0.003 | 0.003 | −0.001 |

**THE NOISE FLOOR IS LARGER THAN THE EFFECT.** Changing the seed moves clutter_edge by up to
**0.054** at one fixed epoch. Training from 15 to 30 epochs moves it by **0.006**.

**Correction to the previous entry.** I recorded the epoch 30/43/50 climb on clutter_edge
(0.664 → 0.705 → 0.711, spread 0.047) as *"the shape of continued improvement, not scatter"*. That
was wrong. It sits entirely inside the seed noise: at epoch 30 alone, seeds 0 and 1 both give 0.664
while seed 2 gives 0.718. The 50-epoch run was seed 0, so its 0.711 at epoch 50 is as well explained
by which seed it was as by how long it trained. Three correlated points from one trajectory could
not have distinguished these, which the entry itself flagged — I then over-read them anyway.

**What it changed.**
1. **Question (b) is unanswerable as posed** — it presumes a climb the data does not support. A
   100-epoch run would spend six hours producing another number inside the same ±0.054 band.
2. **The stopping-epoch question is closed.** Nothing moves more than 0.010 between epochs 15 and
   30, well under the noise. Anything from ~15 onward is equivalent, and early stopping on val loss
   remains unnecessary rather than harmful.
3. **Buy seeds, not epochs.** The missing quantity is an error bar, not a longer curve. 8 seeds
   brings the standard error on clutter_edge to ~0.011, which is what makes a 0.022 clutter gap
   testable at all.
4. **The `pos_weight` ablation bar is now known**: single-seed it would have to beat 0.054 on
   clutter_edge to mean anything. It must be seed-averaged whatever it finds.

### 2026-08-12 — STEP 2 (revised): 8 seeds × 50 epochs  🔄 RUNNING

    for S in 0..7: python scripts/train_tinycnn.py --epochs 50 --seed $S --tag n$S \
        --pos-weight 974 --save-at 15,30,50 --threads 2 --resume --skip-gates

**Question:** what are the per-condition mean ± standard error at epoch 50, and does the DL clutter
deficit against CFAR survive once error bars are attached?

**Config:** identical to the 50-epoch baseline except the seed — TinyCNN width 16, Adam lr 1e-3,
batch 32, `pos_weight=974` fixed (not measured per run, so it cannot drift between seeds), all 8,400
train / 1,600 val frames. Gates skipped after the first verified run; `--resume` on, so a shutdown
costs at most 5 epochs. Eight processes at 2 threads each.

**Comparison target — CFAR on the same split** (`bench2_val`, matched measured Pfa 1e-4): clutter
GO **0.149** · clutter_edge OS 0.443 · isolated CA/OS 0.710 · masking OS 0.645 · mixed GO 0.426 ·
multi CA 0.680 · unresolved CA/OS 0.485.

*(Results appended when the run completes.)*

### 2026-08-13 — `clutter_edge` +0.261 attributed: it is a reference-window effect, and it survives
**Question.** Before reporting the DL margin on `clutter_edge`, is it an edge effect, or an artefact
of that stratum's dilution (Milestone 22)? Script: `scripts/attribute_clutter_edge.py`, `bench2_val`,
CFAR from cached peaks, DL averaged over all 8 seeds, matched measured Pfa 1e-4 swept **per band**.

**Check 1 — against OS specifically, not "best classical".** OS is the method designed for edge
contamination, so a shrinking margin against it would narrow the finding. It does not shrink:
**OS *is* the best classical method here** — OS 0.443, GO 0.426, CA 0.419 — so +0.261 was always
measured against OS. Against CA it is +0.285, against GO +0.278; OS's edge design buys it only
0.017–0.024 over its siblings. Consistent with Milestone 22: at matched *measured* Pfa an
edge-robust design shows up as fewer false alarms, not higher Pd, and matching the rate cancels it.

**Check 2 — where the margin falls relative to the boundary.** Stratified by **target clearance
beyond the ridge edge in Doppler bins**, which is fixed by the data and cannot be changed by any
window or detector, so every method is scored on identical frames. Ridge ±4 bins, training ring at
offsets ±3–4 → clearance 2–4 overlaps the ridge, clearance ≥5 never does (**28%** of frames on val,
against Milestone 22's 24.5% frame-level on test).

| clearance | true edge | CA | GO | OS | best CFAR | DL (8 seeds) | DL − best |
|---|---|---|---|---|---|---|---|
| 2 (closest) | ✓ | 0.215 | 0.215 | 0.292 | 0.292 | 0.619 | **+0.327** |
| 3 | ✓ | 0.329 | 0.314 | 0.400 | 0.400 | 0.736 | **+0.336** |
| 4 | ✓ | 0.512 | 0.512 | 0.512 | 0.512 | 0.724 | +0.212 |
| 5–6 (farthest) | ✗ | 0.571 | 0.610 | 0.545 | 0.610 | 0.724 | +0.114 |

**Verdict: H-window (reference-window contamination), supported.** CFAR collapses as the target
approaches the ridge — OS 0.545 → **0.292**, a drop of 0.253 — while DL barely moves, 0.724 → 0.619,
a drop of 0.105 confined to the closest band. That asymmetry *is* the signature: CFAR is handicapped
exactly where its training ring overlaps the ridge, and a detector with no reference window is not.

**Dilution cannot explain it — it works the other way.** The 28% non-edge frames carry the
*smallest* margin (+0.114), so they drag the pooled number **down**. The reported +0.261 therefore
**understates** the true-edge effect, which is ≈ +0.28 frame-weighted. The +0.114 on non-edge frames
sits in the same range as DL's general advantage elsewhere (isolated +0.022, multi +0.031, masking
+0.049, mixed +0.086) — exactly what one expects on frames that are not really edges.

**Strength, stated honestly.** The near-vs-far contrast is ~2.7σ allowing for CFAR's own per-band
sampling error (~0.055 at 40–57 frames), and the two middle bands invert slightly (+0.327 vs +0.336),
well within that. What makes it convincing is the near-monotone gradient across four bands *plus*
the mechanism agreeing — not the endpoint contrast alone.

**What it changed.** `clutter_edge` moves from "flagged, do not report" to **reportable**, with the
mechanism stated and the per-band counts (40–57 frames) noted. Full write-up, tables and figure in
`notebooks/cnn_development_log.ipynb` §Open/O1.

### 2026-08-14 — `pos_weight` ablation: 974 was the wrong choice, and the clutter finding retracts
**Question.** Does a lower `pos_weight` recover the `clutter` deficit (DL 0.125 vs GO 0.149), and
what does it cost on the other six conditions?

**Setup.** 8 seeds × 50 epochs at each of `pos_weight` = 100, 200, 974 — 24 runs, everything else
identical; the 974 arm reuses the existing 8-seed run. Evaluated at epoch 50 on `bench2_val`,
per-condition sweep to matched measured Pfa 1e-4. Seeds shared across arms, so comparisons are
**paired**. Pd only — final val losses were ~0.118 / 0.191 / 0.520, almost exactly the ratio the
weights predict, which is the loss scale moving rather than performance.

| condition | CFAR best | pw100 | pw200 | pw974 |
|---|---|---|---|---|
| clutter | **0.149** GO | 0.144 ± .005 | 0.144 ± .003 | 0.125 ± .003 |
| clutter_edge | 0.443 OS | **0.731** ± .004 | 0.724 ± .004 | 0.704 ± .004 |
| isolated | 0.710 CA | 0.732 ± .002 | 0.731 ± .002 | 0.732 ± .001 |
| masking | 0.645 OS | 0.694 ± .003 | 0.698 ± .002 | 0.694 ± .004 |
| mixed | 0.426 GO | **0.541** ± .003 | 0.530 ± .002 | 0.512 ± .002 |
| multi | 0.680 CA | 0.719 ± .002 | 0.721 ± .002 | 0.711 ± .002 |
| unresolved | 0.485 CA | 0.486 ± .001 | 0.487 ± .001 | 0.487 ± .001 |

**Result — not a trade, a strict improvement.** The `clutter` gap against GO is **−0.006 (p = 0.27)**
at 100 and **−0.005 (p = 0.15)** at 200, against **−0.025 (p = 0.00013)** at 974. Paired by seed,
100 vs 974: mixed **+0.029** (p = 3e-05), clutter_edge **+0.026** (p = 0.002), clutter **+0.019**
(p = 6e-06), multi +0.008 (p = 0.019); isolated, masking and unresolved unchanged (p ≥ 0.35).
**Four conditions improve, three are flat, none degrades.**

The absence of a trade is itself the finding: the expected tension between detecting rare positives
and suppressing false alarms does not appear, because the operating point is chosen by sweeping the
threshold afterwards — `pos_weight` only has to make the *ranking* learnable, which is exactly what
`dl/train.py`'s own docstring argued and had never been tested at this sensor.

100 and 200 are near-equivalent; only `mixed` separates them (+0.011, p = 0.0015 favouring 100).

**⚠️ RETRACTION — the Milestone 23 headline.** "CFAR genuinely beats the network on `clutter`" was a
statement about a badly-chosen hyperparameter, not about the detectors. `pos_weight = 974` was set
from the measured inverse class frequency and called a principled default, while noting that exact
balancing is a choice rather than a derivation — and that caveat is precisely the one that should
have blocked drawing a conclusion about the detectors from it.

**Consequence, stated plainly:** the network now ties or beats CFAR on all seven conditions, which
revives Milestone 17's concern that a credible learned detector normally loses *somewhere* and ours
loses nowhere. That is a **weaker** position to report from, not a stronger one.

**What it changed.**
1. `pos_weight` moves to **100**.
2. Every Milestone-23 number is superseded — they were all measured at 974 and must be re-run.
3. The `clutter_edge` attribution (2026-08-13) also needs re-checking at 100: its +0.261 was measured
   at 974, and that condition scores 0.731 rather than 0.704 at 100.

Full write-up, tables and figure in `notebooks/cnn_development_log.ipynb` §8, with the retraction as
Correction C3.

### 2026-08-14 — Error bars on both sides: the `isolated` win withdraws, and the shape improves
**Question.** Every DL-vs-CFAR p-value so far used `ttest_1samp(dl_seeds, cfar_value)` — treating
CFAR's Pd as an exact constant when it is one draw over 200 frames. What survives when CFAR gets its
own sampling error, and when the 7 simultaneous comparisons are corrected?

**Setup.** `scripts/bootstrap_comparison.py` · `bench2_val` · paired frame bootstrap (resample
frames, recompute both sides on the same frames so difficulty cancels), 2,000 resamples, percentile
CI · Bonferroni α = 0.05/7 = **0.0071** · cached CFAR peaks + 8 existing DL checkpoints, no
retraining. *Limitation stated:* thresholds are computed once per condition, so the intervals cover
Pd sampling error but not threshold-selection variability — mildly optimistic.

| condition | CFAR | DL (pw100) | diff | 95% CI | p | survives |
|---|---|---|---|---|---|---|
| clutter_edge | OS 0.443 | 0.731 | **+0.288** | [0.239, 0.336] | <0.001 | ✓ |
| mixed | GO 0.426 | 0.541 | **+0.116** | [0.086, 0.148] | <0.001 | ✓ |
| masking | OS 0.645 | 0.694 | **+0.049** | [0.031, 0.070] | <0.001 | ✓ |
| multi | CA 0.680 | 0.719 | **+0.038** | [0.025, 0.052] | <0.001 | ✓ |
| isolated | CA 0.710 | 0.733 | +0.023 | [0.002, 0.046] | 0.031 | ✗ |
| clutter | GO 0.149 | 0.144 | −0.006 | [−0.030, +0.017] | 0.688 | ✗ |
| unresolved | CA 0.485 | 0.486 | +0.001 | [−0.008, +0.010] | 0.911 | ✗ |

**⚠️ RETRACTION — the `isolated` win.** Previously reported at **p = 6.9e-07**; it is **p = 0.031**,
short of the corrected threshold. The DL side had 8 seeds and visible scatter, so the uncertainty
*looked* accounted for — while all the uncertainty on the other side of the comparison was silently
set to zero. Eight seeds of precision on one arm does not license a point estimate on the other.

**This improves the project's position.** Milestone 17 records that a well-built CNN should **tie**
CFAR in homogeneous noise (RadCNN S4, Gusland), and flagged our win there as *"the strongest single
indication that something still favours the network."* With correct statistics it **is** a tie. The
result now reads: **DL wins decisively where interference or clutter is present, ties on clean
noise, on unresolvable pairs, and inside the clutter ridge** — the shape the literature predicts.

**Also checked (clean): target-level leakage.** Frame-level leakage was already zero by content
hash; this tests the subtler case of the same target *parameters* appearing in both splits.
Zero exact matches over 16,122 train and 3,115 val targets. 0.42% of val targets have a train
neighbour within 0.01 normalised distance, against **~0.5% expected by chance** (median NN distance
0.060, density scaling as r³) — ordinary crowding of independent draws, not leakage.
`scripts/verify_target_leak.py`.

**What it changed.** All prior DL-vs-CFAR p-values superseded. The same defect applies to
`notebooks/cfar_baseline.ipynb`, whose CFAR method-to-method comparisons are bare point estimates
judged against a rule of thumb — being corrected there too.

### 2026-08-15 — Pre-lock robustness block: 6 checks, 3 conclusions changed
A **closed-ended** block run against a list of objections, under a rule adopted because the project
had been drifting into open-ended fixing: *a new finding goes on the limitations list, not the work
queue, unless it changes a number already published here.* Six items, no seventh opened.

#### #4 SO-CFAR added — the baseline was already adequate
`SO` (smallest-of) is the multiple-target specialist and was missing while we claimed a `masking`
win. Added to `src/cfar/cfar.py` (`np.minimum` of the half-rings, mirroring GO's `maximum`), and
`bench2_val` re-scored with four classical methods.

| condition | CA | GO | OS | **SO** |
|---|---|---|---|---|
| masking | 0.515 | 0.522 | **0.645** | 0.638 |
| clutter | 0.134 | **0.149** | 0.137 | 0.065 |
| clutter_edge | 0.419 | 0.426 | **0.443** | 0.205 |
| mixed | 0.400 | **0.426** | 0.373 | 0.213 |

**SO does not beat OS on masking** (0.638 vs 0.645), so the DL margin there stands. SO is severely
worse wherever clutter appears — the exact mirror of GO, since taking the *smallest* half-ring
underestimates noise at an edge and matched-Pfa scoring then forces its threshold up. GO 0.149 vs
SO 0.065 on `clutter` is the cleanest statement of GO/SO complementarity in the project.
**SO is never the best classical method on any condition, so no conclusion changes.**

#### #1 Tolerance sweep — the ±1 gate is NOT doing the work (except on `clutter`)
The gate credits a detection within ±1 bin, and the label mask is 3×3, so the CNN is trained to fire
on exactly the region the evaluator rewards (Milestone 16 asymmetry #5, never tested). Re-scored at
gates of ±1, ±2, ±3 — each with its own threshold sweep, since the gate also sets the guard region.

DL − best CFAR:

| condition | ±1 | ±2 | ±3 | change |
|---|---|---|---|---|
| clutter_edge | 0.288 | 0.294 | 0.294 | +0.006 |
| mixed | 0.116 | 0.120 | 0.114 | −0.001 |
| masking | 0.049 | 0.059 | 0.062 | +0.013 |
| multi | 0.038 | 0.045 | 0.039 | +0.001 |
| isolated | 0.022 | 0.026 | 0.026 | +0.004 |
| unresolved | 0.001 | 0.004 | 0.004 | +0.003 |
| **clutter** | **−0.006** | **−0.022** | **−0.038** | **−0.033** |

**Six of seven margins hold** (all ≤0.013). The localisation-to-the-label objection is **falsified**
for every condition where we claim a win.

⚠️ **`clutter` is the exception and it moves against DL.** The deficit triples as the gate widens.
The mechanism is the one asymmetry #5 predicted, pointing the other way: on `clutter` the RD peak is
within 1 bin of the label only **77%** of the time (Milestone 22), so CFAR — which detects the
*physical* peak — was being denied credit at ±1, while DL, trained on label positions, already sat
inside the tight gate. **At ±1 the clutter comparison is biased in DL's favour**, and "clutter is a
tie" may understate CFAR. Needs bootstrapping at ±3 before any claim: the ±1 CI was [−0.030, +0.017]
and −0.038 sits at its edge.

#### #2 NMS sweep — `unresolved` is capped by physics; the NMS choice is conservative on `masking`

| `unresolved` | nms 1 | nms 2 | nms 3 | nms 5 | | `masking` | nms 1 | nms 2 | nms 3 | nms 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| CA | 0.495 | 0.485 | 0.485 | 0.485 | | CA | 0.515 | 0.515 | 0.515 | 0.515 |
| OS | 0.512 | 0.485 | 0.485 | 0.485 | | OS | 0.725 | 0.705 | 0.682 | 0.645 |
| DL | 0.554 | 0.483 | 0.483 | 0.483 | | DL | 0.804 | 0.770 | 0.739 | 0.697 |

**`unresolved` is flat from nms 2 to 5 for every method** — shrinking the window 11×11 → 5×5 changes
nothing, because the pairs are 0.26–1.79 range bins apart, below the measured 0.39 m limit. Only the
disqualified `nms_size=1` lifts it, and it lifts DL most (0.554) while raising DL's
detections-per-target to 1.22 against 1.16 for the others — the free-attempts regime step 6d rejected.
**The 0.5 cap is physics.**

**On `masking` the NMS width matters, and our choice costs us.** DL−OS is +0.052 at nms 5 and +0.079
at nms 1: `nms_size=5` suppresses resolvable neighbours and **shrinks** the claimed DL advantage
rather than manufacturing it. CA and GO are flat because they cannot resolve these pairs at any
width. *Caveat on our own metric:* detections-per-target is dominated by false alarms (~1.3/frame at
Pfa 1e-4), so only the nms=1 rise above the other methods is interpretable as duplication.

#### Block outcome
Scripts: `bootstrap_comparison.py`, `verify_target_leak.py`, `tolerance_sweep.py`, `nms_sweep.py`,
SO in `cfar/cfar.py`. Conclusions changed: the `isolated` win withdrawn (see the 2026-08-14 entry),
the `clutter` tie flagged as gate-sensitive, and the `masking`/`unresolved` objections resolved in
our favour. Nothing else moved.

### 2026-08-15 — The `clutter` claim at ±3: CFAR does win there
**Question.** §11's tolerance sweep showed `clutter` is the one condition whose margin depends on
the matching gate, and it moves *against* DL. "Clutter is a tie" was measured at ±1 — the gate that
penalises CFAR for detecting the physical peak rather than the labelled bin. What is the claim at
±3? `scripts/bootstrap_comparison.py --tolerance 3`, same paired bootstrap, Bonferroni α = 0.0071.

| gate | CFAR (GO) | DL | diff | 95% CI | p |
|---|---|---|---|---|---|
| ±1 | 0.149 | 0.144 | −0.006 | [−0.030, +0.017] | 0.688 |
| **±3** | **0.192** | 0.153 | **−0.038** | **[−0.069, −0.007]** | 0.013 |

**CFAR gains three times as much from the wider gate** (0.149 → 0.192 against DL's 0.144 → 0.153) —
the 77% label-to-peak alignment on this condition (Milestone 22) being paid back. The CI now
excludes zero, but p = 0.013 does **not** survive correction.

**Report as:** *"CFAR is better on targets inside the clutter ridge; the effect is at the edge of
significance once corrected for seven comparisons."* This partly restores what Milestone 17 says a
credible learned detector should show — a condition where it loses. Once CFAR is not penalised for
detecting the physical peak, the clutter ridge is that condition.

**⚠️ `isolated` is borderline and no claim should rest on it.** p = 0.031 at ±1, p = 0.001 at ±3,
while the estimate moves only +0.022 → +0.026. A 30× swing in p from a 0.004 shift is tail
sensitivity, not evidence. State it as "small positive, borderline".

**Decision:** the pre-registered **±1 gate remains the headline**, with ±3 reported beside it as the
robustness check. Two claims are gate-qualified rather than flat: `clutter` and `isolated`.

## Current position
Phase A ✅. Phase B ✅. Data audit ✅. DL Model 0 ✅. Coverage study ✅. Stratified benchmark ✅.
Literature corpus ✅. **Experiments ~10 days ahead of `SCHEDULE.md`** (Week 2 milestone met 30 Jul,
due 9 Aug). **Writing has not started — that is the real risk, not the experiments.**

**NEXT = attribution, not U-Net.** A U-Net built on an unattributed margin would inherit the problem.
Plan in `NEXT_SESSION.md`; ranked ideas in `docs/literature_corpus_review.md` §5.
- **Tier A (numbers must be trustworthy):** A-1 bracket mainlobe integration from both sides
  (guard-constrained CNN *and* MF+CFAR — settles H1 vs H2) · A-2 measure whether our detector is
  actually CFAR (Diskin's definition: FPR invariant to nuisance params at fixed threshold) ·
  A-3 matching-tolerance sweep ±1/±2/±3 · A-4 publish the measured-Pfa table · A-5 add an S3 stratum
  (the condition where RadCNN's CNN loses).
- **Tier B:** heavy-tailed clutter (Weibull/K, exact recipe now available) · stronger classical
  baseline (two-stage CFAR with per-region PDF estimation) · loss ablation (Dice/Tversky/Lagrangian) ·
  Gaussian label smoothing · mutual radar interference.
- **Tier C (the paper):** the **CFAR→CNN ablation ladder** L0…L5, decomposing the total margin into
  named relaxations. Nobody in the corpus does this, and Banerjee's conv+ReLU equivalence is what
  makes it rigorous. It converts our fairness problem into the contribution.

**Guiding principle:** a smaller, well-understood DL advantage is worth far more than a large
unexplained one. "DL ≈ CFAR once CFAR gets equivalent pre-integration" is a perfectly good result.

*(Note: `Plan.md` was deleted; this file is now the single roadmap record.)*

### Optional stretch ideas (Phase D, only if time remains)
- **CFAR training-footprint shape ablation** — square ring vs cross/plus vs diamond. A real
  CFAR design axis but out of proposal scope; clutter orientation is already handled more
  cheaply via the Doppler-axis GO/OS split. Mention in the viva; build only as an extra.

## Roadmap (short)
A: SNR calibration → eval module.  B: CFAR (CA/OS/GO) → results.  C: frozen split → CNN/U-Net →
train/eval.  D: CFAR-vs-DL comparison + ablations + report.
Future realism sets (to expose more CFAR failure): range-dependent amplitude, heavy-tailed
clutter (K/Weibull), Swerling fluctuation, phase noise. See `Plan.md`.
