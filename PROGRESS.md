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

## Current position
Phase A ✅. Phase B ✅. Data audit ✅. DL Model 0 ✅. Coverage study ✅. Stratified benchmark ✅.
**Ahead of `SCHEDULE.md`** (Week 2 milestone met 30 Jul; due 9 Aug) — spend that slack on the
fairness audit.

**NEXT = the fairness audit, not U-Net.** A U-Net built on an unfair protocol would inherit the
problem. See **`NEXT_SESSION.md`**:
- **Step A (cheap diagnostics):** A1 matched *measured* Pfa via ROC · A2 trivial global-threshold
  baseline (if it ≈ CFAR, the benchmark is too easy) · A3 widen the matching gate to ±2.
- **Step B (real fixes):** B1 give CFAR the matched-filter benefit (**highest value** — tests the
  leading hypothesis; expected to close much of the gap) · B2 tune CFAR per condition.
- **Step C:** perturb the generator at test time and re-test DL without retraining.

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
