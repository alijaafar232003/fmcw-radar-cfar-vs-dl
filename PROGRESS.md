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
