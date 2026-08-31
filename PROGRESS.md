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

### 2026-08-16 — Localization error: the one metric where CFAR clearly beats the network
**Question.** `aggregate()` has always computed localization error, and `cfar_baseline.ipynb`
reports it for CFAR — but it was **never extracted for the DL side**; every script computed it and
read only `pd`. It is also a proposal-required metric. How precisely does each detector place what
it finds? `scripts/localization_table.py --per-condition`, `bench2_val`, 8 DL seeds at pw100,
matched measured Pfa 1e-4, gates ±1 and ±3.

| gate ±1 | Pd | \|Δr\| bins | \|Δv\| bins | range (m) | vel (m/s) |
|---|---|---|---|---|---|
| CA | 0.482 | 0.216 | 0.101 | 0.042 | 0.042 |
| GO | 0.497 | 0.218 | 0.119 | 0.043 | 0.050 |
| **OS** | 0.481 | 0.218 | **0.086** | 0.043 | **0.036** |
| SO | 0.237 | **0.179** | 0.142 | **0.035** | 0.060 |
| **DL** | **0.587** | 0.498 | 0.485 | 0.097 | 0.204 |

**DL's Doppler error is 5.6× the best CFAR and its range error 2.8×** — 0.204 m/s against OS's
0.036 m/s. Identical ordering at ±3, so not a gate artefact.

**Likely cause: the 3×3 label blob.** The network is trained that all nine cells around a target are
"target", so its peak may land anywhere inside that block, while CFAR responds to the sharper
physical peak. This is Milestone 16's asymmetry #5 measured directly rather than argued.

**Why it matters.** The headline is no longer one-sided: **DL detects more, CFAR localizes far
better.** That is a real trade, it is proposal-required, and it is the strongest honest answer to
"does the network beat CFAR?" — *at finding targets in interference, yes; at placing them, no.*

Side observations: SO has the **best** range localization (0.179 bins) despite the worst Pd, which
fits a low threshold surviving only on sharp peaks; and DL's error barely moves between gates,
meaning its displaced detections were already inside the tight gate. Suggests a later experiment:
a smaller or Gaussian-weighted label instead of a flat 3×3 block.

### 2026-08-16 — `TinyCNNPool` added (pooling control, not yet trained)
Built to test an assertion `models.py` has been making without evidence: *"down/up-sampling risks
displacing the peak bin"*. Pool by 2, convolve at half resolution, upsample back to 256×64.

**Parameter count lands at exactly 5,041 — identical to `TinyCNN`** — so capacity is not a confound.
**The receptive field is not.** Measured **13×13** against TinyCNN's **21×9**: range *shrinks*
21→13 while Doppler *grows* 9→13, since a 3×3 kernel at half resolution spans twice the input and
the range-stretching dilations are gone. So a localization difference cannot be attributed to
downsampling alone, and this model **no longer matches the CFAR window** — it is a control for one
question, not a valid arm of the CFAR-vs-DL comparison. Both facts are in the class docstring.

Prediction to test: localization error rises. The bar is the no-pooling baseline of 0.498/0.485
bins, and the ±1 gate will convert displacement beyond one bin into lost Pd rather than larger
error — so **both** effects should be reported, at ±1 and ±3.

### 2026-08-16 — Epoch flatness at pw100: Pd flat, but localization degrades with training
**Question.** Two gaps. The flatness evidence ("nothing moves between epochs 15 and 30") came from
the `pos_weight=974` arm — the value later shown to be wrong — and had never been demonstrated at
`pos_weight=100`. And localization had only ever been measured at epoch 50, so its dependence on
training length was unknown.

**Setup.** 4 seeds × `--save-at 15,25,34,50` at pw100 (same seeds as the reported config, so the
trajectories are identical; only the intermediate checkpoints are new). `scripts/epoch_flatness.py`,
`bench2_val`, matched measured Pfa 1e-4, gate ±1.

| epoch | Pd | SE | range err (bins) | Doppler err (bins) |
|---|---|---|---|---|
| **15** | 0.5825 | 0.0020 | **0.467** | **0.385** |
| 25 | 0.5846 | 0.0022 | 0.457 | 0.421 |
| 34 | 0.5875 | 0.0015 | 0.468 | 0.460 |
| 50 | 0.5858 | 0.0031 | 0.517 | 0.454 |

**Pd is flat** — range 0.0050 against mean SE 0.0022 — so flatness now holds at the *reported*
configuration.

**But localization degrades with training**: Doppler +18% (0.385 → 0.454), range +11%. Not an
artefact of pooling the conditions — **all seven degrade monotonically**, `unresolved` 0.278 → 0.364
through `clutter` 0.667 → 0.748. Seven of seven agreeing is much stronger than the ~2 SE of the
pooled figure alone.

**Mechanism: the 3×3 label, again.** With longer training the network grows confident across the
whole nine-cell block, so its peak inside it becomes less sharp. Longer training teaches the *label
shape*, not the target position — same root cause as the localization finding above.

**⚠️ DECISION — the frozen epoch count moves from 50 to 15.** The earlier argument ("50 is fine,
Pd is flat") considered only Pd. On the full picture epoch 15 costs **0.003 Pd (~1 SE, marginal)**
and buys **15% better localization**. The epoch count is now a measured choice, not an arbitrary one.

*Limitation:* nothing below epoch 15 was tested, so the optimum may be earlier. Recorded rather than
swept — the real fix is a better label (smaller or Gaussian-weighted), not a shorter run.

### 2026-08-16 — Full epoch sweep, epochs 1–25: the stopping point read off the whole curve

**Question.** The entry above chose epoch 15 from four sampled points {15, 25, 34, 50}, and flagged
its own limitation: *nothing below 15 was ever evaluated*. Four points cannot tell a curve that is
flat from one that was merely sampled where it happens to be flat. This measures every epoch from
1 to 25, so the stopping point is read off a curve rather than interpolated between four samples.

Two secondary questions it is sized to answer:
1. **Does localization degrade monotonically from epoch 1, or start flat and then turn?** Monotone
   from the very start supports the 3×3-label explanation — the network learns the label *shape*
   from the first epoch. A flat-then-turn shape would point at something else (late overfitting).
2. **How far back does Pd stay flat?** Nothing below epoch 15 has ever been tested.

**Setup.** 4 seeds × 25 epochs at `pos_weight=100`, checkpointing **every** epoch, all 100
checkpoints evaluated on `bench2_val` at matched measured Pfa 1e-4, gate ±1. Same seeds and config
as the reported arm, so the trajectories are identical to the `epflat` runs — verified: seed 0
epoch 1 reproduces train 0.20474 / val 0.14189 exactly.

    for S in 0..3: python scripts/train_tinycnn.py --epochs 25 --pos-weight 100 --seed $S \
        --tag epsweep_s$S --save-at 1,2,...,25 --threads 4 --keep-awake --skip-gates --resume
    python scripts/epoch_sweep.py --tag epsweep --seeds $S      # 4 procs, then --summarise

**⚠️ THE SELECTION RULE, FIXED BEFORE ANY RESULT EXISTS.** Written here while the sweep is still
running, precisely so it cannot be re-negotiated once the curve is on screen:

> **plateau** = the maximum over epochs of the pooled seed-mean Pd.
> **tolerance** = the SE of the pooled seed-mean Pd *at the plateau epoch*.
> **chosen epoch** = the **earliest** epoch whose pooled seed-mean Pd ≥ plateau − tolerance.

"Earliest within 1 SE of the plateau" encodes the actual preference: among epochs that detect
equally well, take the least-trained one, because localization is the metric that degrades with
training and localization is where CFAR wins. Ties in Pd are therefore broken by the metric we are
losing on, without that being a post-hoc choice.

`scripts/epoch_sweep.py` is new. It reproduces `epoch_flatness.py`'s published numbers **exactly**
(all 32 seed-0 rows, all three metrics, zero difference) before being used for anything new, and its
batched-inference path is asserted identical to the per-frame path it replaces (`--verify-cache`).

**Result — the rule selects epoch 21.**

| | |
|---|---|
| plateau | Pd **0.5862** at epoch 23 |
| tolerance (1 SE at the plateau epoch) | **0.0023** → floor 0.5839 |
| **chosen epoch** | **21** |
| Pd there | **0.5844 ± 0.0019** |
| range error there | **0.442 ± 0.022** bins |
| Doppler error there | **0.409 ± 0.036** bins |

Epochs inside the 1-SE band: **21, 22, 23, 25** (24 misses at 1.08 SE — the band is noisy at this
resolution, which is exactly why the rule takes the earliest qualifying epoch rather than the max).

**⚠️ "Pd is flat" was an artefact of where the four points were sampled.** Pooled Pd rises
**+0.059** from epoch 1 (0.5272) to the plateau — about **26 SE**, not a flat curve at all. The
previous entry sampled {15, 25, 34, 50}, *all four of which sit in the flat top*, and the entire
rise happens below 15 where nothing had ever been measured. The two runs agree exactly where they
overlap (epoch 15: Pd 0.5825, range 0.4667, Doppler 0.3848 — identical to the entry above; epoch 25
likewise), so this is not a disagreement between measurements, it is a sampling blind spot.

**Q2 — where does Pd stop being flat going earlier?** Deficit from the plateau, in SE:

| epoch | 1 | 5 | 10 | 14 | 15 | 18 | 20 | 21 |
|---|---|---|---|---|---|---|---|---|
| deficit (SE) | 25.7 | 9.8 | 3.9 | 2.4 | **1.6** | 1.8 | 1.1 | **0.8 ✓** |

Epoch 15 sits **1.6 SE below the plateau** — outside the pre-registered band. Flatness in the strict
1-SE sense begins at **21**; in a looser 2-SE sense it reaches back to ~14. Below 10 the curve is
unambiguously still climbing.

**Q1 — does localization degrade monotonically from epoch 1? Not quite, and the two axes differ.**

| axis | min at | epochs 1–8 | 8–15 | 15–25 | total 1→25 |
|---|---|---|---|---|---|
| range | epoch **1** | **+0.141** (p = 0.0014) | +0.016 (p = 0.025) | −0.010 (p = **0.51**) | +0.147 |
| Doppler | epoch **4** | +0.038 | **+0.145** (p < 1e-4) | +0.036 (p = 0.0005) | +0.220 |

**Range degrades from epoch 1 and then saturates** — flat from 15 onward (slope indistinguishable
from zero, p = 0.51). **Doppler improves slightly to epoch ~4, then turns and never saturates** —
still climbing significantly over 15–25. Both patterns are consistent across **all seven
conditions** (range minimum at epoch 1 in 7/7; Doppler minimum at epoch 2–4 in 7/7), so this is
structure, not seed noise.

**What that does to the 3×3-label explanation: partial support, and one thing it does not predict.**
The "monotone from the very start" signature holds cleanly on **range**. It does **not** hold on
Doppler, which has a real early *improving* phase. More awkwardly, a 3×3 label is **symmetric**, so
it predicts the same timing on both axes — instead Doppler degrades 1.5× as much (+0.220 vs +0.147)
and is still going when range has stopped. The label story survives as *a* mechanism but is no
longer sufficient on its own; something axis-specific is also acting. Candidates, none tested here:
the receptive field is **21×9**, itself asymmetric; Doppler has 64 bins against range's 256, so the
label block covers a larger fraction of the axis; and the clutter ridge is a Doppler-axis structure
(`clutter` has by far the worst Doppler error, 0.711 bins at epoch 21, and the steepest climb).
**Logged as an open question, not resolved** — it does not change a published number.

**Epoch 21 against the two previously-argued values, paired by seed:**

| comparison | Pd | range (bins) | Doppler (bins) |
|---|---|---|---|
| 21 vs **15** | +0.0019 (p = 0.14) | −0.025 (p = 0.37) | +0.024 (p = 0.20) |
| 21 vs **50** | −0.0014 (p = 0.77) | **−0.075 (p = 0.004)** | **−0.045 (p = 0.036)** |

So **21 vs 50 is a real improvement** — same Pd, significantly better localization on both axes,
which vindicates the direction of the previous entry. But **21 vs 15 is inside the noise on every
metric**: the practical cost of having chosen 15 was small. What was wrong was the *reason* given
for it ("Pd is flat at 15"), not badly wrong the number itself. See Correction **C5**.

*Caveat on the rule, stated because it is a property of the rule and not of the data:* the plateau
is the max over 25 noisy estimates, so it is biased upward by roughly the winner's curse — 0.5862
against a mean of 0.5845 over epochs 20–25, a gap of ~0.7 SE. That bias makes the rule **push the
chosen epoch later**, i.e. conservative in the Pd direction and against our localization preference.
Noted rather than corrected: the rule was pre-registered and re-deriving it now is exactly what
pre-registration exists to prevent.

**⚠️ DECISION — the frozen epoch count moves from 15 to 21.** `scripts/epoch_sweep.py`,
`results/dl/epoch_sweep_epsweep.csv`, `results/epoch_sweep_epsweep.png`.

### 2026-08-16 — `--threads` is a reproducibility parameter, not a speed knob

**Found while checking** whether the `epsweep` seeds 0–3 could be reused as the reported seeds 0–3
at epoch 21. They cannot: the two runs have the **same seed, the same config, and byte-identical
training code** (`git diff 27a1f46 b1db7ee -- src/dl/ src/eval/` is empty), yet different weights.
The only difference recorded in their `config.json` files is **`threads`: 2 vs 4**.

Torch's CPU kernels reduce in a different order at a different thread count, so the floating-point
sums differ, and the difference compounds over tens of epochs of Adam. `seed_everything()` cannot
prevent this — it is not an RNG effect.

**The divergence is not cosmetic:** max |weight difference| across the 5,041 parameters is **0.38**.
These are not the same model with rounding noise on it — they are genuinely different points in
parameter space that happen to detect equally well, a last-bit perturbation in epoch 1 amplified by
50 epochs of a momentum optimizer. That is the real reason the arms cannot be interchanged, and it
is a stronger statement than the metric table below, which only shows they *perform* alike.

**Measured cost, same seed 0, same epoch 50, threads 2 (`pw100`) vs 4 (`epflat`):**

| pooled metric | difference | for scale |
|---|---|---|
| Pd | 0.0013 | well under the 0.0023 seed SE — negligible |
| range error | **0.0224** | ≈ 1 seed SE (0.0217) |
| Doppler error | **0.0284** | ≈ 1 seed SE (0.0363) |

Per condition it reaches 0.103 bins (range, `isolated`) and 0.079 (Doppler, `clutter`).

**Why it lands on localization and not Pd.** The matched-*measured*-Pfa threshold shifts slightly,
which changes **which marginal detections are included**. Pd moves by one or two borderline targets
out of ~1,700; localization is averaged over exactly those newly-included marginal detections, which
are the badly-localised ones. So the same perturbation is invisible in Pd and ~1 SE in localization.

**This is one draw, not a variance estimate** — one seed, one epoch. It shows the effect is of order
one seed's noise; it does not establish a distribution, and nothing here suggests either thread count
is *better*. It is a nuisance parameter to hold fixed, not to optimise.

**What it changed.**
1. **The epoch sweep is unaffected** — every `epsweep` run and the epoch-50 arm it is compared
   against used `threads=4`, so §16's within-experiment conclusions hold throughout.
2. **Arms may not be mixed.** `epsweep` seeds 0–3 are a *different numerical realization* from
   `pw100` seeds 0–3 and cannot stand in for them. The epoch-21 re-freeze therefore re-runs **all
   eight** seeds at one fixed thread count rather than reusing four.
3. **`threads` joins the frozen config** — it belongs in the reproducibility contract beside seed,
   lr and `pos_weight`. It was already written to `config_*.json` (which is how this was caught);
   what was missing was treating it as *load-bearing*.
4. Retro-qualifies §15's parenthetical *"same seeds as the reported config, so the trajectories are
   identical"* — true at fixed threads, not true across the 2-vs-4 boundary that actually separated
   those runs. It does not change §15's numbers, which were internally consistent at 4 threads.

### 2026-08-17 — The headline re-scored at epoch 21, and the configuration is FROZEN

**Question.** Every reported DL-vs-CFAR number was measured at epoch 50. Epoch 21 is now the frozen
stopping point, so all of them describe a model we no longer report. What survives?

**Setup.** 8 seeds × 21 epochs at `pos_weight=100`, **`--threads 2` throughout**, from commit
`50755c9`. Scored on `bench2_val`, matched measured Pfa 1e-4, paired frame bootstrap (2,000
resamples), Bonferroni across the 7 conditions.

**✅ The frozen arm IS the reported arm, stopped earlier.** At `threads=2` the new runs reproduce
the original `pw100` trajectories **bit-for-bit** — epoch-21 val loss identical to 9 decimal places
on all 8 seeds. So this is not a re-training that happens to be similar; it is the same 8
trajectories truncated, which makes the 21-vs-50 comparison paired by construction and confirms the
thread-count diagnosis from the entry above.

**Pd at the ±1 gate — structure completely unchanged.** 4 of 7 significant, the same 4:

| condition | ep50 Δ | ep21 Δ | ep21 p | verdict |
|---|---|---|---|---|
| clutter_edge | +0.288 | **+0.274** | <0.001 | win, unchanged |
| mixed | +0.116 | **+0.106** | <0.001 | win, unchanged |
| masking | +0.049 | **+0.051** | <0.001 | win, unchanged |
| multi | +0.038 | **+0.037** | <0.001 | win, unchanged |
| isolated | +0.023 | +0.019 | 0.030 | tie (short of 0.0071), unchanged |
| clutter | −0.006 | −0.008 | 0.505 | tie, unchanged |
| unresolved | +0.001 | +0.003 | 0.623 | tie, unchanged |

**⚠️ One conclusion changed, at the ±3 gate: `clutter` is now a *significant* loss.**
−0.0429, p = **0.005**, which clears the Bonferroni threshold of 0.0071 — where at epoch 50 it was
−0.038, p = 0.013, explicitly reported as "edge of significance". **This strengthens the project's
position:** Milestone 17 records that a credible learned detector should lose *somewhere*, and the
one place ours loses is now established rather than marginal. Everything else at ±3 is unchanged.

**Localization improves, and the headline ratio moves — measured, not extrapolated.**

| pooled, gate ±1 | epoch 50 | epoch 21 |
|---|---|---|
| DL range error | 0.498 | **0.473** bins |
| DL Doppler error | 0.485 | **0.393 ± 0.025** bins |
| **Doppler vs best CFAR (OS)** | **5.6×** | **4.6×** |
| range vs best CFAR (SO) | 2.79× | 2.65× |

It improves in **8 of 8** groups on **both** axes. **The "5.6× worse Doppler" figure is retired and
replaced by 4.6×** — it must be quoted at 21, never carried over from 50. The ~5× figure estimated
in `NEXT_SESSION.md` on 16 Aug was a cross-thread-count extrapolation and is also superseded: the
measured value is 4.6×. Note the ratio's denominator is a single CFAR run carrying its own
frame-sampling error, which this ratio does **not** propagate — `bootstrap_comparison.py` is the
source for intervals that do.

**One-line summary, still true and now better supported:** *at finding targets in interference DL
wins; at placing them CFAR does* — by 4.6× rather than 5.6×.

**🔒 THE CONFIGURATION IS FROZEN** → **`configs/frozen_dl.yaml`**, validated field-by-field against
the `config_frozen_s0.json` actually produced by the run. It records model, training, data,
evaluation, CFAR baseline, environment and the exact commands, including `threads: 2` as a
load-bearing field. Artefacts: `results/dl/bootstrap_frozen_ep021_tol{1,3}.csv`,
`localization_frozen_ep021.csv`, `tinycnn_frozen_s{0..7}_ep021.pt`.

The epoch-50 tables (`bootstrap_pw100*.csv`, `localization_pw100.csv`) are **kept on disk
unmodified** as the superseded record.

**`bench2_test` is now unblocked** — it is the next thing to touch, once, with the table
pre-registered first.

### 2026-08-19 — `bench2_test`, the one-shot: DL wins 6 of 7, and loses nowhere

**The pre-registered read**, `docs/test_preregistration.md` (committed 17 Aug, before anything was
seen). Frozen config, 8 seeds at epoch 21, `threads=2`. Tables T1–T5 as specified. Run once.

**T4 first, because everything rests on it.** Measured Pfa at the matched operating point is within
**2%** of the 1e-4 target for **every method on every condition** (worst 1.02×). The operating points
genuinely matched, so the Pd comparisons are like-for-like.

**T1 — Pd at the ±1 gate. 6 of 7 significant after Bonferroni (α = 0.0071):**

| condition | best CFAR | CFAR Pd | DL Pd | Δ | p | verdict |
|---|---|---|---|---|---|---|
| clutter_edge | OS | 0.449 | 0.751 | **+0.302** | <0.001 | DL win |
| mixed | GO | 0.426 | 0.529 | **+0.103** | <0.001 | DL win |
| isolated | CA | 0.675 | 0.733 | **+0.058** | <0.001 | DL win |
| masking | OS | 0.623 | 0.674 | **+0.051** | <0.001 | DL win |
| multi | CA | 0.687 | 0.732 | **+0.045** | <0.001 | DL win |
| unresolved | CA | 0.464 | 0.476 | **+0.012** | 0.001 | DL win |
| clutter | GO | 0.171 | 0.169 | −0.002 | 0.79 | **tie** |

**T2 — ±3 gate: the same 6 of 7.** `clutter` −0.019, p = 0.068 — a **tie**, not a loss.

**⚠️ Three pre-registered predictions failed. All three are recorded here because they were written
down in advance; none is a re-analysis.**

1. **`clutter` at ±3 was predicted a LOSS and came out a TIE.** On val it was −0.043, p = 0.005 —
   significant after correction, and the 17 Aug entry called it "a strengthening" because Milestone
   17 says a credible learned detector should lose *somewhere*. On test it is −0.019, p = 0.068.
   **It does not replicate.** See Correction **C6**.
2. **`unresolved` was predicted a TIE and came out a significant win** (+0.012, p = 0.001, at both
   gates). Absolutely tiny — 1.2 percentage points — and **O2's reasoning still stands**: both arms
   sit at the ~0.5 two-targets-in-one-mainlobe physics cap, and 800 targets makes a negligible
   effect detectable. **It should not be reported as a win**; it is a significant non-difference.
3. **SO was predicted to remain the best classical method on range localization.** On val it was
   best on 5 of 7 conditions; **on test only 2 of 7**, and **OS** takes the pooled range *and*
   Doppler minimum. The §5 decision to run SO was still right — it is best on `clutter` and `multi`,
   so excluding it would still have biased T3 — but the specific claim was wrong.

`isolated` crossed into significance, which §4 explicitly flagged as uncertain in advance, so it is
not a failed prediction. Worth noting anyway that the delta **tripled** (+0.019 → +0.058), which is
more than the √2 tighter error bars explain.

**T3 — localization. CFAR still wins decisively, by less than predicted.**

| pooled, gate ±1 | DL (8 seeds) | best CFAR | ratio |
|---|---|---|---|
| range error | 0.476 ± 0.021 | 0.198 (OS) | **2.40×** |
| Doppler error | 0.382 ± 0.026 | 0.100 (OS) | **3.83×** |

Predicted 2.65× and 4.57× from val; both came out **lower**. DL's own errors landed within 0.012
bins of the val values, so the movement is on the **CFAR** side — its localization is slightly worse
on test. DL is worse than the best CFAR on **every condition and both axes**, as predicted;
`unresolved` is DL's best condition and `clutter` its worst, also as predicted.

**The headline ratio is now 3.83×, measured on test.** It supersedes 4.6× (val, epoch 21) and 5.6×
(val, epoch 50). All three are kept on record.

**T5 — computational cost** (test, `threads=2`, batch 1, CPU, median over 200 frames):

| method | ms/frame | vs DL |
|---|---|---|
| CA-CFAR | 0.86 | 0.14× |
| **DL (TinyCNN)** | **6.04** | 1.00× |
| GO-CFAR | 7.79 | 1.29× |
| SO-CFAR | 7.82 | 1.29× |
| OS-CFAR | 69.95 | **11.6×** |

TinyCNN is 4,896 MAC/cell (80.2 MMAC/frame), 5,041 params; CFAR has none. **DL is ~7× slower than
CA and ~11.6× faster than OS.** Against GO/SO it is within noise — DL was *slower* than both on the
val run and *faster* on the test run, so that ordering is machine-load dependent and must not be
claimed either way. The §4.2 premise correction is confirmed: our CFAR is vectorised, so DL is not
beating a per-cell loop; it beats OS specifically because rank-selection vectorises badly.

**Where this leaves the headline.** *At finding targets in interference DL wins; at placing them
CFAR does, by ~4× in Doppler; and it runs at roughly GO/SO cost, far cheaper than OS.* But **DL now
loses nowhere**, which is the weaker position to report from and revives the Milestone 17 concern
directly. That is the honest state of the evidence and it goes in the report as such.

**Scope, stated in §7 before the read:** `bench2_test` comes from the same generator as `bench2_val`
— different frames, different seed, no shared content, but not a different world. This establishes
**sampling generalization only**. It is not evidence that any finding survives a different clutter
model, a different sensor, or real measurements.

Artefacts: `results/dl/bootstrap_frozen_ep021_test_tol{1,3}.csv`, `localization_frozen_ep021_test.csv`,
`measured_pfa_frozen_ep021_test_tol1.csv`, `runtime_cost_frozen_ep021_test.csv`.
`scripts/measured_pfa_table.py` and `scripts/runtime_cost.py` are new.

**`bench2_test` is now spent.** Any further read must be recorded as a second read, with its reason.

### 2026-08-19 — Pd vs SNR against theory: the absolute check, and it found something

**Question.** Every other result is *relative* — DL against CFAR at a matched operating point. This
is the one **absolute** check: is either detector anywhere near the physics? `bench2_snr_sweep`,
1,300 frames, one target each, SNR swept 0–24 dB, no clutter and no interference.
Theory is the square-law detector that **knows** the noise level: `Pd = Pfa^(1/(1+SNR))`.

**⚠️ The first run was invalid, and the script's own sanity check caught it.** At the project's ±1
gate **every method — all four CFAR variants and the network — sat ~0.3 Pd ABOVE theory**, reaching
Pd 0.5 some 1.7–2.8 dB earlier than theory says is possible. That is not five superior detectors.
Theory is the probability that **one specific cell** exceeds the threshold; the ±1 gate counts a hit
anywhere in a **3×3** neighbourhood — nine chances, and the mainlobe spans ~2 bins so the neighbours
carry real target energy. **Theory is only a bound at gate 0.**

Re-scored at **gate 0 (exact cell)** the anomaly vanishes entirely:

| | theory | CA | GO | OS | SO | DL |
|---|---|---|---|---|---|---|
| SNR at Pd 0.5 | **10.89 dB** | 11.37 | 11.55 | 11.68 | 11.52 | **never reaches 0.5** |
| detection loss | — | **+0.47** | +0.65 | +0.78 | +0.63 | — |

**CFAR loss of 0.47–0.78 dB is textbook** — exactly the price expected for estimating the noise
level from a finite reference window. The pipeline is validated against physics, which is the point
of the figure.

**Two things the gate-0 panel shows that nothing else did.**

1. **DL's localization deficit, stated as a detection number.** On exact-cell matching the network
   plateaus at **Pd 0.41** even at 24 dB, where CFAR reaches ~0.85. The network finds the target and
   then puts its peak *somewhere in the 3×3 label block* rather than on the target cell. This is
   §14/§16/§19-T3 measured a completely different way, on a different split, and it is the most
   direct evidence yet that the 3×3 label — not detection ability — is the limitation.
2. **CFAR itself tops out at ~0.85, not 1.0.** That is the ~93% label-to-peak alignment of
   Milestone 22 showing up as a ceiling: the label is not always exactly on the physical peak, so
   exact-cell matching cannot reach 1 for anyone.

**Reported as a two-panel figure** — gate 0 (the valid comparison, with the CFAR-loss number) beside
gate ±1 (the project's operating point, with theory drawn only as a reference shape and explicitly
labelled *not a bound*). Publishing only the ±1 panel would have shown five detectors beating
physics.

`scripts/pd_vs_snr_figure.py`, `results/pd_vs_snr_frozen_ep021.png`,
`results/dl/pd_vs_snr_frozen_ep021{,_summary}.csv`. CFAR on this split:
`scripts/cfar_test_baseline.py --split bench2_snr_sweep` → `results/cfar_snr_sweep/`.

*(Note: that script's final summary pivot fails on this split because `bench2_snr_sweep` frames
carry no `condition` label, so the groupby is empty. Every artefact this figure needs — peaks,
frames.csv, truths.csv — is written before that point. Not fixed: the split has one condition by
construction, so a per-condition summary is meaningless for it.)*

**This was the last outstanding experiment.** Everything remaining is writing.

### 2026-08-19 — `cfar_baseline.ipynb` restructured, and SO had been missing from it entirely

**The correctness fix, and it was not cosmetic.** The notebook declared
`METHODS = ["CA", "GO", "OS"]`. **SO-CFAR was absent from every table, plot and colour map**, even
though `peaks_SO.npz` had been on disk since 17 Aug. So:

* every **"best method"** column was a maximum over **three** variants, not four;
* the **localization** table omitted SO entirely — the variant that is best on the range axis in
  most conditions on `bench2_val`;
* the runtime table understated the classical baseline's cheap end.

Fixed throughout: `METHODS`, `COLOR`, peaks/runtime loading, and every table and figure. Verified
`peaks_SO.npz` loads and carries `ms_per_frame` (6.57 ms).

**⚠️ What adding SO changed — the headline claim is now weaker, and correctly so.**

The old separability test compared **best against worst**, which passed on four conditions. With SO
included that column is dominated by *how bad SO is* (SO scores 0.063 on `clutter` against GO's
0.171), so it stopped measuring what it was for. The meaningful test is **best against
runner-up**, and on that test:

| condition | winner | runner-up | margin | SE of winner | separable? |
|---|---|---|---|---|---|
| clutter | GO | CA | 0.0050 | 0.0141 | ❌ |
| clutter_edge | OS | GO | 0.0339 | 0.0212 | ❌ |
| isolated | CA | **SO** | 0.0100 | 0.0235 | ❌ |
| masking | OS | **SO** | 0.0138 | 0.0128 | ❌ |
| mixed | GO | CA | 0.0178 | 0.0179 | ❌ |
| multi | CA | OS | 0.0055 | 0.0139 | ❌ |
| unresolved | CA | GO | 0.0012 | 0.0066 | ❌ |

**Not one winner is separable from its runner-up at 2 SE.** Note also that SO is the *runner-up* on
two conditions — it was invisible before, and it is not a weak method everywhere.

**But the mechanism prediction still scores 7 of 7.** Every point estimate orders exactly as CFAR
theory says it should — CA on homogeneous noise, OS when an interferer is in the ring, GO inside a
ridge, OS at its edge (the opposite geometry), SO nowhere. Seven-for-seven directional agreement
with an independently derived mechanism is unlikely to be chance, but it is **not** the same as
having measured each gap. **The reportable claim is "the variants are complementary in the direction
theory predicts", not "OS beats GO on masking by 0.09".** Separating them would need substantially
more than 400 frames per condition.

*Adding SO did not create this ambiguity — it exposed it.* The three-method version reported four
"real" differences that were artefacts of comparing against the worst method rather than the second
best.

**Second bug found, smaller.** `summary.csv` recorded the **full-ring** `alpha` for SO, while
`cfar.py` uses the **half-ring** value (`alpha_go()`) for it — SO is a half-ring statistic like GO.
Display metadata only: no detection or reported Pd ever depended on it, and correcting it moves SO's
`clutter` threshold ratio only 26.6 → 25.7. `scripts/cfar_test_baseline.py` is fixed for future
runs; the notebook recomputes rather than trusting the stale column, so **`bench2_test` was not
re-read**. (SO genuinely does need ~25× its design threshold on clutter — taking the *minimum* of
two half-rings biases the noise estimate down, so its design Pfa is optimistic. That is a property
of SO, not the bug.)

**The restructure: 11 sections → 7 + appendix.**

| new | contents |
|---|---|
| §1 | the frame: split composition, geometry, **the metric hierarchy stated once**, and why *measured* Pfa. Fixes the `nms_size=5` vs 11×11 ambiguity — it is the **half-width** |
| §2 | **how the CFAR window was chosen** — entirely new, from Milestone 19 step 2a and step 6 |
| §3 | the baseline result: headline table with bootstrap CIs, the win-by-condition table with a **mechanism column**, and the `clutter_edge` dilution moved next to it |
| §4 | why the methods differ: noise-estimate map, design-vs-measured threshold, false-alarm locations |
| §5 | validation against physics, **promoted** from §6: Pd vs SNR with per-method CFAR loss in dB, plus the resolution ablation |
| §6 | secondary metrics: localization for all four methods, bootstrapped, at both gates; runtime with measurement conditions |
| §7 | what this establishes + limitations |
| A | appendix: per-target record, representative ROCs |

**Removed:** the pairwise method-agreement section (§2d) — it supported no claim the win table and
separability test do not already make. **Shrunk:** ROC-per-condition to three representative
conditions (clean / interference / edge), remainder to the appendix.

**Three new figures in §2, which is where the notebook was weakest:**

1. **The measured PSF profile** on a log scale, with the guard boundary and *both* window edges
   marked (range ±10 and Doppler ±4 — one line would have been wrong for one of the two curves).
   The −30 dB spread sits inside the guard and the ring starts below −40 dB: the whole guard = 2
   justification in one picture. New script `scripts/export_psf_profile.py` persists the profile so
   nothing is pasted.
2. **The geometry sweep per condition** as a heatmap plus a gain bar — the pooled sweep alone is the
   artefact (pooled Pd correlates r = 0.98 with *usable Doppler bins*), and the per-condition view
   is what separated real mechanism from artefact.
3. **The 21×9 window drawn to scale on a real `clutter_edge` frame**, with the ridge extent shaded.
   The first draft's title claimed the ridge was inside the ring; it is not, and the figure showed
   that. Corrected to what is actually true and what makes the mechanism work: **the ridge centre is
   outside the window, but 2 of the ring's training columns overlap the ridge's ±4-bin extent**,
   which is enough to inflate the noise estimate.

⚠️ **Discrepancy recorded, not resolved.** `results/cfar_geometry_percondition.csv` reproduces the
*direction and ordering* of the step-6 sweep but not its absolute values (`isolated` 0.272 → 0.306
in the artefact against 0.654 → 0.629 in the step-6 entry) — it is a different run of the sweep. The
notebook plots the artefact and describes the artefact, with a banner warning against quoting the
two together. The conclusions the window selection rested on hold in both: gain concentrated in
clutter-bearing conditions (9.6× the mean elsewhere) and `masking` regressing at 29×9.

`bench2_test` was **not** re-read — SO's detections already existed; this was display only.

### 2026-08-19 — CFAR alpha audit: **none of the four variants is calibrated**, SO least of all

**The question.** SO was added on 15 August, after the Milestone 9 calibration work. Does a
calibrated alpha exist for it, or does `CfarConfig` fall back to a closed form?

**The answer is worse than "SO was missed".** Three findings, in order of severity:

1. **`cfar.py` never imports `calibrate.py`, and there is no cached alpha table.** Every variant
   runs on a **closed-form** alpha. The Milestone 9 values (CA 9.74, GO 9.09, OS 6.98) exist in that
   entry and nowhere in the code path.
2. **Those values were for a different geometry anyway** — `calibrate_all()` takes scalar
   `guard=2, train=8`, i.e. the square 21×21 window, superseded by the rectangular 21×9 chosen in
   step 6.
3. **SO was never put through the procedure at all**: `calibrate_all()` still defaults to
   `kinds=("CA", "OS", "GO")`.

**The direct test** — `scripts/cfar_alpha_audit.py`, 200 pure-noise frames per seed through the real
processing chain, calibration seed 0 and verification seed 9000, the same pair Milestone 9 used.
On pure noise every variant should achieve 1e-4 at its own alpha.

| method | alpha in use (closed form) | calibrated alpha | achieved per-cell Pfa @ alpha in use (seed 9000) | **× target** |
|---|---|---|---|---|
| CA | 9.474 | 10.450 | 2.37e-4 | **2.37×** |
| GO | 9.808 | 9.534 | 7.76e-5 | **0.78×** |
| OS | 9.474 | 7.623 | 1.32e-5 | **0.13×** |
| SO | 9.808 | 12.888 | 6.55e-4 | **6.55×** |

**Not one lands on 1e-4.** OS is ~8× too *conservative*, SO ~6.5× too *permissive* — a spread of
about 50× between the two extremes. At the **calibrated** alpha, on the independent verification
seed, all four land at 0.90–1.01 × target. **The procedure works; it was simply never applied to the
shipped configuration.**

**Is it the alpha or the post-NMS offset? The alpha.** The NMS survival fraction (post-NMS Pfa ÷
per-cell Pfa) is **0.66–0.83** across the four — a roughly *common* factor. The per-cell deviations
are **0.13×–6.55×** and method-*specific*. A common factor cannot produce a method-specific spread,
so the `noise`-row gap in §4.2 (CA 1.05, GO 0.93, OS 0.76, SO 1.25) is the uncalibrated alpha.
Note the orderings agree: the two methods over-firing on noise (CA, SO) are the two whose measured
threshold has to be pushed *above* design, and OS, which under-fires, is the one pulled *below*.

**⚠️ What this does and does not affect.**

* **No reported Pd changes.** Every comparison in this project sweeps the threshold to matched
  *measured* Pfa **per condition**, which discards the closed-form alpha entirely. §3, §5, §6 of the
  CFAR notebook, the whole DL-vs-CFAR comparison, and `bench2_test` are all unaffected.
* **§4.2 is affected**, being the one place that reads the design threshold — and its SO row
  specifically, which was the largest distortion.
* **`bench2_test` was not re-read.** The audit runs on freshly generated noise, not on any split.

**Correction C7 — "alpha is calibrated empirically, and the empirical value is the one used".**
That was §2.4's claim in `cfar_baseline.ipynb`, and it was never true of the shipped code. The
reasoning error: `calibrate.py` was written, validated and documented in Milestone 9, and its
existence was then treated as evidence that it was *in use*. Nothing checked that the value reached
`CfarConfig`. **A module that exists is not a module that runs** — the same class of error as
assuming a flag is honoured because it is defined.

**§2.4 rewritten with the evidence** (it was markdown-only, no code, no numbers): closed-form vs
calibrated alpha per method with the ratio, achieved Pfa on both seeds, the NMS-vs-alpha separation,
and a figure of achieved Pfa vs alpha per method with the 1e-4 target and both alphas marked — every
star on the target line, every circle off it.

**§4.2 fixed**, three ways:
* the figure title claimed "theory is wrong by up to ~4×" while the chart showed 30× — the same
  wrong number sat in §7 point 5. Both now read from the data;
* **log y-axis**, since SO spans two decades and was flattening CA/GO/OS into an unreadable band;
* **each condition's ratio is now normalised by that method's own `noise` row**, separating the
  calibration offset from the clutter effect. The check that this works: the clean conditions
  collapse to **1.00–1.02**. What remains on the clutter rows is the clutter effect proper —
  **CA 2.27×, GO 1.75×, OS 2.75×, SO 24.2×**.

**§7 point 2 gained the other half of its argument.** It said no winner is separable from its
runner-up at 2 SE, which is true and was only half the story: **7 of 7 winners match a mechanism map
specified in advance.** Under the most conservative null — a toss-up between the top two on every
condition — that is (1/2)⁷ ≈ 0.008; against a uniform choice among four variants, (1/4)⁷ ≈ 6e-5.
Individually weak, jointly hard to get by luck, exactly like the 7-of-7 monotone localization result
in the DL log. Both halves are now stated together.

**Not fixed, deliberately:** the calibrated alphas are **not** wired into `CfarConfig`. Doing so
would change every design-threshold number in the project and would require re-running the CFAR
baselines to keep `summary.csv` consistent — for no change to any reported Pd, since everything is
scored at matched measured Pfa. It is recorded as a limitation with the measured values available in
`results/cfar_alpha_audit.csv` for anyone who wants them.

### 2026-08-19 — Post-NMS vs per-cell Pfa: the CFAR loss was understated by 0.1–0.2 dB

**The gap.** §5 of `cfar_baseline.ipynb` compares measured Pd against `Pd = Pfa^(1/(1+SNR))`, which
is a **per-cell** statement. Both arms are matched at a **post-NMS** Pfa of 1e-4, counted over
surviving local maxima. Those are not the same operating point.

Using the NMS survival fractions measured in the alpha audit (0.657–0.829), a post-NMS rate of 1e-4
corresponds to a per-cell rate of **1.21–1.52e-4**, so the detectors were being scored at a slightly
*easier* point than theory was drawn at. Inverting the theory curve at Pd 0.5 is closed form
(`SNR = ln(Pfa)/ln(0.5) − 1`), so the corrected anchor is computed rather than estimated:

| | CA | GO | OS | SO |
|---|---|---|---|---|
| NMS survival | 0.712 | 0.771 | 0.829 | 0.657 |
| implied per-cell Pfa | 1.40e-4 | 1.30e-4 | 1.21e-4 | 1.52e-4 |
| theory anchor (dB) | 10.718 | 10.760 | 10.798 | 10.675 |
| CFAR loss **as plotted** | 0.273 | 0.427 | 0.553 | 0.644 |
| CFAR loss **corrected** | **0.450** | **0.562** | **0.650** | **0.864** |

Anchor as drawn: **10.895 dB**. **The reported loss is understated by 0.10–0.22 dB**, and the
corrected range is **0.45–0.86 dB** against 0.27–0.64 as plotted. **The conclusion is unchanged** —
still a fraction of a dB, still comfortably below the known-noise bound — but it is now stated
rather than left for a reader to work out.

**Not corrected in the figure, deliberately.** The survival fraction is measured on *pure noise*,
where every surviving peak is a false alarm; on target-bearing frames the mix differs, so applying a
single per-method scalar to the plotted curves would swap a stated bias for an unstated one.

**⚠️ This affects the absolute comparison ONLY.** The DL-vs-CFAR result is untouched: both arms use
the same `all_peaks()` with the same `nms_size=5` and are matched at the same post-NMS Pfa, so the
offset is **common-mode and cancels in the difference**. It can move where both sit relative to
*theory*; it cannot move where they sit relative to *each other*. Noted in
`cfar_vs_dl_comparison.ipynb` §6, which is the only place that notebook makes an absolute claim.

**Also this session, checked rather than assumed:** §7 points 2 and 5 of `cfar_baseline.ipynb` had
already been corrected in the previous commit — point 5 reads 1.75–2.8× / ~24×, and point 2 already
carried the 7-of-7 joint-pattern argument. A sweep for the stale "~4×" claim found no surviving
instance: the remaining `~4×` hits are the DL Doppler ratio (3.83×) or the unrelated NMS
candidate-peak figure. §7 point 4 now quotes both the plotted and the corrected loss range.

**C7 promoted to a first-class limitation.** The uncalibrated-alpha finding was recorded only as a
Correction; it is now **§9.8 of `cfar_vs_dl_comparison.ipynb`** (the report-facing notebook) and is
in `NEXT_SESSION.md`'s known-open list. It changes no reported Pd — everything is scored at matched
measured Pfa — but a reader who opens `CfarConfig` will find closed-form alphas, and should meet the
disclosure first rather than discover it.

### 2026-08-19 — The geometry-sweep discrepancy resolved: it was `nms_size = 2`

**The problem.** `results/cfar_geometry_percondition.csv` disagreed with the step-6 table in this
file on the **sign** for two conditions, not just on absolute values — `isolated` −20.025 recorded
against +0.036 in the CSV, `multi` +0.006 against −0.015. The note added to the notebook on 19 Aug
claimed the artefact "reproduces the direction and ordering" and then cited `isolated` as its
example, which is one of the two that flip. **The note contradicted itself and has been removed.**

**Cause — three scoring differences, the first decisive.**

1. **`nms_size`.** `cfar_geometry_percondition.py` calls `all_peaks(ratio, valid)` with no explicit
   size. At the time it was written that defaulted to **2**, a transient state during the step-6d
   NMS investigation; it now defaults to **5**, the project standard. **The script therefore no
   longer reproduces its own CSV** — the artefact is from a detector configuration that no longer
   exists anywhere else in the project.
2. **Interpolated Pd.** It reads Pd off the ROC by interpolation. The project later rejected that:
   interpolation finds the *threshold*, and Pd is MEASURED at it, because interpolated Pd carries up
   to 0.017 of error and is biased optimistic on the clutter strata — exactly the strata this sweep
   is about.
3. **Threshold grid.** Uniform quantiles, rather than the false-alarm-anchored grid adopted after
   Milestone 20b.

**Resolution: regenerated, not reconciled.** `scripts/cfar_geometry_resweep.py` re-runs both sweeps
on `bench2_tune` through the same path as `cfar_test_baseline.py` (NMS 5, FA-anchored thresholds, Pd
measured at the matched threshold). The result **matches the recorded step-6 run**, and both sign
flips disappear:

| condition | step-6 (recorded) | old CSV (nms=2) | **v2 (current protocol)** |
|---|---|---|---|
| clutter_edge | +0.296 | +0.129 | **+0.294** |
| mixed | +0.185 | +0.060 | **+0.193** |
| multi | +0.006 | −0.015 ✗ | **+0.000** |
| masking | +0.003 | +0.011 | **−0.003** |
| unresolved | +0.001 | +0.018 | **+0.000** |
| clutter | −0.016 | +0.001 | **−0.013** |
| isolated | −0.025 | +0.036 ✗ | **−0.013** |

`masking` at 29×9: recorded −0.041, v2 **−0.043**. Absolute levels agree too — `clutter_edge`
0.095 → 0.390 against the recorded 0.094 → 0.390.

**`results/cfar_geometry_percondition_v2.csv` is authoritative**; the original is kept, marked
non-authoritative, and the notebook now reads v2. The window choice never depended on this: the
conclusions — gain concentrated in clutter-bearing conditions, `masking` regressing at 29×9 — hold
in every version. What changed is which numbers we quote. On the corrected data the concentration is
sharper: **+0.244 on clutter-bearing against −0.006 elsewhere**, and the clean conditions do not
merely fail to gain — `isolated` and `clutter` **lose** 0.013 each, which is the opposite of the
uniform shift a pooling artefact would produce.

**Two figures added to §2.2, both previously missing evidence.**

* **The candidate windows drawn to scale**, one row of panels, with the clutter ridge shaded and the
  count of ring columns falling inside it: 21×21 puts **8** training columns in the ridge, 21×9 puts
  **2**, and 29×9 has the same Doppler width but a visibly taller range extent — which is what costs
  it `masking`. The choice is legible before any number is read.
* **The train-cell sweep**, which justified `train = 8` in Milestone 9 but never had its data
  persisted. Regenerated on `bench2_tune`: pooled Pd rises +0.0053 per step up to `train_r = 8` and
  +0.0010 per step after, changing sign — a noise floor. The knee is at 8, and past it the ring is
  no longer improving the estimate, only making it less local.

### 2026-08-21 — Calibrated alphas wired into `CfarConfig`; **`bench2_test` was NOT re-read**

Closes the C7 finding: alpha is now measured rather than assumed, and the claim §7 had been making
since before it was true is true (Correction **C8**).

**What was done.**
* `calibrate_all()` fixed to include **SO** and to take the rectangular geometry — it defaulted to
  `kinds=("CA","OS","GO")` and to the square `guard=2, train=8`, so SO had never been calibrated and
  the others had been calibrated for a window the project does not use.
* All four calibrated by Monte Carlo on 300 pure-noise frames (seed 0) and **verified on held-out
  seed 9000**. A variant is only written to the table if it passes within 25% of target.
* Cached to **`configs/cfar_alpha_table.json`**, keyed by kind + geometry + Pfa, and read by the new
  `CfarConfig.resolved_alpha()`. All four `cfar_ratio_map` call sites use it; it falls back to the
  closed form when a geometry is absent, and `calibrated_alpha_available()` reports which was used.

| | CA | GO | OS | SO |
|---|---|---|---|---|
| closed form | 9.474 | 9.808 | 9.474 | 9.808 |
| **calibrated** | **10.459** | **9.546** | **7.579** | **12.849** |
| achieved/target, closed | 2.37× | 0.78× | **0.13×** | **6.55×** |
| **achieved/target, calibrated (held-out seed)** | **0.96×** | **1.01×** | **0.95×** | **1.00×** |

A ~50× spread between variants collapses to 0.95–1.01×. The held-out seed is the point: these were
fitted on seed 0 and land on target on seed 9000, so they generalise rather than fitting the
calibration draw.

**⚠️ The re-run of `bench2_test` was authorised and was NOT taken, because it cannot matter.**
`cfar_ratio_map` returns `power / noise`; **alpha is the threshold applied to that map, not an input
to it**. Asserted directly — `scripts/cfar_calibrate_table.py --verify-invariance` confirms the ratio
map is bit-identical under alphas differing by 2×, for all four variants. So the cached detections
could not change, and the summaries were re-derived from them with `--score-only`, which reads no
frame data at all.

**Verification (the step-3 gate):**

| | result |
|---|---|
| `pd`, `bench2_test` | **max \|diff\| = 0.000e+00** — bit-identical |
| `pd`, `bench2_val` | **max \|diff\| = 0.000e+00** |
| `measured_pfa`, `tp`, `fp`, `fn`, `fa_per_frame` | all 0.000e+00 on both splits |
| `threshold` | 3.6e-15 (one ulp of float64 on a value of ~10 — the same number re-computed) |
| `peaks_*.npz`, `bench2_test` | **byte-identical, md5-verified** |
| `design_alpha`, `design_pd`, `design_pfa`, `design_fa_per_frame` | changed, as intended |

Nothing moved that should not have, so the gate did not trip. **This is not a second read of
`bench2_test`** — no frame of it was opened — and the pre-registration's one-shot is intact.

**§4.2 simplified, and the simplification is itself the evidence.** That section carried a
normalisation step dividing each condition by its method's own `noise` row, to separate the
calibration offset from the clutter effect. With calibrated alphas the `noise` row runs **0.952 to
0.959** — a spread of **0.007**, against 0.76–1.25 before. The offset is no longer method-specific,
so the normalisation measures nothing and has been **removed**; the raw ratios are now directly
comparable. The residual common ~0.95 is the **post-NMS** offset (alpha is per-cell, the thresholds
are matched post-NMS), not miscalibration.

What remains above the line is the clutter effect proper: **CA 2.25×, GO 1.74×, OS 2.74×, SO 24.2×**
— still a different factor per method, which is still the reason results are reported at matched
measured Pfa.

**§2.4 rebuilt** around the procedure rather than the outcome, with a two-panel figure: achieved Pfa
vs alpha per method with an arrow from each variant's closed-form position to its calibrated one
(the arrows point in **opposite** directions — OS's alpha had to come down, SO's up, which is why one
closed form could never serve four statistics), and a second panel showing the held-out-seed result.

**§7 point 1 corrected.** It claimed geometry, guard *and alpha* were selected by measurement. The
geometry half was always honest; the alpha half was not, until now. Both corrections are recorded as
**C7** (the false claim) and **C8** (making it true) in `cnn_development_log.ipynb`, which is the
canonical retraction list — C7 had previously been logged only here.

### 2026-08-22 — The two CFAR-loss numbers are two different measurements, now labelled

**The question.** §7 of `cfar_baseline.ipynb` reports CFAR loss of **0.27–0.64 dB**; the SNR-sweep
figure reports **0.47–0.78 dB** (CA 0.47, GO 0.65, OS 0.78, SO 0.63). Same quantity twice, or two
quantities?

**Two quantities.** Confirmed from the artefacts rather than inferred:

| | measurement A | measurement B |
|---|---|---|
| artefact | `results/dl/pd_vs_snr_frozen_ep021_summary.csv` | computed live in `cfar_baseline.ipynb` §5 |
| split | **`bench2_snr_sweep`** | **`bench2_test`**, clean conditions only (`isolated` + `multi`) |
| targets | 1,300 — **one per frame**, SNR swept uniformly 0–24 dB | 2,215 — **1–2 per frame** |
| mean cell SNR | **11.24 dB** | **15.19 dB** (p90 28.9 dB) |
| CFAR loss, gate 0 | **0.47–0.78 dB** | **0.27–0.64 dB** |

Different splits, different target populations, and `multi` contributes mild mutual interference
that the one-target-per-frame sweep has by construction. They are not expected to agree and neither
supersedes the other.

**Did SO or the alpha calibration move either? No — checked, not assumed.**

* **Alpha calibration:** re-ran `pd_vs_snr_figure.py` after wiring the calibrated table in and
  diffed against the pre-calibration artefact. `snr50`, `loss_db` and `pd_max` all **max \|diff\| =
  0.000e+00**. Measurement B likewise unchanged (0.273 / 0.427 / 0.553 / 0.644 both sides). This is
  the expected result and now the *measured* one: CFAR loss is derived from Pd at matched **measured**
  Pfa, which sweeps the threshold and cannot depend on alpha.
* **Adding SO:** SO appears as its own row in both artefacts; the CA/GO/OS values are computed
  per-method and were untouched by its addition.

**Labelled in five places** so the two cannot be read as one range: `cfar_baseline.ipynb` §5 intro
and §7 point 4, `cnn_development_log.ipynb` §20, `cfar_vs_dl_comparison.ipynb` §6 and its scorecard
row, and `NEXT_SESSION.md`. Each now names its split and points at the other measurement.

**✅ Decided 22 Aug — which number the report quotes.** The **Results chapter uses measurement B**
(`bench2_test` clean conditions), because every other reported number comes from that split and
mixing splits inside one chapter invites the confusion these labels exist to prevent. It is quoted
as **0.27–0.64 dB as plotted, 0.45–0.86 dB once the post-NMS-vs-per-cell Pfa offset is corrected**
— §5 computes both. **Measurement A stays as the supporting validation figure**: one target per
frame with SNR swept uniformly is the cleaner design for that particular curve, so it is the better
*picture* even though it is not the headline *number*.

*Also corrected while there:* `NEXT_SESSION.md` still described the alphas as uncalibrated (closed
21 Aug) and counted seven Corrections rather than eight.

### 2026-08-24 — Evaluation-methodology notebook; four parked fixes closed, one queue note retracted

**What was built.** `notebooks/evaluation_metrics_overview.ipynb` (102 cells), the companion to
`dataset_overview.ipynb`: that notebook is the dataset, this one is the ruler. It documents the
scoring protocol end to end — score map → detections, valid region, matching, TP/FP/FN, metric
definitions, ROC and the matched operating point, uncertainty, multiplicity, limitations — so the
Method chapter's evaluation section can be written from it without opening source.

Generated by `scripts/build_evaluation_metrics_nb.py`, per the never-hand-edit rule. The builder
compile-checks every code cell before writing; that check caught three real escape bugs during
development. Verified by full `Run All` each time: 102 cells, 13 figures, 21 tables, zero errors,
no stored outputs.

**`bench2_test` was NOT re-read.** §6.1, §7.1 and §8 read cached peaks and result CSVs only.
Illustrative curves come from `bench2_val` so a methodology figure cannot read as a second look at
the reported split. This follows the 21 Aug precedent: reading cached detections is not a read of
the split.

#### Four of the six parked fixes are closed

**Item 1 — localization on the common detected subset.** `scripts/localization_common_subset.py`
existed but had never completed a run; it has now (~75 min, 8 seeds × 8 groups).

**The 3.83× Doppler gap survives.** Pooled it moves to **3.63×**, and range 2.40× → **2.31×**, on a
common subset that is **94.3%** of the CFAR arm's detected targets.

| | own detected sets | common subset |
|---|---|---|
| pooled Doppler ratio | 3.83× | **3.63×** |
| pooled range ratio | 2.40× | **2.31×** |

The ratio shrinks on 5 of 7 conditions and grows on 2. That is the selection effect, measured: the
CFAR arm's detections are largely a subset of the network's, so intersecting removes mostly DL-only
targets — the harder ones DL found and CFAR missed — which improves DL's average. The effect exists
and it is small. **`clutter` moves the other way and substantially (6.44× → 8.80×)**: it is the
condition where the arms overlap least and the only one where they tie on Pd, so the intersection is
a genuinely different population there and the ratio is correspondingly unstable. That condition's
localization carries the caveat.

⚠️ **Post-hoc.** Not in `docs/test_preregistration.md`; listed there and in the comparison notebook
§9.4 as a limitation deliberately not run. Reported beside the pre-registered number, never instead
of it. Artefact: `results/dl/localization_common_frozen_ep021_test.csv`.

**Item 2 — re-select the best CFAR variant inside each bootstrap replicate.** New script
`scripts/bootstrap_best_cfar_reselect.py`. Both procedures run on the *same* resamples, so any
difference is attributable to the re-selection rather than to bootstrap noise.

**0 of 7 verdicts changed, and the intervals did NOT widen** — ratios 0.952 to 1.000.

The selection uncertainty is real: the globally chosen winner is displaced in **29%** of replicates
on `unresolved` and **21%** on `clutter`. But it is immaterial, and the reason is worth recording
because it is not luck. The ranking only flips when the variants are within noise of each other,
which is precisely where swapping one for another changes Pd by almost nothing. **The original
procedure's assumption was not load-bearing**: the fixed-winner intervals were not overconfident,
because the quantity they conditioned on is insensitive to the condition.

⚠️ Re-selection is conservative for DL by construction — taking a maximum over four variants on the
same frames used to score them biases the CFAR arm up. It cannot manufacture a DL win.
⚠️ Neither procedure propagates the variability of *choosing the threshold*; that was true of the
original too and is stated in both. Artefacts:
`results/dl/bootstrap_reselect_frozen_ep021_test_tol1.csv` and `..._diffs.npz`.

**Item 3 — the three Pfa notions** are defined once, in §5.1, with the per-cell/post-NMS relation
measured from `cfar_alpha_audit.csv` rather than asserted (NMS survival 0.66–0.83, so a post-NMS
1e-4 is a per-cell 1.21e-4 to 1.52e-4).

**Item 5 — `p = 0` is never printed.** `fmt_p()` reports anything below the two-sided resolution
floor (2/2000) as `p < 0.001`, and is used at every p-value site in the notebook.

**Items 4 and 6 are written up** as §7.2 (the oracle-threshold wording) and §9 (Bonferroni stated
rather than applied silently) but still have to land in the Method chapter itself.

#### ⚠️ Retraction of a queue note — `unresolved` DOES survive Bonferroni

`NEXT_SESSION.md` item 6 stated that `unresolved` (+0.012) "does not survive correction", and used
that as a second independent reason to read it as a tie. **That is false.** Measured:
p = 0.001 < α = 0.0071 at **both** gates, and the `sig (Bonf)` column of
`bootstrap_frozen_ep021_test_tol{1,3}.csv` has said `True` since the 19 Aug run.

**The reasoning error:** the parked note inferred a statistical property from a physical one. Because
`unresolved` was correctly read as a tie at the physics cap, the note assumed the statistics would
agree. They do not, and they did not need to.

**No published number changes, so this is not a new Corrections-log entry.** C1–C8 are retractions
of reported results; this is an error in a to-do note, and every artefact and every table in
`PROGRESS.md` Milestone 23 already reported `unresolved` as significant. Recording it here so the
queue note cannot be believed later.

**The correct reading, now in §9.2:** `unresolved` is **statistically significant and physically a
tie** — ~1 percentage point against a ~0.5 ceiling both arms are pinned against, with a p-value
sitting at the bootstrap's own resolution floor. It should still not appear in a win list. This is
the stronger version of the point, because the multiplicity correction does *not* rescue the
reading — judgement about effect size has to. It is also the better teaching example for the report:
a correction is not a substitute for asking whether an effect is large enough to mean anything.

#### Protocol checks, all computed live in the notebook

* **0 forgiven duplicate detections out of 6,519** at the matched operating point across 3
  conditions × 5 methods on `bench2_val`. The guard-region convention (§4) is a safety net that
  currently catches nothing — NMS removes duplicates before the evaluator has to forgive them. That
  is the intended state and it is now measured rather than assumed.
* **12 evaluator edge-case checks pass** (empty truths, empty detections, duplicates, one detection
  between two truths, out-of-tolerance, border and valid-mask handling, localization conversion,
  threshold application, ROC monotonicity, gate monotonicity), with an assert. There is no pytest
  suite in this project; these cells *are* the regression check and run on every rebuild.
* **The two detection paths agree** — `extract_detections` (threshold inside) vs `all_peaks` +
  filter (the path every reported number uses) — on 15 frames at each of the 12 CFAR matched
  operating points, α 7.21–308.74, all returning 2–246 detections so the agreement is not vacuous.
  They are *not* identical by construction: `_collapse_plateaus` keeps one representative per
  connected region of the peak mask, and that region can differ when a plateau straddles the
  threshold. Hence the check.

#### New limitation — the DL ROC cannot be swept past sigmoid 0.5

`all_peaks` keeps cells with `score > 0`. Measured on one frame: it removes **0 of 115** CFAR local
maxima (the ratio is positive by construction) and **67 of 80** DL ones, because on logits
`score > 0` means sigmoid > 0.5.

**No reported number is affected** — everything removed sits at or below 0 against operating cuts of
1.39–1.70 logits, so those peaks would have failed the cut anyway. But the DL ROC is *structurally*
unable to sweep past that point toward higher Pfa, and the network's "candidate count" is a
post-floor number rather than every local maximum.

Per the 15 Aug rule this goes on the limitations list, not the work queue — it changes no published
number. It would only start to matter for an experiment wanting a **high-Pfa** DL operating point,
which the fixed-threshold / CFAR-ness experiment might, since one threshold held across conditions
could push some conditions well above 1e-4. Worth a line in the write-up so nobody later reads a
truncated DL ROC as a detector property. Documented in appendix A.2.

#### Two corrections made inside the notebook during review

Both were caught by checking rather than by reasoning, and both are recorded because the pattern is
the point.

1. **The stage figure drew the score threshold before NMS.** The pipeline does the opposite:
   `all_peaks()` extracts every local maximum from the raw score map and the cut is applied
   afterwards by the caller. §0's diagram was already right, so the notebook contradicted itself.
   Redrawn; two other places that implied threshold-first ordering (§5.1's Pfa funnel, §0's caption)
   were fixed with it.

2. **The `masking` ceiling was attributed to the NMS-first ordering. It is not.** Ordering is
   **computational** — peaks are extracted once so the ROC sweeps a cut over a fixed candidate set
   instead of re-running the 11×11 max filter ~95 times — and gives the same detections either way.
   The ceiling comes from the **footprint**: radius 5 is an 11×11 suppression neighbourhood, so
   resolvable targets within ±5 bins (0.98 m in range, 2.10 m/s in Doppler) suppress one another,
   and that holds under either ordering. The two claims are now stated separately.

   *A caption claim was also retracted.* It called the missing weak `masking` target "a threshold
   outcome, not an NMS one". Checked: **neither arm produced a candidate peak within ±5 bins of it**.
   It was not rejected by the cut, because it never reached the cut — it never became a local
   maximum. Nor is it the pairwise NMS interaction, since the pair is 8 range bins apart, outside
   radius 5. Three candidate explanations were on the table and the true one was none of them. The
   assertion is replaced by a computed check.

#### The matching gate and the guard region are COUPLED — measured, and it affects how ±3 is described

Found while writing notebook §3.1. `evaluate_frame` builds the guard region from the **same
`MatchConfig` that defines the matching gate**, so the two are not independent knobs: widening the
gate from ±1 to ±3 also widens the guard, and the false-alarm denominator shrinks with it.

| | gate ±1 | gate ±3 | change |
|---|---|---|---|
| guard cells, one truth | 9 | 49 | 5.4× |
| `cells_tested`, `isolated` (`bench2_val`) | 5,282,800 | 5,266,800 | **−0.30%** |
| `cells_tested`, `multi` | 5,270,065 | 5,197,517 | **−1.38%** |

The size of the effect scales with target density, as it must — `multi` carries 4–5 targets a frame
against `isolated`'s one.

**This does not invalidate anything, and the reason matters.** Within a single gate the denominator
depends only on the truths, the valid mask and the tolerance — **never on the detector** — so both
arms are measured against identical denominators and the matched operating point is genuinely
matched. No number reported at ±1 is affected by what ±3 would have done.

**What it does mean is a wording change for the report.** ±1 and ±3 are **two different operating
points**, not one result viewed two ways: the threshold achieving a measured Pfa of 1e-4 differs
slightly between them, because the background it is measured over differs. The ±3 row is therefore a
*re-run of the entire protocol at a looser tolerance*, which is a stronger robustness check than it
would be if the two shared a denominator. **The rule: never read a Pd from one gate against a Pfa
from the other.** Any sentence in the report that describes ±3 as "the same experiment re-scored"
needs correcting to that.

A §10 check now asserts the coupling exists, so it cannot change silently (13 checks, all pass).

*Also corrected in §3 while there:* the caption claimed the ±3 gate converts **two** misses into hits
in the toy example. It converts **one** (TP 2→3, and one FP reclassified as forgiven). The count is
now computed from the figure's own `FrameEval` objects rather than written by hand. And "±1 bin is
one resolution cell in each dimension" was wrong twice over — the region is **nine** cells, and a
range bin (0.195 m) and a Doppler bin (0.420 m/s) are different physical quantities that should not
be collapsed into one phrase. It now reads "±1-bin tolerance in each axis, forming a 3×3 gate", with
the physical asymmetry stated as a consequence to be aware of.

#### ⚠️ Retracted: the "NMS offset is common-mode" fairness argument

Notebook §5.1 argued that the post-NMS-vs-per-cell Pfa offset **cancels** in the CFAR-vs-DL
comparison "because both arms run the same `all_peaks()` at the same NMS radius". **The reasoning is
invalid.** Running the same NMS code does not imply the same NMS *survival fraction*: survival
depends on how peaked a detector's score map is, which is a property of the detector, not of the
suppression routine.

**The project's own data already refuted it.** All four CFAR variants call the same `all_peaks()`
at the same radius, and their measured survival fractions on pure noise still span **0.66–0.83** — a
factor of 1.26. §1.1 separately shows the network's above-threshold response is far broader than
CFAR's (~10 cells per detection against ~2), so its survival fraction has every reason to differ
again.

**The conclusion stands; only the justification was wrong.** The DL-vs-CFAR comparison is unaffected
for a simpler reason that needs no cancellation assumption: **both arms are matched directly at the
same final post-NMS, guard-excluded Pfa** (notion 3), measured on each arm's own reported
detections. Whatever per-cell rates sit behind those matched post-NMS rates may differ; that is
irrelevant, because no reported comparison is made at the per-cell level. Fairness comes from
equalising the quantity that is actually reported.

✅ **The same flawed justification in `cfar_baseline.ipynb` §5 is now fixed too** (24 Aug). It
carried "the offset is common-mode and cancels in the difference" from the same premise. Replaced
with the direct-matching argument and a cross-reference to `evaluation_metrics_overview.ipynb`
§5.2–5.3, so the two notebooks state the same thing. Rebuilt and re-executed: 66 cells, 14 figures,
zero errors. Its conclusion — that only §5 carries the offset, because only §5 compares against an
absolute per-cell bound — is unchanged.

*Also corrected in §5:* the design Pfa was described with numerator/denominator columns as though it
were measured; it is a **target probability** used to derive or calibrate α, and its cells now say
so. The post-NMS Pfa's interpretation changed from "chance a background cell raises an alarm" to
"false detections per tested background cell", and its row is now named for the notion it is.

*And a claim narrowed:* "notion 3 ≤ notion 2" was stated generally. It does not hold generally — the
numerators nest, but notion 3 also divides by a **smaller** denominator (guard cells removed), and
two ratios shrinking top and bottom cannot be ordered without knowing by how much. The ordering is
now stated only where it holds: **target-free frames**, where there is no guard region and both
notions share a denominator. That is the pure-noise calibration setting the survival fractions are
measured in.

#### Two protocol descriptions corrected in §7, and plain-prose summaries added throughout

**The matched threshold is chosen per METHOD per CONDITION, not per condition.** §7.2 said "the
threshold is selected separately for each condition, using that condition's own measured ROC". That
understates the oracle: `_at_measured_pfa` is called once for every (condition, method) pair, so each
*method* also gets its own threshold on each condition. The evaluation-oracle warning is
correspondingly stronger than it read — a deployed system would need not just per-scene retuning but
per-scene retuning *of that particular detector*. Corrected wording: "selected separately for each
method and each condition, using that method-condition pair's own measured ROC."

**"The dots sit on one vertical line" overclaimed.** The achieved Pfa is close to 1e-4, not equal to
it — 0.977–1.012 of target on `bench2_val` (§7.1), 0.981–1.017 on `bench2_test` (T4). The caption now
says "approximately the same vertical line" and points at the achieved values printed beside each
operating point. This matters because the whole comparison rests on that near-alignment, so the
figure should not imply a precision the table does not support.

**Plain-prose summaries added** at the end of §§4, 5, 6, 6.1, 7, 7.1 and 7.2, written to be liftable
into the Method chapter. They restate each section's protocol in continuous prose rather than in the
bullet-and-figure form the notebook uses for explanation. The §5.3 and §7.2a paragraphs in particular
are the clearest statements of the two fairness arguments — matching at the final post-NMS Pfa, and
the oracle-threshold scope limit — and should go into the report close to as-written.

#### ⚠️ The best-CFAR re-selection claim narrowed — how to state it in the report

The 24 Aug entry above concluded from the re-selection bootstrap that "**the original procedure's
assumption was not load-bearing**: the fixed-winner intervals were not overconfident". **That is
broader than the evidence supports and is narrowed here.**

The re-selection experiment varied exactly one thing: which CFAR variant the difference is taken
against. It found that varying it changes nothing material. That licenses a statement about *that*
source of uncertainty, and no more. It does **not** license a statement about whether the intervals
are well calibrated overall, because a second source — **the variability of choosing the operating
threshold** — is still not propagated by either procedure. The threshold is computed once per method
per condition; re-deriving it inside each replicate would cost roughly 200× the compute and has not
been done.

**Correct wording for the report:** *accounting for uncertainty in which CFAR variant is best did not
materially change the confidence intervals.* Not: *the intervals are not overconfident.*

The reasoning error is one worth naming, because it is the same shape as the `unresolved` Bonferroni
error recorded above: concluding that a quantity is sound because the one thing tested about it came
back clean. Both times the fix was to state what was measured rather than what it seemed to imply.
The measured result itself is unchanged — 0 of 7 verdicts changed, width ratios 0.952–1.000.

*Also in §8:* the plain-English rewrite makes explicit what the bootstrap resamples (frames only —
models and thresholds stay fixed), and what "paired" means (both arms use the same resampled frame
indices, so a difficult frame drawn twice is drawn twice for both).

#### §9 rewritten in plain English, and the `unresolved` bug story connected to the physics check

**Two claims corrected.**

*Bonferroni does not require independence.* §9 said "testing seven **independent** comparisons at
α = 0.05 gives roughly a 30% chance of at least one spurious result", which reads as though the
correction depends on the tests being independent. It does not — Bonferroni holds regardless of
dependence; it is the ~30% figure that assumes independence. The two are now separated explicitly,
because the seven conditions share a trained model and a generator and are certainly not independent.

*"Significant non-difference" replaced.* That phrase was compact but obscure. `unresolved` is now
described as a **statistically significant but practically negligible difference**, which says the
same thing without needing decoding.

**The `unresolved` bug story is now told where it belongs.** §10 mentioned the NMS plateau defect but
never connected it to `unresolved` or to the number that exposed it. §9.2 now carries the full
callout, verified against Milestone 16: DL scored **0.547** on `unresolved`, above the 0.5
two-targets-in-one-mainlobe ceiling; the physics sanity check caught it; `all_peaks` was marking
every cell of a plateau as a local maximum, so DL emitted 1.90 detections per single-target frame
against CA's 0.83, each duplicate forgiven by the guard region; after `_collapse_plateaus` all four
methods tied at **0.490**.

⚠️ **Those two figures are pre-re-specification** (`bench_test`, the old 256×128 sensor, Milestone
15/16) and are **not** comparable with the current 0.464 / 0.476. A bracketed note in §9.2 says so,
because a reader would otherwise try to reconcile 0.490 with 0.476 and conclude something had
drifted. The ceiling argument itself survives the re-specification: it follows from two targets
sharing one mainlobe, not from the sensor.

**Worth carrying into the report.** This is the strongest single piece of evidence that the
`unresolved` stratum earns its place. It does not decide a winner — both arms sit at the cap — but it
is the condition that turned a plausible-looking "sub-resolution recovery" claim into a retraction
(the 19% → 8% correction, Milestone 16). A stratum whose job is to be impossible is how an
implausible result gets caught, and that is a better justification for including it than any Pd
number it produces.

*Also in §9:* the p-value floor explanation is rewritten in plain terms — `p = 0` means no replicate
of 2,000 landed on the opposite side, not that the probability is zero — with the reporting rule
(`p < 0.001`) tied to the two-sided resolution 2/2000.

#### §10: a missing regression test found, and a claim about the existing one corrected

**The plateau fix had no regression test, despite a note claiming otherwise.** §10 stated that its
duplicate-detection check "is Milestone 16's non-maximum-suppression defect". It is not. That check
feeds *ready-made* duplicate detections to `evaluate_frame` and confirms they are **counted**
correctly — one TP, the rest set aside. **It would have passed throughout the Milestone 16 period**,
because the counting was never the broken part: the defect was upstream in `all_peaks`, which marked
every cell of a flat plateau as a local maximum and so *produced* the duplicates.

So the project believed it had a guard against its most consequential evaluator bug and did not.

**Now added** (14 checks, up from 13): a plateau-collapse regression test calling `all_peaks`
directly. Flat plateaus of 1×1, 3×3 and 5×5 must each yield exactly one candidate, and two separated
plateaus must yield two. The two checks now cover the *production* and the *counting* halves of the
same failure mode; only together do they cover it.

**And the more useful lesson, now stated in §10:** neither check is how the defect was actually
caught. It was caught by DL scoring 0.547 on `unresolved`, above the ~0.5 physical ceiling — a
physics sanity check finding what a unit test had no reason to look for. Worth making this point in
the report: unit tests confirm the rules you thought to write down, and impossible-by-construction
benchmark conditions catch the ones you did not.

**ROC monotonicity widened from one curve to all of them.** It previously checked a single
`masking`/CA curve. It now runs over every exported curve — 3 conditions × 5 methods, all swept
points — and names any violator. Zero violations, verified before the assert was written.

**Framing corrected:** the section described its checks in terms that implied they establish
correctness. They are targeted regression and sanity checks, and the text now says so, along with
the plain statement that passing them does not prove the evaluator is bug-free.

*Also:* gate monotonicity restated precisely as "for a **fixed set of detections**, widening the gate
cannot remove an existing TP" — the previous phrasing did not say the detection set was held fixed.

#### §11: two summary claims corrected against the sections they summarise

**"Both gates are reported because neither is privileged" contradicted the pre-registration.** §3.1
states that ±1 is the pre-registered headline gate and ±3 is a robustness check; §11's limitations
list said neither is privileged. That is the paragraph a reader is most likely to quote, so it was
quietly undoing the pre-registration in the most visible place. Corrected to: "The ±1 gate is the
pre-registered headline definition, while ±3 is reported as a robustness check."

**"The choice of best classical variant carried inside the interval" implied the re-selected CIs are
the reported ones. They are not.** The headline intervals come from `bootstrap_comparison.py` with
the winner fixed on the full condition; the re-selection is a separate robustness run
(`bootstrap_best_cfar_reselect.py`). §11 now describes it as a robustness check that *confirms
winner-selection uncertainty does not materially change the intervals*, and states explicitly that
**the reported intervals remain the fixed-winner ones**. This also keeps §11 consistent with the
narrowed §8.1 claim recorded above.

*Also corrected:* the localization bullet still said the gate "censors the localization metric at its
own width" — the phrasing §6 replaced with the precise per-axis bound. And three labels were made
specific: "Honest uncertainty" → "Paired frame-sampling uncertainty"; "A working evaluator" →
"Targeted evaluator checks", with §10's caveat that they cannot prove the absence of every bug;
"one `all_peaks()` function object" → "the same `all_peaks()` implementation and the same NMS
radius". The "Shows" list now opens by stating that everything in it concerns the pipeline **after**
score-map generation — the score maps being the one stage deliberately not shared, since they are
what is being compared.

**Pattern worth noting.** Both corrections were summary text drifting from the section it summarises,
and both drifted in the direction of claiming more. Worth re-reading the report's own summary
sections against their bodies before submission for the same reason.

#### `dl_development.ipynb` — the DL side rebuilt as an explanation; `PLAN.md` created

**New notebook, old one preserved.** `notebooks/dl_development.ipynb` (70 cells, built by
`scripts/build_dl_development_nb.py`) reorganises the DL material as a design narrative: why a
deliberately small network, what it receives, what it predicts, the architecture, the receptive
field, training, sanity checks, seed variability, epoch selection, localization, the decision
history, the frozen card, and what the work does and does not establish.
`cnn_development_log.ipynb` is **unchanged** and remains the dated research log — the new notebook is
the record of *why*, the old one of *when*.

**Everything recomputed from source rather than copied**: 5,041 parameters (verified layer by layer),
receptive field analytic (21,9) **equal to** measured, growth 3x3 → 7x5 → 13x7 → 21x9, the
`pos_weight` ablation from `checkpoint_history.csv`, and the epoch rule from
`epoch_sweep_epsweep_selection.csv`.

**Two findings sharpened by recomputing rather than quoting.**

*The `pos_weight` ablation is 5 better / 1 tied / 1 worse, not "6 of 7".* `masking` is an exact tie
(0.0000) and `unresolved` the single loss (−0.0013, at the physics ceiling). More useful: the three
largest gains from pw974 → pw100 are **`mixed` +0.0292, `clutter_edge` +0.0264, `clutter` +0.0190** —
all three clutter-bearing conditions. That is the mechanism behind Correction C3: exact class
balancing degraded precisely the clutter conditions, so the "DL loses in clutter" reading was a
property of the loss weighting rather than of the detector.

*The epoch-21 margin is tight.* Stopping at 21 rather than the plateau at 23 gives up **+0.0018** Pd
against a 1 SE tolerance of **0.0023**. The rule is satisfied, but only just — worth knowing when
defending the choice rather than presenting it as comfortable.

**Localization degradation quantified over the full sweep:** Doppler error **0.201 → 0.421 bins**
(2.1x) and range **0.310 → 0.457** from epoch 1 to 25, while Pd rises only **+0.057**. The clearest
single statement of the detection/localization trade in the project.

**Contradictions found and reported, not silently fixed:**

1. ⚠️ **`src/dl/models.py` docstring overclaims.** It states the two detectors "see identical context
   and differ only in the DECISION RULE". Matched *extent* is not matched *information* — CFAR
   excludes its guard region by design and the network need not. The notebook states the correct,
   weaker version throughout; **the source docstring is still wrong and should be corrected.**
2. ⚠️ **Stale dimension in the same docstring**: "the output must stay 256x128". The map is 256x64.
3. The epoch sweep uses **4 seeds**, not the 8 of the frozen arm; stated in the notebook.
4. **`TinyCNNPool` has never been trained or scored.** The model exists and its RF is measured
   (13x13, differing from TinyCNN on both axes) but there is no pooling *experiment*. Presented as
   built-not-run, not as a completed control.

**`PLAN.md` created** — future work only, in priority order: (1) label/localization attribution,
(2) mainlobe/information-access attribution, (3) fixed-threshold CFAR-ness, (4) secondary ablations;
then the CARRADA roadmap R0–R4, beginning with a domain audit and a frozen zero-shot evaluation.
⚠️ This reintroduces a file previously deleted, when `PROGRESS.md` was made "the single roadmap
record". `PLAN.md` opens by reconciling that: it holds **only** work not yet done, and `PROGRESS.md`
remains authoritative for everything completed.

*Two notebook bugs worth remembering:* a figure loop bound `M` and silently overwrote
`FROZEN["model"]`, failing nine cells later; and a filtered DataFrame kept its original index, so
`iterrows()` produced out-of-range list lookups. Both are runtime faults the builder's compile check
cannot catch — only executing the notebook finds them.

#### Artefacts and how to regenerate

`results/` is gitignored, so the three cached inputs are not in the repository:

```
python scripts/roc_curve_export.py              # ROC/FROC curves + operating points (bench2_val)
python scripts/bootstrap_best_cfar_reselect.py  # best-CFAR re-selection (~10 min)
python scripts/localization_common_subset.py    # common detected subset (~75 min)
```

Commits: `0b1c97d` (notebook + scripts), `15dec28` (stage figure reordered), `7ea40f2` (ordering vs
footprint separated, caption retracted, A.1 strengthened), on branch
`evaluation-methodology-notebook`.

### 2026-08-24 — Integrated CFAR: mainlobe energy explains NONE of the DL margin

**Standalone attribution experiment.** Everything lives in `experiments/integrated_cfar/`; nothing in
`src/`, `scripts/`, `notebooks/` or `results/` was modified, and **`bench2_test` was not touched**.
This does not alter the project's primary results or conclusions. Pre-registered in
`experiments/integrated_cfar/PREREGISTRATION.md`, written before any result was computed.

**The question.** How much of TinyCNN's advantage over CFAR is explained by the CNN integrating
neighbouring mainlobe energy that single-cell CFAR discards? Standard CFAR tests one cell; a CNN with
a 21×9 receptive field can combine the whole mainlobe. If that is the mechanism, giving CFAR the same
energy should close most of the gap.

**Method.** Same CFAR noise estimate — the estimators are *imported* from `src/cfar/cfar.py` and
called unchanged, verified by asserting `power / our_noise` reproduces the shipped `cfar_ratio_map`
output to **0.000e+00** for all four variants — with an integrated CUT. Two statistics: uniform sum,
and PSF-weighted (the matched filter). Scored through the unmodified protocol on `bench2_val`.

**Region derived, not assumed.** Averaged normalised power patch over 50 high-SNR `isolated` targets:
**1×1 captures 27.1%** of target energy, **3×3 captures 94.2%**, 5×5 captures 99.7%. 5×5 buys the last
5.5 points with 16 more noise cells, and the guard is ±2, so 3×3 was chosen and sits strictly inside
the guard — the numerator can never overlap the reference ring. The raw patch was 11.6% asymmetric
(finite sampling over off-grid targets); the kernel is symmetrised and that figure reported.

**Calibration.** No closed form applies to a sum of correlated cells, so alpha was Monte-Carlo
calibrated on 300 pure-noise frames and verified on held-out seed 9000: achieved Pfa **0.96–1.13×**
target across all eight (kind × statistic) combinations. Alpha affects only the design point; every
Pd below is at matched *measured* Pfa, where the threshold is swept.

#### The result: integration makes CFAR WORSE, on every condition

| statistic | mean Δ_integration | negative on | significant (Bonferroni α = 0.0071) |
|---|---|---|---|
| uniform | **−0.0366** | **7 / 7** | 4 / 7 |
| PSF-weighted | **−0.0246** | **7 / 7** | 4 / 7 |

`delta_remaining` is therefore the entire original gap and slightly more. Measured Pfa across all
arms 0.971–1.021× target, so the operating points genuinely matched.

**Against the three pre-registered outcomes: "Integrated ≈ CFAR → integration explains none of it,
and the learned decision rule is the live explanation."** Observed, and more strongly — integration
is not merely neutral, it is harmful.

**The registered prediction was correct and stronger than predicted.** Uniform summation was
predicted to possibly lose; it loses everywhere. But the **PSF-weighted matched filter — the variant
with the theoretical reason to help — also loses**, by less. Performance is monotone in how sharply
the kernel concentrates on the centre cell:

| kernel | centre weight | pooled Pd rank | pooled Doppler error |
|---|---|---|---|
| single-cell CFAR | 1.000 | best | 0.1228 bins |
| PSF-weighted | 0.288 | middle | 0.1954 (×1.59) |
| uniform | 0.111 | worst | 0.3065 (×2.50) |

**Why.** The measured −3 dB width is ~1.6 bins, so the response is concentrated in very few cells and
the centre dominates. Summing nine cells adds little signal and much noise — and because the Blackman
window correlates adjacent cells, the noise sum grows faster than √K, so the deflection falls rather
than rises. The matched filter loses less precisely because it down-weights the noisy outer cells.

**What this means for the attribution programme.** The DL margin is **not** access to mainlobe energy
CFAR discards. The **learned decision rule** is now the live explanation, and PLAN.md's E2 has
returned its answer for zero training. Under the plan's own logic this raises the value of E4 (the
guard-masked patch CNN), because the cheap explanation has been eliminated rather than confirmed.

⚠️ **What it does not show.** Not that the CNN gains nothing from its receptive field — only that
*this* form of integration, with the noise estimate held fixed, does not reproduce its advantage. A
network can weight its 21×9 field in ways a fixed 3×3 kernel cannot, including adaptively.

#### ⚠️ Unplanned finding — a lead for Open/O4

A **symmetric** 3×3 kernel degrades **Doppler localization ×2.50 against range ×1.37**. Open/O4 asks
why the CNN's localization degrades asymmetrically between range and Doppler when its 3×3 label is
symmetric. Here the same asymmetry appears in a **purely classical detector with no learning
involved**, produced by a symmetric operator.

That makes "the network's training causes the asymmetry" less likely and points instead at something
about the RD geometry. **Suggestive, not conclusive** — a fixed box filter is not what the network
does, and this was not the experiment's question. Recorded as a lead, not an answer, and not chased
(standing rule: it changes no published number).

**Artefacts** (all inside `experiments/integrated_cfar/results/`, none in the project `results/`):
`psf_patch.npy`, `psf_energy_capture.csv`, `kernels.npz`, `calibration.csv`,
`peaks_int_{CA,GO,OS,SO}_{uniform,psf}.npz`, `summary_pd.csv`, `deltas.csv`, `localization.csv`.
Notebook: `integrated_cfar_experiment.ipynb` (22 cells, 4 figures, zero errors).

**Whether this warrants a `bench2_test` read is a separate decision**, to be recorded before it
happens. The result is a *negative* one on the validation split, so there is a reasonable argument
that it does not need the reported split at all.

### 2026-08-27 — Label ablation: the 3×3 label CAUSES the localization deficit, and it is fixable

**Standalone attribution experiment** (PLAN.md E3), isolated in `experiments/label_ablation/`.
Nothing in `src/`, `scripts/`, `data/`, `notebooks/` or `results/` was modified; **`bench2_test` was
not touched**. Pre-registered in `experiments/label_ablation/PREREGISTRATION.md`, committed
(`cc342c5`) **before any checkpoint existed**. This does not alter the project's primary results or
conclusions.

**The question.** The flat 3×3 label is the documented suspect behind three findings — DL localizes
3.83× worse than CFAR in Doppler, localization degrades monotonically with training, and DL
collapses to Pd 0.41 against CFAR's 0.85 at gate 0. All three share one explanation: the network is
told nine cells are equally correct. Does a better label fix it?

**Design.** Three arms × 8 seeds × 21 epochs, `threads=2`, otherwise identical to
`configs/frozen_dl.yaml`: **point** (1×1), **flat3x3** (incumbent), **gaussian** (same 3×3 footprint,
weights from the measured PSF — σ derived from `results/psf_profile.csv` as −3 dB width / 2.3548,
off-bin worst case: σ_r 0.6895, σ_d 0.7007 bins). 24 runs, 9.95 h wall-clock at 6 concurrent
processes.

**`pos_weight` recomputed per arm, and the rule stated.** `BCEWithLogitsLoss` weights each cell's
positive term by `w·y`, so total positive gradient mass ∝ `w·Σy`; that product is held constant,
anchored to the incumbent: **point 899.37, flat3x3 100.00, gaussian 307.51**, constant to 0.000%.
Anchored to 100 rather than re-derived from class frequency because exact class balancing was
already measured worse here (Correction C3). The point arm's mass ratio is 0.1112× rather than
exactly 1/9 — 3×3 blocks clip at the border and overlap between close targets.

#### The control is verified twice, and both checks pass

* **Labels:** `flat3x3` is built by the project's own `make_label_mask` and is byte-identical to
  `data/bench2_train/nn_cache_y.npy` on 40 sampled frames.
* **Weights:** retraining `flat3x3` at `threads=2` reproduces the frozen checkpoints **bit-for-bit
  at epoch 21 on all 8 seeds** (max |ΔW| = **0.000e+00**).

The second also validates the training driver, which writes into the experiment folder instead of
`results/dl/` yet produces numerically identical models. **The control IS the frozen arm**, so any
difference between arms is attributable to the label alone.

#### Result — the deficit is caused by the label, and it is fixable at ~zero Pd cost

Pooled, `bench2_val`, gate ±1, epoch 21, 8-seed means:

| arm | Doppler error | vs best CFAR | range error | vs best CFAR | Pd | Δ Pd |
|---|---|---|---|---|---|---|
| flat 3×3 | 0.3933 bins | **4.57×** | 0.4729 | 2.65× | 0.5848 | — |
| gaussian | 0.1708 | **1.99×** | 0.3105 | 1.74× | 0.5871 | **+0.0023** |
| point | 0.1097 | **1.27×** | 0.2197 | 1.23× | 0.5750 | −0.0098 |

**The mechanism is confirmed.** Doppler error from epoch 5 → 21 rises **×1.98** under the flat label,
**×1.25** under the Gaussian, **×1.04** under the point label — monotone in how flat the supervision
is. That was the pre-registered sharpest test and it lands exactly as the hypothesis predicted.

**At gate 0 the incumbent gets WORSE with training**: exact-cell Pd falls **0.278 → 0.191** over 21
epochs, while the Gaussian holds ~0.33 and the point label ~0.39. Flat supervision does not merely
cap exact-cell accuracy — more training actively destroys it.

#### The registered predictions

1. **"Gaussian matches 3×3 on Pd while improving localization" — CONFIRMED.** +0.0023 Pd (within
   noise, significant on 1 of 7 conditions), Doppler 4.57× → 1.99×.
2. **"Point may train poorly at ~1:8500, which would be the first evidence the 3×3 choice was
   reasonable" — FAILED PREDICTION.** Point trained stably on all 8 seeds and gave the *best*
   localization and the *best* gate-0 Pd. **The 3×3 choice is not vindicated — the opposite.** Its
   only cost is −0.0098 Pd at gate ±1, concentrated significantly in the clutter-bearing conditions
   (`clutter` −0.025, `mixed` −0.024, `clutter_edge` −0.022), consistent with sparser supervision
   giving less gradient signal where the task is hardest.
3. **"A Pd gain from Gaussian would be unexpected — flag it."** Flagged. The +0.0023 at gate ±1 is
   significant on `multi` only and is **not** reported as a detection win. The large gate-0 gains
   (+0.06 to +0.31, significant on all 7 for both arms) are **the localization improvement expressed
   as exact-cell Pd** — the same quantity measured differently, not an independent detection gain.

#### What this changes

The localization deficit is **caused by the supervision and is fixable**, which converts a documented
limitation into a design choice with a measured alternative. It also retires the two competing
explanations recorded alongside it: it is **not** the network's capacity, and **not** the matching
gate.

⚠️ **Scope.** Validation split only. No reported number changes and the frozen configuration is
untouched. Whether to adopt a different label — and whether that would justify a second
`bench2_test` read — is a separate decision to be recorded before it is taken. Note the arms differ
in *where* they win: the Gaussian is the safe choice (localization halved, Pd unchanged), while the
point label buys more localization at a real cost in clutter.

**Artefacts** (all under `experiments/label_ablation/`, arm-tagged, none in the project `results/`):
96 checkpoints, `epoch_curve.csv`, `per_condition.csv`, `deltas.csv`, `cfar_reference.csv`,
`pos_weight_derivation.csv`, `sigma.json`, `DERIVED_VALUES.md`, and
`label_ablation_experiment.ipynb` (17 cells, 4 figures, zero errors).

### 2026-08-28 — E4: guard access buys ~1.5 pp of Pd and costs a great deal of localization

**Standalone attribution experiment** (PLAN.md E4), isolated in `experiments/e4_cut_guard_mask/`.
Pre-registered before any masked model existed. **The frozen TinyCNN was never retrained**;
16 protected files re-hashed unchanged, no git change outside the experiment folder. `bench2_test`
never opened. Does not alter the project's primary results or conclusions.

**Feasibility finding, reported before training.** CUT-relative guard masking is **impossible on a
dense fully-convolutional network**: the guard is ±2, so cell (r+1,v) sits inside CUT (r,v)'s guard
while being its own CUT, and one input map cannot hold it both hidden and visible. A decimated
lattice blanks ~90% of the map; exact dense masking needs 189 passes per frame. The masked arm must
therefore be a CUT-centred patch model — sound for inference (frozen dense logit = 21×9 patch-centre
logit to **1.9e-06** in eval mode) but different in training (BatchNorm over 21×9 not 256×64; one
supervised cell per patch, not 16,384 per frame). So the primary contrast is **masked-patch vs
unmasked-patch**, making those differences common-mode, with the frozen model as a read-only
reference. 16 new models trained (8+8), 20.4 h.

#### The two gates disagree, and that is the result

| | gate ±1 (registered primary) | gate 0 (exact cell) |
|---|---|---|
| mean ΔPd = Pd(unmasked) − Pd(masked) | **+0.0150** | **−0.157** |
| sign | positive **7/7** | negative **7/7** |
| significant (Bonferroni) | 3/7 | **7/7** |

Measured Pfa 0.992–1.001× target across every arm and condition.

**Registered outcome A holds at the headline gate**: guard/mainlobe information does contribute —
significantly on `multi` (+0.038), `masking` (+0.025), `clutter` (+0.024). But the effect is
**small**: removing 24 of 189 input cells — the ones carrying ~64% of the frozen model's occlusion
sensitivity — costs ~1.5 percentage points of Pd.

**The unregistered finding is larger than the registered one.** At gate 0 the sign reverses on every
condition: the **masked** model is far better at putting its peak on the exact target cell
(`isolated` 0.511 vs 0.260 — nearly double). Mainlobe energy in the guard cells makes the response
**broad**: enough to match within ±1, not enough to identify the peak. Deny those cells and the
response sharpens.

**E3 and E4 have now found the same failure from opposite directions.** E3: flat 3×3 *labels*
broaden the response. E4: access to mainlobe *input* broadens it too. Both are fine at ±1 and both
collapse at gate 0. **The localization deficit has two independent contributing causes** — one in the
supervision, one in the input.

#### Combined with E2

E2 gave CFAR the mainlobe energy and it **hurt** (7/7 worse). E4 takes it from the CNN and detection
drops **slightly** while localization improves **sharply**. Together:

> The guard cells carry information the learned detector genuinely uses for detection, but that
> information is not simply *energy* — integrating the same energy into CFAR degrades it. And the
> network's use of those cells is what makes it place targets imprecisely.

**Explicitly not supported:** "the CNN wins because it sums more target energy." E2 rules it out, and
E4 shows the guard cells buy only ~1.5 pp of Pd at a large localization cost.

⚠️ **Limitations.** Both patch arms are undertrained relative to the frozen dense model — uniform
256-CUT sampling preserves the class distribution (so `pos_weight` keeps its meaning) but gives ~67×
fewer positives per epoch. Common-mode, so ΔPd is unbiased, but absolute Pd sits below the dense
model's. Validation split only.

### 2026-08-31 — CARRADA real-data validation, Phase 1: experiment isolated, dataset verified, no detector run

**Real-data external validation begins** (PLAN.md Part B), isolated in
`experiments/r_carrada_real_data_validation/`. **This is Phase 1 only — setup and integrity.**
No detector was run, no threshold tuned, no detection statistic computed, and nothing in `src/`,
`scripts/`, `notebooks/`, `configs/`, `data/bench2_*` or `results/` was modified. No published
number changes.

**Experiment folder created** with `README.md`, `isolation_manifest.json`, and a placeholder
notebook containing only the title, research question, isolation statement and Phase 1 notes.

**Protected assets hashed.** `isolation_manifest.json` pins SHA-256, size and category for
**628 files (2.88 GB)**: the 8 frozen TinyCNN epoch-21 checkpoints and their run configs, all
DL/CFAR/eval/radar/data source, `frozen_dl.yaml`, `cfar_alpha_table.json`, the six dataset configs
and their provenance and caches, the headline result tables, the CFAR baseline outputs, and the
complete E2/E3/E4 outputs. Re-verified at end of phase: **0 changed, 0 missing** across every
category except `project_record_append_only`, which is this file. Git baseline: `main` at
`1ed2aab`, working tree clean before the experiment folder was created.

**CARRADA integrity verified — all checks pass.** 30 sequences on disk, matching both
`validated_seqs.txt` and `data_seq_ref.json`. 12,666 `range_doppler_numpy` and 12,666
`range_doppler_raw` frames, with identical filename sets in 30/30 sequences and 0 frames missing
against the light list. All **25,332** `.npy` headers scanned: every array is `(256, 64)` float64,
0 unreadable; 600 full-load spot checks, 0 failures. All 12 top-level metadata files present in both
the source tree and the `data/carrada_meta/` backup; box annotations present in 30/30 sequences.

**Annotation terminology resolved — four quantities had been conflated.**

| quantity | count |
|---|---|
| total frames | 12,666 |
| annotated frames | 7,193 |
| empty frames | 5,473 |
| frame-level object instances ("object-frames") | **8,750** |
| unique tracks | **77** |

**The previously quoted "84" is not an object count.** It is the raw sum of the `instances` list
lengths in `data_seq_ref.json`: 84 − 2 duplicate entries (`001109` twice in `2020-02-28-13-08-51`,
`001143` twice in `2020-02-28-13-12-42`) = 82 unique ids, − 5 merged away by
`instance_exceptions.json` = **77**, confirmed independently by both `annotations_frame_oriented`
and `annotations_instance_oriented`. ⚠️ `PLAN.md` R3 says "78 instances"; the correct value is 77.
PLAN.md is protected and was not edited. **R1 statistics must use 8,750 as the object denominator**,
not 77 and not 12,666. Class split by object-frames: car 3,741, pedestrian 3,239, cyclist 1,770.
Objects per frame: 0 → 5,473, 1 → 5,636, 2 → 1,557 — **never more than two**.

**Official sequence-level splits verified and adopted unchanged** (mirrored to
`configs/official_splits.json`): Train 18 seqs / 8,088 frames / 4,319 annotated / 5,318
object-frames / 49 tracks · Validation 6 / 2,448 / 1,483 / 1,804 / 15 · Test 6 / 2,130 / 1,391 /
1,628 / 13. Every sequence assigned exactly one split, sets disjoint. **No new split was created and
no frame-level split will be used.**

**Representation facts recorded as prior information, not re-derived in this phase:**
`range_doppler_numpy` is the primary representation and is power in dB up to a constant scale;
`_raw` behaves as an amplitude; the eventual CFAR input is proportional to `P = 10^(numpy/10)`, with
the unknown constant cancelling because CFAR is a ratio test; `_numpy` is the representation
CARRADA's own annotation pipeline used. **`_raw` is retained** as an independent cross-check and was
not deleted.

**Four methodological questions are open and must be settled before Phase 2**, recorded in the
experiment README rather than resolved unilaterally: (1) the folder ID **collides with PLAN.md's
Part-A E5**, the unbuilt four-way comparison; (2) CARRADA truth is an extended **segmentation blob**,
median car footprint 14×9 bins, so a truth-point definition must be pre-registered before any R1
number exists; (3) **Pfa is not the same quantity** across the two domains, since only the 77 tracked
objects are annotated and unannotated real reflectors would score as false alarms; (4) with at most
2 objects per frame there is **no close-target regime**, so `multi`/`masking`/`unresolved` have no
real-data counterpart.

**Next phase: R0**, the synthetic-vs-CARRADA domain audit, before any detector touches a real frame.

### 2026-08-31 — CARRADA real-data validation, Phase 2 (R0): the domain audit, and no detector run

**Phase 2 of the real-data external-validation track.** Everything lives in
`experiments/r_carrada_real_data_validation/`. **No detector was run** — no TinyCNN inference, no
CFAR pass, no threshold, no Pd, no Pfa, no alpha recalibration. No published number changes and no
frozen asset was modified.

**Name resolved.** The Phase 1 folder was briefly `e5_carrada_real_data_validation`, which collided
with the **E5** that `PLAN.md` reserves for the unbuilt Part-A four-way comparison. It is now
`r_carrada_real_data_validation` — `r_` marks the real-data track, and it is deliberately not called
`R1` because it holds both R0 and R1. `PLAN.md` was **not** edited.

**Phase 1 commit:** `15f3aab2c9690a22dcbe55799d78ab6d69451c09`.

#### Representation — `range_doppler_numpy` is the primary input, and the conversion is exact

240 frames across all 30 sequences. Per-frame regression of `numpy` on `10log10(raw)`:
slope **2.0046 ± 0.0183**, intercept −47.42 dB, residual **0.330 dB** with essentially no level
dependence (r = +0.091) and no range/Doppler structure. Slope 2 is the signature of `20log10`, so
**`_raw` is an amplitude and `_numpy` is power in dB**. Confirmed independently on the eligible
background: CV of `10^(numpy/10)` matches CV of `raw²` to **10.6%** but differs from CV of `raw` by
**252%**.

* **CFAR input:** `P ∝ 10^(numpy/10)` is valid; the unknown constant cancels because CFAR is a ratio
  test.
* **TinyCNN input:** the frozen preprocessing is `10log10(P / median P)`. Because dB is monotone in
  power, `median(P) = 10^(median(dB)/10)` **exactly**, so this equals `dB − median(dB)` — the
  adapter is exact in the dB domain and the unknown constant cancels identically.
* `_raw` offers no advantage (identical noise statistics; peak lands inside the annotated mask
  97.34% vs 96.28% over 752 instances) and is **kept on disk** as a cross-check.

#### Geometry transfers cleanly

CARRADA RD is `(256, 64)` float64 on all 12,666 frames; annotation coordinates span range bins
0–255 and Doppler bins 0–63, indexing the array directly. **No resampling or interpolation.**

#### R0 background statistics — FINAL, and they supersede every preliminary value

Protocol (`src/r0_background.py`): object support dilated ±3 bins, zero-Doppler band ±4 bins, and
the 21×9 detector border all excluded. **No percentile trimming anywhere.** Frames combined by
**median within group**, then mean ± SE **across groups** — 30 CARRADA sequences, 8 simulator
conditions — never across correlated frames. CARRADA uses all 12,666 frames; the simulator uses all
1,600 frames of `bench2_val` (**`bench2_test` was not opened**).

| statistic | CARRADA | simulator | single-look ideal |
|---|---|---|---|
| CV(power) | **0.4734 ± 0.0157** | **1.0053 ± 0.0032** | 1.000 |
| Gamma shape k (moments) | **4.771 ± 0.223** | **0.990 ± 0.006** | 1.00 |
| Gamma shape k (log-moment) | 4.269 ± 0.077 | 0.9958 ± 0.0014 | 1.00 |
| background std [dB] | **2.212 ± 0.011** | **5.573 ± 0.005** | 5.57 |
| lag-1 range correlation | 0.5635 ± 0.0031 | 0.5731 ± 0.0010 | — |
| lag-1 Doppler correlation | **0.2265 ± 0.0055** | **0.5740 ± 0.0029** | — |
| zero-Doppler ridge | **24.0 dB**, half-width ±1 | **14.8 dB**, half-width ±4 | — |

**⚠️ Four preliminary values are hereby superseded, and two of them reversed direction.** The
preliminary pass reported CARRADA k 2.555, CV 0.720, dB std 3.207, lag-1 range 0.765, lag-1 Doppler
0.587, ridge 29.3 dB. The defects, each found and fixed in sequence: (1) an early pass trimmed the
top decile, biasing CV downward; (2) another left the ridge in, inflating CV enormously; (3) the
preliminary pass measured **far range only** (bins 200–255) on empty frames, which is not the
population a detector sees; (4) the first R0 run had a **band-selector bug** — it asked whether
*any* bin in the candidate window exceeded +3 dB, which the centre bin always satisfies, so the
ridge band grew to ±15 and excluded 31 of 64 Doppler bins.

**Two conclusions reverse:**

* **Range correlation is NOT higher in CARRADA.** 0.5635 vs 0.5731 — statistically indistinguishable
  overall. The preliminary claim was an artefact of measuring only far range.
* **Doppler correlation is LOWER in CARRADA, not higher.** 0.227 vs 0.574.

A **range-block decomposition** (new, `r0_range_blocks.csv`) explains why: CARRADA's background is
**non-stationary in range** — k runs 5.86 / 4.74 / 7.39 / 7.52 / 2.91 across blocks 10–60, 60–110,
110–160, 160–210, 210–246, and lag-1 range runs 0.512 → 0.745. The far-range block (200–246) gives
k 3.36 and lag-1 range 0.713, reproducing the preliminary numbers. **The simulator is stationary**
in range: k 0.97–0.99 and lag-1 range 0.569–0.574 in every block.

**Unplanned finding — the background is contaminated by unannotated real reflectors.** The
per-frame CV distribution is heavy-tailed on CARRADA: median 0.495, p99 6.81, max 41.4, with
**5.85%** of frames above CV 2 and 0.73% above 10 (simulator: 1.19% and 0.25%). Averaging over that
tail broke the summary — mean-aggregated CV said 1.00 while mean-aggregated k said 3.97, which
cannot both hold because **k = 1/CV² is exact per frame** (verified to 4e-15). Medians agree
exactly, which is why median aggregation was adopted; mean-based values are retained in
`r0_audit.json` as `meanagg_*`. Because only the **77 tracked objects** carry annotations, those
bright frames are almost certainly genuine unannotated reflectors — **direct evidence that a CARRADA
"Pfa" is not the same quantity as the synthetic one.**

**Simulator validated as designed.** k = 0.990 ± 0.006 against a theoretical 1.0 and background std
5.573 ± 0.005 dB against a theoretical 5.57 dB, stationary across every range block. The
compound-Gaussian clutter model behaves as single-look exponential speckle, which had never been
checked against theory this directly. **This is an independent validation of the simulator, not a
CARRADA result.**

**Frame-median audit.** Per-frame median spread is 0.593 dB (CARRADA) vs 5.305 dB (simulator).
Frame-median normalization removes the global scale difference **exactly**, so it cannot explain any
residual domain gap — whatever remains is a **texture/correlation** shift, which normalization
cannot touch. That separation is the point of the audit.

#### Object support — CARRADA truth is extended, and two extents exist

| class | object-frames | dense range × Doppler (median) | box range × Doppler (median) | dense cells |
|---|---|---|---|---|
| pedestrian | 3,239 | 9 × 9 | 5 × 5 | 74 |
| cyclist | 1,770 | 11 × 11 | 7 × 7 | 93 |
| car | 3,741 | **18 × 13** | 14 × 9 | 147 |

⚠️ **The `box` field is not the bounding box of the dense mask** — it is strictly smaller in
**8,750 of 8,750** instances on both axes, so it bounds the *sparse* annotation. Earlier notes
quoting "car 14×9" were quoting the box; the segmentation support is 18×13. Both are now recorded.
A synthetic target occupies **one** bin.

#### Domain-shift conclusion

CARRADA is **not** simply another sample from the synthetic domain. Geometry and representation
transfer cleanly, but the background is markedly smoother (k ≈ 4.8 vs 1.0), less correlated in
Doppler, carries a ridge ~9 dB stronger and far narrower, is non-stationary in range, and contains
unannotated reflectors. Conservative wording to use in the report: *CARRADA background statistics
are consistent with an effective Gamma shape near 4.8* — **not** "CARRADA is a 4.8-look radar",
which the dataset authors document nowhere.

**Recorded, not acted on:** the frozen CFAR alphas were calibrated under single-look exponential
statistics, so on CARRADA the same alpha will produce a different achieved false-alarm rate. **R1's
CFAR arm is therefore a zero-shot transfer result, and a drop in its recall would not by itself show
CFAR is intrinsically worse.** No alpha was recalibrated. Likewise TinyCNN faces three simultaneous
shifts (background distribution, Doppler correlation, ridge strength) that frame-median
normalization cannot absorb — **no conclusion about TinyCNN performance is permitted until R1.**

#### Limitations

"Object support" means a 3×3 point-target label in the simulator and a human-drawn blob in CARRADA —
identical dilation, different object. The simulator has no sequence structure, so its groups are the
8 scenario conditions rather than sequences. The Gamma shape is descriptive, not a claim about the
sensor's processing chain. `bench2_val` only. Absolute amplitude is not comparable, since CARRADA's
constant offset is unknown. The ±3 dB ridge-band threshold is a choice, made before any background
statistic was computed.

**Artefacts** (all under `experiments/r_carrada_real_data_validation/`):
`r_carrada_real_data_validation.ipynb` (30 cells, §0–§16, executed with **zero errors**),
`src/r0_background.py`, `scripts/{run_r0_audit,make_r0_figures,build_notebook,verify_carrada,build_isolation_manifest}.py`,
`results/r0_audit.json` plus 7 CSV/NPZ caches, and 6 figures in `figures/`.

**Next: R1 pre-registration — not yet written, and no detector runs until it is committed.**

### 2026-08-31 — CARRADA real-data validation, Phase 3: R0 committed, R1 pre-registration frozen

**Phase 3 of the real-data external-validation track.** Two deliverables only: commit the completed
R0 audit, and freeze the R1 pre-registration. **No detector was run on CARRADA, and no CARRADA label
of any split — train, validation or test — was used to choose any quantity.** No published number
changes; no frozen asset was modified.

**Phase 2 commit (the frozen R0 baseline):** `a3a3b88e4f26dcffa14d97e2fc8f4c03e08eff04`.
Verified before committing: 0 frozen files changed, 0 missing, 0 new in the protected set, with
`PROGRESS.md` confirmed insertion-only against `git show HEAD`.

**Pre-registration:** `experiments/r_carrada_real_data_validation/PREREGISTRATION.md`,
SHA-256 **`aaad1ff7d122bb41faad6bff40dfdcda2d9ca49f77c831df4d63316ad009ed85`**, 20,400 bytes, frozen
2026-08-31T16:36:04Z against that baseline commit. Freeze record in
`results/preregistration_freeze.json`; notebook §17–§28 (45 cells, executes with zero errors).

**R1 research question.** *Do the already-frozen TinyCNN and classical CFAR detectors transfer
zero-shot from the synthetic development domain to real CARRADA Range–Doppler data?* This is
**external validation**: it does not replace or revise the synthetic benchmark, and a poor R1 result
would retract no synthetic finding — it would bound its external validity.

**Frozen truth definition.** The **CARRADA dense segmentation mask**. No point target is invented;
centroid, box centre, brightest-cell-in-mask and sparse-point centroid are all explicitly rejected as
primary, because each is a construct of this project rather than an annotation CARRADA supplies. R0
established that the `box` field is not the bounding box of the dense mask (strictly smaller in
8,750 of 8,750 instances). The detection question is *"did the detector place a peak inside the
annotated object support?"*

**Frozen eligibility and matching.** The unchanged frozen valid region — margin 10 range / 4 Doppler,
236 × 56 = 13,216 cells; `valid_support = dense_mask ∩ valid_region`; an object is eligible if that
intersection is non-empty. Matching is one-to-one and identical for both arms: maximum-cardinality
bipartite assignment, tie-broken first on total detector score and then on deterministic
`(range_bin, doppler_bin, object_id)` order. Seven matching unit tests must pass before any test
scoring.

**Primary metric.** **Mask-based object recall** — matched eligible objects over total eligible
objects, overall and per class, identical definition for both arms. Tolerance ladder: exact mask is
the headline, with ±1 and ±3 bin dilations secondary. **Support-distance is excluded from
confirmatory metrics**: it is identically zero for any successful detection, so it cannot
discriminate among successes, and a nearest-of-many rule would reward whichever detector emits more
peaks. **No localization metric comparable to the synthetic one is reported at all**, because CARRADA
supplies no point truth. Registered wording: synthetic Pd and CARRADA mask recall measure related
capability under **different truth definitions** and are never differenced or shown in one column.

**Unmatched-output terminology.** **Synthetic-style Pfa is not reported on CARRADA in any form.** R0
measured that 5.85% of frames carry bright returns belonging to no annotation, and only the 77
tracked objects are annotated at all, so a peak outside every mask is not a known false alarm. The
burden metric is **unmatched detections per frame**, never renamed `Pfa`, `false-positive rate`,
`false alarm rate` or `FPR`. Secondary: unmatched detections per **frame with no CARRADA
tracked-object annotation** — described in exactly those words, never as "true background-only
frames".

**Detector threshold provenance.** ⚠️ **The frozen project has no global TinyCNN threshold** —
`configs/frozen_dl.yaml` specifies *"matched MEASURED Pfa, swept per condition"*, an evaluation
oracle that cannot transfer. One had to be defined, and this is recorded as a **new protocol choice,
not a transcription**: one threshold **per seed**, derived on **`bench2_val` only**, pooled across
all 8 scenario conditions, the lowest threshold whose pooled measured per-cell Pfa ≤ 1e-4, at the ±1
gate, using frozen `all_peaks` and the frozen valid region. No CARRADA data enters the derivation.
**Consequence registered in advance: R1 runs under a different operating-point protocol from the
synthetic headline and its numbers are not poolable with the synthetic tables.** CFAR is **strict
zero-shot** on frozen alpha, with the interpretation registered in advance that R0 showed those
alphas were calibrated for single-look statistics while CARRADA sits near k ≈ 4.8, so **poor CFAR
recall would not establish that CFAR is intrinsically inferior.** No alpha recalibration is
performed, and none will be invented to improve CFAR's numbers.

**Frozen NMS.** `all_peaks`, `nms_size = 5`, shared by both arms, not retuned on CARRADA. All four
CFAR variants (CA/GO/OS/SO) are run and reported; the benchmark's "best CFAR per condition" cannot
transfer because selecting it needs labels. **Primary CFAR arm is OS**, registered a priori on R0's
heterogeneous-background finding.

**Class analysis.** Car / pedestrian / cyclist on dense-mask truth, with the caveat registered now
that class masks differ greatly in size (median support 18×13, 11×11, 9×9), so a larger mask is a
larger area for a peak to land inside and **class differences must not be read as detector
sensitivity alone.**

**Secondary stratifications**, all fixed before scoring: range blocks reuse the **exact** R0 blocks
(10–60, 60–110, 110–160, 160–210, 210–246); zero-Doppler uses the R0 common ±4 band (Doppler bins
28–36), with the ridge **not** removed from the primary analysis; plus strict eligibility, sparse/box
sensitivity, a recall–unmatched-burden curve (x-axis is **not** Pfa and it is **not** an ROC), and
empty-frame burden.

**Sequence-aware statistics.** The test split has **6 sequences, and that is the effective sample
size.** Bootstrapping RD cells, or treating the 1,628 test object-frames as independent, is
prohibited. Primary inference is a **paired bootstrap over the 6 test sequences**, 10,000
replicates, percentile CI, with TinyCNN averaged over its 8 seeds per sequence. **Seed variability is
reported separately and never merged with sequence variability into a single SE.** With n = 6 the
emphasis is on effect sizes and CIs; low-n class comparisons are labelled exploratory.

⚠️ **Two pre-registered choices are new and are flagged for confirmation** rather than silently
adopted (`PREREGISTRATION.md` §14): the global TinyCNN threshold rule, and OS as the primary CFAR
arm. Neither can be resolved by looking at CARRADA data without destroying the pre-registration.

**Both real-vs-simulator RD comparison figures are preserved** for the final notebook —
`r0_fig1_rd_examples.png` (bounding box plus statistical callouts) and
`r0_fig1b_rd_examples_dense_overlay.png` (dense support drawn cell-by-cell). Both carry the label
*ILLUSTRATIVE EXAMPLES — NOT STATISTICAL EVIDENCE*, with notebook §5 stating that CARRADA's
background is smoother in the sense of **lower cell-to-cell fluctuation, not less physical clutter**,
while simultaneously carrying stronger localized structure — so the real domain is **different**,
not simply cleaner or harder.

**No R1 detector result has been computed or viewed.** Next: Phase 4, implementation and sanity
checks C1–C8, of which C8 uses CARRADA **Train** sequences only and is diagnostic, not reportable.

### 2026-08-31 — CARRADA real-data validation, Phase 4: R1 pipeline implemented, zero-shot thresholds derived, no test scoring

**Phase 4 of the real-data external-validation track.** Implementation and sanity checks only.
**NO CARRADA TEST DETECTOR RESULT HAS BEEN COMPUTED.** Every check ran on synthetic data or on
CARRADA **Train** sequences. No published number changes and no frozen asset was modified.

**Phase 3 commit (pre-registration frozen):** `9c9a3e25e60414309d5677e92ce2e85bfbeea079`.
**Final pre-registration SHA-256:** `e800ad84af5a909d595a9c1a5b32bfe308efc4830a1bfe17d8ea793a1bba98a8`
(FINAL PRE-SCORING FREEZE, 23,472 bytes), superseding the earlier draft `aaad1ff7…` that named OS
primary.

**Final rulings applied before the freeze.** (1) TinyCNN threshold rule approved as proposed.
(2) **Primary CFAR changed from OS to CA** — canonical baseline, continuity with the main
benchmark, and it avoids choosing the comparator because R0 hinted OS might suit CARRADA; GO/SO/OS
run as secondary with **OS highlighted as the robust heterogeneous-background comparison but never
promotable to primary after results.** (3) **NMS frozen at radius 5** with the extended-object
limitation registered. (4) Objects spanning range blocks assigned by the **median range index of
their valid support** — annotation-only, so no detector output can move an object's stratum.

#### TinyCNN zero-shot thresholds — synthetic validation only

Per frozen seed, from `bench2_val` alone, pooled across all 8 synthetic conditions, the lowest
threshold whose pooled measured per-cell Pfa ≤ 1e-4 at the ±1 gate, using frozen extraction, frozen
NMS 5 and the frozen valid region.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| threshold | 1.8049 | 1.8334 | 1.6981 | 2.0172 | 1.3329 | 1.9364 | 1.7510 | 1.9862 |
| achieved pooled val Pfa (×1e-5) | 8.74 | 9.74 | 9.52 | 9.03 | 9.85 | 8.76 | 8.90 | 9.98 |

All eight ≤ 1e-4; spread 1.333–2.017, sd 0.218; pooled validation Pd 0.578–0.589. Re-running seeds
0, 4 and 7 reproduced the thresholds to every printed digit, so the derivation is deterministic. A
runtime **file-access audit hook** recorded every file opened: **only `bench2_val`**, 3,203 data
files, **zero forbidden accesses** — machine-verified rather than asserted. ⚠️ These thresholds are
**not** part of the frozen benchmark; they are a **different operating-point protocol** and must
never be pooled numerically with the per-condition matched-Pfa synthetic tables.

#### Verification results

* **C1 adapter equivalence — PASS.** The pipeline feeds `mag = 10^(D/20)` to the **frozen**
  `normalize_rd`, so the registered adapter `D − median(D)` is realised by the frozen code itself
  and no preprocessing is reimplemented. Over 108 CARRADA Train frames:
  max |(D−median D) − 10log10(P/median P)| = **2.845e-08 dB**, max |frozen − registered| =
  **2.384e-07** (float32 output). Tolerances were set from float precision (≈2 ulp of float64 and
  float32), not chosen to make the test pass.
* **C2 CFAR scale invariance — PASS.** 192 combinations (4 variants × Train frames × scales
  0.01/1/100/1e6): max |ratio(P·s) − ratio(P)| = **1.455e-11**, peak-count difference **0**. CFAR
  source was not modified.
* **C3 checkpoint integrity — PASS**, 8/8 hash-match the isolation manifest.
* **C4 matching — PASS**, all 8 cases including maximum-cardinality on overlapping masks (not
  greedy-by-score) and the 18×13 car mask giving 1 matched / 2 unmatched.
* **C5 dense-mask reconstruction — PASS**, all 5,318 Train objects rebuild from JSON, covering all
  three classes, 999 two-object frames, 3,769 empty frames, 89 edge-touching masks.
* **C6 valid region — PASS**, 13,216 cells, rows 10–245 × cols 4–59, **imported** from the frozen
  `dl.infer.cfar_valid_mask` rather than reimplemented.
* **C7 metric tests — PASS**, including a 1-bin-off peak correctly missed by the exact mask and
  recovered by ±1, and an out-of-region object correctly excluded from the denominator.
* **C8 non-test dry run — PASS**, 48 CARRADA Train frames, 0 non-finite arrays, 0 structural
  problems. Aggregate recall deliberately not computed.

**A real defect was caught by the visual sanity check and fixed before any scoring.** The harness
applied the zero-shot threshold to TinyCNN but left CFAR **unthresholded** — `all_peaks` keeps every
local maximum with ratio > 0 and never applies alpha — so the arms sat at incomparable operating
points (CFAR ~116 peaks/frame against TinyCNN's ~9). `cfar_peaks_at_alpha` now applies the frozen
calibrated alpha (CA 10.459, GO 9.546, SO 12.849, OS 7.579), giving ~7 and ~9 peaks/frame.

#### Eligibility denominators (annotation-only, no detector)

| split | objects | eligible | excluded | strictly inside |
|---|---|---|---|---|
| Train | 5,318 | 5,317 | 1 | 4,689 |
| Validation | 1,804 | 1,804 | 0 | 1,589 |
| **Test** | **1,628** | **1,627** | **1** | **1,557** |

Test eligible by class: pedestrian 955, car 513, cyclist 159 — **cyclist results will be labelled
exploratory.** By range block: 10–60 → 533, 60–110 → 853, 110–160 → 111, 160–210 → 79, 210–246 → 51.

⚠️ **New and consequential: 1,365 of 1,627 eligible Test objects (84%) fall inside the ±4
zero-Doppler band.** CARRADA's road users are largely slow-moving and therefore sit inside the very
ridge R0 measured at 24 dB. The zero-Doppler stratification is not a minor sub-analysis — it covers
most of the test set, and R1's headline number will be dominated by objects buried in the strongest
clutter feature of the real domain.

#### Extended-object / frozen-NMS characteristic — measured, not acted on

At each arm's frozen operating point on CARRADA Train: **≥2 post-NMS peaks inside a single object's
own support in 35.4% of objects for CA-CFAR and 47.9% for TinyCNN** (median 1, max 5). Under
one-to-one matching exactly one is matched and the surplus counts as **unmatched**, so
**unmatched-detection burden is inflated for both arms and is not pure detector noise.** It is
common-mode in the TinyCNN−CFAR difference but not in the absolute burden. **NMS was not altered in
response** (PREREGISTRATION §7.3).

#### Leakage audit — CLEAN

No CARRADA Test RD frame was passed through TinyCNN or CFAR; no Test detector result exists; no
CARRADA label influenced any threshold; `bench2_test` was never opened; no experiment source line
loads it. One caveat recorded rather than buried: **R0's background statistics did include the Test
sequences** — domain-level, not detector-level, since R0 ran no detector, but it fed two
pre-registered choices. Checked directly: recomputing the ridge band from **Train sequences only**
gives the **identical ±1 band** (peak 23.9 dB vs 24.5 dB on Test), so the pre-registration would be
unchanged.

**Artefacts** (all under `experiments/r_carrada_real_data_validation/`): `src/{carrada_input,
eligibility,matching,metrics,detectors,stats,schema}.py`; `scripts/{derive_tinycnn_thresholds,
validate_adapters,run_sanity_checks,dry_run_r1,make_sanity_figures,leakage_audit}.py`;
`configs/tinycnn_zero_shot_thresholds.json`; `results/phase4_*.json` and
`phase4_multipeak_characteristic.csv`; 3 non-test sanity figures in `figures/phase4_sanity/`.
Notebook now 58 cells (§0–§43), executes with zero errors.

**Next: Phase 5 — score the 6 CARRADA Test sequences. Nothing above may change.**

## Decision history — how the configuration got here

Kept deliberately: the report needs the journey, not just the endpoint. Every superseded value below
is still recorded with its date, its number, and what displaced it. Nothing here has been deleted.

| parameter | path | why it moved |
|---|---|---|
| **epochs** | 12 → 50 → **15** (16 Aug) → **21** (16 Aug) | 12 was arbitrary with val loss still falling; 50 was "flat Pd so it does not matter"; 15 came from four sampled points once localization was found to degrade; **21** from the full 1–25 sweep under a pre-registered rule, which showed Pd was never flat below 15 (Correction C5) |
| **`pos_weight`** | 200 → 974 (13 Aug) → **100** (14 Aug) | 200 was tuned for the old 1:1563 ratio; 974 was the measured inverse class frequency, adopted as a "principled default"; 100 won the 3-way ablation and **retracted the `clutter` loss** it had manufactured (Correction C3) |
| **`clutter_edge` trend** | "still climbing past the plateau" → **seed noise** (12 Aug) | three correlated points from one trajectory, spread 0.047, against a 0.054 seed spread at fixed epoch (Correction C1) |
| **`isolated` win** | "DL beats CFAR, p = 6.9e-07" → **tie** (15 Aug) | CFAR's Pd had been treated as exact; with error bars on both sides and Bonferroni, p = 0.031 (Correction C4) |
| **gate** | ±1 pre-registered, ±3 added beside it | ±3 reported as robustness, not as a replacement — it is where the `clutter` loss appears |
| **`threads`** | incidental → **frozen** (16 Aug) | changes weights at fixed seed; ~1 SE on localization (entry above) |

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
