# Next Session — Fairness Audit of the CFAR vs DL Comparison

**Status: the current DL-wins-everywhere result is NOT yet trustworthy. Do not put it in the report
until the checks below are done.**

Written 30 July 2026, after finding and fixing one real bug and identifying several unresolved
methodological asymmetries.

---

## 1. Where things stand

### Fixed already ✅
**NMS plateau bug.** `all_peaks` used `score == maximum_filter(score)`, which marks *every* cell of
a plateau as a peak. A sigmoid output has flat/tied regions; CFAR's continuous ratio almost never
ties. So **DL emitted multiple detections per target** (1.90 per single-target frame vs CA's 0.83),
and because the evaluator ignores unmatched detections inside a truth's guard region, the duplicates
carried **no false-alarm penalty** — several free attempts per target.

Fixed in `src/cfar/detect.py` via `_collapse_plateaus()`: one detection per connected peak region,
applied identically to all detectors. **Retracted:** the "19 % sub-resolution recovery" claim (it was
the bug; corrected to 8 %, and that is probably a matching artifact).

### Corrected results (post-fix, `bench_test`, 150 frames/stratum)

| Condition | CA | OS | GO | DL |
|---|---|---|---|---|
| isolated | 0.867 | 0.867 | 0.867 | **0.880** |
| masking | 0.493 | 0.643 | 0.493 | **0.727** |
| unresolved | 0.490 | 0.490 | 0.490 | 0.490 (tie) |
| clutter | 0.666 | 0.686 | 0.618 | **0.723** |
| multi | 0.852 | 0.855 | 0.848 | **0.870** |
| multipath | 0.831 | 0.831 | 0.818 | **0.840** |
| mixed | 0.783 | 0.798 | 0.768 | **0.804** |

Mechanism metrics: masking weak-target Pd **DL 0.500 vs OS 0.333**; clutter-band FA/frame
**DL 2.47 vs GO 4.83**.

### Why this is still suspect
A 5,041-parameter CNN beating well-implemented CFAR in **every** condition — including homogeneous
noise, where cell-averaging CFAR is near-optimal — is not credible at face value. Two structural
asymmetries most likely explain it, and **both are our own methodological choices, not fraud**:

1. **The CNN gets an implicit matched filter; CFAR does not.** A target spreads over ~5×5 cells.
   CFAR tests a **single** cell against its neighbours; the CNN convolves across the whole mainlobe,
   effectively integrating that energy — a genuine several-dB gain unrelated to "learned vs
   hand-derived decision rule". In real radar you integrate *before* CFAR. **This alone could
   explain a win in every condition.**
2. **CFAR's geometry was tuned on the wrong condition.** guard=2 / train=8 was chosen using
   `snr_sweep` (clean, single target — where geometry barely matters) and then applied everywhere.
   In the clutter stratum a 21×21 window is far wider than the ~4-bin clutter ridge, diluting the
   noise estimate and dropping the threshold inside the ridge. That likely accounts for much of CA's
   7.91 band false alarms.

Additional open issues:
3. **Pfa is not actually matched.** Measured: DL 4.8e-5, GO 1.2e-4, CA 1.5e-4, OS 1.9e-4 — a 4×
   spread. Thresholds were calibrated on pure noise but evaluated on clutter/targets. (It happens to
   *handicap* DL, so it does not inflate the win — but "matched Pfa" is currently an overstatement.)
4. **DL adapts to the data; CFAR does not.** DL was retrained on the new distribution; CFAR's
   parameters were not re-tuned for it. Indefensible as it stands.
5. **Matching gate may penalise CFAR.** DL is trained to hit the *label* bin; CFAR finds the
   *physical* peak. A ±1-bin gate may be measuring localisation, not detection.
6. **Train and test share one generator** — identical PSF and clutter model, so the network can
   exploit generator regularities absent from real data.

---

## 2. Plan for the next session

Work top-down; **stop and reassess after Step A**, since it may change the conclusion.

### Step A — Cheap diagnostics (do these first; they localise the problem)

**A1. Compare at matched *measured* Pfa.**
Build the full ROC for each method and interpolate Pd at a common **measured** per-cell Pfa
(e.g. 1e-4), instead of using each method's design threshold. Removes issue 3.
*Files:* `src/dl/infer.py::dl_roc`, `src/eval/detection.py::roc_curve` (both exist).

**A2. Add a trivial baseline — global threshold + peak-find, no adaptivity.**
If it scores near CFAR, the benchmark is too easy and the comparison is uninformative. This is the
fastest way to detect a rigged-easy setup. *~20 lines.*

**A3. Widen the matching gate to ±2 bins and re-score.**
If CFAR's Pd rises more than DL's, the tight gate was penalising CFAR's localisation (issue 5).
*Files:* `MatchConfig(tol_r=2, tol_v=2)` — one-line change.

**Expected outcome:** A1–A3 tell us how much of the DL margin is real versus protocol artifact.

### Step B — The real fairness fixes

**B1. Give CFAR the matched-filter benefit** *(highest value — tests the leading hypothesis)*.
Apply CFAR to a mainlobe-integrated map (e.g. convolve the power map with the measured PSF, or a
small boxcar/Gaussian) so both detectors see integrated energy. Compare "CFAR" vs "matched-filter +
CFAR" vs "DL".
*Prediction: this closes a large part of the gap.* If so, the honest headline becomes
**"the CNN's advantage comes substantially from implicit mainlobe integration, not from a smarter
decision rule"** — a more defensible and more interesting finding than "DL wins everywhere".

**B2. Tune CFAR per condition**, as DL was retrained per distribution. Sweep guard/train (and OS
rank) *within each stratum* and report each method at its own best geometry. Removes issue 4.

### Step C — External validity (after B)

**C1. Perturb the generator at test time** — change clutter strength, slightly alter the PSF or
window — and re-test DL **without retraining**. If it collapses, it was exploiting generator
regularities (issue 6). This is also a genuine "future work / limitations" result.

---

## 3. What to report regardless of outcome

* The NMS bug, how it was found (DL exceeded the physical 0.5 cap on unresolved pairs), and the fix.
  **Finding your own bug via a physics sanity check is a strength, not an embarrassment.**
* Whatever the corrected margin turns out to be — even if DL's advantage shrinks substantially.
* The matched-filter asymmetry, whether or not B1 is completed: it is a real limitation of the
  comparison as originally framed.

**Principle to hold onto:** a smaller, well-understood DL advantage is worth far more than a large
unexplained one. If the honest conclusion is "DL and CFAR are comparable once CFAR is given
equivalent pre-integration", that is a perfectly good — arguably better — thesis result.

---

## 4. Quick reference

| Item | Location |
|---|---|
| Corrected results | `docs/results_stratified.md` (see the ⚠️ box at the top) |
| Coverage study (evidence → diagnosis → redesign) | `docs/dataset_coverage_study.md` |
| DL design record | `docs/dl_design.md` |
| Model 0 results + pros/cons | `docs/results_model0.md` |
| Notebook (36 cells, 10 figures) | `notebooks/dl_cnn_experiment.ipynb` |
| Per-condition evaluator | `src/dl/stratified_eval.py` |
| NMS fix | `src/cfar/detect.py::_collapse_plateaus` |
| Trained checkpoints | `results/dl/tinycnn.pt` (frozen data), `results/dl/tinycnn_bench.pt` (stratified) |
| Raw per-condition CSV | `results/dl/stratified_after_FIXED.csv` |

**Schedule:** Week 2 milestone (CFAR-vs-DL with real numbers) is met, ~10 days ahead of
`SCHEDULE.md`. That slack should be spent on this fairness audit **before** moving to Model 1
(U-Net) — a U-Net built on an unfair protocol would just inherit the problem.

**Note:** the notebook still shows the pre-fix figures in §13 (it was executed before the NMS fix).
Re-execute it after Step A so the numbers there match the corrected results.
