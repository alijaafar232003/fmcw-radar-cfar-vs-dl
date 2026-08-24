# Next Session — pick up here

> ## ⚡ STATE AS OF 19 AUGUST 2026 — read this block first
>
> Everything below the horizontal rule predates the DL work and is kept for the CFAR reasoning.
> The current state is here. Full detail: `PROGRESS.md` Milestone 23 (dated entries) and
> `notebooks/cnn_development_log.ipynb` (§1–18, Corrections, Open, Next experiments).
>
> ### 🔒 The configuration — FROZEN 17 Aug → **`configs/frozen_dl.yaml`**
> That file is the contract; this table is the summary. It was validated field-by-field against the
> `config_frozen_s0.json` the run actually produced.
>
> | | |
> |---|---|
> | model | `TinyCNN`, 5,041 params, RF **21×9** = the CFAR window |
> | `pos_weight` | **100** — measured; 974 (exact class balancing) was worse on 4 of 7 conditions |
> | epochs | **21** — the full 1–25 sweep under a pre-registered rule (earliest epoch within 1 SE of the Pd plateau). Supersedes "15": Pd is *not* flat below 15, it rises +0.059 from epoch 1 |
> | **threads** | **2 — LOAD-BEARING.** Changes the trained weights at fixed seed (max \|ΔW\| 0.38, ~1 SE on localization). Arms at different thread counts may not be pooled (§17) |
> | optimizer | Adam, lr 1e-3, batch 32 |
> | NMS | `nms_size=5`, shared with CFAR |
> | gate | **±1** pre-registered; ±3 reported beside it as robustness |
> | CFAR | 21×9 window; **CA/GO/OS/SO** (SO added 15 Aug) |
> | operating point | matched **measured** Pfa 1e-4, swept per condition |
>
> ### 🏁 HEADLINE — `bench2_test`, 8 seeds @ epoch 21, pre-registered, run once (19 Aug)
> These are **the reportable numbers**. The split is now spent.
>
> **Pd, gate ±1 — DL wins 6 of 7 after Bonferroni:** `clutter_edge` **+0.302**, `mixed` +0.103,
> `isolated` +0.058, `masking` +0.051, `multi` +0.045, `unresolved` +0.012 (all p ≤ 0.001).
> `clutter` **ties** (−0.002, p = 0.79). Gate ±3: the same 6 of 7.
>
> **⚠️ DL loses NOWHERE.** The `clutter` loss at ±3 did **not** replicate (val −0.043 p = 0.005 →
> test −0.019 p = 0.068) → **Correction C6**. This is the *weaker* position to report from and
> revives the Milestone 17 concern. Report it as such.
>
> **⚠️ `unresolved` is significant but is NOT a win** — +0.012 at the ~0.5 physics cap where both
> methods are stuck. See Open/O2. Do not put it in a wins list.
>
> **Localization — CFAR wins, and this is DL's real weakness:** DL Doppler **0.382 ± 0.026** vs
> **0.100 (OS)** = **3.83×**; range 0.476 ± 0.021 vs 0.198 (OS) = **2.40×**.
> **Quote 3.83×.** It supersedes 4.6× (val ep21) and 5.6× (val ep50) — both kept on record.
>
> **Cost:** DL **6.04 ms/frame** vs CA 0.86, GO 7.79, SO 7.82, OS 69.95 — ~7× slower than CA,
> **11.6× faster than OS**, within noise of GO/SO (the ordering flipped between runs, so do not
> claim it either way). TinyCNN 4,896 MAC/cell, 5,041 params.
>
> **Protocol check:** measured Pfa within **2%** of 1e-4 for every method on every condition.
>
> One-line summary: *at finding targets in interference DL wins; at placing them CFAR does, by ~4×
> in Doppler; and it runs at roughly GO/SO cost, far cheaper than OS.*
>
> **Scope:** `bench2_test` shares a generator with val — this is **sampling generalization only**,
> not evidence any finding survives a different clutter model, sensor, or real data.
>
> *(The `bench2_val` epoch-21 figures this supersedes are kept in
> `results/dl/bootstrap_frozen_ep021_tol{1,3}.csv` and `localization_frozen_ep021.csv`, §18.)*
>
> ### Do next, in order
> 1. ~~Freeze the config~~ ✅ **DONE 17 Aug** — `configs/frozen_dl.yaml`.
> 2. ~~`bench2_test`, ONCE~~ ✅ **DONE 19 Aug** — §19. **The split is spent.** Any further read is a
>    second read and must be recorded as one, with its reason.
> 3. ~~Pd-vs-SNR figure~~ ✅ **DONE 19 Aug** — §20, `results/pd_vs_snr_frozen_ep021.png`.
>    CFAR loss against theory at gate 0. **Two separate measurements — do not merge the ranges:**
>    **`bench2_test` clean conditions → 0.27–0.64 dB** (`cfar_baseline.ipynb` §5), and
>    **`bench2_snr_sweep` → 0.47–0.78 dB** (`cnn_development_log.ipynb` §20). Different splits and
>    target populations (mean cell SNR 15.2 vs 11.2 dB); both valid.
>
>    **✅ DECIDED 22 Aug — which to quote where.** The **Results chapter quotes the `bench2_test`
>    figure**, because every other reported number comes from that split and mixing splits inside one
>    chapter invites exactly the confusion this note exists to prevent. Quote it as
>    **"0.27–0.64 dB as plotted, 0.45–0.86 dB once the post-NMS-vs-per-cell Pfa offset is
>    corrected"** — §5 computes both. The **`bench2_snr_sweep` figure stays as the supporting
>    validation figure**: it has the cleaner design for this particular curve (one target per frame,
>    SNR swept uniformly), so it is the better *picture* even though it is not the headline *number*.
>    ⚠️ That is measured at a **post-NMS** Pfa against a **per-cell** theory curve, so it is
>    understated by ~0.1–0.2 dB (quantified in `cfar_baseline.ipynb` §5). Common-mode in the
>    DL-vs-CFAR delta, so it cancels there.
>    ⚠️ **Use the two-panel version.** At the ±1 gate every method sits ABOVE theory (3×3 = nine
>    chances vs theory's one), so theory is a bound only at gate 0. The ±1 panel alone would show
>    five detectors beating physics. Bonus finding: at gate 0 DL plateaus at **Pd 0.41** where CFAR
>    reaches ~0.85 — the localization deficit as a detection number, on a third split.
> 4. **Six evaluation-methodology fixes — ✅ FOUR CLOSED 24 Aug**, in
>    `notebooks/evaluation_metrics_overview.ipynb` (built by
>    `scripts/build_evaluation_metrics_nb.py`). Full record: `PROGRESS.md` 24 Aug.
>    1. ~~Localization on the common detected subset~~ ✅ **DONE 24 Aug** —
>       `scripts/localization_common_subset.py` completed its first run. **The gap survives:**
>       pooled Doppler **3.83× → 3.63×**, range 2.40× → 2.31×, on a common subset that is 94.3% of
>       the CFAR arm's detected targets. Shrinks on 5 of 7 conditions — that is the selection
>       effect, measured, and it is small. **`clutter` moves the other way (6.44× → 8.80×)**: the
>       arms overlap least there and tie on Pd, so the intersection is a different population and
>       the ratio is unstable. ⚠️ Still **post-hoc** — report beside the pre-registered number,
>       never instead of it. → notebook §6.1,
>       `results/dl/localization_common_frozen_ep021_test.csv`.
>    2. ~~Re-select best CFAR inside each bootstrap replicate~~ ✅ **DONE 24 Aug** —
>       `scripts/bootstrap_best_cfar_reselect.py` (new). **The prediction was wrong: 0 of 7 verdicts
>       changed and the intervals did NOT widen** (ratios 0.952–1.000). Selection uncertainty is
>       real — the winner is displaced in 29% of replicates on `unresolved`, 21% on `clutter` — but
>       immaterial, because the ranking only flips when the variants are within noise of each other,
>       where swapping them changes Pd by almost nothing. The original procedure's assumption was
>       not load-bearing. → notebook §8.1.
>    3. ~~Define the three Pfa notions once~~ ✅ **DONE 24 Aug** — notebook §5.1, with the
>       per-cell/post-NMS relation measured (NMS survival 0.66–0.83) rather than asserted.
>    4. **Precise wording:** "Detector discrimination is compared at matched measured Pfa = 1e-4,
>       obtained independently from each condition's ROC" — plus a clause noting a per-condition
>       threshold is an oracle a deployed system lacks, which frames the fixed-threshold experiment
>       as closing a stated gap. **Written up as notebook §7.2; still to land in the Method chapter.**
>    5. ~~Never print `p = 0`~~ ✅ **DONE 24 Aug** — `fmt_p()` in the notebook reports below the
>       two-sided floor (2/2000) as `p < 0.001`, used at every p-value site.
>    6. **Confirm Bonferroni on the DL side** and state it in Methods rather than applying it
>       silently. **Written up as notebook §9; still to land in the Method chapter.**
>       ⚠️ **This item previously said `unresolved` (+0.012) "does not survive correction". That was
>       WRONG and is retracted (24 Aug).** Measured: p = 0.001 < α = 0.0071 at **both** gates, and
>       the `sig (Bonf)` column of `bootstrap_frozen_ep021_test_tol{1,3}.csv` has said `True` since
>       19 Aug. The reasoning error was inferring a statistical property from a physical one — the
>       tie-at-the-cap reading is right, so the note assumed the statistics would agree.
>       **The correct reading:** `unresolved` is **statistically significant and physically a tie**
>       — ~1 percentage point against a ~0.5 ceiling both arms are pinned against, p sitting at the
>       bootstrap's own resolution floor. Still not a win, and still not in a win list. This is the
>       *stronger* version for the report: the correction does **not** rescue the reading, so
>       judgement about effect size has to. No published number changes, so it is **not** a new
>       Corrections-log entry (C1–C8 are retractions of reported results; this was a to-do note).
>       → notebook §9.2.
>
> 5. **WRITING — the whole critical path now.** All experiments are done. `SCHEDULE.md` targeted
>    20 Aug; nothing is drafted; the realistic target is the **1 Sep hard deadline**. Order:
>    Background → Method (both largely in `docs/`) → Results → Discussion → Intro/Conclusion/
>    Abstract → slides → rehearse.
>
> **For the report, do not bury these:** the Corrections log C1–C8 (eight retractions with the
> reasoning error named in each), the decision history (epochs 12→50→15→21, `pos_weight`
> 200→974→100), and the fact that three pre-registered predictions failed on the test split and are
> reported as failures. That is stronger viva material than a clean-looking result.
>
> **Deliberately cut** (logged, not chased — they change no published number): the `TinyCNNPool`
> control, N2/N3, the Open/O4 localization asymmetry, Milestone-18 attribution, CARRADA, and the
> remaining proposal ablations. Every one of them costs a writing day that does not exist.
>
> ### Rule adopted 15 Aug, worth keeping
> *A new finding goes on the limitations list, not the work queue, unless it changes a number
> already published in `PROGRESS.md`.* Adopted because the project was drifting into open-ended
> fixing; the six-item robustness block was run under it and closed cleanly.
>
> ### Known-open, deliberately not chased
> ~~no CFAR alpha is calibrated~~ **closed 21 Aug** — all four calibrated by Monte Carlo, verified
> on a held-out seed, cached in `configs/cfar_alpha_table.json` (Corrections C7→C8). Achieved
> per-cell Pfa went from 0.13×–6.55× of target to 0.95–1.01×; no reported Pd changed ·
> `os_rank_frac` never swept · no learned linear baseline · `pos_weight` < 100 untested (100 was the
> edge of the range) · ~~nothing below epoch 15 tested~~ **closed 16 Aug — epochs 1–25 all
> measured** · localization degrades *asymmetrically* between range and Doppler, which a symmetric
> 3×3 label does not predict (notebook Open/O4) · one simulator only (criterion 8) ·
> `clutter_edge` stratum 34% diluted · H1/H2 attribution (Milestone 18) not started ·
> **the DL ROC cannot be swept past sigmoid 0.5** — `all_peaks` keeps `score > 0`, which on
> logits removes 67 of 80 local maxima on a sample frame (0 of 115 for CFAR, whose ratio is
> positive by construction). No reported number is affected: operating cuts sit at 1.39–1.70
> logits, so everything removed would have failed the cut anyway. It would only matter for a
> **high-Pfa** DL operating point, which the fixed-threshold / CFAR-ness experiment might need
> (added 24 Aug; notebook appendix A.2).

---


Current as of 11 August 2026, and everything below is committed. `PROGRESS.md` holds the full
record with reasoning; this file is the short version plus what happens next.

> **If you are a session with no prior context: read this file first, then `PROGRESS.md`.**
> Start at §2.1. Everything in §4 is settled and should not be re-derived — several of those
> decisions cost a measurement each to establish.

**What changed most recently:** the radar was re-specified to match the CARRADA dataset, and
*everything downstream was rebuilt and re-verified at the new sensor* — clutter model, CFAR window,
CNN receptive field, all five data splits. Results in `docs/results_model0.md` and
`docs/results_stratified.md` predate that and are banner-marked as superseded; their **methods**
carry over, their **numbers** do not.

---

## 1. Where things stand

The radar was re-specified to match the **CARRADA** dataset so that a simulation-trained detector
can eventually be tested on real measurements without a resolution mismatch confounding it.
Everything downstream was rebuilt and re-verified at the new sensor. **Steps 1–7 are done.**

### The configuration, in one place

| | |
|---|---|
| Radar | 77 GHz, **B = 768 MHz**, **Tc = 72.4 µs**, N = 512, **M = 64** |
| RD map | **256 × 64** — 0.195 m and 0.42 m/s per bin; 50 m and ±13.44 m/s |
| Mainlobe | 2 bins (−30 dB half-width), set by the Blackman window, **measured** |
| Resolution limit | 0.39 m equal-strength, 0.59 m for a weak partner; 1.16 m/s in Doppler |
| **CFAR window** | **21 × 9** (guard 2/2, train 8/2) — 164 ref cells, α = 9.474 at Pfa 1e-4 |
| **CNN receptive field** | **21 × 9**, measured by backprop — matches CFAR exactly |
| CNN | TinyCNN, 5,041 params, dilations range (1,2,3,4) / Doppler (1,1,1,1) |
| NMS | 11 × 11 (`nms_size=5`), shared by CFAR and DL |
| Clutter | compound-Gaussian: Weibull texture (shape swept 1.1–2.0) × Rayleigh speckle, CNR-calibrated |

### The five splits (no shared frames, verified by content hash)

| split | frames | strata | seed | role |
|---|---|---|---|---|
| `bench2_train` | 8,400 | 7 × 1200 | 92001 | network training (`unresolved` excluded) |
| `bench2_val` | 1,600 | 8 × 200 | 93001 | overfitting check |
| `bench2_test` | 3,200 | 8 × 400 | 94001 | **the reported numbers — touch once** |
| `bench2_tune` | 400 | 8 × 50 | 91001 | CFAR geometry (already used) |
| `bench2_snr_sweep` | 1,300 | 1 condition | 95001 | Pd-vs-SNR diagnostic, evaluation only |

### CFAR baseline, already measured on `bench2_tune`

Pd at matched **measured** Pfa 1e-4, at the 21×9 window:

| condition | CA | GO | OS |
|---|---|---|---|
| isolated | 0.632 | 0.623 | 0.632 |
| multi | 0.595 | 0.590 | 0.591 |
| masking | 0.484 | 0.494 | **0.534** |
| mixed | 0.390 | **0.400** | 0.354 |
| clutter_edge | 0.390 | 0.390 | 0.390 |
| clutter | 0.145 | **0.167** | 0.147 |
| unresolved | 0.471 | 0.471 | 0.470 |

*(These are on the tuning split. The reportable CFAR numbers come from `bench2_test`.)*

---

## 2. What to do next — three tasks, in this order

### 2.1 CFAR on `bench2_test` — ✅ DONE (11 August)

All 3,200 frames. `scripts/cfar_test_baseline.py` → `results/cfar_test/` (detections cached, so any
operating point can be re-derived without re-running CFAR); presented end-to-end in
`notebooks/cfar_baseline.ipynb`. Numbers and analysis in `PROGRESS.md` Milestone 21.

**This is now the number the network has to beat.** Do not re-run it, and do not touch
`bench2_test` again until the network is final.

Headline: OS wins masking (0.623 vs 0.533) and clutter_edge (0.465 vs 0.407); everything else is a
tie within sampling noise. In clutter, ~100% of every method's false alarms land in the
zero-Doppler ridge — the specific weakness a learned detector should attack.

<details><summary>Original brief (kept for the reasoning)</summary>

CFAR has been run on `bench2_tune` (to pick the window) but **never on the test split**. Do it
*before* the network exists, for three reasons: it is then a pre-registered baseline that cannot be
unconsciously tuned against; it exercises the evaluation pipeline end-to-end while there is no
network to blame if something breaks; and CFAR has no learnable parameters, so touching the test
split with it cannot contaminate anything.

Produce per-condition Pd at matched **measured** Pfa for CA/GO/OS at the 21×9 window. Run once,
record it, do not iterate.

✅ **The evaluator was fixed first (11 August) — use `src/dl/stratified_eval.py`.** It now matches
at **measured** Pfa per condition, closing Milestone 16's asymmetry #3, and is the single path for
both halves of the comparison. Set `METHODS = ("CA", "GO", "OS")` to run it CFAR-only.

Validation: on the `noise` stratum the swept threshold lands at CA **10.173** against the
independently Monte-Carlo-calibrated **10.163**, and it reproduces the step-6 baseline to ≤0.017
with every ordering intact. Three interpolation bugs were found and fixed en route — see
`PROGRESS.md` Milestone 20b.

`eval.detection.pd_vs_snr()` now requires a keyword-only `threshold` and rejects unscored
detections, so the Pd-vs-SNR figure cannot silently be produced at "detect everything".

⚠️ `scripts/cfar_geometry_percondition.py` still uses the coarse grid and linear interpolation. Its
window decision stands (the 21×9 gains were 10× the discrepancy), but do not use it for new numbers.

</details>

Deferred deliberately: scoring DL on **logits** rather than sigmoid probabilities. Saturation
compresses the ROC's low-Pfa tail and leaves `_collapse_plateaus` picking an arbitrary cell from a
tied region — but the duplicate-detection failure it used to cause was closed in Milestone 16, so
this is root-cause tidying, not a fix. It belongs with the attribution-phase `nms_size` sweep, since
the two interact through the evaluator's guard-region forgiveness.

### 2.1b GO and the clutter edge — ✅ INVESTIGATED (11 August), answer is "no"

Full record in `PROGRESS.md` Milestone 22. Short version:

**GO genuinely loses `clutter_edge` on Pd at matched measured Pfa, and the margin grows once the
data is cleaned up.** The stratum turned out to be **34% diluted** — the ridge is ±4 Doppler bins,
targets sit at offsets 6–10, and the 21×9 window reaches only ±4, so a third of the frames are
ordinary noise frames wearing the label. Split by whether the window actually straddles: at a **true
edge OS leads GO by +0.068** (0.439 vs 0.372); GO wins only the *non*-edge frames (0.555 vs 0.487).

GO is working as designed — it takes the greatest half-ring, so at an edge its noise estimate is
**2.1× CA's** while OS's is **0.51×**. GO's purpose is false-alarm control (2× better than CA at a
fixed threshold), and matching the measured Pfa cancels that by construction.

⚠️ **Do not "fix" this by switching to 21×7.** Pooled, 21×7 hands GO the `clutter_edge` win — but
restricted to targets that genuinely sit at an edge, OS still leads at every window and GO never
beats CA by more than 0.023. 21×7 straddles only 41.5% of the stratum against 21×9's 65.6%, so it
was winning the frames that had stopped being edges.

✅ **The step-6 window choice is vindicated** (this reverses an earlier suspicion in this file).
Stratifying by target clearance from the ridge — which no window can change — Pd at the hardest
band runs 0.065 → 0.276 from 21×21 to 21×9, a 4× gain at *fixed* physical difficulty, where dilution
cannot explain anything. Narrowing the Doppler window genuinely helps.
`results/cfar_window_true_edge_sweep.csv`, `scripts/cfar_window_true_edge_sweep.py`.

**21×9 is kept** — not because 21×7 is artefactual, but because across all seven conditions 21×7 is
a wash: +0.039 `clutter_edge` and +0.010 `mixed` against −0.020 `clutter`, −0.010 `masking`,
−0.010 `multi`, mean **+0.004**, inside noise. No overall gain, so the incumbent stands and the test
baseline needs no re-run.

**Deferred, cheap, would remove the problem at the source:** tighten the `clutter_edge` velocity band
from 2.5–4.2 m/s to ~2.5–3.4 m/s so every frame straddles the 21×9 window. Needs regeneration of
that stratum, so not on the critical path.

**Not yet tested** (dropped when the dilution explained the result, but still valid controls):
GO with `go_axis=0` as a negative control, and quantifying GO's homogeneous-noise bias
(max-of-two-estimates) separately from its edge behaviour.

### 2.2 Data-quality re-check — ✅ DONE (11 August)

All nine criteria re-measured on the frozen splits by `scripts/audit_data_quality.py` (new).
**8/9 pass**; full record in `PROGRESS.md` Milestone 20, write-up in `docs/data_quality_criteria.md`.
The three items listed here as outstanding are now closed:

| Criterion | Status |
|---|---|
| Clutter distribution | ✅ `c = 1.51` measured **in the frozen data**, not just the model |
| Coverage | ✅ 0–5 targets, 8–45 m, ±8 m/s, no empty range decile |
| Difficulty calibration | ✅ ~48–50% of targets in the theory Pd 0.1–0.9 band |
| **Noise is exponential** (KS test) | ✅ KS **0.006** on target-free frames |
| **Targets off-grid** | ✅ fractional offset std **0.290** vs 0.289 ideal uniform, KS 0.007 — the `close_pair` snapping removal is confirmed to have worked |
| **No learnable shortcuts** | ✅ no frame statistic predicts the label (max \|corr\| 0.21); 8400/8400 unique seeds; per-frame median spans 945–8166, so nothing is per-frame normalised |

Also verified beyond the three: the radar equation fits at exponent **−1.003** with the residual
spanning exactly the config's 4–16 dB draw window — the loss is applied once, at the right exponent.

**The one open criterion is 7 (benchmark neutrality):** `bench2_test` is eight *equal* strata,
37.5% deliberately adversarial. Per-condition is the headline; a pooled Pd needs its composition
printed beside it. That bears directly on 2.1 below.

### 2.3 Retrain TinyCNN

#### Four fixes to `src/dl/train.py`, all identified in the step-6d audit

The previous run stopped at a fixed 12 epochs with the validation loss **still falling**, kept
whatever the last epoch produced, and used only half the validation set. None of that is defensible
in a report.

1. **Early stopping** — train until validation loss stops improving (patience ~5), not a fixed count
2. **Save the best epoch**, not the last
3. **Use all 1,600 validation frames** (the previous run used 800)
4. **Revisit `pos_weight=200`** — class imbalance moved **1:1563 → ~1:900–1000** when the map
   halved (step 6d measured 1:888; the 11 Aug audit measured 1:943–1:1027 on a different sample —
   two samples of one quantity, not a disagreement). Per `train.py`'s own note the exact value
   matters less than intuition suggests, since the ROC sweeps the threshold, but it should not
   silently inherit a value tuned for a different ratio. `train.py`'s docstring still quotes 1:1563
   and needs updating too.

Note: there is **no training entry point** — the loop lived in `dl_cnn_experiment.ipynb`, deleted in
`14b38bd`. `scripts/plot_loss_curves.py` reads `results/dl/train_bench.log` and parses
`epoch N train X val Y`, so whatever replaces it must write that format.

#### Re-run the Milestone-13 gates at the new geometry

- dataset shapes / finiteness / per-frame median ≈ 0 *(already confirmed in the audit)*
- **orientation check** — no transpose bug
- **receptive field measured 21 × 9** *(already confirmed)*
- **overfit-one-batch** — loss must collapse and peaks land within 1 bin

#### Then evaluate

- Per-condition Pd at matched **measured** Pfa on `bench2_test`
- Pd-vs-SNR on `bench2_snr_sweep`, overlaid on theory `Pd = exp(−T/(1+SNR))`

---

## 3. After those three

**Attribution (the original fairness audit, unchanged in intent).** Six named hypotheses with
falsifying experiments — see `PROGRESS.md` "Milestone 18". Start with H2's analytic bound, then the
guard-constrained CNN, the CFAR-property test, and the tolerance sweep.

**Proposal ablations still outstanding:** RD resolution (bandwidth/chirps — needs a `radar:` block
in the dataset YAML, ~6 lines, then vary M ∈ {32, 64, 128}, CFAR-only), DL model size, and
magnitude vs I/Q input (the complex map is saved, so this needs no regeneration).

**CARRADA**, when the dataset arrives: test-only first, then hybrid (real background + injected
simulated targets), then optionally fine-tuning. **Split by sequence, never by frame** — 12,666
frames but only 30 sequences and 78 instances, so frame-level splitting leaks badly.

**Writing.** This is the only genuinely behind track. `docs/` holds most of the background and
method material already.

---

## 4. Things not to re-derive

- **Guard = 2 is measured, not chosen.** It equals the −30 dB mainlobe half-width, which is set by
  the Blackman window and does not move with the sensor.
- **Match at *measured* Pfa, never design.** Clutter drives the achieved rate 60–125× above design,
  by different factors per method — a comparison at "design 1e-4" would compare CA at 1.2e-2
  against GO at 8.7e-3 against OS at 3.6e-2.
- **`nms_size=5` was investigated and kept.** It caps the masking stratum near Pd 0.69, which is a
  documented limitation; shrinking it is measurably worse because NMS also filters noise, and
  `nms_size=1` re-introduces free attempts. Full reasoning in `src/cfar/detect.py`.
- **CARRADA's published 4 GHz bandwidth is wrong** for reproducing its bin geometry; 768 MHz is
  correct. Evidence in `docs/radar_config_carrada.md`.
- **Alignment is ~93%, not 100%, and that is right.** It is 100% on isolated and multi, and falls
  only where interference is the point of the condition. Clean-map regeneration puts every label
  exactly on the peak.

---

## 4b. 📊 Figures earmarked for the report and presentation

Flagged as worth using — regenerate at final quality when writing.

| Figure | Source | Why it earns a slide |
|---|---|---|
| **8-panel strata sheet** | `results/bench2_debug_strata.png` (`scripts/preview_strata.py`) | Shows the whole experimental design in one image — every condition, targets ringed. Best single "here is what I built" figure. |
| **CFAR noise estimate vs clutter ridge** | `results/cfar_sanity_threshold.png` (`scripts/cfar_sanity.py`) | Makes the window problem *visible*: the ridge is narrow, the estimate is 5× wider. This is what motivated the 21×9 window. |
| **Difficulty vs theory** | `notebooks/dataset_overview.ipynb` §9 | Target SNR histogram against the Pd 0.1/0.5/0.9 lines — shows the data sits where detectors actually differ. |
| **Difficulty by scenario** | `dataset_overview.ipynb` §6 | Boxplot per stratum against the same theory lines. |
| **Split distributions overlaid** | `dataset_overview.ipynb` §5 | Evidence train/val/test are identically distributed. |
| **Swept parameters** | `dataset_overview.ipynb` §10 | Separation and clutter-tail coverage; motivates the two sweep analyses. |
| **Pd vs SNR against theory** | *not yet made* — `bench2_snr_sweep` | The validation figure: measured curve on the theoretical one, with the gap being CFAR loss. |
| **Loss curves** | *regenerate* — `scripts/plot_loss_curves.py` | Reads the training log directly so it cannot drift from the run. |

All plots use the same palette and styling (`dataviz` conventions), so they will look like one set.

## 4c. Verification already done — do not repeat

Two scripts gate the data, and both pass. Re-run them if anything changes; there is no need to
re-derive what they check.

**`scripts/verify_splits.py`** — *is anything broken?*
Balanced strata · **zero shared frames between any of the ten split pairs**, checked by hashing
frame contents rather than by comparing seeds · alignment 92.3–93.6% · zero targets in the untested
border · zero non-finite values.

**`scripts/audit_datasets.py`** — *is it what we asked for?*
Every stratum matches its config (masking separations 0.41–2.49 m against a 0.40–2.50 spec,
`unresolved` 0.05–0.35 exactly, `clutter_edge` velocities 2.50–4.20 exactly, `clutter` below 1.2
m/s) · `radar_config_used.json` matches the code in all five splits · 1047/1047 sampled labels have
bin indices consistent with their physical coordinates · all 16 metadata fields present.

⚠️ **Both sample frames RANDOMLY, deliberately.** The generator assigns conditions cyclically
(frame *i* gets condition *i* % n_strata), so an evenly-spaced `linspace` sample aliases against that
cycle and can miss whole strata — it once returned **2** `unresolved` targets against 155–576 for
every other condition, in a table that otherwise looked entirely plausible. Do not "simplify" this
back to `linspace`.

## 4d. Repository state after cleanup

Removed (all recoverable from git history): four notebooks that ran against the previous sensor
(`cfar_comparison`, `data_quality_audit`, `dl_cnn_experiment`, `validate_rd_alignment`),
`src/01_generate_rd_dataset.ipynb` (legacy — defined its own `RadarConfig`, so it kept the old
radar parameters alive), the streamlit demo's 620 MB venv, `docs/RADDet.pdf`, and `results/dl/`
(checkpoints and caches from models trained on deleted data).

Kept: `notebooks/InteractivePipeline.ipynb` (the CFAR prototype the vectorised implementation was
ported from), the streamlit demo minus its venv (wanted for the presentation), and — deliberately —
`results/cfar_geometry_sweep.csv` and `results/cfar_geometry_percondition.csv`, which are the raw
evidence behind the 21×9 window choice and exist nowhere else, since `results/` is gitignored.

Nothing referencing the old sensor remains outside `docs/results_*.md`, which are banner-marked as
superseded and kept as the experimental record.

## 5. Reference

| | |
|---|---|
| Full record + reasoning | `PROGRESS.md` |
| Radar config derivation | `docs/radar_config_carrada.md` |
| DL design record | `docs/dl_design.md` |
| 19-paper corpus review | `docs/literature_corpus_review.md` |
| Scripts | `scripts/measure_psf.py`, `verify_clutter.py`, `cfar_sanity.py`, `cfar_calibrate.py`, `cfar_geometry_sweep.py`, `cfar_geometry_percondition.py`, `verify_splits.py`, `preview_strata.py`, `roc_curve_export.py`, `bootstrap_best_cfar_reselect.py`, `localization_common_subset.py` |
| Evaluation protocol | `notebooks/evaluation_metrics_overview.ipynb` (built by `scripts/build_evaluation_metrics_nb.py`) |
