# Next Session — pick up here

Rewritten 10 August 2026. Supersedes the previous version entirely (that one described the
fairness audit as it stood *before* the sensor was re-specified to match CARRADA; its plan survives
as the attribution phase below, but its numbers do not).

**Everything below is current and committed.** `PROGRESS.md` holds the full record with reasoning;
this file is the short version plus what happens next.

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

## 2. STEP 8 — retrain TinyCNN (next task)

### Four fixes to `src/dl/train.py`, all identified in the step-6d audit

The previous run stopped at a fixed 12 epochs with the validation loss **still falling**, kept
whatever the last epoch produced, and used only half the validation set. None of that is defensible
in a report.

1. **Early stopping** — train until validation loss stops improving (patience ~5), not a fixed count
2. **Save the best epoch**, not the last
3. **Use all 1,600 validation frames** (the previous run used 800)
4. **Revisit `pos_weight=200`** — class imbalance moved **1:1563 → 1:888** when the map halved.
   Per `train.py`'s own note the exact value matters less than intuition suggests, since the ROC
   sweeps the threshold, but it should not silently inherit a value tuned for a different ratio.

### Re-run the Milestone-13 gates at the new geometry

- dataset shapes / finiteness / per-frame median ≈ 0 *(already confirmed in the audit)*
- **orientation check** — no transpose bug
- **receptive field measured 21 × 9** *(already confirmed)*
- **overfit-one-batch** — loss must collapse and peaks land within 1 bin

### Then evaluate

- Per-condition Pd at matched **measured** Pfa on `bench2_test`
- Pd-vs-SNR on `bench2_snr_sweep`, overlaid on theory `Pd = exp(−T/(1+SNR))`

---

## 3. After step 8

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

## 5. Reference

| | |
|---|---|
| Full record + reasoning | `PROGRESS.md` |
| Radar config derivation | `docs/radar_config_carrada.md` |
| DL design record | `docs/dl_design.md` |
| 19-paper corpus review | `docs/literature_corpus_review.md` |
| Scripts | `scripts/measure_psf.py`, `verify_clutter.py`, `cfar_sanity.py`, `cfar_calibrate.py`, `cfar_geometry_sweep.py`, `cfar_geometry_percondition.py`, `verify_splits.py`, `preview_strata.py` |
