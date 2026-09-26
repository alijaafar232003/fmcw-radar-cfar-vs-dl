# Defence presentation asset pack

Every figure here is generated from a **final frozen artefact** in this repository at build
time. No value is typed into a script. Re-run any of the three builders and the numbers come
straight from the same files the thesis reads.

```bash
venv/Scripts/python.exe presentation_assets/build_diagrams.py   # d01-d05, b09, b10
venv/Scripts/python.exe presentation_assets/build_figures.py    # f06-f16  + data/values.json
venv/Scripts/python.exe presentation_assets/build_backup.py     # b01-b08  + data/values_backup.json
```

**Format.** Every asset is written twice: `.png` at 150 dpi (1920 px wide for full-width
figures) and `.svg`. Canvas is 16:9. Shared style in `_style.py`.

**Visual language.**

| Arm | Colour |
|---|---|
| CA-CFAR | `#c0392b` red |
| GO-CFAR | `#d19a13` gold |
| OS-CFAR | `#2e8b57` green |
| SO-CFAR | `#8e44ad` violet |
| TinyCNN, original synthetic training | `#2c6fbb` blue |
| TinyCNN, CARRADA-informed synthetic training | `#7b52ab` purple |
| TinyCNN, real-domain fine-tuned | `#0e5b5b` teal |
| clairvoyant reference | black dashed |

One fixed classical baseline; one learned detector evolving blue → purple → teal through its
three training domains. Same colours as the thesis figures.

---

## Main deck

### d01_fmcw_pipeline — *diagrams/*
**Shows** Transmit chirps → receive echoes → range FFT → Doppler FFT → RD map → detector →
target peaks, with the detector marked "this thesis".
**Source** conceptual; no data.
**Values** none.
**Slide** Problem setting (slide 2–3). Under 20 seconds.

### d02_cfar_window — *diagrams/*
**Shows** The 21 × 9 CFAR footprint: CUT, ±2 guard cells, reference ring, with axis labels.
**Source** `configs/` frozen geometry (`CfarConfig`: guard 2, train 8 range / 2 Doppler). Cell
counts are computed in the script and asserted, not typed.
**Values** 189 cells = 1 CUT + 24 guard + 164 reference. 21 range × 9 Doppler.
**Slide** CFAR explained.

### d03_cfar_failure_modes — *diagrams/*
**Shows** Two conceptual panels: target masking (a strong neighbour raises the threshold) and
clutter edge (the window spans two background levels).
**Source** conceptual; synthetic illustration drawn from a fixed seed. **Not a result.**
**Values** none.
**Slide** Why a learned rule might help.

### d04_cfar_vs_tinycnn — *diagrams/*
**Shows** Side-by-side footprints. CFAR: fixed rule, guard cells **excluded** from the
background estimate. TinyCNN: 21 × 9 receptive field composed layer by layer, **may** use the
whole field. Title says "Comparable spatial extent — different decision rule".
**Source** frozen architecture constants.
**Values** 21 × 9; 4 conv layers; range dilation 1, 2, 3, 4; Doppler dilation 1; 5,041
parameters. The four nested rectangles are the composed receptive field after each layer
(range half-widths 1, 3, 6, 10; Doppler 1, 2, 3, 4).
**Slide** The controlled comparison.

### d05_evaluation_pipeline — *diagrams/*
**Shows** RD frame → detector score map → local peak extraction (NMS, 5 × 5) → score threshold
→ one-to-one matching → metrics, with the note "same post-processing and matcher for both
detector families".
**Source** `configs/final_protocol.yaml` (radius 2).
**Values** NMS radius 2 → 5 × 5 neighbourhood. The obsolete 11 × 11 appears nowhere.
**Slide** Evaluation protocol.

### f06_synthetic_headline — *figures/*
**Shows** ΔPd per condition, established gains in blue, tie/sanity cases greyed inside a shaded
band.
**Source** `experiments/e6_nms_audit/results/final/bootstrap_final_r2_tol1.csv`
**Values** clutter edge +0.2672 · mixed +0.0997 · masking +0.0770 · isolated +0.0491 · multi
+0.0311 | unresolved +0.0080 · clutter +0.0005. Blue/grey split is read from the file's
`sig (Bonf)` column, not hard-coded.
**Slide** Stage 1 headline.

### f07_roc_operating_point — *figures/*
**Shows** Pd against measured per-cell Pfa, log x-axis, all five arms, with the 10⁻⁴ region
shaded and annotated "compared HERE".
**Source** `experiments/e6_nms_audit/results/final/roc_final_r2_tol1.csv`
**Values** pooled Pd at 10⁻⁴: TinyCNN 0.5937 · GO 0.4879 · OS 0.4747 · CA 0.4740 · SO 0.2486.
**Slide** Why "matched measured Pfa" matters. AUROC is deliberately not discussed.

### f08_pd_vs_snr — *figures/*
**Shows** Pd vs cell SNR, two gates, with the full clairvoyant curve and the 1.06 dB margin as
a shaded band between the two Pd = 0.5 crossings.
**Source** `final/pd_vs_snr_final_r2.csv` + `final/pd_vs_snr_final_r2_summary.csv`
**Values** TinyCNN 8.06 dB · CA-CFAR 9.12 dB · margin 1.06 dB. Exact cell: CA-CFAR 11.36 dB,
clairvoyant 10.89 dB, CFAR loss 0.47 dB.
**Slide** The result in decibels.

### f09_localisation — *figures/*
**Shows** Combined RMSE and Doppler error, CA-CFAR vs TinyCNN.
**Source** `final/localization_final_r2.csv`, row `gate=1, group=ALL`
**Values** RMSE 0.544 (CA) vs 0.921 (TinyCNN) = 1.69× worse. Doppler 0.101 vs 0.378.
**Slide** The cost of the synthetic gain.

### f10_attribution_summary — *figures/*
**Shows** Three stacked panels, one per attribution experiment, each with its conclusion in the
panel title.
**Source** `p10_attribution/e2/deltas.csv` (gate 1 mean over 7 conditions),
`p10_attribution/e3/epoch_curve.csv` (gate 1, epoch 21 — the **pooled** row),
`p10_attribution/e4/deltas.csv` (mean over 7 conditions per gate).
**Values** E1 uniform −0.0376, PSF −0.0237 · E2 flat 0.3946, Gaussian 0.1719, point 0.1116 ·
E3 ±1 bin +0.0173, exact cell −0.1597.
**Slide** Stage 2. Headline: "Not simply more energy — learned local context and supervision
matter." No unique causal mechanism is claimed.

### f11_domain_gap — *figures/*
**Shows** Synthetic vs CARRADA across seven measured properties, ending "CARRADA is different,
not simply harder."
**Source** `r_carrada_real_data_validation/results/r0_audit.json` and
`r0_object_footprint.csv`
**Values** Gamma k 0.990 / 4.771 · dB spread 5.573 / 2.211 · lag-1 range 0.573 / 0.563 · lag-1
Doppler 0.574 / 0.226 · ridge 14.8 dB ±4 bins / 24.0 dB ±1 bin · point target / extended
object · median support 1 × 1 / 9×9, 11×11, 18×13.
**Slide** Why transfer failed.

### f12_zeroshot_all_detectors — *figures/*
**Shows** Recall-vs-burden scatter, every arm labelled with both numbers.
**Source** `e6_nms_audit/results/p4_carrada/r1_test/r1_summary.json`
**Values** TinyCNN 0.8721 / 23.48 · CA 0.8838 / 12.28 · GO 0.9219 / 30.29 · SO 0.9687 / 30.36 ·
OS 0.9662 / 31.87.
**Slide** Stage 3, the honest picture. "TinyCNN does not dominate the classical family."

### f13_zeroshot_primary — *figures/*
**Shows** TinyCNN vs CA on recall and on burden, with the CI in a callout.
**Source** same `r1_summary.json`
**Values** 0.8721 vs 0.8838; sequence-balanced −0.0136, 95 % CI [−0.0269, +0.0045];
23.48 vs 12.28 (1.9×). Never called false alarms.
**Slide** Stage 3 headline.

### f14_domain_alignment — *figures/*
**Shows** Recall recovers (left), output density explodes (right, log axis).
**Source** `carrada_informed_domain/results/test_analysis_r5.json` — **see the note below on
r5 vs r2**.
**Values** original 0.8721 / 23.48 · CARRADA-informed 0.9449 / 109.28 · CA-CFAR 0.8838 /
12.28 · B − A = +0.0629, CI [+0.0353, +0.0879]. Near range 0.8455 → 0.9468 vs CA 0.9475.
**Slide** Stage 4.

### f15_real_adaptation — *figures/*
**Shows** Recall-vs-burden scatter on CARRADA **validation**, with arrows original →
domain-informed → fine-tuned and CA-CFAR as a fixed crosshair baseline.
**Source** `carrada_real_finetune/results/validation_analysis.json`
**Values** A 0.9033 / 22.50 · B 0.9577 / 109.59 · C 0.9611 / 11.17 · CA 0.8725 / 11.64.
Footnote carries C − B = +0.0022, CI [−0.0090, +0.0138], and states that the operating point
was **also** selected on held-out CARRADA TRAIN sequences under a burden constraint.
**Slide** Stage 5 — the strongest visual in the deck.

### f16_five_stage_story — *figures/*
**Shows** The five stages, one line of outcome each, then the two-line conclusion.
**Source** narrative; no data.
**Values** none.
**Slide** Closing slide.

---

## Simulated scene — a deliberately difficult frame

Built by its own script, separate from the three builders above. It imports the frozen
simulator and runs it once:

```bash
venv/Scripts/python.exe presentation_assets/generate_challenging_rd_map.py
```

| Asset | Shows |
|---|---|
| `figures/s01_challenging_rd_map` | the frame, axes and colourbar only — the slide version |
| `figures/s02_challenging_rd_map_annotated` | the same frame with ground-truth centres, roles and cell SNRs |
| `figures/s03_challenging_rd_map_truth_mask` | the 3x3 label mask the detectors are scored against |
| `data/s01_challenging_rd_map.npz` | `rd_db_above_noise` (plotted), `rd_db_rel_max`, `rd_mag`, `label_mask`, both axes |
| `data/s01_challenging_rd_map.json` | scene, every target, the ghost, the validation checks, the seed |

**What it contains.** Four true targets and one multipath ghost, on a narrow zero-Doppler
ridge. It reads as three obvious returns; it is not.

| id | role | range | velocity | bins | cell SNR |
|---|---|---:|---:|---:|---:|
| `pair_a` | unresolved pair | 34.00 m | −5.60 m/s | (174, 19) | 20.52 dB |
| `pair_b` | unresolved pair | 34.28 m | −5.42 m/s | (176, 19) | 20.52 dB |
| `multipath` | multipath primary | 13.00 m | +6.40 m/s | (67, 47) | 24.43 dB |
| `weak` | weak / masked | 10.50 m | −0.45 m/s | (54, 31) | 17.33 dB |
| `multipath_ghost` | **not a target** | 15.00 m | +6.40 m/s | (77, 47) | 16.33 dB |

Seed 20260915. Reference cell SNR 6 dB. Ridge CNR 16 dB, `width_bins` 1, Weibull shape 1.6.
Multipath delay 2.0 m, attenuation −8 dB. Radar equation on, referenced to 25 m.

**Everything is produced by the simulator, and the script asserts it:** the two-call
multipath split reproduces a single call exactly (max abs error 0.0); the clutter/noise
path reproduces `simulate_cpi` bit-for-bit (0.0); switching multipath off drops the ghost
cell by 14.3 dB; switching clutter off drops the ridge by 14.3 dB; the pair is 0.28 m
apart, below the measured 0.39 m equal-strength limit, and leaves **one** local maximum in
the range cut; the weak target is in the label mask and sits 3.7 dB over its local ridge
background.

The ridge runs along the **velocity = 0 column**, i.e. a vertical band spanning the range
axis, because the thesis plots range on y and velocity on x.

---

## Domain gap — frames, not statistics

```bash
venv/Scripts/python.exe presentation_assets/build_domain_frames.py
```

| Asset | Shows |
|---|---|
| `figures/s04_domain_frames` | 2 rows x 6 frames: synthetic (top) vs real CARRADA (bottom) |
| `figures/s05_domain_frames_three_rows` | the same, with the CARRADA-informed synthetic domain as a middle row |
| `data/s04_domain_frames.json` | display convention, the frame-selection rule, and every frame id |

No markers, no boxes, no target annotation — background texture only. One display
convention throughout: dB above **that frame's own** median, magma, −5 to +35 dB,
identical 256 x 64 axes. Selection is a fixed rule, not a choice: the first frame id in
sorted order for each of isolated / multi / masking / mixed / clutter / clutter_edge, and
the middle annotated frame of the first sequences of each CARRADA recording day.

What the eye should catch, and the numbers behind it
(`r_carrada_real_data_validation/results/r0_audit.json`):

| | synthetic | CARRADA |
|---|---:|---:|
| background dB spread | 5.573 dB | 2.211 dB |
| Gamma shape k | 0.990 | 4.771 |
| zero-Doppler ridge | 14.8 dB, ±4 bins | 24.0 dB, ±1 bin |

Coarse single-look speckle and a wide, blocky ridge on one side; a smooth background and a
razor-thin, much brighter ridge on the other. `s05` then shows what the Stage 4
intervention actually changed.

**Slide** Stage 3 → 4, next to `f11_domain_gap`. This is the picture; `f11` is the
measurement.

---

## Attribution experiments — design diagrams

```bash
venv/Scripts/python.exe presentation_assets/build_attribution_diagrams.py
```

No results on any of these; they explain the design only, so each can go up before its
result slide.

### d06_label_ablation — *diagrams/*
**Shows** The three supervision targets on a 3x3 footprint: point, flat 3x3 (the incumbent),
Gaussian from the measured PSF. Truth drawn off-grid, because it is.
**Source** `experiments/label_ablation/DERIVED_VALUES.md`
**Values** sigma_range 0.6895, sigma_Doppler 0.7007 bins, derived from `results/psf_profile.csv`
(off-bin worst case, -3 dB width / 2.3548). Gaussian weights 1.00 / 0.361 / 0.349 / 0.126.
pos_weight 899 / 100 / 308, scaled so pos_weight x sum(y) is identical across arms.
**Slide** Stage 2, before the label result.

### d07_masked_cnn — *diagrams/*
**Shows** The three arms: frozen dense TinyCNN (reference, not retrained), unmasked 21x9 patch,
guard-masked patch with the 24 guard cells blanked and the CUT still visible.
**Source** `experiments/e4_cut_guard_mask/PREREGISTRATION.md`
**Values** 21x9 receptive field, guard +/-2 range and +/-2 Doppler = 24 cells, masked cells set
to 0.0 which `normalize_rd` maps to the frame's own median. No mask channel is added.
**Slide** Stage 2. The subtitle carries the reason the contrast is patch-vs-patch rather than
dense-vs-dense, which is the first question anyone asks.

### d08_integrated_cfar — *diagrams/*
**Shows** The same 21x9 CFAR window with the numerator widened from one cell to 3x3, plus both
kernels. Everything else — ring, guard, matcher, protocol — unchanged.
**Source** `experiments/integrated_cfar/results/kernels.npz`, read at build time
**Values** uniform 0.111 everywhere; PSF-weighted 0.288 centre, 0.126 / 0.122 edges,
0.054 corners. 3x3 fits inside the guard, so numerator and denominator stay disjoint.
**Slide** Stage 2, before the integration result.

---

## Backup — for Q&A only

| Asset | Shows | Source | Key values |
|---|---|---|---|
| `b01_cfar_geometry_sweep` | Why 21 × 9 | `results/cfar_geometry_percondition_v2.csv` | mean Pd per candidate window; 21 × 9 selected on `bench2_tune` |
| `b02_nms_radius_sweep` | Why radius 2 | `e6_nms_audit/results/p1_selection/nms_selection_sweep_val.csv` | duplicates vanish at r ≥ 2; masking Pd falls as r grows |
| `b03_tolerance_pm1_pm3` | ±1 vs ±3 gate | `final/bootstrap_final_r2_tol1.csv`, `..._tol3.csv` | only clutter changes sign |
| `b04_carrada_range_blocks` | Zero-shot recall by range | `r1_summary.json` → `by_range` | near-range deficit −0.1020 (0.8455 vs 0.9475) |
| `b05_mask_size_by_class` | Why class recall is confounded | `r0_object_footprint.csv` | 9×9 / 11×11 / 18×13; 74 / 93 / 147 dense cells |
| `b06_per_seed_carrada` | Per-seed zero-shot recall | `r1_summary.json` → `primary.tinycnn_per_seed` | 0.8457–0.8980, sd 0.0166, 2 of 8 above CA |
| `b07_burden_matched` | Recall at equal output | `carrada_informed_domain/results/burden_matched.json` | CA leads both learned arms at every matched level |
| `b08_stage5_burden_curve` | Recall vs burden, validation | `validation_analysis.json` → `burden_curve` | at CA's 11.64/frame: fine-tuned 0.9613, informed 0.8637, CA 0.8725 |
| `b09_dense_vs_box_truth` | Point truth vs extended support | conceptual + `r0_object_footprint.csv` | why the synthetic metric cannot transfer |
| `b10_dataset_splits` | Every split and its role | `configs/`, `numbers.tex` counts | holdout 3,200 read once; CARRADA 8,088 / 2,448 / 2,130 |

---

## data/

* `values.json` — every number that reaches a main-deck figure, with the artefact it came from.
* `values_backup.json` — the same for the backup figures.
* `roc_final_r2_tol1.csv` lives in `experiments/e6_nms_audit/results/final/` (results are
  gitignored); regenerate with `scripts/export_roc_final.py`.

---

## Provenance notes you should be able to answer in the viva

**1. Stage 4 reads `test_analysis_r5.json`, not `test_analysis_r2.json`.** This looks alarming
and is not. `r5` names the **threshold set**, not the suppression radius. Both files are
evaluated at radius 2; they differ only in whether the *thresholds* were derived under the
default-argument defect described in the thesis appendix. The thesis reports the `r5` figures as
its headline (0.9449 / 109.28 / +0.0629) and the `r2` figures as the corrected sensitivity check
(0.9445 / 107.87 / +0.0625). The pack follows the thesis. Same sign, same conclusion, same
significance either way.

**2. E3's Doppler errors are the pooled row, not the mean of per-condition rows.**
`e3/epoch_curve.csv` at gate 1, epoch 21 gives 0.3946 / 0.1719 / 0.1116 — the values the thesis
quotes. Averaging `e3/per_condition.csv` instead gives 0.4291 / 0.1778 / 0.1075, which is a
different (unweighted) statistic and does **not** match the thesis.

**3. The ROC in f07 is pooled with one global threshold**, while the headline in f06 sweeps each
condition separately. The two need not agree exactly; at 10⁻⁴ they come out close (0.5937 vs a
0.580 unweighted per-condition mean), which is reassuring but not a cross-check.

**4. TinyCNN's ROC curve stops near 8 × 10⁻⁴** because the shared extractor keeps only positive
logits. Documented in the thesis, stated in the figure footnote.

**5. Unmatched detections are never called false alarms.** CARRADA annotates only the tracked
road users, so no false-alarm rate exists there. Every real-data axis label says "output burden
— not Pfa".

## Deliberately not used

`PROGRESS.md`; `results/dl/` radius-5 tables; `results/cfar_test*`; `results/cfar_test_backup_11aug`;
the superseded `bench2_test` split; `report/figures/_superseded_nms5/`;
`carrada_informed_domain/results/*_r5.csv` pilot sweeps; every notebook output. Where a notebook
had a better visual idea (the Pd-vs-SNR panel with the theory curve, §6 of
`cfar_vs_dl_comparison.ipynb`), the **design** was reused and the data regenerated from the final
artefacts.
