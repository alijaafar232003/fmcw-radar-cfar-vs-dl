# CARRADA Real-Data External Validation — **COMPLETE**

*Does the frozen synthetic DL-vs-CFAR result transfer zero-shot to real automotive radar?*

> ## Headline
>
> **It reverses.** In the matched synthetic domain TinyCNN beats CFAR on 6 of 7 conditions. On real
> CARRADA data, frozen **CA-CFAR 0.844** vs frozen **TinyCNN 0.712** exact dense-mask object recall
> — paired Δ **−0.109**, 95% CI **[−0.166, −0.062]** over the 6 test sequences, **CFAR ahead in all
> six**, and no individual seed closes the gap.
>
> The gap concentrates near CARRADA's strong zero-Doppler ridge (0.669 vs 0.823, n=1,365). **Away
> from it the two are close** (0.936 vs 0.950, n=262).

⚠️ **Synthetic $P_d$ and CARRADA mask recall are different measurements** — point truth with a ±1-bin
gate versus extended segmentation support. They are never pooled. The comparable statement is the
**ranking**, not the numbers.

---

## Scientific question

Every prior number in this project came from one simulator, so the headline established *sampling*
generalization only. This experiment asks whether the frozen detectors transfer to real measurements.
It **does not** replace the synthetic benchmark; it **bounds its external validity**.

## Phase structure

| phase | content | commit |
|---|---|---|
| 1 | isolation, dataset integrity | `15f3aab` |
| 2 | R0 synthetic-vs-real domain audit | `a3a3b88` |
| 3 | frozen R1 pre-registration | `9c9a3e2` |
| 4 | implementation, adapters, thresholds, sanity checks | `30936d6` |
| 5 | one-shot frozen CARRADA Test evaluation | `9bfa805` |
| 6 | documentation, cross-domain figures, closeout | *this commit* |

## Dataset

CARRADA at `C:/Users/USER/Desktop/CARRADA/Carrada`, **read-only**, reduced 95.86 GB → 3.52 GB before
the experiment. 30 sequences · 12,666 frames · 7,193 annotated · **8,750 frame-level object
instances** · **77 unique tracks**. Official **sequence-level** split used unchanged (18/6/6); never
split by frame. Metadata backup in `data/carrada_meta/`.

Primary representation `range_doppler_numpy` = power in dB up to a constant.
CFAR input `P ∝ 10^(D/10)`; TinyCNN adapter `D − median(D)`, realised by the **frozen**
`normalize_rd` fed `mag = 10^(D/20)`.

## Frozen detector provenance

* **TinyCNN** — `results/dl/tinycnn_frozen_s{0..7}_ep021.pt`, 8 seeds, evaluated independently,
  never ensembled, never retrained or fine-tuned.
* **CFAR** — 21×9 window (guard 2/2, train 8/2), frozen calibrated alphas
  CA 10.4586 · GO 9.5458 · SO 12.8494 · OS 7.5786. **CA is the primary comparator**, fixed in
  advance; GO/SO/OS secondary and never promoted after results.
* **NMS** — `all_peaks`, radius 5, shared by both arms, not retuned.
* **TinyCNN thresholds** — the frozen project had none, so one per seed was derived from
  **`bench2_val` alone** at pooled measured per-cell $P_{fa}$ ≤ 1e-4. A runtime file-access audit
  confirmed only `bench2_val` was opened.

## Pre-registration

`PREREGISTRATION.md` — **FINAL PRE-SCORING FREEZE**
SHA-256 `e800ad84af5a909d595a9c1a5b32bfe308efc4830a1bfe17d8ea793a1bba98a8`
Frozen 2026-08-31T17:11:45Z, before any CARRADA detector inference existed.

## R0 conclusion

Geometry and representation transfer exactly; the background does not.

| statistic | CARRADA | simulator |
|---|---|---|
| Gamma shape k | 4.771 ± 0.223 | **0.990 ± 0.006** (single-look, as designed) |
| background std [dB] | 2.212 | 5.573 (theory 5.57) |
| lag-1 Doppler corr | 0.227 | 0.574 |
| zero-Doppler ridge | 24.0 dB, ±1 bin | 14.8 dB, ±4 bins |

CARRADA is **different**, not simply easier or harder. Its background is smoother in the sense of
lower cell-to-cell fluctuation — *not* less physical clutter — while carrying stronger localized
structure. 84% of eligible test objects sit inside the ±4 zero-Doppler band.

## Important caveats

1. **Unmatched detections/frame is NOT $P_{fa}$** (TinyCNN 13.14 vs CA 8.73). CARRADA annotates only
   its 77 tracked objects; R0 found bright unannotated returns in 5.85% of frames; and the frozen
   point-target NMS leaves ≥2 peaks inside one object's support for 35–48% of objects.
2. **No localization metric is reported** — CARRADA supplies no point truth.
3. **±1 / ±3 is mask-support tolerance, not localization accuracy.**
4. **Class recall is confounded with mask size** (car 18×13, cyclist 11×11, pedestrian 9×9).
5. **n = 6 test sequences.** Cyclist (n=159, 2 sequences) is exploratory.
6. **No causal claim** that the ridge caused the loss — stratification locates it only.

## Layout

```
r_carrada_real_data_validation.ipynb   the full record, 85 cells, §0–§71, executes clean
PREREGISTRATION.md                     frozen protocol
isolation_manifest.json                628 protected files, SHA-256
configs/   official_splits.json · tinycnn_zero_shot_thresholds.json
src/       carrada_input · eligibility · matching · metrics · detectors · stats · schema
scripts/   run_r1_test · build_tables_from_raw · analyze_r1_test · make_*_figure
           derive_tinycnn_thresholds · validate_adapters · run_sanity_checks
           dry_run_r1 · leakage_audit · build_isolation_manifest · build_notebook
results/r1_test/   *_peaks.npz (immutable) · test_output_manifest.json · run_manifest.json
                   access_log.json · per_object.csv · per_frame.csv · r1_summary.json
figures/           r0_fig1…fig6 (domain audit) · r1_test/ (results) · phase4_sanity/
```

## Key figures

| figure | file |
|---|---|
| **cross-domain 4-panel — the headline** | `figures/r1_test/r1_fig5_crossdomain_4panel.png` |
| RD examples, synthetic vs real | `figures/r1_test/r1_fig6_rd_examples.png` |
| primary result + per sequence | `figures/r1_test/r1_fig1_primary_and_per_sequence.png` |
| strata (class / Doppler / range) | `figures/r1_test/r1_fig2_strata.png` |
| tolerance, burden, seeds | `figures/r1_test/r1_fig3_tolerance_burden_seeds.png` |
| multi-peak + recall–burden curve | `figures/r1_test/r1_fig4_multipeak_and_burden_curve.png` |
| R0 simulator vs CARRADA (boxes) | `figures/r0_fig1_rd_examples.png` |
| R0 simulator vs CARRADA (dense overlay) | `figures/r0_fig1b_rd_examples_dense_overlay.png` |

## Reproduction — no detector re-run needed

```bash
python scripts/analyze_r1_test.py
python scripts/make_r1_figures.py
python scripts/make_crossdomain_figure.py
python scripts/make_rd_examples_figure.py
python scripts/build_isolation_manifest.py --verify isolation_manifest.json
```

Re-scoring from scratch (~10 min) is `python scripts/run_r1_test.py` followed by
`build_tables_from_raw.py`. Environment: Python 3.11.9 · numpy 2.4.6 · pandas 3.0.3 ·
torch 2.13.0+cpu · scipy 1.17.1 · matplotlib 3.11.0.

---

## Status

**COMPLETE.** No frozen TinyCNN, CFAR, synthetic result, E2, E3 or E4 asset was modified at any
point.

Any **CARRADA training, fine-tuning, simulator-realism change, domain randomization, CFAR
recalibration, or NMS adaptation is a NEW experiment**, with its own pre-registration and isolation
manifest. None of it may be attached to this one.
