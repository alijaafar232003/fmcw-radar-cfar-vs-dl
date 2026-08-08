# Benchmarking Conventions (SNR definition + Scoring Rulebook)

Reference notes for the FMCW CFAR-vs-DL project. These define **how difficulty is set**
and **how detectors are scored**, so CFAR and deep learning are compared fairly.
Written to be copied into the FYP report / notebook later.

---

## Part 1 — SNR definition  ✅ IMPLEMENTED

> **Status: done.** Implemented in `src/radar/calibration.py`; per-target `cell_snr_db` is
> stored in every `labels.csv`. `scene.snr_db` in a config means *requested cell SNR for a 0 dB
> reference target*; a target of amplitude `a_db` lands near `snr_db + a_db`, and its measured
> value is recorded per target. The rest of this Part documents the reasoning.
>
> ⚠️ **Processing gain must be re-measured (4 Aug 2026).** The 43.38 dB figure below was measured
> on the previous radar configuration (B = 300 MHz, Tc = 32 µs, M = 128). The sensor is now
> matched to CARRADA (M = 64), which drops the coherent gain by 3.01 dB —
> `10log10(512×128) = 48.16 dB` → `10log10(512×64) = 45.15 dB` — so the post-window figure should
> land near **40.4 dB**. `calibration.py` derives the gain analytically from the windows, so the
> cell-SNR calibration self-corrects; but **do not carry the old number into the report.**
> See `docs/radar_config_carrada.md`.
>
> *Superseded on the old configuration:* measured processing gain 43.38 dB (on-bin error 0.02 dB,
> noise model within 1.1%); acceptance check passed with Pd rising 0.00 → 1.00 across the sweep,
> transition ~9–12 dB. Both need re-running.

### The problem this solved

### What SNR means here
SNR (signal-to-noise ratio) is how strong a target is compared to background noise.
We use it as the **difficulty dial**: low SNR = hard, high SNR = easy. We want the dataset
to span easy → hard so the Pd-vs-SNR curve actually *moves*.

### The bug in the current definition
The simulator currently sets SNR on the **raw time samples** before any FFT
(`signal_power = mean(|dechirped CPI|^2)`, then noise scaled to hit `snr_db`).

The problem: the two FFTs (range + Doppler) perform **coherent integration**, which boosts
a target's peak far above the noise. The gain is roughly:

```
processing gain  ≈  10·log10(N · M) − window loss
              ≈  10·log10(512 · 128) − ~2.4 dB (Blackman)
              ≈  48.2 − 2.4  ≈  43 dB
```

So a frame labelled **−10 dB** (per-sample) actually has a peak in the RD map at about
**−10 + 43 = +33 dB** cell SNR — a giant, trivially detectable dot.

**Consequence:** the current sweep (−10 .. +20 dB per-sample) sits entirely in the
"always detectable" zone. Both CFAR and DL will score Pd ≈ 1 everywhere → flat ROC curves →
the comparison proves nothing. The detection *transition* (Pd going 0 → 1) happens around
**per-sample SNR ≈ −37 .. −27 dB**, which the current sweep never reaches.

### The fix (recommended): report SNR at the detector, not at the ADC
Define and report **cell SNR** = signal power in the target's RD peak cell ÷ average noise
power per RD cell. This is the physically meaningful quantity, and it's exactly what CFAR
detection theory (Pd/Pfa formulas) is written in terms of. Then sweep cell SNR over the
range where Pd actually transitions (about **0 .. 18 dB**).

### Calibration recipe (empirical — no fragile hand-math)
For each frame (or for the SNR-sweep set, per single target):

1. Generate the **clean** CPI (targets, no noise) → process to RD map → take power
   `|RD|^2`. Record the target's **peak-cell power** `P_sig`.
2. Generate a **noise-only** CPI with unit input noise (σ² = 1) → process identically →
   measure the **mean per-cell power** `P_noise_unit`. (This captures the FFT + window
   noise gain once and for all.)
3. To hit a desired cell SNR `S` (linear, e.g. S = 10^(S_dB/10)):

   ```
   required input noise variance  σ²  =  P_sig / (S · P_noise_unit)
   ```

   Add complex white noise with that σ².
4. Store the **defined cell SNR** `S_dB` in the frame/label metadata — this becomes the
   x-axis of every Pd-vs-SNR plot. Keep the old per-sample `snr_db` too for traceability.

### Also store per-target cell SNR
For multi-target frames, a single "frame SNR" is meaningless (a strong target hides a weak
one). So compute and store **each target's own cell SNR** in `labels.csv`. Benefit: you can
plot Pd vs SNR using *every* target across *all* datasets, not just the single-target sweep.

### Minimal alternative (if short on time)
Keep the per-sample SNR definition but **re-pick the sweep range empirically**: run a trivial
peak-threshold detector over candidate ranges and choose the span where Pd climbs from ~0 to
~1 (likely around −45 .. −20 dB per-sample). Less interpretable than cell SNR, but unblocks
the project. The cell-SNR approach above is preferred for the report.

### Acceptance check (do this before CFAR)
Generate the SNR-sweep set with the new definition, run a simple peak-vs-threshold detector,
and confirm the Pd-vs-SNR curve **rises from near 0 to near 1**. If it's flat, the dial is
still wrong.

---

## Part 2 — The Scoring Rulebook (shared by CFAR and DL)

One rulebook, applied **identically** to both detectors. Both methods must first be reduced
to the same output type: **a list of detected points** in `(range_bin, velocity_bin)` per frame
(CFAR: threshold → local maxima; DL: heatmap → peak-find + non-maximum suppression).

### Ground truth
The labelled target points `(range_bin, velocity_bin)` from `labels.csv`.

### Matching rule (association)
A detection matches a truth point if it lies inside a **tolerance gate**:

```
|Δrange_bin| ≤ G_r   AND   |Δvelocity_bin| ≤ G_v
```

Recommended start: `G_r = 1` (≈ one range resolution cell), `G_v = 1`. Justify by the
window mainlobe width. Do **one-to-one** matching (greedy nearest, or Hungarian) so one
detection can't claim two truths.

- Detection matched to a truth → **True Positive (TP)**
- Truth with no match → **False Negative (FN)** = missed detection
- Detection matched to nothing → **False Positive (FP)** = false alarm

### Guard region (avoid double-counting)
Cells within the gate of any truth are **excluded from the false-alarm denominator**: a
detection 1 bin off a real target is scored as the TP (or ignored), never as an FP.

### Metrics
```
Pd  = TP / (TP + FN)                         # fraction of true targets found
Pfa(per-cell)   = (false-alarm points) / (number of noise cells tested)
FA-per-frame    = mean number of FP per RD map     # human-friendly secondary metric
Localization error = distance(detection, matched truth), over TPs only
                     reported in (range bins, velocity bins) and meters & m/s (mean/median/RMSE)
```

**Pfa convention:** use **per-cell Pfa** as the primary x-axis (matches CFAR theory and the
proposal's "Pd at fixed Pfa"), and also report **FA-per-frame** for intuition. Pick the
definitions once and use the SAME ones for CFAR and DL.

### ROC / threshold sweep
- CFAR: sweep design Pfa (or the α multiplier) → each setting yields a measured (Pfa, Pd) point.
- DL: sweep the heatmap detection threshold → (Pfa, Pd) points.
- Plot CA-CFAR, OS-CFAR, DL on the **same axes**. Report **Pd at a fixed Pfa** (e.g. 1e-3).

### Aggregation for curves
- **ROC:** aggregate TP/FP/FN over the whole frozen test set (or per scenario).
- **Pd vs SNR:** bin targets by cell SNR, compute Pd in each bin at a **fixed operating
  threshold** (fixed Pfa).

### Edge cases
- **Empty (0-target) frames:** contribute only FP / Pfa (no TP/FN possible). This is the
  job of `noise_only_debug`.
- **Two truths inside one resolution cell** (very close pairs): physically unresolvable, so
  Pd is capped at 0.5 no matter the detector. Either (a) keep both truths and discuss the
  resolution limit, or (b) for clean scoring, keep the minimum separation in
  `close_targets` ≥ the gate size and run "unresolvable" as a separate, clearly-labelled study.
- **Multipath ghosts are intentionally unlabelled** → they correctly count as false alarms
  for both detectors. Document this so it isn't mistaken for a labelling bug.

### Fairness note for DL
Train the DL detector on a mix that **includes** clutter / multipath / close targets, so it
faces the same conditions as CFAR at test time. Otherwise poor DL clutter performance would
just be distribution shift, not a real CFAR-vs-DL difference.

---

## Part 3 — Scenario set review (the 7 configs)

| Config | Purpose | Verdict |
|---|---|---|
| `clean_debug` | clean, high SNR, separated targets — sanity baseline | keep |
| `noise_only_debug` | 0 targets — false-alarm / Pfa testing | keep (essential) |
| `snr_sweep_debug` | single target, Pd-vs-SNR | keep — **fix SNR calibration**; needs many frames per SNR point in the real (Phase C) set |
| `clutter_debug` | near-zero-Doppler clutter — CFAR's weakness | keep |
| `close_targets_debug` | close pairs — resolution / masking | keep |
| `multi_target_debug` | 0–5 random targets — general / DL training | keep |
| `multipath_debug` | ghost reflections — extra false-alarm source | keep |

**Approval:** all 7 are well chosen, each maps to a proposal ablation, none are redundant —
**keep all of them.**

**Optional additions (later, not now):**
- A **combined "hard" scenario** (clutter + multipath + multiple targets + low SNR) — the
  realistic worst case where DL should most clearly beat CFAR. This gives the headline result.
- A **clutter-strength sweep** (cycle `clutter.strength_db`) to directly support the
  proposal's "clutter strength" ablation.
- RD-resolution ablation needs radar params in YAML (currently hardcoded) — defer until needed.

**Note on sizes:** these are intentionally small *debug* sets (10–28 frames). The Phase C
train/val/test sets must be much larger (hundreds–thousands), with the **test set frozen**
(fixed seeds) and shared by CFAR and DL.
