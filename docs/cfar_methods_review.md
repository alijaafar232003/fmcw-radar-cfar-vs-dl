# CFAR Detection Methods — Reference Review

Background review of the main CFAR techniques for the FMCW CFAR-vs-DL project: mechanism,
pros, cons, and where each succeeds and fails. Citations are real and report-ready.

---

## The shared CFAR framework (all methods share this)

Slide a reference window across the RD map. For the **Cell Under Test (CUT)**:
- **Guard cells** around it are skipped (keep the target's own spread out of the noise estimate).
- **Reference / training cells** further out — often split into a **leading** and **lagging**
  half — estimate the local noise/clutter level `Z`.
- Threshold `T = α · Z`, with `α` set to hit a desired `Pfa`.
  For CA-CFAR with `N` reference cells (square-law detector): `α = N·(Pfa^(−1/N) − 1)`.
- Decision: **CUT > T → detection.**

**Every variant differs only in how the reference cells become `Z`.** That single choice
determines its strengths and weaknesses.

---

## 1. CA-CFAR — Cell-Averaging (Finn & Johnson, RCA Review, 1968)
- **Mechanism:** `Z =` mean of all reference cells (leading + lagging averaged).
- **Pros:** simplest, cheapest; **optimal (max Pd)** in homogeneous background; lowest CFAR loss.
- **Cons:** fails at **clutter edges** (window straddles a boundary → false-alarm burst or missed
  targets); **target masking** — a second target in the window inflates `Z`, raises the
  threshold, and hides the real target.
- **Succeeds:** uniform noise/clutter. **Fails:** clutter edges, multiple/close targets.

## 2. GO-CFAR — Greatest-Of (Hansen & Sawyers, IEEE T-AES, 1980)
- **Mechanism:** `Z = max(mean_leading, mean_lagging)`.
- **Pros:** controls **false alarms at clutter edges** (keeps threshold high on the clutter side).
- **Cons:** **worse than CA at multiple targets** — an interferer in one half raises the
  threshold further → more masking. Slightly higher homogeneous loss.
- **Succeeds:** clutter edges (best classical). **Fails:** multi-target masking (worst).

## 3. SO-CFAR — Smallest-Of (Trunk, IEEE T-AES, 1978)
- **Mechanism:** `Z = min(mean_leading, mean_lagging)`.
- **Pros:** **resolves closely-spaced targets** — picks the clean half when one half is
  contaminated by an interferer.
- **Cons:** **disastrous at clutter edges** (threshold collapses → false-alarm explosion);
  fails if interferers are in both halves; higher homogeneous loss.
- **Succeeds:** multi-target when interferers are one-sided. **Fails:** clutter edges.
- *(GO and SO are mirror images — each fixes what the other breaks.)*

## 4. OS-CFAR — Ordered-Statistic (Rohling, IEEE T-AES, 1983) — *in the proposal's references*
- **Mechanism:** sort the `N` reference cells ascending; take the **k-th ranked value**
  `Z = x(k)` (typically `k ≈ 0.75N`). Interfering targets are high outliers excluded by the rank.
- **Pros:** **excellent in multiple / closely-spaced target** scenes; tolerates up to `N−k`
  interferers without masking; robust to outliers.
- **Cons:** heavier (sort per cell); ~0.5 dB extra loss vs CA in homogeneous; choice of `k` is a
  trade-off; weaker than GO at clutter edges; masks if interferers exceed `N−k`.
- **Succeeds:** multi/close-target. **Fails/limits:** clutter edges, sort cost.
- *Recommended second method — should beat CA exactly where CA fails.*

## 5. VI-CFAR — Variability Index (Smith & Varshney, IEEE T-AES, 2000) — advanced *classical*
- **Mechanism:** compute a **Variability Index (VI)** (≈ variance/mean²) and a **Mean Ratio (MR)**
  of leading vs lagging halves; hypothesis tests classify the window as *homogeneous*,
  *multi-target*, or *clutter-edge*, then switch estimator: CA / SO-like / GO-like accordingly.
- **Pros:** **robust across all three regimes** without being told which; low loss when homogeneous.
- **Cons:** more complex; VI/MR thresholds need tuning; still model-based — degrades under
  heavy-tailed (K/Weibull) clutter far from the assumed model.
- **Succeeds:** mixed/unknown environments (best classical all-rounder). **Fails:** strongly
  non-Gaussian clutter; adds tuning.

## 6. Deep-learning CFAR (modern frontier — the project's premise)
- **Representative work:**
  - *Deep learning-based lightweight radar target detection method*, J. Real-Time Image
    Processing (Springer), 2023 — labeled RD dataset from real ATC radar; CNN uses both axes;
    outperforms 1-D/2-D CFAR, especially at low SNR.
  - *(VI)CFAR-CNN cascade + dual-detection-map fusion*, Remote Sensing (MDPI), 2024 — hybrid:
    CFAR proposes candidates, CNN refines; keeps CFAR's Pfa control plus CNN robustness.
- **Mechanism:** a CNN learns the 2-D appearance of targets vs clutter/noise from labeled data
  (input RD map / RAD cube → detection heatmap), instead of a 1-D sliding statistic.
- **Pros:** learns complex non-homogeneous clutter with no analytic model; uses 2-D spatial
  context; strong at low SNR and dense multi-target; can jointly detect + classify.
- **Cons:** needs large labeled data; **no analytically guaranteed constant Pfa** (the CFAR
  property is lost — measure it empirically); domain-shift/generalization risk; heavier,
  black-box.
- **Succeeds:** complex clutter, low SNR, multi-target. **Fails/risks:** unseen distributions,
  data hunger, no guaranteed Pfa. *This is the exact tension the thesis quantifies.*

---

## Threshold calibration (CA analytic; OS/GO Monte-Carlo)

The detection threshold is `T = alpha * Z` (Z = local noise estimate); `alpha` sets the design
false-alarm rate `Pfa`.

**CA has a closed form.** Its estimate is the arithmetic mean of `N` exponential (square-law)
cells, so the CUT/estimate ratio has a known distribution and

    alpha_CA = N * (Pfa^(-1/N) - 1).

**OS and GO do not.** OS uses an order statistic (the k-th ranked cell) and GO the maximum of
two half-window means — different statistics with different distributions. Reusing `alpha_CA`
for them gives the wrong Pfa (with guard 2 / train 8 it left OS and GO far too conservative,
≈ 0 false alarms on pure noise, because their estimates sit above the mean: OS ≈ 1.39× = the
0.75 quantile of exponential noise, GO ≈ 1.09×).

**Empirical (Monte-Carlo) calibration** (`src/cfar/calibrate.py`) — works for any statistic:
1. Process many pure-noise CPIs through the identical RD chain → noise power maps.
2. Run the CFAR ratio map (CUT ÷ local estimate) on each; pool the ratio over valid cells.
3. `alpha(Pfa)` = the **(1 − Pfa) quantile** of that pooled ratio distribution.

*Validation:* the empirical CA threshold matches the analytic formula to ~3–5% (the small
positive bias is the window's mild inter-cell correlation, which the empirical method captures
and the idealised formula does not). *Verification:* on independent noise, all three then hit
the target Pfa (e.g. at Pfa = 1e-4: CA 9.4e-5, OS 9.8e-5, GO 9.8e-5).

Calibrated multipliers (guard 2, train 8, N_ref = 416):

| Pfa | CA | GO | OS |
|---|---|---|---|
| 1e-3 | 7.17 | 6.69 | 5.17 |
| 1e-4 | 9.74 | 9.09 | 6.98 |

OS has the lowest multiplier because its noise estimate is the highest.

## Window-geometry tuning (guard / training cells)

- **Guard = 2** covers the mainlobe (measured: −20 dB at ±2 bins, < −90 dB by ±3). A sweep
  confirmed guard 2 vs 3 changes Pd by < 1% — it is set by physics, not tuning.
- **Training = 8 per side (N_ref = 416)** sits at the knee of the CFAR-loss vs locality
  trade-off: on the graded SNR sweep at matched Pfa, Pd is flat beyond ≈ 8 training cells,
  while larger windows sacrifice locality in the clutter scenarios.

## Summary

| Method | Noise estimate `Z` | Best at | Worst at | Cost |
|---|---|---|---|---|
| CA | mean of all ref cells | homogeneous (optimal) | edges, multi-target | lowest |
| GO | max(lead, lag) | clutter edges | multi-target (worst) | low |
| SO | min(lead, lag) | close targets | clutter edges (worst) | low |
| OS | k-th sorted value | multi/close targets | edges; sort cost | medium |
| VI | adaptive CA/GO/SO | mixed/unknown scenes | non-Gaussian clutter; tuning | medium |
| DL | learned from data | complex clutter, low SNR | data need, no guaranteed Pfa | high |

## Predicted results for this project (hypotheses to test)
- `clean_debug` → all ≈ equal (CA slightly best).
- `clutter_debug` (edge/ridge) → CA degrades; **GO best classical**; DL strong.
- `multi_target` / `close_targets` → CA masks; **OS wins classical**; DL strong.
- Overall → no single classical method wins everywhere (CA↔GO↔OS trade off) → the gap DL fills.

**Plan:** proposal mandates **CA (must)** + **OS (recommended)**. Add **GO** as a cheap third
baseline (makes the clutter-edge story concrete). Treat **VI** and **DL** as the advanced tier.

## References
- Finn, H.M. & Johnson, R.S. (1968), "Adaptive detection mode with threshold control as a
  function of spatially sampled clutter-level estimates," *RCA Review*, 29, 414–464. (CA-CFAR)
- Trunk, G.V. (1978), "Range resolution of targets using automatic detectors," *IEEE T-AES*,
  14(5), 750–755. (SO-CFAR)
- Hansen, V.G. & Sawyers, J.H. (1980), "Detectability loss due to greatest-of selection in a
  cell-averaging CFAR," *IEEE T-AES*, 16(1), 115–118. (GO-CFAR)
- Rohling, H. (1983), "Radar CFAR thresholding in clutter and multiple target situations,"
  *IEEE T-AES*, 19(4), 608–621. (OS-CFAR) — https://ui.adsabs.harvard.edu/abs/1983ITAES..19..608R/abstract
- Smith, M.E. & Varshney, P.K. (2000), "Intelligent CFAR processor based on data variability,"
  *IEEE T-AES*, 36(3), 837–847. (VI-CFAR) — https://ieeexplore.ieee.org/document/869503/
- Deep learning-based lightweight radar target detection method (2023), *J. Real-Time Image
  Processing* — https://link.springer.com/article/10.1007/s11554-023-01316-5
- (VI)CFAR-CNN dual-detection fusion (2024), *Remote Sensing* — https://www.mdpi.com/2072-4292/16/2/332
- Tutorial: An Introduction to Radar Sliding-Window Detectors (CA/GO/SO/OS) — https://arxiv.org/pdf/1709.09786
