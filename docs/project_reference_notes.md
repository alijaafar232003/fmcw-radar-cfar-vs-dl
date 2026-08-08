# FMCW Radar Target Detection — Master Reference Notes

A single, ordered place for everything you need to understand and write down: the radar
physics, the detection theory, what you have built so far and why each piece matters, and
where the project is going. Read top to bottom — each part builds on the previous one.

> Project in one line: a **simulation-based benchmarking study** that builds a physically
> faithful FMCW radar signal pipeline and fairly compares **classical CFAR** against a
> **deep-learning detector** for finding targets on Range–Doppler maps.

---

# PART 0 — Orientation: what this project is

- **Type:** applied/experimental computational study. You implement *known* methods and
  *measure* them on *controlled* synthetic data.
- **It is NOT:** a real car radar, new hardware, or new mathematical theory.
- **Application context:** automotive-style FMCW radar (77 GHz), but you study the *detection
  algorithms*, not the vehicle system.
- **Real-time?** You *measure* runtime per frame and discuss feasibility; you do not build a
  live system.
- **The contribution:** CFAR is the decades-old standard but degrades in clutter / low SNR /
  multi-target masking. DL might do better. You quantify **whether and when** — with proper
  metrics, on identical data.

**The whole pipeline in words:**
`radar settings → simulate chirps → target echoes → dechirp → Range FFT → Doppler FFT →
Range–Doppler (RD) map → detect targets (CFAR vs DL) → score (Pd, Pfa, ROC, error, runtime)
→ compare.`

---

# PART 1 — Radar fundamentals (the physics you need)

## 1.1 What radar measures
Radar transmits an electromagnetic wave, it reflects off a target, and the echo returns. From
that echo you extract **range** (how far) and **radial velocity** (how fast toward/away).

- **Range** comes from the **time delay** of the echo: `τ = 2R/c` (round trip), `c` = speed of
  light. Far target → larger delay.
- **Velocity** comes from the **Doppler effect**: a moving target shifts the echo's phase/
  frequency slightly. Measuring that shift gives radial velocity.

## 1.2 Why FMCW (Frequency-Modulated Continuous Wave)
Instead of a short pulse, FMCW transmits a continuous **chirp**: a tone whose frequency ramps
linearly with time. Benefits: low peak power, simple hardware, and you get **both range and
velocity** from cheap signal processing. It is the dominant automotive radar type.

- **Chirp:** frequency goes from `fc` up by bandwidth `B` over chirp time `Tc`.
- **Chirp slope:** `slope = B / Tc` (Hz per second). *Your value:* `300e6 / 32e-6 = 9.375 ×
  10¹² Hz/s`.

## 1.3 Getting range — the beat frequency (dechirping)
Mix (multiply) the received echo with the transmitted chirp. Because the echo is a
**delayed** copy of the chirp, the mix produces a low-frequency **beat tone** whose frequency
is proportional to the target's range:

```
f_beat = (2 · R · slope) / c          →        R = (c · f_beat) / (2 · slope)
```

So a single **Range FFT** of the beat signal turns beat frequency into a range axis. This
"mix then low-frequency" trick is called **dechirping** (or "stretch processing").

> Sanity check with your numbers: a target at the max range 50 m gives
> `f_beat = 2·50·1.0608e13 / 3e8 = 3.54 MHz`, which is exactly half your sampling rate (Nyquist).
> That is why 50 m is your maximum range.

## 1.4 Getting velocity — Doppler across chirps
One chirp gives range. To get velocity you send **many chirps** (a burst) and look at how the
target's **phase changes from chirp to chirp**. A moving target advances its phase by a fixed
amount each chirp:

```
Δφ (per chirp) = (4π · v · Tc) / λ          λ = c / fc  (wavelength)
```

Taking a **Doppler FFT** *across the chirps* (at a fixed range bin) converts that phase
progression into a velocity axis. *Your wavelength:* `λ = 3e8 / 77e9 ≈ 3.9 mm`.

## 1.5 The CPI and the two-axis data cube
A burst of `M` chirps, each sampled `N` times, forms one **Coherent Processing Interval
(CPI)** — a 2-D array:

- **Fast time** (within a chirp, `N` samples) → after Range FFT → **range axis**.
- **Slow time** (across chirps, `M` of them) → after Doppler FFT → **velocity axis**.

*Your values:* `N = 512`, `M = 64`. Two FFTs turn the `M×N` CPI into the **Range–Doppler map**.

## 1.6 The Range–Doppler (RD) map
The final image. One axis = range, other axis = velocity. A target is a **bright peak** at its
`(range, velocity)` cell. *Your RD map is 256 × 64*:
- Range bins = `N/2 = 256` (only the positive-frequency half of the Range FFT is kept).
- Velocity bins = `M = 64`.

## 1.7 The performance equations (memorise these — they define your radar)
| Quantity | Formula | Your value | Meaning |
|---|---|---|---|
| Wavelength | `λ = c/fc` | 3.893 mm | sets Doppler sensitivity |
| Chirp slope | `B/Tc` | 1.0608×10¹³ Hz/s | steepness of the ramp |
| Sample rate | `fs = N/Tc` | 7.072 MHz | ADC speed |
| **Range resolution** | `c/(2B)` | **0.195 m** | smallest separable range gap |
| **Max range** | `c·fs/(4·slope)` | **50.0 m** | Nyquist limit on beat freq |
| **Velocity resolution** | `λ/(2·M·Tc)` | **0.420 m/s** | smallest separable speed gap |
| **Max velocity** | `λ/(4·Tc)` | **±13.44 m/s** | unambiguous Doppler limit |
| Frame time | `M·Tc` | 4.634 ms | one CPI |

**Key intuition:** range resolution depends on **bandwidth** (more `B` = finer range);
velocity resolution depends on **how long you observe** (more chirps `M·Tc` = finer velocity).

## 1.8 Windowing (why your map uses Blackman)
A raw FFT of a finite signal produces **sidelobes** — false ridges around a strong target that
can hide weak ones. Multiplying by a **window** (Blackman) before the FFT suppresses sidelobes,
at the cost of a slightly **wider mainlobe** (each target smears over a few bins). This is a
deliberate trade: cleaner background vs slightly worse resolution. (You saw the effect in
validation: close targets within a few bins merge.)

## 1.9 Noise, SNR, and processing gain (very important for your project)
- Real receivers add **thermal noise** → modelled as complex **AWGN** (additive white Gaussian
  noise).
- **SNR** = signal power ÷ noise power. It is your "difficulty dial."
- **Processing gain:** the two FFTs *coherently integrate* the target's energy while noise
  averages out, so the target's peak in the RD map is **far stronger** than in the raw samples.
  Gain ≈ `10·log10(N·M)` minus a few dB of window loss ≈ **~44 dB** for your radar.
  - *Measured proof:* a frame labelled −10 dB showed a peak 33.7 dB above the noise floor →
    gain ≈ 43.7 dB. **This is why "raw-sample SNR" is misleading and must be recalibrated**
    (see Part 3 finding #2).

## 1.10 The things that make detection hard (clutter, multipath, non-idealities)
- **Clutter:** unwanted reflections from the environment (ground, rain, trees). Often appears
  near **zero Doppler** (stationary) as a bright ridge. Non-uniform clutter is what classically
  **breaks CFAR**.
- **Multipath:** the signal bounces via a second path → a **ghost** target at the wrong range.
- **Non-idealities:** oscillator **phase noise**, **frequency offsets**, quantization — they
  raise the noise floor and smear targets.

---

# PART 2 — Detection theory (needed for CFAR and DL)

## 2.1 The detection problem
Given the RD map, decide for each cell: **target or just noise?** You compare the cell value to
a **threshold**. Above → "detection"; below → "nothing."

- **Pd (probability of detection):** fraction of real targets correctly found.
- **Pfa (probability of false alarm):** rate of declaring a target where there is none.
- There is always a **trade-off**: lower the threshold → catch more targets (higher Pd) but
  also more false alarms (higher Pfa).

## 2.2 Why a fixed threshold fails → CFAR
The noise/clutter level **varies across the map**, so one fixed threshold gives too many false
alarms in bright regions and misses targets in quiet ones. **CFAR (Constant False Alarm Rate)**
fixes this by estimating the **local** noise level around each cell and setting the threshold
*relative* to it — keeping the false-alarm rate constant everywhere.

- **Cell Under Test (CUT):** the cell being judged.
- **Guard cells:** skipped neighbours right next to the CUT (so the target's own energy doesn't
  leak into the noise estimate).
- **Training cells:** cells further out, used to estimate the local noise.
- **Threshold** = (noise estimate) × α, where **α is chosen from the desired Pfa**.
- **CA-CFAR** (Cell-Averaging): noise estimate = average of training cells. Simple; fails when
  clutter is non-uniform.
- **OS-CFAR** (Ordered-Statistic): noise estimate = a chosen ranked value (e.g. 75th
  percentile). More robust to clutter edges and nearby targets. *(These are Part B work.)*

## 2.3 Why deep learning might beat CFAR
A CNN/U-Net can learn the *shape* of targets vs clutter from many examples, instead of relying
on a local-average rule. The hope: it handles non-uniform clutter and multi-target masking
better. The risk: it needs training data and may not generalise.

## 2.4 The metrics you will report (define once, use for both methods)
- **Pd vs Pfa / ROC curve:** sweep the threshold, plot Pd against Pfa. Better detector = curve
  closer to the top-left.
- **Pd at a fixed Pfa:** a single headline number (e.g. Pd at Pfa = 10⁻³).
- **Localization error:** how far detections land from true targets (bins, and m / m·s⁻¹).
- **Runtime per frame + model size:** practicality.

## 2.5 The fairness rules (non-negotiable)
- Both detectors scored with the **same matching rule** and on the **same frozen test set**.
- CFAR's threshold/Pfa is **tuned on train/val data, never on the test set**.
- (Full rulebook: `docs/benchmarking_conventions.md`.)

---

# PART 3 — What you have built so far, and why it matters

## 3.1 Radar configuration (`src/radar/config.py`)
Defines the radar (`fc, B, Tc, N, M`) and computes the derived numbers in Part 1.7.
**Significance:** every later result depends on these; fixing them defines your whole
experiment. *Current: 77 GHz, 768 MHz, 72.4 µs, 512, 64 → 256×64 RD map.*
These are matched to the CARRADA dataset so simulated and real maps are directly
comparable — see `docs/radar_config_carrada.md` for the derivation.

## 3.2 The signal simulator (`src/radar/simulator.py`)
This is the physics engine. Map each function to Part 1:
- `generate_tx` → the transmitted chirp (1.2).
- `generate_dechirped_cpi` → builds each target's **delayed** echo (`τ = 2R(t)/c`), sums them,
  and **dechirps** (`tx · conj(rx)`) → the beat signal (1.3). Velocity enters via the moving
  range `R(t) = R + v·t` (1.4). Includes the **multipath ghost** (1.10).
- `add_awgn` → thermal noise at a chosen SNR (1.9).
- `add_near_zero_doppler_clutter` → the zero-Doppler clutter ridge (1.10).
- `simulate_cpi` → ties it together (targets → clutter → noise).
**Significance:** this is the "ground truth physics." If it's right, everything is trustworthy
(you verified it — see 3.6).

## 3.3 Range–Doppler processing (`src/radar/rd_processing.py`)
Applies the Blackman windows, the Range FFT (keeps positive half), the Doppler FFT, and
`fftshift`. Outputs:
- `rd_mag` (linear magnitude) — **CFAR uses this** (its statistics need linear/power values).
- `rd_db` (log, normalized) — for **display and DL input**.
**Significance:** turns raw signals into the RD image; the two output forms exist because CFAR
and DL need different scales.

## 3.4 Labels (`src/radar/labels.py`)
- `make_target_labels` → each target's exact `(range_bin, velocity_bin)` = the **answer key**.
- `make_label_mask` → a small mask marking target cells (for DL segmentation training).
**Significance:** synthetic data's superpower is *perfect labels*. These are what make fair
Pd/Pfa possible.

## 3.5 Dataset generator + scenarios (`src/data/`, `configs/`)
`create_dataset.py` reads a YAML scenario, generates `num_frames` frames with reproducible
seeds, and saves per-frame arrays + `labels.csv` + `metadata.json` + previews, plus a copy of
the config used (provenance). **The 7 scenarios and their purpose:**

| Scenario | What it stresses | Why it exists |
|---|---|---|
| `clean_debug` | easy, separated, high SNR | sanity baseline |
| `noise_only_debug` | 0 targets | measure false alarms (Pfa) |
| `snr_sweep_debug` | one target, SNR varied | the Pd-vs-SNR curve |
| `clutter_debug` | zero-Doppler clutter | **where CA-CFAR should fail** |
| `close_targets_debug` | targets a few bins apart | **resolution / masking limit** |
| `multi_target_debug` | 0–5 random targets | realism + DL training variety |
| `multipath_debug` | ghost reflections | extra false-alarm source |

**Significance:** isolating one effect per set is exactly how you find **where and why CFAR
fails** — the heart of your comparison.

## 3.6 Validation (`notebooks/validate_rd_alignment.ipynb`)
Confirms, frame by frame, that the **RD peak ≈ label bin ≈ mask cell**. Result: **7/7 datasets
pass**; every isolated target within ±1 bin; no NaN/Inf.
**Significance:** proves your labels are spatially correct *before* you build detectors on top
of them. Without this, every later number could be silently wrong. (It also produced the
3-panel figure and the summary CSV in `reports/validation/`.)

## 3.7 The two findings so far (write these down — they're report-worthy)
1. **Labels are verified correct** — the simulation pipeline has no axis/transpose/off-by-one
   bug. (3.6)
2. **The SNR definition is misleading** — because of ~44 dB processing gain, frames labelled
   "−10 dB" are still trivially detectable. The nominal (raw-sample) SNR does **not** reflect
   detectability, so it must be redefined as **post-integration cell SNR** and recalibrated
   before it can drive a meaningful Pd-vs-SNR curve. (Part 1.9; fix = item #2.)

---

# PART 4 — Where the project is going

**Phase A — Lock the foundation (current):**
- [x] Validate RD-peak/label alignment (done, 7/7).
- [ ] **Fix SNR calibration** (cell-SNR + radar-equation amplitude) — *next*.
- [ ] Write the shared evaluation/scoring module (`docs/benchmarking_conventions.md` → code).

**Phase B — CFAR baseline:** implement CA-CFAR then OS-CFAR (on `rd_mag`), evaluate → ROC / Pd
/ Pfa / localization.

**Phase C — Dataset + DL:** generate the **frozen** train/val/test split; build a small CNN/
U-Net; train; run on the same test set.

**Phase D — Comparison + report:** CFAR vs OS-CFAR vs DL on identical data; ablations; write up.

## 4.1 Realism roadmap (planned extra scenarios to find where CFAR fails)
*You asked to keep maximising scenarios that expose CFAR's weaknesses. These are deferred
realism upgrades, kept here on purpose. The rule: realistic noise/clutter, but keep target
geometry (and therefore labels) known.*

- **(High value) Range-dependent amplitude (radar equation, ∝ 1/R⁴)** — always-on physics;
  folds into the SNR work.
- **(High value) Heavy-tailed / non-uniform clutter (K-distributed or Weibull, with Doppler
  spread)** — the scenario most likely to break CA-CFAR; its own set.
- **(High value) Swerling RCS fluctuation** — textbook target-amplitude fluctuation; its own
  toggle.
- **(Optional) Phase noise / frequency offset** — your proposal's non-idealities; its own set.
- **(Optional) Extended targets, mutual interference** — more realism, more effort; note the
  label complication for extended targets.

Final structure: baseline physics on everywhere → one isolated set per effect → one combined
"realistic-hard" set where DL should most clearly beat CFAR.

---

# PART 5 — Quick reference

## 5.1 Symbols
| Symbol | Meaning | Your value |
|---|---|---|
| `fc` | carrier frequency | 77 GHz |
| `B` | bandwidth | 768 MHz |
| `Tc` | chirp duration | 72.4 µs |
| `N` | samples per chirp (fast time) | 512 |
| `M` | chirps per CPI (slow time) | 64 |
| `λ` | wavelength | 3.893 mm |
| `slope` | chirp slope `B/Tc` | 1.0608×10¹³ Hz/s |
| `fs` | sample rate `N/Tc` | 7.072 MHz |
| `c` | speed of light | 3×10⁸ m/s |

## 5.2 Formula cheat-sheet
```
λ            = c / fc
slope        = B / Tc
fs           = N / Tc
f_beat       = 2·R·slope / c          (range  → beat frequency)
R            = c·f_beat / (2·slope)
Δφ per chirp = 4π·v·Tc / λ            (velocity → Doppler phase)
range_res    = c / (2·B)
range_max    = c·fs / (4·slope)
vel_res      = λ / (2·M·Tc)
vel_max      = λ / (4·Tc)
proc. gain   ≈ 10·log10(N·M) − window loss   (≈ 44 dB here)
SNR          = signal power / noise power
threshold    = noise_estimate × α     (CFAR; α set by desired Pfa)
Pd           = TP / (TP + FN)
```

## 5.3 Related docs in this repo
- `Plan.md` — the live project plan / roadmap.
- `docs/benchmarking_conventions.md` — SNR definition + scoring rulebook.
- `notebooks/validate_rd_alignment.ipynb` — the validation walkthrough.
- `reports/validation/` — alignment summary + preview figure.

---

*Suggested writing order for your notebook:* Part 0 → Part 1 (physics) → Part 2 (detection) →
Part 3 (what you did) → Part 4 (plan). Parts 1–2 are your "Background" chapters; Part 3 is your
"Method/Implementation"; Part 4 is your "Plan/Future work."
