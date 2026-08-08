# Radar configuration — matched to CARRADA

**Why this document exists:** the simulator's sensor parameters were re-specified to match the
CARRADA dataset, and the published CARRADA parameter table is internally inconsistent. This
records what was matched, what was *not* taken at face value, and the evidence — so the choice
can be defended in the report and the viva rather than re-derived from memory.

Written 4 August 2026, at the point of switching. Code: `src/radar/config.py`.

---

## 1. What was matched, and why

The goal is that a target looks **identical** in a simulated RD map and a CARRADA RD map, so
that a detector trained in simulation can be tested on real measurements without a resolution
mismatch confounding the result. That requires matching the **bin geometry**, because a detector
operates in bins:

| | CARRADA | Simulator (new) |
|---|---|---|
| RD map size | 256 range × 64 Doppler | 256 × 64 ✓ |
| Range span | 0 – 50 m | 0 – 49.97 m ✓ |
| Range per bin | 0.195 m | 0.1952 m ✓ |
| Velocity span | ±13.43 m/s | ±13.44 m/s ✓ |
| Velocity per bin | 0.42 m/s | 0.4201 m/s ✓ |
| Frame time | 4.635 ms | 4.634 ms ✓ |

## 2. Sensor parameters

| Parameter | Symbol | Value |
|---|---|---|
| Carrier frequency | `fc` | 77 GHz |
| Bandwidth | `B` | **768 MHz** |
| Chirp duration | `Tc` | **72.4 µs** |
| Samples per chirp | `N` | 512 |
| Chirps per frame | `M` | **64** |

Derived:

| Quantity | Formula | Value |
|---|---|---|
| Wavelength | `c/fc` | 3.8934 mm |
| Chirp slope | `B/Tc` | 1.0608 × 10¹³ Hz/s |
| Sampling frequency | `N/Tc` | 7.0718 MHz |
| Sample period | `1/fs` | 141.4 ns |
| Range resolution | `c/(2B)` | 0.19518 m |
| Max range | `c·fs/(4·slope)` | 49.965 m |
| Velocity resolution | `λ/(2·M·Tc)` | 0.42013 m/s |
| Max velocity | `λ/(4·Tc)` | ±13.444 m/s |
| Frame time | `M·Tc` | 4.6336 ms |
| Coherent processing gain | `10·log10(N·M)` | 45.15 dB (before window loss) |

**Nyquist check:** a target at 49.97 m produces a beat frequency of 3.536 MHz, exactly `fs/2`.
That is what sets the maximum range.

---

## 3. ⚠️ The published CARRADA table is inconsistent

CARRADA's Table I (Ouaknine et al., ICPR 2020, arXiv:2005.01456), verbatim:

| Parameter | Value |
|---|---|
| Frequency | 77 GHz |
| **Sweep Bandwidth** | **4 GHz** |
| Maximum Range | 50 m |
| FFT Range Resolution | 0.20 m |
| Maximum Radial Velocity | 13.43 m/s |
| FFT Radial Velocity Resolution | 0.42 m/s |
| Field of View | 180° |
| FFT Angle Resolution | 0.70° |
| Chirps per Frame | 64 |
| Samples per Chirp | 256 |

**The 4 GHz cannot be the bandwidth that produced these range bins.** Four independent checks:

### 3.1 Their own formula contradicts their own number
The paper states `δd = c/(2B)`. With `B = 4 GHz`:

```
δd = 3e8 / (2 × 4e9) = 0.0375 m
```

The same table prints **0.20 m**. A factor of 5.2 apart, within one table.

### 3.2 The angle entry reveals what "FFT Resolution" means
The paper states `δα = c/(fc·N_Rx·h·cos α)`. With 8 virtual antennas at half-wavelength spacing
that gives roughly **14°**. The table prints **0.70°**.

But `180° / 256 angle bins = 0.703°`.

So the entries labelled "FFT … Resolution" are **bin spacings**, not physical resolutions.

### 3.3 Applying that reading to range is self-consistent
`50 m / 256 range bins = 0.195 m` ≈ the 0.20 m printed. ✓

### 3.4 A genuine 4 GHz sampled sweep would make the dataset impossible
With `N = 256` samples and `B = 4 GHz`:

```
R_max = c·N / (2B) = 3e8 × 256 / 8e9 = 9.6 m
```

But CARRADA's own Figure 1 caption describes *"a pedestrian at approximately 8 m … and a car in
the background at approximately 33 m"*. The map plainly extends past 9.6 m, so **50 m is real and
4 GHz-as-sampled is not**.

### 3.5 What the 4 GHz most likely is
The configured RF sweep of the front-end (77–81 GHz is a standard automotive band). Only the
portion landing inside the sampled/low-pass-filtered beat band contributes to the range FFT.
Reproducing the observed 0.195 m bin spacing over 50 m in 256 bins requires an effective

```
B_eff = c·N_bins / (2·R_max) = 3e8 × 256 / (2 × 50) ≈ 768 MHz
```

### 3.6 The velocity side is fully self-consistent
Unlike range, the Doppler numbers check out exactly, which is a useful cross-validation that the
rest of the table is trustworthy:

- From `δv_R = c/(2·fc·T)` with `δv_R = 0.42 m/s` → frame time **T = 4.635 ms**
- Divided by 64 chirps → **Tc = 72.4 µs**
- `v_max = λ/(4·Tc)` = **13.44 m/s** ✓ against their stated 13.43

---

## 4. Consequences to remember

**Processing gain drops ~3 dB.** Coherent gain falls from `10log10(512×128) = 48.16 dB` to
`10log10(512×64) = 45.15 dB`, purely from halving the chirps. `src/radar/calibration.py` derives
the gain analytically from the windows, so cell-SNR calibration self-corrects — but the injected
noise-variance figures will differ from the previous configuration. **Re-measure, do not carry
the old 43.38 dB figure forward.**

**The CFAR window must be re-tuned.** The Doppler axis halves (128 → 64 bins). A 21×21 window
covered 16% of the old velocity axis and would cover 33% of the new one, and border exclusion
would leave only 44 of 64 Doppler bins testable. The guard/train geometry is re-measured
empirically rather than carried over. A non-square window (wider in range than Doppler) may be
required, which `CfarConfig` does not currently support.

**The CNN's dilations must follow the window.** TinyCNN's dilations `(1,2,3,4)` were chosen to
give a receptive field of exactly 21×21, matching CA-CFAR. That match is the basis of the
controlled experiment. Whatever geometry the sweep selects, the dilations are re-derived to match
it — this happens *after* the sweep, not before.

**Scenario definitions must be re-derived in physical units.** A separation quoted in bins means
a different distance now: 5 bins was 2.5 m, it is now 0.98 m. Scenarios are specified in metres
and m/s, then converted.

---

## 5. For the report

Suggested wording:

> The simulated sensor reproduces the range-Doppler bin geometry of the CARRADA dataset
> (256 × 64 bins; 0.195 m and 0.42 m/s per bin; 50 m and ±13.4 m/s span), so that simulated and
> measured maps are directly comparable and a simulation-trained detector may be evaluated on
> real data without a resolution mismatch confounding the comparison.
>
> The bandwidth was derived from the published bin geometry rather than from the 4 GHz sweep
> bandwidth quoted in the dataset's parameter table. That figure is inconsistent with the range
> resolution stated in the same table under the range-resolution formula the authors themselves
> give, and would imply a maximum range of 9.6 m rather than the stated 50 m. The table's angle
> entry confirms that its quoted "FFT resolutions" are bin spacings rather than physical
> resolutions: 180°/256 = 0.70°, against a physical angular resolution of roughly 14° for an
> eight-element virtual array. An effective bandwidth of 768 MHz reproduces the observed 0.195 m
> bin spacing across 50 m in 256 bins.

This is worth including: it demonstrates the dataset specification was validated rather than
copied, and it pre-empts an examiner asking why the simulator does not use 4 GHz.

---

## 6. Sources

- Ouaknine, Newson, Rebut, Tupin, Pérez, *CARRADA Dataset: Camera and Automotive Radar with
  Range-Angle-Doppler Annotations*, ICPR 2020 — https://arxiv.org/abs/2005.01456
- HAL copy — https://telecom-paris.hal.science/hal-03324378v1
- Dataset & code — https://arthurouaknine.github.io/codeanddata/carrada
- `valeoai/carrada_dataset` — https://github.com/valeoai/carrada_dataset

**Download note:** `Carrada.tar.gz` (23 GB) holds the camera + radar views and annotations, which
is what this project needs. `Carrada_RAD.tar.gz` (176 GB) holds the full range-angle-Doppler
tensors and is only required if the angle dimension is used.
