# Pre-registration — Integrated CFAR attribution experiment

**Written 2026-08-24, before any result of this experiment was computed or seen.** The run script
was launched from a clean `results/` folder and this file was committed alongside it.

## The question

> How much of TinyCNN's advantage over CFAR is explained by the CNN integrating neighbouring
> mainlobe energy that single-cell CFAR discards?

## The prediction being registered

> With a −3 dB width of ~1.6 bins the target energy is concentrated in one or two cells, so
> **uniform summation adds mostly noise to the numerator and may perform WORSE than single-cell
> CFAR**. PSF-weighted is the statistic with a theoretical reason to help. **If uniform loses, that
> is a result, not a failure — report it.**

## The competing consideration, also registered

Stated so that whichever way it lands, the reasoning was on the record beforehand. Two effects pull
in opposite directions for the uniform statistic:

* **For it:** summing K cells raises the signal by the captured energy fraction while the noise mean
  grows by K and its standard deviation only by √K if the cells were independent — which would be a
  net gain in deflection.
* **Against it:** the Blackman window **correlates adjacent cells**, so the noise sum's standard
  deviation grows faster than √K, eroding that gain. Combined with the measured energy being
  concentrated in very few cells, the uniform sum may add more noise than signal.

Which dominates is an empirical question, and it is the reason both statistics are run rather than
only the one with the cleaner theory.

## Decision rules fixed in advance

* **Integration region** is chosen from the *measured* energy-capture table, not by convention, and
  must fit inside the CFAR guard region (±2, ±2) so that the numerator cannot overlap the reference
  ring the denominator uses.
* **Alpha is calibrated by Monte Carlo** on pure noise and verified on a held-out seed. The closed
  form does not apply to a sum of correlated cells, and the project's cached alphas are meaningless
  for this statistic. *Alpha does not affect any Pd reported at matched measured Pfa* — it fixes the
  design point only, and is reported so that an uncalibrated design point cannot pass unnoticed.
* **Evaluation is the unmodified project protocol**: `bench2_val`, Pd at matched measured Pfa 1e-4
  swept per condition, gates ±1 and ±3, shared `all_peaks` at `nms_size=5`, Hungarian matching,
  guard-excluded Pfa denominator, paired frame bootstrap with Bonferroni across the 7 scored
  conditions.
* **`bench2_test` is not touched.** It was spent on 19 Aug. Whether any of this warrants a second
  read is a separate decision to be made after seeing the validation result, and recorded before
  it happens.
* **Localization is reported for the integrated variants too.** If integration improves Pd but
  worsens placement, that bears directly on the CNN's own localization deficit.

## Interpretation fixed in advance

Per condition and pooled, for **both** the uniform and PSF-weighted variants:

    delta_integration = Pd(Integrated CFAR) − Pd(CFAR)
    delta_remaining   = Pd(TinyCNN)         − Pd(Integrated CFAR)

| outcome | reading |
|---|---|
| Integrated ≈ TinyCNN | the advantage is largely **mainlobe energy** |
| Integrated between CFAR and TinyCNN | mainlobe energy explains **part** of it |
| Integrated ≈ CFAR | integration explains **none** of it, and the **learned decision rule** is the live explanation |

## Scope

This is a standalone attribution experiment. It does **not** alter the project's primary results or
conclusions, and it is scored on the validation split, so it cannot restate the pre-registered test
result either way.
