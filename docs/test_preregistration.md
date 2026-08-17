# Pre-registration — the `bench2_test` evaluation

**Written 17 August 2026, BEFORE the DL side of `bench2_test` is read.** Committed first, on
purpose. The whole value of a held-out split is that the analysis was fixed before the numbers
existed; anything decided after seeing them is tuning, however it is described afterwards.

The configuration was frozen the same day in `configs/frozen_dl.yaml` and is not touched again.

---

## 1. What has and has not been touched

| | |
|---|---|
| CFAR on `bench2_test` | **already measured**, 11 Aug, `PROGRESS.md` Milestone 21 → `results/cfar_test/` |
| DL on `bench2_test` | **never** — this is the one-shot |

CFAR was deliberately run first, while no network existed, so it could not be tuned against. CFAR
has no learnable parameters, so reading the test split with it cannot contaminate anything — that
argument is recorded in `NEXT_SESSION.md` §2.1 and it is the same argument that permits the SO
addition in §5 below.

## 2. The frozen configuration being evaluated

From `configs/frozen_dl.yaml`, unchanged: `TinyCNN` width 16 (5,041 params, RF 21×9) ·
`pos_weight` 100 · **epoch 21** · Adam lr 1e-3 · batch 32 · **threads 2** · seeds 0–7 ·
`nms_size` 5 · matched **measured** Pfa 1e-4 swept per condition · gate ±1 pre-registered, ±3
reported beside it.

Checkpoints: `results/dl/tinycnn_frozen_s{0..7}_ep021.pt`. **No re-training, no threshold
re-selection, no seed re-selection.** All eight seeds are reported whatever they show.

## 3. The tables to be produced — fixed now

**T1 — per-condition Pd, DL vs best CFAR.** 7 scored conditions (`noise` is excluded; it holds no
targets and exists to measure Pfa). DL = 8-seed mean ± SE. CFAR = the best of CA/GO/OS/SO *on that
condition*, chosen by Pd exactly as on `bench2_val`. Difference with a **paired frame bootstrap**,
2,000 resamples, 95% percentile CI, **Bonferroni across 7 conditions → α = 0.0071**. At gate ±1.

**T2 — the same at gate ±3.** Reported beside T1, never instead of it. This is where DL's `clutter`
loss appears, so omitting it would hide the one condition DL loses.

**T3 — localization**, per condition and pooled, for CA/GO/OS/SO/DL at both gates: mean |Δrange|,
|Δdoppler| in bins, plus metres and m/s. DL as 8-seed mean ± SE. Includes the headline ratio
**DL Doppler ÷ best-CFAR Doppler**.

**T4 — the measured Pfa actually achieved** per condition per method, to show the operating points
really were matched.

Commands (the same scripts, pointed at the test split — no new analysis code):

    python scripts/bootstrap_comparison.py --tag frozen --epoch 21 --tolerance 1   # and 3
    python scripts/localization_table.py  --tag frozen --epoch 21 --per-condition

## 4. Predictions, recorded before the fact

From `bench2_val` at epoch 21, 8 seeds. **`bench2_test` has 400 frames per condition against val's
200, so its standard errors should be ≈ √2 tighter** — marginal effects may harden into significance
purely from the extra frames, and that is not a new finding.

| condition | val Δ (±1) | val verdict | **predicted on test** |
|---|---|---|---|
| clutter_edge | +0.274 | win, p < 0.001 | **win, large** |
| mixed | +0.106 | win, p < 0.001 | **win** |
| masking | +0.051 | win, p < 0.001 | **win** |
| multi | +0.037 | win, p < 0.001 | **win** |
| isolated | +0.019 | tie, p = 0.030 | **genuinely uncertain — may cross into significance on 2× the frames.** Called in advance so it cannot be presented as a discovery |
| unresolved | +0.003 | tie, p = 0.62 | **tie** — both methods sit at the ~0.5 physics cap |
| clutter | −0.008 | tie, p = 0.51 | **tie at ±1** |
| clutter (±3) | −0.043 | **loss, p = 0.005** | **loss** |

**Localization:** DL is expected to remain clearly worse, Doppler ratio in the region of **4.6×**
(the val value at epoch 21). The ratio's denominator is a single CFAR run whose frame-sampling error
is not propagated, so it is a headline, not an inference.

**Absolute Pd will differ from val on both sides.** The CFAR test baseline already measured is:

| | clutter | clutter_edge | isolated | masking | mixed | multi | unresolved |
|---|---|---|---|---|---|---|---|
| best CFAR (test) | GO 0.171 | OS 0.449 | CA 0.675 | OS 0.623 | GO 0.426 | CA 0.687 | CA 0.464 |

`isolated` and `masking` differ noticeably from val (CA 0.710, OS 0.645). **The comparison is
within-split**, so a shift in both arms together is expected and is not a result.

## 5. One gap, and how it is handled

**SO-CFAR was added on 15 August, after the 11 August test baseline**, so `results/cfar_test/` holds
CA/GO/OS only. Two options were available and the choice is made **now**, before seeing anything:

> **Run SO on `bench2_test` as part of this same one-shot.** It is CFAR — no learnable parameters,
> nothing to tune against, and the non-contamination argument of §1 applies unchanged. Reporting a
> "best classical" that silently excluded one of the four methods would be worse.

On `bench2_val` SO was much the weakest on Pd (pooled 0.237 against CA 0.482) and was never the
best method on any condition, so it is **not expected to change "best CFAR" anywhere**. If it does,
that is a reportable finding and it will be reported.

## 6. Rules for reading the result

1. **The tables above are reported as they come out**, including any that contradict §4.
2. **No re-runs.** If something looks wrong, the finding is that it looks wrong. Any second read of
   `bench2_test` must be recorded as such, with the reason, in `PROGRESS.md`.
3. **No new conditions, metrics, gates or methods** invented after seeing the numbers. Anything that
   suggests itself goes to the limitations list, per the rule adopted 15 Aug.
4. **A prediction in §4 that fails is written up as a failed prediction**, not quietly dropped, and
   gets a Correction entry if it changes a published conclusion.
5. `bench2_val` remains the source for every *development* claim. Test numbers replace val numbers
   **only in the reported headline**, and the val figures stay on record.

## 7. What would count as a problem

Stated in advance so it cannot be rationalised later:

- **A reversal** — any condition where DL wins on val and loses on test at corrected significance.
  None is expected; the two splits are drawn from the same generator with no shared frames.
- **Pd collapsing on the clean conditions** (`isolated`, `multi`) would suggest an evaluation-path
  bug on the new split rather than a property of the detector, since those have no interference.
- **Measured Pfa far from 1e-4** in T4 would mean the operating point did not match, invalidating
  every comparison — this is the check that the whole protocol rests on.
