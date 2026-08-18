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
addition in §5, carried out in §8.

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

**T5 — computational cost**, DL against CA/GO/OS/SO. Detailed in §3.3.

### 3.1 "Best CFAR" is a maximum over four methods — name it, do not hide it

Every DL-vs-CFAR delta in T1/T2 is measured against **the best of CA/GO/OS/SO on that condition,
selected by Pd**. That is a **maximum over four correlated estimates**, so it is biased *upward* by
the usual winner's-curse argument, and our reported delta is therefore biased **downward**. The
classical arm gets the benefit of per-condition method selection that a deployed system would have
to make in advance, without knowing the condition.

This is kept because it is exactly what was done on `bench2_val`, and changing it now to flatter the
network would be indefensible. But it is a **choice, not a neutral baseline**, and it is named here
so no reader mistakes "best CFAR" for a single fixed detector.

The same applies with more force in **T3**, where "best CFAR" localization is a **minimum over four
methods, taken independently per condition and per axis** — a still more generous selection. See
§4.1 for the additional conditioning effect that makes those particular ratios read high.

### 3.2 What each gate is testing — neither is neutral

Both gates are reported because **each is biased, in opposite directions**. Reporting only one would
be a choice about who wins.

**±1 (strict, pre-registered as primary).** The gate region is 3×3 cells — **exactly the 3×3 label
block the network is trained on**. The network is therefore rewarded for hitting precisely the
region it was taught to fire on, which is as favourable an alignment as the DL side can get. It also
penalises CFAR for label misalignment: labels sit on the true peak only ~93% of the time (Milestone
22), and CFAR responds to the *physical* peak, so a correct CFAR detection can fall outside the gate
through a labelling artefact rather than a detection failure.

**±3 (permissive, reported beside it).** Widening to 7×7 recovers those CFAR label-misalignment
losses — but it stops penalising placement altogether, so a badly-localised DL detection now scores
as a hit. It converts DL's localization deficit from lost Pd into no penalty at all.

**±3 was not added because ±1 looked bad.** ±1 was pre-registered as primary from the start and
remains primary. And the direction of the effect rules out that reading anyway: **±3 is the gate
where DL *loses* `clutter`** (−0.043, p = 0.005 on val), so it cannot be a gate chosen to flatter
the network. If anything the honest summary is that ±1 favours DL and ±3 favours CFAR, and the
conclusions that survive both are the ones worth reporting.

### 3.3 T5 — computational cost, and how it is measured

Three quantities, because a single wall-clock number on one laptop is not a portable claim:

1. **Wall-clock ms per frame**, on the same machine, same thread count, same process — measured
   back-to-back so thermal and load conditions are shared.
2. **A hardware-independent arithmetic count**: multiply–accumulates per RD cell, derived from the
   architecture rather than timed.
3. **Model size**: 5,041 parameters for DL; CFAR has none.

**Measurement conditions, which decide the number and must be reported with it:** CPU only
(no GPU anywhere in this project), `threads=2` to match `configs/frozen_dl.yaml`, batch size 1
(one 256×64 frame at a time, the honest per-frame latency rather than a throughput-flattering
batch), `torch.no_grad()`, model in `eval()` mode, excluding disk I/O and NMS — NMS is shared by
both arms and would cancel. CFAR timings are the `ms_per_frame` already recorded in
`results/cfar_test/peaks_*.npz`, measured on the same machine.

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

### 4.1 Localization, pre-registered per condition — not just the pooled ratio

Localization is now half the headline, so predicting only the pooled 4.6× would leave it far less
pre-registered than Pd. **Predicted values on test, from `bench2_val` at epoch 21** (DL = 8-seed
mean ± SE; "best CFAR" = the lowest error among CA/GO/OS/SO on that condition **and that axis**):

| condition | DL range | best CFAR range | ratio | DL Doppler | best CFAR Doppler | ratio |
|---|---|---|---|---|---|---|
| **pooled** | 0.473 ± 0.017 | 0.179 (SO) | **2.65×** | 0.393 ± 0.025 | 0.086 (OS) | **4.57×** |
| clutter | 0.534 ± 0.032 | 0.038 (SO) | 13.9× | 0.708 ± 0.034 | 0.037 (CA) | 19.1× |
| clutter_edge | 0.470 ± 0.027 | 0.180 (SO) | 2.60× | 0.422 ± 0.020 | 0.049 (SO) | 8.59× |
| isolated | 0.491 ± 0.023 | 0.148 (OS) | 3.32× | 0.380 ± 0.031 | 0.057 (GO) | 6.64× |
| masking | 0.494 ± 0.017 | 0.227 (SO) | 2.17× | 0.383 ± 0.027 | 0.039 (CA) | 9.85× |
| mixed | 0.490 ± 0.021 | 0.209 (SO) | 2.34× | 0.403 ± 0.024 | 0.098 (OS) | 4.10× |
| multi | 0.498 ± 0.019 | 0.221 (CA) | 2.25× | 0.375 ± 0.033 | 0.079 (CA) | 4.77× |
| unresolved | 0.306 ± 0.013 | 0.182 (SO) | 1.68× | 0.323 ± 0.029 | 0.077 (OS) | 4.18× |

**Predicted, concretely:** DL's per-condition errors land within roughly ±0.05 bins of the values
above; DL is worse than the best CFAR on **every condition and both axes**; `unresolved` remains
DL's best condition on range and `clutter` its worst on both axes.

**⚠️ Two things inflate these ratios, and both must be stated wherever they are quoted.**

**(a) The denominator is a minimum over four methods, per condition per axis.** SO supplies the best
range error on 5 of 7 conditions while being the worst detector by a wide margin; CA and OS supply
most of the Doppler minima. No single classical detector achieves the "best CFAR" column — it is an
envelope, and no deployable system is that envelope.

**(b) Localization is conditioned on detection, which flatters whoever detects less.** The error is
averaged over *detected* targets only. On `clutter` (val, epoch 21) CFAR detects Pd **0.149** against DL's **0.141** at a
matched operating point, but CFAR's detections are concentrated on the strongest, cleanest targets,
which are also the easiest to place — hence a 19× ratio that is **partly a selection effect, not
purely a precision gap**. The pooled 2.65× / 4.57× figures are the defensible headline; the
per-condition extremes, especially `clutter`, must not be quoted as bare precision ratios.

A clean way to see this in the report, **not run here** (it would be a new analysis invented after
the fact): restrict both arms to the targets *both* detect, and recompute. Logged as a limitation.

### 4.2 T5 — cost, predicted

**The premise needs correcting before the prediction is worth anything.** The intuition "DL does the
whole map in one forward pass while CFAR works per cell" **does not describe our CFAR**. Ours is
fully vectorised with `scipy.ndimage` (`src/cfar/cfar.py`): `uniform_filter` computes the box mean in
**O(1) per cell regardless of window size** via separable running sums, so CA is not paying 164
operations per cell. Both arms are whole-map array operations, and the real question is arithmetic
intensity and how well each vectorises — not per-cell versus batched.

Measured arithmetic, from the architecture:

| | per RD cell | per 256×64 frame |
|---|---|---|
| **TinyCNN** (4 conv layers, 1→16→16→16→1, k3×3) | **4,896 MAC** | 80.2 MMAC (160 MFLOP) |
| CA-CFAR | ~O(1) via separable box filter | — |
| OS-CFAR | rank-selection per cell (`rank_filter`) | — |

DL does **~3 orders of magnitude more arithmetic per cell than CA**. Against that, its arithmetic is
dense, regular and BLAS-friendly, while OS's per-cell rank selection vectorises badly.

**Predicted, and deliberately split rather than a single optimistic claim:**

- **DL is FASTER than OS-CFAR** (already measured at **57.8 ms/frame** on test — by far the most
  expensive classical method, because of the rank filter).
- **DL is SLOWER than CA** (**2.34 ms/frame**), on the arithmetic argument above. Predicting
  otherwise would be wishful.
- **DL against GO/SO (6.6 ms/frame) is genuinely uncertain** — called now, not after.

**The RadCNN contrast still holds, and is the point worth making.** RadCNN classifies one patch per
cell, i.e. a network invocation per cell, and reports ~10× OS-CFAR. Ours is one fully-convolutional
pass that shares all overlapping computation, so it should be far cheaper than that — but that is an
advantage over *other CNN designs*, not over our own vectorised CFAR, and the report must not
conflate the two.

**If DL is slower than everything except OS, that is the result and it gets reported as such.** A
detector that wins Pd in interference at a few times CA's cost is still a defensible finding;
overclaiming speed would not be.

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

**SO matters more than "completeness" suggests, and an earlier draft of this document understated
it.** On Pd, SO is much the weakest method (`bench2_val` pooled 0.237 against CA 0.482) and is never
the best on any condition — so for **T1/T2 it is not expected to change "best CFAR" anywhere**.

But on **localization it is the best classical method on the range axis** (`bench2_val`, ±1:
SO **0.179** bins against CA 0.216, GO 0.218, OS 0.218). The headline range ratio in T3 —
"DL's range error is 2.65× the best CFAR" — therefore *divides by SO*. Omitting SO would not have
made T3 incomplete, it would have made it **wrong**, by silently comparing against a weaker
denominator (OS, giving ~2.2× instead of 2.65×).

So SO is run on `bench2_test` before the DL read, and the T3 range ratio is defined against the best
of all four. Predicted: **SO remains the best classical method on range localization and the worst
on Pd.** If either flips, that is a reportable finding and it will be reported.

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

**What this read does NOT establish.** `bench2_test` is drawn from the **same simulator, with the
same generator settings**, as `bench2_train` and `bench2_val` — different frames and different seed,
no shared content, but not a different world. So this is a test of **sampling generalization**:
whether the result holds on frames the network never saw. It is **not** evidence that any finding
survives a different clutter model, a different sensor, or real measurements — nothing here can
distinguish a genuine detection advantage from one that depends on our simulator's particular
statistics. That is a **limitation of scope, not a flaw in the protocol**, and it is the reason
CARRADA remains the stretch goal. It should be stated in the report's limitations wherever the test
numbers are presented as the headline.

---

## 8. Addendum — SO-CFAR added to the test baseline (17 Aug, before the DL read)

Run as specified in §5, **before** any DL evaluation of the split.

    python scripts/cfar_test_baseline.py --methods SO              # detection pass
    python scripts/cfar_test_baseline.py --methods SO --score-only # merge into summary.csv

**Integrity, verified:** `peaks_CA/GO/OS.npz` are byte-identical to the 11 Aug backup (md5), and all
14 numeric columns of their `summary.csv` rows are unchanged. Only SO rows were added. The 11 Aug
state is preserved in `results/cfar_test_backup_11aug/`.

**Pd at matched measured Pfa 1e-4, `bench2_test`:**

| | clutter | clutter_edge | isolated | masking | mixed | multi | unresolved |
|---|---|---|---|---|---|---|---|
| CA | 0.166 | 0.397 | **0.675** | 0.535 | 0.408 | **0.687** | **0.464** |
| GO | **0.171** | 0.415 | 0.662 | 0.529 | **0.426** | 0.680 | 0.462 |
| OS | 0.163 | **0.449** | 0.662 | **0.622** | 0.388 | 0.682 | 0.462 |
| SO | 0.063 | 0.197 | 0.665 | 0.609 | 0.210 | 0.682 | 0.462 |

**The §5 prediction held: SO is the weakest on Pd and is the best method on no condition**, so
"best CFAR" is unchanged everywhere and T1/T2 are unaffected by the addition. SO's value here is
T3, where it is expected to be the best classical method on the range axis.

The `--score-only` path was added while doing this: the detections were always cached precisely so
an operating point could be re-derived without re-running CFAR, but nothing exposed that. A
guard also now refuses to overwrite `frames.csv`/`truths.csv` if they disagree with the existing
run, since every cached peaks file is keyed on those indices. It compares floats with a tolerance —
the CSVs round-trip float64 to within one ulp (3.6e-15 on `cnr_db`), which is serialisation, not a
different dataset; NaN patterns and all integer/string columns must still match exactly.

**`bench2_test` remains untouched on the DL side.** The one-shot has not been taken.
