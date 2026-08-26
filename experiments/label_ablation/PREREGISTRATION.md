# Pre-registration — Label ablation

**Written 2026-08-24, before any training run of this experiment was started.** Committed alongside
the label caches and before the first checkpoint exists.

## The question

The 3 × 3 label block is the documented suspect behind three separate findings:

* DL localizes **3.83× worse than CFAR in Doppler** (pooled, `bench2_test`, gate ±1);
* localization degrades **monotonically with training epochs** while Pd plateaus;
* at **gate 0** DL collapses to Pd ≈ 0.41 where CFAR reaches ≈ 0.85.

All three have the same proposed explanation: the network is told nine cells are equally correct, so
it never learns which one is the peak.

> **Does a better label fix this?**

## The three arms

| arm | label | note |
|---|---|---|
| **point** | 1.0 at the quantised truth bin only | precedent: RadCNN (Yavuz 2021) |
| **flat3x3** | 1.0 on a 3 × 3 block | the incumbent; built by the project's own `make_label_mask`, asserted identical to the frozen run's supervision |
| **gaussian** | 3 × 3 footprint, weights from the **measured** PSF | precedent: CFARNet (Liang 2025) |

**σ is derived, not chosen.** From `results/psf_profile.csv` (the output of
`scripts/measure_psf.py`), the −3 dB width of the measured response is taken on each axis and
converted by `sigma = FWHM / 2.3548`. The **off-bin worst case** is used: it is the case the project
already quotes (1.62 / 1.65 bins) and targets here are placed off-grid by design, so an on-bin target
is the exception. This gives σ_range ≈ 0.690, σ_Doppler ≈ 0.701 bins.

**Multi-target combination for the Gaussian arm is elementwise MAXIMUM**, not the sum CFARNet uses.
A sum lets two nearby targets saturate into a flat plateau — precisely the flat-supervision failure
under test.

Everything else is identical to `configs/frozen_dl.yaml`: TinyCNN width 16, Adam, lr 1e-3, batch 32,
21 epochs, `threads=2`, seeds 0–7, same splits.

## The `pos_weight` rule, fixed in advance

`BCEWithLogitsLoss(pos_weight=w)` weights each cell's positive term by `w · y`, so the loss's total
emphasis on "this is a target" is proportional to `w · Σy`. **That product is held constant across
arms**, anchored to the incumbent:

    w_arm = 100 · ( Σy_flat3x3 / Σy_arm )

**Anchored to the incumbent deliberately.** The obvious alternative — re-deriving `pos_weight` per
arm from class frequency (`w = N_neg / N_pos`, i.e. exact class balancing) — has already been
**measured to be worse in this project**: `pos_weight` 974 lost the three-way ablation on 6 of 7
conditions and manufactured a `clutter` deficit that had to be retracted (Correction C3). Anchoring
to the empirically chosen 100 and scaling by positive mass preserves the operating point shown to
work while still removing label shape as a confound with class weighting.

Exact values are computed by `build_label_cache.py` and recorded in
`results/pos_weight_derivation.csv`. The point arm carries ~1/9 the positive mass of the flat arm,
so its `pos_weight` will be ~9× larger (≈ 900); the Gaussian arm has the same 9-cell support but
lower total mass, so it will fall between.

## Predictions being registered

1. **Gaussian matches flat3x3 on Pd while IMPROVING localization.** It keeps the dense gradient
   signal that makes training work, but gives the network a reason to prefer the centre cell.
2. **Point may train poorly or unstably** at roughly 1:8500 positive:negative. If it does, **that is
   a result** — and it is the first actual evidence that the 3 × 3 choice was reasonable rather than
   merely argued.
3. **A Pd GAIN from Gaussian would be unexpected.** If it appears it must be flagged as such, not
   reported as a straightforward win.

**Failed predictions will be reported as failed predictions.**

## The sharpest test of the mechanism

Checkpoints at **epochs 5, 10, 15, 21**, with **both** Pd and localization extracted at every one.

Localization is known to degrade monotonically with training under the flat label. If flat
supervision causes that drift, **weighted supervision should reduce or remove it**. A Gaussian curve
that is flatter than the flat3x3 curve is direct evidence for the mechanism; one that is equally
steep falsifies it, whatever the endpoint values are.

## Evaluation, fixed in advance

`bench2_val`. Pd at matched measured Pfa 1e-4 swept per condition. **Gates 0, ±1 and ±3.** Shared
`all_peaks` at `nms_size=5`, Hungarian matching, guard-excluded Pfa denominator. 8-seed mean ± SE,
paired frame bootstrap, Bonferroni across the 7 scored conditions.

**Gate 0 matters most here.** It is the exact-cell criterion, where the 3 × 3 label costs the most,
so it is where a better label should show the largest gain. Reported **beside** ±1, never instead
of it.

Localization reported in bins and physical units, and as a **ratio to the best CFAR**, so it is
directly comparable to the existing 3.83× / 2.40×.

**`bench2_test` is not touched.** This is attribution on validation.

## Scope

Standalone attribution experiment. It does **not** alter the project's primary results or
conclusions. All artefacts stay under `experiments/label_ablation/`, arm-tagged so nothing can
collide with the frozen run's naming.
