# Pre-registration — E4, CUT-centred guard-masked CNN attribution

**Written 2026-08-27, before any masked model was trained and before any Pd was measured for either
patch arm.** A 2-epoch pilot of the *unmasked* arm was run first to confirm the training loop
converges at all; no detection metric was computed from it.

## The question

> Does the existing TinyCNN obtain part of its detection advantage because it can use
> target/mainlobe information inside the cells that classical CFAR treats as guard cells?

## What the design had to change, and why

**CUT-relative guard masking is impossible on a dense fully-convolutional network.** The guard is
±2, so cell (r+1,v) lies inside CUT (r,v)'s guard while being its own CUT. One input map cannot
hold it hidden and visible at once. Proven in `scripts/verify_control.py`; masking on a decimated
lattice instead blanks ~90% of the map, and exact dense masking needs 189 passes per frame.

**So the masked arm must be a CUT-centred patch model.** That is sound for inference — the centre
output of a 21×9 patch draws on exactly those 21×9 inputs and never on padding, and the frozen
dense model's logit equals its patch-centre logit to **1.9e-06** in eval mode.

**But patch *training* differs from dense training in two further ways**: BatchNorm normalises over
21×9 rather than 256×64 (measured: +0.128 vs +0.203 centre logit for identical weights in train
mode), and the loss supervises one cell per patch rather than 16,384 per frame.

**Therefore the primary contrast is masked-patch vs unmasked-patch**, both trained identically, so
those two differences are common-mode and cancel. The frozen dense TinyCNN is retained as a
*reference point* for the patch/dense gap, **not** as the arm being differenced against. This
departs from the original instruction to train no unmasked control, and the departure was approved
after the blocker was reported.

## Arms

| arm | input | training |
|---|---|---|
| **frozen dense** | full map | **not retrained** — read-only control/reference |
| **unmasked patch** | 21×9 patch, nothing hidden | new, 8 seeds |
| **masked patch** | 21×9 patch, 24 guard cells set to neutral, **CUT visible** | new, 8 seeds |

Identical `TinyCNN` (width 16, 5,041 params, RF 21×9, dilations (1,2,3,4)/(1,1,1,1)), imported
read-only. Identical frozen hyperparameters: `pos_weight` 100, Adam, lr 1e-3, 21 epochs,
`threads=2`, seeds 0–7, flat 3×3 labels (**E3's alternatives are deliberately not used**).

**Both patch arms see the same CUT positions**: the sampling RNG is keyed on `seed + epoch` only,
never on the arm.

**Masked cells are set to 0.0**, which `normalize_rd` makes the frame's own median power — the
neutral noise-floor value. No mask channel is added; one input channel stays one input channel.

## Primary metric

    dPd = Pd(unmasked patch) - Pd(masked patch)

at matched **measured** post-NMS Pfa ≈ 1e-4, swept per condition on `bench2_val`, gate ±1, through
the project's unmodified evaluation pipeline. Paired by seed (0↔0, …, 7↔7), paired frame bootstrap,
Bonferroni across the 7 conditions.

## Registered interpretation

| outcome | reading |
|---|---|
| **Pd(unmasked) > Pd(masked)** | guard/mainlobe information materially contributes to the learned detector |
| **Pd(unmasked) ≈ Pd(masked)** | direct access to those cells is not an important explanation of the DL advantage |
| **Pd(masked) > Pd(unmasked)** | guard information is not responsible and may even distract the learner |

**This interpretation will not be changed after results are visible.**

## A prior from the occlusion diagnostic, registered as such

Run before training, on the frozen model: **63.7%** of the CUT logit's occlusion-sensitivity mass
sits in the 24 guard cells for target CUTs (36.2% for background), against 24.3% in the 164
reference cells — about **18× more influence per cell**. That is a *descriptive* result on
out-of-distribution inputs and does **not** predict the causal outcome. It is registered here so it
cannot later be presented as a prediction that was confirmed.

**A null or reversed result would be the more interesting outcome**, because it would mean the
network's function leans on information it does not actually need.

## Combination with E2, fixed in advance

E2 gave CFAR integrated mainlobe energy and it **did not** recover the DL advantage — integration
made CFAR worse on 7/7 conditions. If E4 now finds that removing guard access **does** cost the CNN,
the combined reading is:

> Mainlobe/guard information matters to the learned detector, but simply integrating that energy
> into CFAR is insufficient — so the learned rule exploits spatial/contextual structure in those
> cells rather than their energy alone.

**Explicitly not licensed:** "the CNN wins because it sums more target energy." E2 already argues
against that.

## Known limitations, stated before results

* **Both patch arms will be undertrained relative to the frozen dense model.** Uniform sampling of
  256 CUTs/frame preserves the dense class distribution (so `pos_weight` keeps its meaning and is
  not retuned) but delivers roughly **67× fewer positive examples per epoch**. At the frozen 21
  epochs both arms therefore sit at a lower operating point than the dense model. This is
  common-mode and does not bias ΔPd, but it means E4 measures the guard's contribution *within the
  patch regime*, not the absolute size of the effect in the dense model.
* Sampling is uniform and fixed in advance; indices are reproducible from `seed + epoch` and are
  not tuned on performance.
* `bench2_test` is never read. Asserted in the loader.
