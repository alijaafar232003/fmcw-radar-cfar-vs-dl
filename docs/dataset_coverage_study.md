# Training-Coverage Study — why the benchmark dataset was redesigned

An experimental record of the sequence **evidence → diagnosis → redesign**. It documents why the
training distribution was changed from simple random sampling to a stratified design, and why that
is an experimental-design decision rather than a patch.

Reproduced by `notebooks/dl_cnn_experiment.ipynb` §10–12.

---

# 1. The question

Sections 7–9 of the DL experiment compared all four detectors on the **neutral mixed test set**
(`frozen_test`) and the network came out ahead at low Pfa. But a neutral aggregate answers only
*"who is best overall"*. The project's central claim is stronger:

> **Can one learned model handle every regime, whereas the classical methods are complementary
> (each strong in one regime and weak in another)?**

Testing that requires evaluating on the **diagnostic scenarios**, each of which isolates one
mechanism.

---

# 2. Evidence — the home-turf comparison

All four detectors at **matched Pfa = 1e-4**. The DL probability threshold was Monte-Carlo
calibrated on pure noise using exactly the CFAR protocol (the dB-above-median normalization is
scale-invariant, so unit-variance noise is representative): threshold = **0.8638**. CFAR
multipliers: CA 9.74, OS 6.98, GO 9.09.

| Scenario (metric) | CA | OS | GO | **DL** |
|---|---|---|---|---|
| **Masking** — weak-target Pd ↑ | 0.15 | **0.65** | 0.12 | **0.40** |
| **Clutter** — clutter-band FA/frame ↓ | 14.15 | 17.10 | 8.35 | **0.95** |
| **Unresolved pairs** — Pd (0.5 = physics cap) | 0.50 | 0.50 | 0.50 | 0.55 |

Supporting numbers: on masking, all methods detect the strong target (Pd = 1.00); DL also had the
*lowest* false-alarm rate there (1.4 vs 2.1–2.5 /frame). On clutter, DL held Pd = 0.86 — equal to
CA/OS and better than GO's 0.79 — while cutting clutter-band false alarms by ~9×.

## Interpretation

**Clutter — DL dominates decisively.** 0.95 clutter-band false alarms per frame against GO's 8.35
(the best classical), i.e. nearly an order of magnitude better, *without* sacrificing Pd. A local
window *statistic* cannot represent clutter; a learned function can recognise its **texture**
(Doppler-spread, spatially correlated) and distinguish it from a target's compact mainlobe.

**Masking — DL is mid-table.** Weak-target Pd 0.40 clearly beats CA (0.15) and GO (0.12), but is
well short of the specialist **OS-CFAR (0.65)**. So the network has *partial* masking capability.

**Unresolved pairs — everything is capped near 0.5.** Two targets inside one mainlobe produce a
single peak, and non-maximum suppression keeps one detection. DL's 0.55 corresponds to **one frame
out of ten** and is within sampling noise — **not** evidence of super-resolution.

## Caveats (stated, not hidden)
* The diagnostic sets are small — 40 / 20 / 10 frames — so these are **preliminary**. The clutter
  effect is far too large to be sampling noise; the "unresolved" column is not interpretable.
* These sets were generated **before** the radar-equation range law was introduced, so their
  amplitude-versus-range statistics differ from the training data. That is a **secondary
  distribution shift** on top of the condition difference, and it is one reason the redesign puts
  every condition in a single dataset with consistent physics.

---

# 3. Diagnosis — performance tracks training coverage

Auditing `frozen_train` (the data the network actually learned from) for how frequently each
condition occurs produces an almost monotonic relationship with the measured results:

| Condition | Coverage in training | DL vs best classical |
|---|---|---|
| Clutter (target within 8 bins of the ridge) | **~16 %** | **DL wins** (~9× fewer edge FA) |
| Masking (resolvable neighbour ≥6 dB stronger, inside the CFAR window) | **~1.1 %** | **DL loses** to OS (0.40 vs 0.65) |
| Unresolved (neighbour ≤3 bins) | **~0.4 %** | no advantage (physics-capped anyway) |

**This is the key finding.** The network's weakness at masking is a **data-coverage problem, not a
capability limit**: it demonstrably *can* learn a condition when adequately represented (clutter,
where it beats every classical method), and it underperforms precisely where training examples were
scarce.

**Why this matters for the conclusions.** Without this diagnosis the comparison would have reported
*"CFAR beats DL at masking"*, implying a fundamental limitation of learned detection. The honest
statement is *"the network was barely trained on masking"* — a completely different claim with a
completely different remedy.

**Root cause.** `frozen_train` used simple random sampling with
`min_separation: {range_bins: 3, velocity_bins: 3}`, which additionally *suppresses* sub-mainlobe
pairs by construction — hence 0.4 % coverage for unresolved cases.

---

# 4. The redesign — stratified sampling

`configs/bench_{train,val,test}.yaml`. Eight named conditions in **equal proportion**, assigned by
**cycling** (`frame_index % 8`) so counts are exactly balanced and reproducible:

| Stratum | Definition | Purpose |
|---|---|---|
| `isolated` | 1 well-separated target | clean baseline |
| `masking` | strong + resolvable weak pair, 5–9 range bins apart | the case OS is built for |
| `unresolved` | pair 1–3 bins apart | the resolution limit |
| `clutter` | 1–3 targets constrained near v ≈ 0, strong ridge | the case GO is built for |
| `multi` | 4–5 targets | dense scenes |
| `multipath` | 1–2 targets, ghosts enabled | extra false-alarm source |
| `noise` | 0 targets | false-alarm measurement |
| `mixed` | 0–5 targets, clutter + multipath (the original recipe) | realistic combination |

Sizes: **train 9,600 / val 1,600 / test 2,400** — 1,200 / 200 / 300 frames per stratum. Splits share
the design and differ only by `random_seed`, so they are identically distributed with no shared
frames.

**Implementation.** The generator gained a `scenarios:` block; per frame it selects a stratum by
cycling and **deep-merges** that stratum's `targets:`/`scene:` overrides onto the dataset defaults.
Crucially, **each frame records its `condition` in metadata** (`metadata.json` and `metadata.csv`),
so stratified evaluation is a direct filter with guaranteed sample counts rather than an inference
from label geometry.

**Design decisions worth noting**
* The **radar-equation range law is enabled in every stratum**, so the physics is consistent across
  conditions — removing the secondary distribution shift present in the old debug sets.
* `mixed` preserves the original realistic recipe, so the realistic case is not lost.
* Verified on a 40-frame trial run: exactly 5 frames per stratum, condition tags recorded, and
  per-stratum behaviour correct (isolated → 1.0 targets/frame, masking → 2.0, multi → 4.4,
  noise → 0.0).

## What was deliberately *not* changed

The SNR distribution is skewed towards high SNR (53 % of targets above 20 dB). This is a
**consequence of the radar equation** — near targets genuinely are stronger — so flattening it would
make the data less physical. It is recorded as a characteristic, not corrected.

---

# 5. Why this is stratified sampling, not a patch

The anticipated question is *"why not design it correctly from the start?"* The answer is a standard
experimental-design distinction, not a resource excuse:

> The first dataset used **simple random sampling**: physically realistic scenes with randomly drawn
> targets. That maximises *realism*. Auditing it against the evaluation conditions showed that
> realistic random sampling necessarily **under-represents rare configurations** — masking at 1.1 %
> of targets, unresolvable pairs at 0.4 %. Reliable *per-condition* detection estimates require
> adequate samples in every stratum, so the design was changed to **stratified sampling**, in which
> strata are sized for **estimation reliability** rather than for realism.

Two supporting points:
1. **The correct proportions were not knowable in advance.** They follow from which conditions the
   evaluation needs to resolve, and from the measured rarity of each — both of which required the
   dataset to exist first. The audit produced the design.
2. **The realistic distribution is retained** as the `mixed` stratum, and `frozen_test` is kept
   intact as a neutral aggregate benchmark. Nothing was discarded; a stratified view was added.

The progression **random sampling → coverage audit → stratified redesign** is how a controlled
experiment is meant to be built. The audit is the method, not an apology for it.

---

# 6. Report-ready statements

> "Detection performance was first compared on a neutral mixed test set, then on diagnostic
> scenarios isolating individual mechanisms. On clutter-dominated scenes the learned detector
> reduced clutter-region false alarms by approximately an order of magnitude relative to the best
> classical variant (0.95 versus 8.35 per frame) at equal detection probability. On target-masking
> scenes, however, it recovered only 40 % of weak targets against 65 % for ordered-statistic CFAR."

> "Auditing the training distribution revealed that these outcomes track condition coverage:
> clutter-affected targets constituted approximately 16 % of training examples, whereas masking
> configurations constituted 1.1 % and unresolvable pairs 0.4 %. The learned detector therefore
> underperformed precisely where supervision was scarce, indicating a limitation of the training
> distribution rather than of the method."

> "The training and evaluation data were consequently redesigned from simple random sampling to a
> stratified scheme comprising eight equally-represented conditions, with the condition recorded per
> frame so that per-condition estimates have guaranteed sample sizes. Strata in a stratified design
> are sized for estimation reliability rather than for representativeness; the realistic mixture is
> retained as one stratum."
