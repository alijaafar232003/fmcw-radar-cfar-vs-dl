# PLAN — future work

**Scope: this file records work that has NOT been done.** Nothing here has been executed, measured,
or scripted. Completed work lives in `PROGRESS.md` (the dated record with reasoning) and in the
notebooks; the immediate queue lives in `NEXT_SESSION.md`.

> **Note on file history.** An earlier `Plan.md` was deleted in the repository cleanup, and
> `PROGRESS.md` was made "the single roadmap record" at that point. This file reintroduces a
> separate roadmap deliberately and with a narrower job: **future experiments only**, kept out of
> `PROGRESS.md` so that completed and planned work cannot be confused when reading either one.
> `PROGRESS.md` remains the authority on everything already done.

**Current position.** All synthetic experiments are complete; the configuration is frozen in
`configs/frozen_dl.yaml`; `bench2_test` has been read once against a pre-registration. The critical
path is **writing**, not more experiments. Everything below is deliberately deferred behind that.

---

## Part A — DL attribution work (synthetic)

The purpose of this block is to explain *why* the learned detector behaves as it does. The headline
comparison already exists; what is missing is the mechanism behind it.

### Priority 1 — Label / localization attribution

**The question.** TinyCNN detects more targets than CFAR in interference but places them less
precisely (pooled Doppler error ≈ 3.8× CFAR's on `bench2_test`). The leading suspect is the
supervision: every target is labelled as a flat 3×3 block of positives, so the network is told that
nine adjacent cells are equally correct. That is a **hypothesis**, not a demonstrated cause.

**The comparison to run.** Train the same TinyCNN, unchanged in every other respect, under three
label representations:

| arm | label | what it tests |
|---|---|---|
| exact-cell | 1×1 — only the true target bin is positive | does removing the flat block sharpen the peak? |
| current | 3×3 flat block | the incumbent, as a control |
| soft | Gaussian / distance-weighted target | does a graded target sharpen it without losing Pd? |

**Purpose.** Determine whether the localization deficit comes from the supervision, and whether
localization can be improved **without losing Pd**. A trade in either direction is a result.

**Design notes to respect.** The 1×1 arm makes the class imbalance ~9× worse, so `pos_weight` is a
confound unless handled — either re-derive it per arm or report the ablation across arms. All arms
must use the frozen thread count and all eight seeds, and must be scored by the existing evaluation
protocol without modification.

### Priority 2 — Mainlobe / information-access attribution

**The question.** TinyCNN and CFAR have the same 21×9 spatial extent, but a matching extent is not
matching information *use*: CFAR imposes a fixed CUT / guard / reference-ring structure and computes
one hand-designed statistic, while the network may use every cell in its receptive field however it
learns to. The margin could therefore come from the learned decision rule, from access to mainlobe
energy CFAR deliberately excludes, or from both. **This is currently unresolved and is the single
biggest open question about the headline result.**

**Possible controls**, either of which narrows it from one side:

* a CNN architecturally prevented from using the cells CFAR treats as guard/CUT, so the learned rule
  is denied the mainlobe energy CFAR excludes;
* CFAR given an equivalent pre-integration step over the mainlobe, so the classical arm gets the
  energy the network can use.

**Purpose.** Separate "learned decision rule is better" from "the learned detector had access to
information the classical one discards".

### Priority 3 — Fixed-threshold false-alarm stability ("is it CFAR-like?")

**The question.** Every reported number uses a threshold chosen per method **and per condition** from
that condition's measured ROC. That is an evaluation oracle: a deployed detector cannot re-derive its
threshold per scene from labelled data. Holding a false-alarm rate constant under changing clutter
*without* that knowledge is precisely the property CFAR is named for, and nothing measured so far
tests whether the network has it.

**The experiment.** Fix one TinyCNN threshold, apply it unchanged across all conditions, and measure
how far the achieved Pfa wanders. Run the same protocol for each CFAR variant as the reference.

**Purpose.** Determine whether the network exhibits anything resembling constant-false-alarm
behaviour. Until this is done, **no claim of CFAR-like behaviour by the network is supported**.

**Note.** This experiment may need DL operating points well above Pfa 1e-4, where the `score > 0`
floor in `all_peaks` becomes relevant — the DL ROC cannot currently be swept past sigmoid 0.5. See
`evaluation_metrics_overview.ipynb` appendix A.2.

### Priority 4 — Secondary DL ablations

Only after Priorities 1–3. In no particular order:

* **receptive-field size** — is 21×9 near an optimum, or merely the CFAR-matched choice?
* **pooling / downsampling** — `TinyCNNPool` is implemented (5,041 params, measured RF 13×13) but has
  never been trained or scored. ⚠️ Its RF differs from TinyCNN's on both axes, so it is a control for
  "does downsampling displace peaks?" and **not** a valid arm of the CFAR-vs-DL comparison.
* **input normalization** — the current input is dB above the frame's own median power; alternatives
  and their effect on the global-statistic dependency.
* **occlusion sensitivity / interpretability**, quantitatively rather than by inspection.

**Explicitly not first: do not start with a U-Net or a larger CNN.** A high-capacity model would gain
from larger context, pooling, and capacity simultaneously, making any margin over CFAR
uninterpretable. The controlled small model is what makes the current comparison mean anything, and
an unattributed margin from a bigger model would inherit every open question above.

---

## Part B — CARRADA / real-data roadmap

Comes **after** the core synthetic attribution work. Recorded here so the order is not lost.

> **Scope reminder, to be repeated wherever this work is written up.** The simulator was configured
> to reproduce CARRADA's **published Range–Doppler grid geometry** — map dimensions and bin spacings
> — so that a target occupies a comparable number of cells. It does **not** reproduce CARRADA's
> waveform, scenes, clutter statistics, or sensor. Matching a grid is not matching a radar.

### R0 — Synthetic vs CARRADA domain audit *(before any detector runs)*

Compare simulated and real RD characteristics before asking how a detector performs, so that any
later performance gap can be attributed:

* RD geometry and physical axes
* target / mainlobe footprint and peak shape
* background amplitude distribution
* clutter structure, including the zero-Doppler ridge
* spatial correlation
* dynamic range
* the normalization statistics the network depends on (notably per-frame median power)

Also audit **what CARRADA's annotations actually support** before deciding which of Pd, Pfa and
localization can be reproduced at all. If the annotation format does not support a per-cell truth
bin, the localization metric used throughout this project does not transfer unchanged.

### R1 — Zero-shot synthetic → CARRADA

The first real-data detector experiment uses the **frozen synthetic TinyCNN with no fine-tuning**.

**Purpose.** Determine whether the learned rule transfers at all to real RD structure.

Keep the fully-frozen zero-shot result **distinct** from any later threshold calibration — once a
threshold is tuned on real data, the zero-shot claim can no longer be made from that run.

### R2 — Real-data failure analysis

Where do TinyCNN and CFAR succeed and fail on CARRADA, and do those failures match the mechanisms
identified in simulation (clutter-ridge false alarms, masking, localization spread)?

### R3 — Limited real-data adaptation

Only after the zero-shot result is recorded. Compare, if the data supports it:

* synthetic-only (frozen)
* synthetic pre-training + limited real fine-tuning
* real-only training, if there is enough labelled data to justify it

**Split by sequence, never by frame.** CARRADA has ~12,666 frames but only 30 sequences and 78
instances; frame-level splitting leaks badly.

### R4 — Mechanism replication on real data

Finally, test whether the findings that matter survive on real measurements:

* the detection / localization trade-off
* information-access effects (Priority 2)
* fixed-threshold behaviour (Priority 3)
* whatever spatial patterns the network is found to rely on

---

## Deliberately cut

Logged so they are not silently rediscovered. None changes a published number, and each costs
writing time that does not exist before the deadline: the `TinyCNNPool` control, the N2/N3 noise
studies, the Open/O4 localization asymmetry, Milestone-18 attribution, and the remaining proposal
ablations (RD resolution, DL model size, magnitude vs I/Q input).

## Standing rule (adopted 15 Aug 2026)

*A new finding goes on the limitations list, not the work queue, unless it changes a number already
published in `PROGRESS.md`.* Adopted because the project was drifting into open-ended fixing.
