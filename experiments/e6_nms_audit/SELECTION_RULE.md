# NMS radius selection rule — FROZEN BEFORE ANY MEASUREMENT

**Status: FROZEN.** This file is written and committed *before* Phase 1 is executed and before any
NMS sweep output exists. It must not be edited after the first Phase 1 run. If the rule turns out
to be wrong, that is recorded as a finding and the rule is superseded in a new file — never edited
in place.

Frozen at git commit: recorded in `MANIFEST.json` at execution time.

---

## What is being selected

A single non-maximum-suppression radius $n$, giving a $(2n+1) \times (2n+1)$ square suppression
neighbourhood (Chebyshev distance $n$), applied identically to both detector arms through
`src/cfar/detect.py::all_peaks`.

Candidates: **$n \in \{1, 2, 3, 5\}$**.

The incumbent is $n = 5$. **No radius is assumed to win, including $n = 2$.**

## Data used for selection

- **Selection split: `bench2_tune`** (400 frames, 50 per condition, seed 91001).
- **Confirmation split: `bench2_val`** (1,600 frames, 200 per condition, seed 93001).

The fresh holdout generated in Phase 2 is **not** used for selection and must not be scored until
the radius is frozen.

A radius is selected on `bench2_tune` alone. `bench2_val` is then used once, to confirm the
selection reproduces. If `tune` and `val` disagree on which radius passes, **the disagreement is
reported and escalated rather than resolved by picking the better-looking split.**

## Quantities measured, per radius

Measured on the frozen TinyCNN checkpoints (8 seeds, epoch 21) and on CA and OS CFAR, at matched
**measured** $P_{fa} = 10^{-4}$, primary $\pm 1$-bin gate.

| Group | Quantity |
|---|---|
| Candidate density | candidate peaks per **noise** frame |
| Duplicates | mean in-gate detections per matched truth (**absolute**) |
| Duplicates | fraction of matched truths with $\ge 2$ in-gate detections (**absolute**) |
| Duplicates | detections per target (**absolute**) |
| Duplicates | each of the above as a **ratio to the $n=5$ value** |
| Detection | `isolated` $P_d$ |
| Detection | `masking` $P_d$ |
| Detection | `unresolved` $P_d$ |
| Operating point | threshold at matched measured $P_{fa}$, and achieved $P_{fa}$ |

All quantities are reported for **every** radius including $n = 5$, in both absolute and relative
form. No quantity is reported only as a ratio.

## Gates

A radius is **admissible** only if it passes all four gates.

### A — Duplicate / free-attempt gate (absolute)

The evaluator ignores unmatched detections lying inside a truth's gate
(`src/eval/detection.py`), so a detector that emits several detections per target receives several
attempts at no cost. This gate bounds that.

- **A1.** Mean in-gate detections per matched truth $\le 1.15$.
- **A2.** Fraction of matched truths with $\ge 2$ in-gate detections $\le 0.15$.

*Justification for the 0.15 ceiling.* The smallest margin this project reports is $+0.012$ in $P_d$
(`unresolved`, Chapter 5). A duplicate mechanism affecting more than roughly one truth in seven is
capable of manufacturing a $P_d$ shift of that order, and would therefore be able to move a
reported conclusion. The ceiling is set before any value is observed.

### B — Duplicate gate (relative, reported but not decisive)

- **B1.** Both A1 and A2 quantities $\le 1.10 \times$ their $n = 5$ values.

**B1 is reported alongside A but is not sufficient on its own, and is not disqualifying on its
own.** A radius that passes A but fails B is not automatically rejected; the discrepancy is
reported and escalated for a human decision. A radius that fails A is rejected regardless of B.

This is a deliberate departure from a purely relative rule: $n = 5$ is the incumbent under
question, so anchoring the duplicate criterion solely to its behaviour would assume the answer.

### C — Isolated-target gate

- **C1.** TinyCNN `isolated` $P_d$ not more than $0.01$ below its $n = 5$ value.

Rationale: an isolated target has no neighbour to merge, so any $P_d$ loss there is attributable to
the candidate-density effect on the matched threshold rather than to merging.

### D — Operating-point stability gate

- **D1.** Achieved measured $P_{fa}$ within $5\%$ of $10^{-4}$ for every arm and every condition
  scored.

## Decision

Among admissible radii, select the **smallest**.

Tie-break, if two admissible radii are indistinguishable on A–D: prefer the one with higher
`masking` $P_d$, then higher `unresolved` $P_d$.

If **no** radius other than $n = 5$ is admissible, the incumbent stands and the audit reports that
the frozen protocol survives scrutiny.

If $n = 1$ is admissible on A but the absolute duplicate figures are large in plain terms — even
while passing the stated ceilings — that is reported explicitly with the raw numbers, and the
decision is escalated rather than taken automatically.

## Post-selection

Once selected, the radius is written to `configs/experiment.yaml`, hashed into `MANIFEST.json`, and
**frozen for every subsequent phase**, including the CARRADA re-evaluation. It is never re-tuned on
the fresh holdout and never re-tuned on CARRADA.

## Limitations accepted in advance

1. **Epoch 21 and `pos_weight` 100 were selected under $n = 5$.** They are held fixed. A different
   radius might imply a different optimal epoch; re-selecting it would require retraining and is
   out of scope. Recorded as a limitation of this audit.
2. **`bench2_tune` is small** (50 frames per condition). It was chosen because it is the least-used
   split, not because it is the most powerful. `bench2_val` confirmation exists for this reason.
3. **No bootstrap in this audit.** Differences are reported as raw deltas. Any delta smaller than
   the confidence intervals already published in Chapter 5 is to be described as *not established*,
   never as *unchanged*.
