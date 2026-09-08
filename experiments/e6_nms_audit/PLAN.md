# E6 — Fast NMS re-evaluation audit

**Status: PLANNED, NOT EXECUTED.** Nothing in this experiment has been run. No dataset, detector
output, figure or result exists yet. This document is the handoff: it is written so the experiment
can be executed later without reconstructing the reasoning.

**Execution begins only on explicit instruction from the user.**

---

## 1. Why

The frozen protocol uses NMS radius 5 (an $11 \times 11$ suppression neighbourhood). Two things
prompted this audit.

1. The justification on record (`PROGRESS.md` Step 6d) is that a smaller radius admits far more
   noise candidates, which at matched measured $P_{fa}$ forces the threshold up and **costs $P_d$
   everywhere, including on `isolated`**, where nothing can be merged. That table reports isolated
   $P_d$ of 0.291 at $n=2$ against 0.521 at $n=5$.

2. A later development-set sweep (8 Sep 2026, `results/dl/nms_sweep.csv`, run on `bench2_val` with
   the development pw100 checkpoints) **confirmed the mechanism but not the consequence**:

   - candidate peaks per noise frame: 845 / 425 / 248 / 164 / **115** / 66 at $n=1,2,3,4,5,7$ —
     the "~4× more at $n=2$" claim is exactly right;
   - but `isolated` $P_d$ is **flat**: TinyCNN 0.729 / 0.730 / 0.730 / 0.731 / 0.731 / 0.731,
     CA-CFAR 0.710 at every radius.

   Meanwhile `masking` moves strongly the other way (TinyCNN 0.804 at $n=1$ down to 0.697 at
   $n=5$), and `unresolved` shows the free-attempts signature only at $n=1$ (1.22 detections per
   target against 1.15 elsewhere).

The discrepancy between (1) and (2) is unexplained. Step 6d predates the Step 7 dataset generation,
and `PROGRESS.md` separately records a period in which "the geometry sweep had silently inherited
`nms_size = 2`", so a confound is plausible but unproven. **This audit does not attempt to
reconstruct what Step 6d measured.** It asks a narrower question:

> Does changing the NMS radius from 5 to a smaller value materially change the main synthetic and
> CARRADA conclusions?

## 2. Scope — deliberately narrow

**In scope:** radius selection on development data; one fresh synthetic holdout; core Chapter 5
outputs at the selected radius and at a radius-5 control; core Chapter 7 outputs at the selected
radius; a conclusion-change audit.

**Explicitly out of scope at this stage:** the 2,000-resample bootstrap · all Chapter 5 and
Chapter 7 figures · attribution regression (E2/E3/E4) · any retraining · a new SNR-sweep split ·
any edit to Chapters 4–7 or the General Conclusion.

Those follow **only** if the audit shows they are needed, and only on separate approval.

## 3. Decisions taken by the user

| Question | Decision |
|---|---|
| Selection split | **`bench2_tune`**, confirmed on **`bench2_val`** |
| Radius-5 control on the fresh holdout | **Include** |
| Holdout size | **3,200 frames** |
| NMS=1 duplicate assessment | Report **absolute** free-attempt behaviour as well as the ratio to $n=5$. **Do not accept or reject on the relative 1.10× rule alone** |
| Timing | Freeze the selection rule first; maintain full isolation |

The selection rule is frozen in `SELECTION_RULE.md`, written and committed before any measurement.

## 4. Isolation contract

Everything created lives under `experiments/e6_nms_audit/`. Nothing outside it is written.

**Must not be modified:** `results/**` · `data/bench2_*/**` · `configs/**` · all
`results/dl/tinycnn_frozen_*.pt` · `experiments/r_carrada_real_data_validation/**` (628 files under
its own isolation manifest) · `report/**` · any existing script.

**Existing code is imported read-only.** Where a script hard-codes an output path it is **copied**
into this folder and the path parameterised in the copy. The originals are never edited.

Scripts requiring a wrapped copy, confirmed by audit:

| Script | Hard-coded path |
|---|---|
| `scripts/nms_sweep.py` | `ROOT/"results"/"cfar_val"`, `ROOT/"results"/"dl"` |
| `scripts/bootstrap_comparison.py` | same (scoring helpers reused, bootstrap not run) |
| `scripts/localization_table.py` | same |
| CARRADA `src/detectors.py` | module constant `NMS_SIZE = 5`; file is manifest-protected |

`src/data/create_dataset.py` needs no wrapper — `output_root` and `random_seed` are read from the
YAML config.

`all_peaks(ratio, valid, nms_size=...)` already takes the radius as an argument, so no core
detection code requires modification.

## 5. Phases

### Phase 0 — Setup

Create the folder structure, wrapper scripts, `configs/holdout.yaml`, and `MANIFEST.json` recording
the git commit, the SHA-256 of `SELECTION_RULE.md`, the frozen checkpoint hashes, and the
environment. Verify the CARRADA isolation manifest passes **before** starting.

### Phase 1 — Selection sweep (development data only)

`bench2_tune` first (400 frames), then `bench2_val` (1,600 frames) to confirm.

- Radii $\{1, 2, 3, 5\}$.
- Conditions: `isolated`, `masking`, `unresolved`, plus `noise` for candidate density.
- Arms: 8 frozen TinyCNN seeds, CA-CFAR and OS-CFAR.
- Score maps computed **once per frame**, then `all_peaks` applied at each radius — this is the
  main cost saving and is how `nms_sweep.py` already works.
- Every quantity listed in `SELECTION_RULE.md`, in absolute and relative form.

Apply the frozen rule. Record the outcome, including the escalation cases the rule defines.

**Output:** `results/p1_selection/`. **Gate: the radius is frozen here and never revisited.**

### Phase 2 — Fresh synthetic holdout

3,200 frames, 8 strata × 400, **seed 96001** (next in the established 9x001 series; existing seeds
are 90001 debug, 91001 tune, 92001 train, 93001 val, 94001 test, 95001 snr_sweep).

`configs/holdout.yaml` is a copy of `configs/bench2_test.yaml` with only `name`, `random_seed` and
`output_root` changed; the diff is recorded in the manifest.

**Overlap check:** content-hash every generated frame and assert zero collisions against all five
existing splits. Seeds are not trusted on their own — the same discipline Step 7 used.

**Output:** `data/holdout_nms/`.

### Phase 3 — Fast synthetic rebenchmark

Score the fresh holdout twice: at the **selected radius** and at **radius 5** (the control). Score
maps are computed once and both radii applied, so the control costs almost nothing.

- All four CFAR variants; all 8 frozen TinyCNN seeds; no retraining.
- Per-condition $P_d$ at matched measured $P_{fa}$.
- Achieved $P_{fa}$ per arm and condition.
- $\pm 1$ and $\pm 3$ gates.
- Basic localisation summary (range, Doppler, RMSE), pooled and per condition.
- **No bootstrap. No figures.**

**Output:** `results/p3_synth/`.

The radius-5 control on the same fresh frames is what separates *protocol change* from *sampling
change*, giving a 2×2: {old radius, new radius} × {old frames, new frames}. The old-frames row is
read from existing artefacts and is not recomputed.

### Phase 4 — Fast CARRADA re-evaluation

**Mandatory first step:** re-derive the TinyCNN zero-shot thresholds at the selected radius **from
`bench2_val` alone**. `run_r1_test.py` records that the frozen thresholds were "derived from
bench2_val alone ... NMS radius 5", so they are radius-dependent. Skipping this would silently
mismatch the operating point. **No CARRADA data is used to set any threshold.**

Then, on the frozen CARRADA test sequences, with no retraining, no fine-tuning and no CARRADA-based
tuning of anything:

- overall TinyCNN vs CA-CFAR dense-mask object recall;
- per-sequence direction (does CA still lead in all 6?);
- exact / $\pm 1$ / $\pm 3$ tolerance ladder;
- near-zero-Doppler vs away-from-zero strata;
- unmatched detections per frame;
- 8-seed spread.

**Output:** `results/p4_carrada/`. The original CARRADA experiment folder is read-only throughout;
its isolation manifest is re-verified afterwards.

### Phase 5 — Conclusion audit

One table. For each conclusion, classify as **UNCHANGED** · **NUMERICALLY CHANGED, SAME
INTERPRETATION** · **MATERIALLY CHANGED** · **REVERSED**.

Rows: clutter-edge advantage · isolated · masking · unresolved · homogeneous clutter · localisation
penalty · CARRADA ordering · zero-Doppler concentration · unmatched-detection burden.

**Any delta smaller than the confidence intervals already published in Chapter 5 is described as
*not established*, never as *unchanged*.** This is the principal cost of skipping the bootstrap and
must be stated in the audit output.

**Output:** `results/p5_audit/audit.md`.

### Phase 6 — Stop

Report: the selected radius and why; whether the main scientific story changes; and an estimate of
whether a full rerun is warranted. **Then stop and wait.**

## 6. Runtime

Per-frame costs measured in `results/dl/runtime_cost_frozen_ep021_test.csv`: CFAR all four
**86.4 ms**, TinyCNN 8 seeds **48.3 ms** (6.04 ms per seed), CA alone 0.86 ms, OS alone 69.95 ms.

| Phase | Estimate | Basis |
|---|---|---|
| 0 Setup | 20–30 min | wrapper authoring, human-paced |
| 1 Selection (`tune` + `val`) | 20–30 min | 2,000 frames of maps, 4 radii applied after |
| 2 Holdout generation | 10–40 min ⚠️ | **least certain** — no recorded generation rate; time 50 frames first and extrapolate |
| 3 Fast rebenchmark | 20–30 min | 3,200 × 134.7 ms = 7.2 min of maps + scoring at 2 radii |
| 4 CARRADA | 20–30 min | threshold re-derivation on val + 2,130 frames × 12 arms |
| 5 Audit table | 10 min | no figures |

**Total 1.5–2.5 hours**, roughly 1–1.5 h of it unattended.

**Sequential:** 1 → *freeze radius* → 2 → 3, and 1 → 4.
**Parallel:** Phases 3 and 4 may run concurrently once the radius is frozen.

**Disk: ~0.8 GB** — holdout 0.72 GB (measured: `bench2_test` is 719 MB for 3,200 frames) plus
~0.1 GB of outputs.

## 7. What is reused without recomputation

| Artefact | Serves as |
|---|---|
| `results/dl/bootstrap_frozen_ep021_test_tol{1,3}.csv` | old $P_d$ per condition, both gates |
| `results/dl/localization_frozen_ep021_test.csv` | old localisation |
| `results/dl/measured_pfa_frozen_ep021_test_tol1.csv` | old achieved $P_{fa}$ |
| `experiments/r_carrada.../results/r1_test/**` | **all** old CARRADA numbers |
| `configs/cfar_alpha_table.json` | alphas are **per-cell and pre-NMS**, provably unaffected — no recalibration needed |

**Not reusable:** every `peaks_*.npz` cache in `results/`, because they are already NMS-applied at
radius 5.

The old NMS=5 column of every comparison therefore comes from frozen artefacts. Only the new column
and the fresh-holdout control are computed.

## 8. Risks

1. **No bootstrap means no significance.** Deltas only. Mitigated by comparing against Chapter 5's
   published intervals and by the *not established* wording rule above.
2. **Two variables change at once** (radius and frames) — mitigated by the radius-5 control on the
   fresh holdout.
3. **`bench2_tune` is small** (50 frames per condition). Chosen as the least-used split rather than
   the most powerful; `bench2_val` confirmation exists for this reason. If they disagree, escalate.
4. **Epoch 21 and `pos_weight` were selected under radius 5.** Held fixed; a stated limitation.
5. **CARRADA thresholds are radius-dependent** and must be re-derived from `bench2_val` before any
   CARRADA scoring.
6. **A negative result is a valid outcome.** If radius 5 survives, the audit reports that the frozen
   protocol withstands scrutiny. Nothing in this plan presumes a change is warranted.
7. **Chapter 4 currently contains an unsupported sentence** — that radius 5 "performed substantially
   better on isolated targets". The 8 Sep sweep contradicts it. It is left standing pending this
   audit **by explicit decision**; if the audit confirms the insensitivity, that sentence must be
   corrected regardless of which radius is selected.

## 9. Open items at execution time

- Time 50 frames of generation before committing to the Phase 2 estimate.
- Re-verify the CARRADA isolation manifest before Phase 4 and after Phase 6.
- Record every command, artefact hash and timestamp in `MANIFEST.json`.
