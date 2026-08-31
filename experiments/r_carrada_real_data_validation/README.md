# CARRADA Real-Data External Validation

> **Phase 1 — isolation and setup. COMPLETE.**
> **No detector has been run. No model, configuration, or prior result has been modified.**

This folder is an **isolated external-validation experiment**. It is self-contained: everything it
produces lives under `experiments/r_carrada_real_data_validation/`. It reads the frozen synthetic
benchmark and the CARRADA dataset, and writes to neither.

---

## Purpose

Every number this project has published comes from one simulator. `bench2_test` and `bench2_val`
share a generator, so the headline result establishes **sampling generalization only** — not that any
finding survives a different clutter model, a different sensor, or real measurements.

This experiment asks whether the findings transfer to **real automotive radar measurements**
(CARRADA), and is the first work in the programme to use data this project did not generate.

## Scientific question

> Does the synthetic DL-vs-CFAR result — TinyCNN ahead on detection in interference, CFAR ahead on
> localization — reproduce on real measured Range–Doppler data, and do the mechanisms identified in
> E2/E3/E4 explain what happens there?

This decomposes into PLAN.md Part B:

| stage | question |
|---|---|
| **R0** | How far apart are the synthetic and real domains, measured before any detector runs? |
| **R1** | Does the frozen synthetic TinyCNN transfer zero-shot, with no fine-tuning? |
| **R2** | Where do TinyCNN and CFAR fail on real data, and do those failures match E5/E6 mechanisms? |
| **R3** | Does limited real-data adaptation help? Split by sequence, never by frame. |
| **R4** | Do the mechanism findings themselves survive on real measurements? |

## Current phase

**Phase 1 — isolation, repository setup, and data integrity. Complete.**

Phase 1 deliberately ran **no detector**, computed **no detection statistic**, and tuned **no
threshold**. Its entire job was to make the later phases auditable.

Next phase: **R0 — synthetic vs CARRADA domain audit**, which must complete before any detector
touches a real frame.

---

## Scope reminder — repeat this wherever the experiment is written up

The simulator was configured to reproduce CARRADA's **published Range–Doppler grid geometry** — map
dimensions and bin spacings — so that a target occupies a comparable number of cells. It does **not**
reproduce CARRADA's waveform, scenes, clutter statistics, or sensor.

**Matching a grid is not matching a radar.**

---

## Dataset

| | |
|---|---|
| location | `C:/Users/USER/Desktop/CARRADA/Carrada` (**read-only** for this experiment) |
| sequences | 30 |
| frames | 12,666 (7,193 annotated · 5,473 empty) |
| RD arrays | `range_doppler_numpy/` and `range_doppler_raw/`, both `(256, 64)` float64 |
| official split | `data_seq_ref.json`, mirrored to `configs/official_splits.json` |
| metadata backup | `data/carrada_meta/` (116 MB, project-local, gitignored) |
| verification | `results/phase1_dataset_verification.json`, `logs/phase1_verify_carrada.log` |

The dataset was reduced from 95.86 GB to 3.52 GB before this experiment began. That cleanup is
recorded in `PROGRESS.md`; **Phase 1 did not repeat it and must not.** Removed: `annotations/dense`
and `annotations/sparse` (proven byte-identical to `annotations_frame_oriented.json` on 360/360
samples), all range-angle and angle-Doppler views, `range_doppler_processed`, DoA and tracking data,
and all but 360 camera images.

### Representation facts — established prior information, not to be re-derived in Phase 1

Carried in from the pre-Phase-1 audit and recorded here so R0 starts from a stated position:

* `range_doppler_numpy` is the **intended primary representation**.
* `_raw` behaves as an **amplitude**.
* `_numpy` is consistent with **power in dB up to a constant scale**.
* The CFAR conversion, when it is eventually applied, is proportional to **`P = 10^(numpy / 10)`**.
  CFAR is a ratio test, so the unknown constant cancels.
* `_numpy` is the representation **CARRADA's own annotation pipeline used**, so the annotations are
  in its coordinates.
* `_raw` is **retained temporarily as an independent cross-check** and must not be deleted yet.

The full statistical and domain audit belongs to **Phase 2 (R0)** and was not repeated here.

---

## Annotation terminology

Four different quantities were previously being called "instances". They are now fixed, and each is
reconciled against at least one independent source:

| quantity | count | what it means |
|---|---|---|
| total frames | 12,666 | every frame with an RD array on disk |
| annotated frames | 7,193 | frames containing ≥ 1 object |
| empty frames | 5,473 | frames with no object — usable as false-alarm-only frames |
| **frame-level object instances** | **8,750** | Σ over frames of objects present ("object-frames") |
| **unique tracks** | **77** | distinct physical objects across the dataset |

**The previously quoted "84" is not an object count.** It is the raw sum of the `instances` list
lengths in `data_seq_ref.json`, which contains 2 duplicate entries and 5 IDs that were merged into
other instances during annotation:

```
84  raw sum of data_seq_ref 'instances' lengths
-2  duplicate entries  (001109 twice in 2020-02-28-13-08-51; 001143 twice in 2020-02-28-13-12-42)
82  unique listed instance ids
-5  merged away by instance_exceptions.json
77  unique annotated tracks
```

77 is confirmed independently by both `annotations_frame_oriented.json` and
`annotations_instance_oriented.json`.

⚠️ **`PLAN.md` R3 says "78 instances". That is incorrect; the value is 77 unique tracks.** PLAN.md is
a protected file and was not edited.

**Use the right denominator.** R1 detection statistics are per **object-frame** (8,750), not per
track (77) and not per frame (12,666).

| class | object-frames | unique tracks |
|---|---|---|
| pedestrian | 3,239 | 18 |
| cyclist | 1,770 | 16 |
| car | 3,741 | 43 |
| **total** | **8,750** | **77** |

Objects per frame: 0 → 5,473 · 1 → 5,636 · 2 → 1,557. **Never more than two.**

## Official splits — sequence-level, used as given

Never re-split, and never split by frame. Mirrored to `configs/official_splits.json`.

| split | sequences | frames | annotated | empty | object-frames | tracks |
|---|---|---|---|---|---|---|
| Train | 18 | 8,088 | 4,319 | 3,769 | 5,318 | 49 |
| Validation | 6 | 2,448 | 1,483 | 965 | 1,804 | 15 |
| Test | 6 | 2,130 | 1,391 | 739 | 1,628 | 13 |
| **total** | **30** | **12,666** | **7,193** | **5,473** | **8,750** | **77** |

Every sequence is assigned exactly one split, and the three sets are disjoint.

---

## Protected asset policy

The frozen synthetic benchmark is **immutable for the duration of this experiment.**

`isolation_manifest.json` records the SHA-256, size, and protection category of **628 files
(2.88 GB)** spanning: the 8 frozen TinyCNN checkpoints and their run configs, all DL / CFAR / eval /
radar / data source, `frozen_dl.yaml`, `cfar_alpha_table.json`, the dataset configs and caches, the
headline result tables, the CFAR baseline outputs, and the complete E2 / E3 / E4 experiment outputs.

To re-check at any time:

```bash
python experiments/r_carrada_real_data_validation/scripts/build_isolation_manifest.py --verify experiments/r_carrada_real_data_validation/isolation_manifest.json
```

`PROGRESS.md` is categorised **`project_record_append_only`** — appending a dated entry is the
project's normal workflow, so a hash change there is expected and legitimate. Every other category
must hash identically. **If any of them changes, stop and report rather than continuing.**

## Layout

```
configs/    official_splits.json           sequence-level split, mirrored from CARRADA
scripts/    build_isolation_manifest.py    build + --verify the protected-asset manifest
            verify_carrada.py              integrity, terminology, split verification
results/    phase1_dataset_verification.json
logs/       phase1_verify_carrada.log
figures/    (empty - Phase 1 produced no figures)
src/        (empty - Phase 1 wrote no experiment code)
```

## Open methodological questions — resolve before Phase 2

1. ~~**Folder ID collision.**~~ **RESOLVED 31 Aug 2026.** The folder was briefly named
   `e5_carrada_real_data_validation`, which collided with the **E5** that `PLAN.md` reserves for the
   unbuilt Part-A four-way comparison. It is now `r_carrada_real_data_validation`: the `r_` prefix
   marks the **real-data / external-validation track**, distinct from the E-numbered synthetic
   ablation sequence. The folder is not called `R1`, because it holds **both R0 and R1** and R1
   names only the later detector-evaluation stage. `PLAN.md` was not edited.

2. **CARRADA truth is a segmentation blob, not a point target.** Median car footprint is 14 × 9
   bins. The project's ±1-bin gate, one-to-one matching, and localization-error-in-bins all assume
   one true cell per target. A truth-point definition (box centre / sparse centroid / brightest cell
   in mask) must be **chosen and pre-registered before any R1 number exists**, because it is a
   definition this project invents rather than something CARRADA supplies.

3. **Pfa is not the same quantity on CARRADA.** Only the 77 tracked objects are annotated; real
   scenes contain unannotated reflectors, so peaks outside the masks include genuine targets.
   Matched-Pfa comparison *across* the synthetic and real domains is not valid and must not be
   presented as one.

4. **No close-target regime exists.** Frames contain at most 2 objects, so `multi`, `masking` and
   `unresolved` — where the E2/E4 mechanisms live — have no real-data counterpart.
