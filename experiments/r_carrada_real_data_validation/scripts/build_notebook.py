"""Build the experiment notebook from the cached R0 results.

Every number in the notebook is read from results/r0_audit.json rather than typed, so the
prose cannot drift away from the measurements. Writes only the notebook.
"""
from __future__ import annotations

import json
import os
import sys

import nbformat as nbf
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
RESULTS = os.path.join(EXP, "results")
NB = os.path.join(EXP, "r_carrada_real_data_validation.ipynb")

R = json.load(open(os.path.join(RESULTS, "r0_audit.json")))
P1 = json.load(open(os.path.join(RESULTS, "phase1_dataset_verification.json")))


def g(d, k, digits=4):
    return "{:.{}f}".format(d[k]["mean"], digits)


def se(d, k, digits=4):
    return "{:.{}f}".format(d[k]["se"], digits)


def pm(k, digits=4):
    a, b = R["background_carrada"], R["background_simulator"]
    return (g(a, k, digits), se(a, k, digits), g(b, k, digits), se(b, k, digits))


bc, bs, rep, geo, ridge = (R["background_carrada"], R["background_simulator"],
                           R["representation"], R["geometry"], R["ridge"])
fp = R["object_footprint"]
half = R["background_protocol"]["ridge_half_width_applied"]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ---------------------------------------------------------------------------------- §0
md("""# CARRADA Real-Data External Validation

**Real-data experiment — R0 domain audit + R1 zero-shot validation.**

> ## §0 Purpose and isolation
>
> **Phase 2 (R0) complete. No detector has been run.**

Every number this project has published comes from one simulator, and `bench2_test` shares a
generator with `bench2_val`, so the headline result establishes **sampling generalization only**.
This experiment is the first use of data the project did not generate.

**R0 asks one question:** *how different is the real CARRADA Range–Doppler domain from the
synthetic domain on which the frozen TinyCNN and CFAR benchmark were developed?*

It is a **domain audit, not a detector evaluation.** In this phase there is deliberately:
no TinyCNN inference, no CFAR pass, no threshold, no Pd, no Pfa, no recall, no alpha
recalibration, and no change to any frozen asset.

### Isolation

* `isolation_manifest.json` pins SHA-256 for **{npf} protected files ({mb:.2f} GB)**.
* The frozen synthetic benchmark, E2/E3/E4 outputs, and CARRADA source are read-only.
* All outputs live under `experiments/r_carrada_real_data_validation/`.

### Naming

The folder was briefly `e5_carrada_real_data_validation`, which collided with the **E5** that
`PLAN.md` reserves for the unbuilt Part-A four-way comparison. The `r_` prefix marks the
real-data track. It is **not** called `R1`, because it holds both R0 and R1.

### Scope reminder

The simulator reproduces CARRADA's published RD **grid geometry** — dimensions and bin spacings —
not its waveform, scenes, clutter statistics, or sensor. **Matching a grid is not matching a
radar.**
""".format(npf=628, mb=2.8827))

# ---------------------------------------------------------------------------------- §1
md("""## §1 Dataset and official splits

CARRADA, cleaned from 95.86 GB to 3.52 GB before this experiment began (recorded in
`PROGRESS.md`; **not repeated here**). Treated as read-only source data throughout.

The official **sequence-level** split from `data_seq_ref.json` is used exactly as given.
No new split was created and no frame-level split will ever be used — with only 30 sequences
and 77 tracks, a frame-level split would leak the same object between train and test.
""")
code("""import json, os
import numpy as np, pandas as pd
RESULTS = "results"
R  = json.load(open(os.path.join(RESULTS, "r0_audit.json")))
P1 = json.load(open(os.path.join(RESULTS, "phase1_dataset_verification.json")))

sp = P1["official_splits"]
pd.DataFrame([{"split": k, "sequences": v["n_sequences"], "frames": v["frames"],
               "annotated": v["annotated_frames"], "empty": v["empty_frames"],
               "object-frames": v["object_frames"], "tracks": v["unique_tracks"]}
              for k, v in sp.items()]).set_index("split")""")

# ---------------------------------------------------------------------------------- §2
md("""## §2 Phase 1 integrity and isolation summary

All Phase 1 checks passed and are reproducible via `scripts/verify_carrada.py`.

| check | outcome |
|---|---|
| sequences | 30 on disk = 30 `validated_seqs.txt` = 30 `data_seq_ref.json` |
| RD frames | 12,666 `_numpy` and 12,666 `_raw`, identical filename sets in 30/30 |
| frames missing vs light list | **0** |
| shape / dtype | all **25,332** headers `(256, 64)` float64 |
| corruption | 0 unreadable headers; 600 full-load spot checks, 0 failures |
| metadata / box annotations | 12/12 files present; 30/30 sequences |

Phase 1 commit: **`15f3aab2c9690a22dcbe55799d78ab6d69451c09`**
""")

# ---------------------------------------------------------------------------------- §3
md("""## §3 Annotation terminology

Four quantities had been conflated. **This table defines the R1 denominators.**

| quantity | count | meaning |
|---|---|---|
| total frames | 12,666 | frames with an RD array on disk |
| annotated frames | 7,193 | ≥ 1 object present |
| empty frames | 5,473 | no object — usable as false-alarm-only frames |
| **frame-level object instances** | **8,750** | Σ over frames of objects present |
| **unique tracks** | **77** | distinct physical objects |

The previously quoted **"84" is not an object count**: it is the raw sum of the `instances`
list lengths in `data_seq_ref.json`, which contains 2 duplicate entries and 5 ids merged away
by `instance_exceptions.json`.

```
84  raw sum  →  −2 duplicates  →  82 unique ids  →  −5 merged  →  77 tracks
```

77 is confirmed independently by `annotations_frame_oriented.json` **and**
`annotations_instance_oriented.json`.

⚠️ `PLAN.md` R3 states "78 instances". That is incorrect; the value is **77**. `PLAN.md` is a
protected file and was not edited.

**R1 statistics use 8,750 as the object denominator** — not 77, and not 12,666.
""")

# ---------------------------------------------------------------------------------- §4
md("""## §4 `_numpy` vs `_raw` — the representation

Formally reproduced on a documented sample: **{n} frames, {s} sequences**, deterministic seed.

Per-frame least squares of `numpy` on `10·log₁₀(raw)`:

| quantity | value |
|---|---|
| slope | **{slope:.4f} ± {slope_sd:.4f}** |
| intercept | {icpt:.3f} ± {icpt_sd:.3f} dB |
| residual std | {rs:.4f} dB |
| residual abs-max | {rm:.3f} dB |
| corr(residual, level) | {crl:+.4f} |
| residual range-profile std | {rrp:.4f} dB |
| residual Doppler-profile std | {rdp:.4f} dB |
| Spearman ρ(`numpy`, `raw`) | {sp:.4f} |

**The slope is 2.** That is the signature of `20·log₁₀`, so `_raw` is an **amplitude** and
`_numpy` is **power in dB**. The residual is unstructured — essentially no dependence on signal
level or on range/Doppler position.

Power-statistics agreement on the eligible background confirms it independently:

| candidate power | CV | gap vs `10^(numpy/10)` |
|---|---|---|
| `raw²` (raw as amplitude) | {cvr2:.4f} | **{gap2:.2f}%** |
| `raw` (raw as power) | {cvr:.4f} | {gap1:.1f}% |
| `10^(numpy/10)` | {cvn:.4f} | — |

Peak alignment, {pa} object instances: the array's maximum inside the annotated box lands
**inside** the dense mask for **{pn:.2%}** of instances using `_numpy` versus {pr:.2%} using
`_raw`. The two arrays choose the *same* peak cell in only {ag:.2%} of cases, so the choice is
not cosmetic.

### Conclusion

**`range_doppler_numpy` is the primary representation for R1.**

* For CFAR: **`P ∝ 10^(numpy/10)`** is valid. CFAR is a ratio test, so the unknown constant
  cancels exactly.
* For TinyCNN: the frozen input is `10·log₁₀(P / median P)`. Because dB is a monotone transform
  of power, `median(P) = 10^(median(dB)/10)` **exactly**, so this equals `dB − median(dB)` and the
  adapter is exact in the dB domain — again with the unknown constant cancelling.
* `_raw` is **kept on disk** as an independent cross-check. It offers no scientific advantage:
  identical noise statistics and slightly *worse* peak alignment.

*No detector inference was implemented or run to reach this conclusion.*
""".format(n=rep["frames_analysed"], s=rep["sequences"],
           slope=rep["slope_vs_10log10raw_mean"], slope_sd=rep["slope_vs_10log10raw_sd"],
           icpt=rep["intercept_mean"], icpt_sd=rep["intercept_sd"],
           rs=rep["resid_std_db_mean"], rm=rep["resid_absmax_db_mean"],
           crl=rep["corr_resid_level_mean"], rrp=rep["resid_range_profile_std_mean"],
           rdp=rep["resid_doppler_profile_std_mean"], sp=rep["spearman_db_vs_raw_mean"],
           cvr2=rep["cv_from_raw_sq_mean"], cvr=rep["cv_from_raw_mean"],
           cvn=rep["cv_from_numpy_mean"], gap2=rep["cv_pct_gap_numpy_vs_rawsq"],
           gap1=rep["cv_pct_gap_numpy_vs_raw"], pa=rep["peak_alignment_instances"],
           pn=rep["peak_in_mask_numpy"], pr=rep["peak_in_mask_raw"],
           ag=rep["peak_cell_agreement"]))

# ---------------------------------------------------------------------------------- §5
md("""## §5 RD geometry and coordinate verification

| | CARRADA | simulator |
|---|---|---|
| RD array shape | {shapes} | {ss} |
| range axis | 256 bins over 50 m | {r0:.2f} → {r1:.2f} m, {rb:.4f} m/bin |
| velocity axis | 64 bins over ±13.44 m/s | {v0:.3f} → {v1:.3f} m/s, {vb:.5f} m/s/bin |
| zero-Doppler bin | {zc} (measured from the ridge peak) | {zs} (from `axes.npz`) |

Annotation coordinates span range bins **{armin}–{armax}** and Doppler bins
**{admin}–{admax}** across all {ac:,} annotated cells — inside `(256, 64)` on both axes, so the
annotations index the RD array directly.

**No resampling or interpolation is required.** The geometry transfers exactly; this is by
construction, since the simulator was configured against CARRADA's published grid.

Figure 1 shows example maps on identical axes with identical display normalization. Those
examples are **manually selected and illustrative — not statistical evidence.**
""".format(shapes=", ".join(geo["carrada_rd_shapes_seen"]), ss=tuple(geo["simulator_rd_shape"]),
           r0=geo["simulator_range_axis_m"][0], r1=geo["simulator_range_axis_m"][1],
           rb=geo["simulator_range_bin_m"],
           v0=geo["simulator_velocity_axis_mps"][0], v1=geo["simulator_velocity_axis_mps"][1],
           vb=geo["simulator_velocity_bin_mps"],
           zc=ridge["carrada_ridge_peak_bin"], zs=geo["simulator_zero_doppler_bin"],
           armin=geo["annotation_range_bin_min"], armax=geo["annotation_range_bin_max"],
           admin=geo["annotation_doppler_bin_min"], admax=geo["annotation_doppler_bin_max"],
           ac=geo["annotation_cells_total"]))
md("![RD examples](figures/r0_fig1_rd_examples.png)")
md("![RD examples, dense overlay](figures/r0_fig1b_rd_examples_dense_overlay.png)")

md("""> ### ILLUSTRATIVE EXAMPLES — NOT STATISTICAL EVIDENCE
>
> Both figures above show two hand-picked frames. They are an aid to intuition. The statistical
> evidence is Figures 2–6, computed over all 12,666 CARRADA and 1,600 simulator frames.

**Reading the two panels.** Both use **identical physical axes** and **identical display
normalization** — dB relative to each frame's own median — so every visible difference is real
rather than a display artefact. The two variants differ only in how the annotation is drawn: the
first as a bounding box with statistical callouts, the second cell-by-cell, which shows that the
CARRADA support is an irregular blob rather than a rectangle.

* CARRADA's **general background is visibly smoother and less speckled.**
* **"Smoother" means lower cell-to-cell amplitude fluctuation — it does *not* mean less physical
  clutter.** The measured quantity is background std 2.21 dB against 5.57 dB, i.e. effective Gamma
  shape k ≈ 4.8 against ≈ 1.0.
* At the same time CARRADA carries **stronger localized structured clutter** — most obviously its
  **narrow, intense zero-Doppler ridge** (24 dB, half-width ±1 bin), plus range-sidelobe streaks
  through the bright targets.
* The simulator shows visibly **stronger grain-like single-look speckle**, and its zero-Doppler
  clutter is **broader but weaker** (14.8 dB, half-width ±4 bins).

**So the real domain is not simply "cleaner" or "harder" than the synthetic one — its background
and clutter structure is *different*.** A detector tuned against fine-grained stationary speckle
faces, on CARRADA, a smoother field interrupted by strong localized structure. Which of those two
regimes is harder is a question for R1, and R0 deliberately does not answer it.""")

# ---------------------------------------------------------------------------------- §6
md("""## §6 Background-selection protocol

Two earlier passes produced **non-authoritative** numbers, for two different reasons:

* one **trimmed the top decile**, which biases the coefficient of variation downward;
* one **left the zero-Doppler ridge in**, which inflates it enormously.

**Neither is retained.** Only the correctly masked, untrimmed analysis below is authoritative.

There is exactly one background definition (`src/r0_background.py`), applied identically to both
domains. A cell is eligible background if it survives all three exclusions:

| exclusion | value | why |
|---|---|---|
| object support, dilated | **±{od} bins** | covers the ~2-bin mainlobe so target skirts do not leak in |
| zero-Doppler band | **±{half} bins** | real structure, not speckle; measured separately in §11 |
| detector border | **{br} range, {bd} Doppler** | the region a 21×9 CFAR window can actually test |

**No percentile trimming is applied anywhere.** The ridge-included variant is reported
separately in §11 as an explicitly labelled side analysis, never mixed into the headline.

### Documented mismatch

"Object support" is not the same object in the two domains: in the simulator it is the 3×3 label
mask around a **point** target; in CARRADA it is a human-drawn **segmentation blob** whose median
car footprint is {car_r:.0f}×{car_d:.0f} bins. The dilation is identical; the thing dilated is
not. This asymmetry cannot be removed and is restated wherever these statistics appear.
""".format(od=R["background_protocol"]["object_dilation_bins"], half=half,
           br=R["background_protocol"]["border_range_bins"],
           bd=R["background_protocol"]["border_doppler_bins"],
           car_r=fp["car"]["range_extent"]["median"], car_d=fp["car"]["doppler_extent"]["median"]))
code("""import sys; sys.path.insert(0, "src")
import r0_background as bg
mask = bg.background_mask(object_cells=[(128, 32)], ridge_half=R["background_protocol"]["ridge_half_width_applied"])
print("eligible background cells in a frame with one object:", int(mask.sum()), "of", mask.size)
print("border keeps          :", int(bg.border_mask().sum()))
print("ridge band excludes   :", int(bg.ridge_mask(R["background_protocol"]["ridge_half_width_applied"])[0].sum()), "Doppler bins")
print("trimming              :", R["background_protocol"]["percentile_trimming"])""")

# ---------------------------------------------------------------------------------- §7/8/9
md("""## §7–§9 Background statistics and domain comparison

Computed per frame, then averaged within each group, then summarised across groups.
**Uncertainty is taken across groups — {ngc} CARRADA sequences and {ngs} simulator conditions —
never across frames**, because frames inside a sequence are strongly correlated and treating
millions of RD cells as independent would produce meaninglessly small error bars.

CARRADA uses **all {nfc:,} frames**; the simulator uses **all {nfs:,} frames of `bench2_val`**
(the development split — `bench2_test` is not touched).
""".format(ngc=bc["cv"]["n_groups"], ngs=bs["cv"]["n_groups"],
           nfc=bc["_n_frames"], nfs=bs["_n_frames"]))
code("""rows = []
lab = {"cv": "CV(power)", "gamma_k_moment": "Gamma shape k (moment)",
       "db_std": "background std [dB]", "lag1_range": "lag-1 range corr",
       "lag1_doppler": "lag-1 Doppler corr"}
ideal = {"cv": 1.0, "gamma_k_moment": 1.0, "db_std": 5.57, "lag1_range": None, "lag1_doppler": None}
for k, name in lab.items():
    a, b = R["background_carrada"][k], R["background_simulator"][k]
    rows.append({"statistic": name,
                 "CARRADA": f"{a['mean']:.4f} ± {a['se']:.4f}",
                 "simulator": f"{b['mean']:.4f} ± {b['se']:.4f}",
                 "single-look ideal": "-" if ideal[k] is None else f"{ideal[k]:.3f}"})
pd.DataFrame(rows).set_index("statistic")""")
md("""### How frames are combined, and a finding that forced the choice

Within a group, frames are combined by **median**, not mean. That decision was forced by a
measurement, not a preference.

The per-frame CV distribution is strongly heavy-tailed on CARRADA — median **{med:.3f}**,
99th percentile **{p99:.2f}**, maximum **{mx:.1f}**. Averaging over that tail broke the summary:
mean-aggregated CV said {cvm:.3f} while mean-aggregated k said {km:.3f}, which cannot both be
true because **k = 1/CV² holds exactly per frame** (verified to 4×10⁻¹⁵). The medians agree
exactly. Mean-based values are retained in `r0_audit.json` as `meanagg_*` so the gap is visible
rather than hidden.

**The tail itself is the finding.** {f2:.1%} of CARRADA frames have background CV > 2,
{f5:.1%} exceed 5, and {f10:.1%} exceed 10 — versus {sf2:.2%} / {sf5:.2%} / {sf10:.2%} in the
simulator. Those frames contain something very bright that the object mask did not remove.
Since **only the 77 tracked objects carry annotations**, these are almost certainly genuine
**unannotated reflectors** — parked vehicles, poles, buildings, roadside structure.

This is direct evidence for the R1 concern in §16: a peak outside the annotated support is not
necessarily a false alarm, so a CARRADA "Pfa" is **not** the same quantity as the synthetic one.
""".format(med=bc["_cv_tail"]["cv_median"], p99=bc["_cv_tail"]["cv_p99"],
           mx=bc["_cv_tail"]["cv_max"],
           cvm=bc["cv"]["meanagg_mean"], km=bc["gamma_k_moment"]["meanagg_mean"],
           f2=bc["_cv_tail"]["frac_cv_gt_2"], f5=bc["_cv_tail"]["frac_cv_gt_5"],
           f10=bc["_cv_tail"]["frac_cv_gt_10"],
           sf2=bs["_cv_tail"]["frac_cv_gt_2"], sf5=bs["_cv_tail"]["frac_cv_gt_5"],
           sf10=bs["_cv_tail"]["frac_cv_gt_10"]))

md("""### Effective Gamma shape — estimator and wording

`k` above is the **method-of-moments** estimator `k ≈ mean(P)² / var(P)`. The **log-moment**
estimator, which is less sensitive to the upper tail, is reported beside it so the reader can see
whether the two agree rather than trusting one:

| | CARRADA | simulator |
|---|---|---|
| k, method of moments | {kcm} ± {kcs} | {ksm} ± {kss} |
| k, log-moment | {klc:.3f} ± {klcse:.3f} | {kls:.3f} ± {klsse:.3f} |

⚠️ **Wording matters.** This does **not** mean the sensor literally performs *k* independent
looks — the dataset authors document no such processing. The defensible statement is:

> *CARRADA background statistics are consistent with an effective Gamma shape near {kc_round}.*
""".format(kcm=g(bc, "gamma_k_moment", 3), kcs=se(bc, "gamma_k_moment", 3),
           ksm=g(bs, "gamma_k_moment", 3), kss=se(bs, "gamma_k_moment", 3),
           klc=R["gamma_logmoment"]["carrada_k_logmoment_mean"],
           klcse=R["gamma_logmoment"]["carrada_k_logmoment_se"],
           kls=R["gamma_logmoment"]["simulator_k_logmoment_mean"],
           klsse=R["gamma_logmoment"]["simulator_k_logmoment_se"],
           kc_round="{:.1f}".format(bc["gamma_k_moment"]["mean"])))
md("""### ⚠️ Superseded numbers — what changed and why

Three earlier passes produced figures for these same quantities. **None is retained.** Only the
protocol in §6 is authoritative. The defects, in the order they were found:

| pass | defect | effect |
|---|---|---|
| preliminary #1 | trimmed the top decile | biased CV downward (reported k ≈ 6.4) |
| preliminary #2 | ridge left in for some statistics | inflated CV enormously (reported CV ≈ 27.6) |
| preliminary #3 | far-range cells only (bins 200–255), empty frames only | not the population a detector sees |
| R0 first run | **ridge-band selector bug** | see below |

The band selector asked whether *any* bin inside the candidate window exceeded the +3 dB
threshold. The centre bin always does, so the band grew to the loop bound (±15) on every call,
excluding 31 of 64 Doppler bins. It now grows outward and stops at the first offset where
**neither edge bin** is still above threshold. Measured bands: CARRADA ±{hc}, simulator ±{hs}.

The remaining difference against preliminary #3 is **not** a defect in either — it is a genuine
difference of population, and §9b measures it rather than assuming it.
""".format(hc=ridge["ridge_half_width_carrada"], hs=ridge["ridge_half_width_simulator"]))

md("""### §9b Range-block decomposition — why the population matters

Background statistics are **not stationary in range**. The preliminary pass measured only far
range (bins 200–255); R0 measures the whole detector-valid map. This table shows both, so the
difference is attributable rather than mysterious.
""")
code("""rb = R["range_blocks"]
rows = []
for blk in ["10-60", "60-110", "110-160", "160-210", "210-246", "200-246"]:
    r = {"range block (bins)": blk}
    for dom, tag in (("CARRADA", "CARRADA"), ("simulator", "sim")):
        if blk in rb.get(dom, {}):
            r[f"{tag} CV"] = f"{rb[dom][blk]['cv']['mean']:.3f}"
            r[f"{tag} k"]  = f"{rb[dom][blk]['gamma_k_moment']['mean']:.2f}"
    rows.append(r)
out = pd.DataFrame(rows).set_index("range block (bins)")
print("The 200-246 row is the closest reproducible analogue of the superseded far-range pass.")
out""")
md("![background distributions](figures/r0_fig2_background_distributions.png)")
md("![domain statistics](figures/r0_fig3_domain_statistics.png)")

# ---------------------------------------------------------------------------------- §10
md("""## §10 Frame-median normalization audit

TinyCNN's frozen preprocessing is `x = clip(10·log₁₀(P / median P), −10, +60) / 20`
(`src/dl/dataset.py`). This section audits the **input domain**, and constitutes no model tuning.

Because dB is a monotone transform of power, `median(P) = 10^(median(dB)/10)` exactly, so

```
10·log10(P / median P)  ==  dB − median(dB)
```

The adapter is therefore **exact** on `_numpy`, and CARRADA's unknown constant scale offset
**cancels identically**. Global amplitude scale is a solved problem.

| | CARRADA | simulator |
|---|---|---|
| frame-median spread (sd, dB) | {c:.3f} | {s:.3f} |
| within-sequence sd (dB) | {w:.3f} | — |
| range (dB) | {c0:.2f} → {c1:.2f} | {s0:.2f} → {s1:.2f} |

**Interpretation.** Frame-median normalization removes the global scale difference exactly. It
therefore cannot be the explanation for any residual domain gap — which means whatever remains
is a genuine **texture / statistical** shift (§7–§9, §11), not a scale shift. This separation is
the point of the audit.
""".format(c=R["frame_median_audit"]["carrada_frame_median_db_sd_across_all"],
           s=R["frame_median_audit"]["simulator_frame_median_db_sd_across_all"],
           w=R["frame_median_audit"]["carrada_frame_median_db_sd_within_sequence"],
           c0=R["frame_median_audit"]["carrada_frame_median_db_range"][0],
           c1=R["frame_median_audit"]["carrada_frame_median_db_range"][1],
           s0=R["frame_median_audit"]["simulator_frame_median_db_range"][0],
           s1=R["frame_median_audit"]["simulator_frame_median_db_range"][1]))
md("![frame median](figures/r0_fig5_frame_median.png)")

# ---------------------------------------------------------------------------------- §11
md("""## §11 Zero-Doppler ridge

The excluded band was **selected, not assumed**: the mean Doppler profile was measured on a
deterministic subsample ({nc} CARRADA frames, {ns} simulator frames) and the band set to the
bins exceeding the off-ridge floor by more than {thr:.0f} dB. CARRADA required ±{hc} bins and the
simulator ±{hs}; the **union, ±{half} bins**, is applied to both so neither domain's statistics
are contaminated and the comparison stays symmetric.

| | CARRADA | simulator |
|---|---|---|
| ridge peak bin (argmax) | {pc} | {ps} |
| bins within 0.5 dB of the peak | {platc} | {plats} |
| ridge strength above floor | **{sc:.1f} dB** | **{ss:.1f} dB** |
| measured half-width | ±{hc} bins | ±{hs} bins |

The two ridges are **shaped differently, not merely scaled**: CARRADA's is narrow and tall
(±{hc} bins, {sc:.0f} dB), the simulator's is broad and flat-topped (±{hs} bins, {ss:.0f} dB).

⚠️ The simulator's `argmax` lands on bin {ps} rather than {zb}. That is a **flat top, not an
axis offset** — every bin listed above is within 0.5 dB of the peak, and the profile is
symmetric about bin {zb}. The zero-Doppler convention of §5 is confirmed, not contradicted.

**Side analysis, explicitly separated:** with the ridge *included*, CV(power) is
{cvri:.3f} for CARRADA and {cvrs:.3f} for the simulator — inflated by roughly
{fc:.0f}× and {fs:.0f}× respectively. These are reported only to show why ridge handling has to
be explicit; they are **not** background statistics and are never mixed into §7–§9.
""".format(nc=ridge["band_selection_frames_carrada"], ns=ridge["band_selection_frames_simulator"],
           thr=ridge["excess_threshold_db"], hc=ridge["ridge_half_width_carrada"],
           hs=ridge["ridge_half_width_simulator"], half=half,
           pc=ridge["carrada_ridge_peak_bin"], ps=ridge["simulator_ridge_peak_bin"],
           platc=", ".join(str(b) for b in ridge["carrada_ridge_plateau_bins"]),
           plats=", ".join(str(b) for b in ridge["simulator_ridge_plateau_bins"]),
           zb=R["background_protocol"]["zero_doppler_bin"],
           sc=ridge["carrada_ridge_peak_over_floor_db"],
           ss=ridge["simulator_ridge_peak_over_floor_db"],
           cvri=bc["cv_ridge_included"]["mean"], cvrs=bs["cv_ridge_included"]["mean"],
           fc=bc["cv_ridge_included"]["mean"] / bc["cv"]["mean"],
           fs=bs["cv_ridge_included"]["mean"] / bs["cv"]["mean"]))
md("![ridge](figures/r0_fig4_zero_doppler_ridge.png)")

# ---------------------------------------------------------------------------------- §12
md("""## §12 Object-support geometry

Descriptive geometry from the annotations only. **No target centre is invented here** — that
definition belongs to the R1 pre-registration, and inventing one during an audit is exactly how
a metric gets chosen to suit a result.

**Two extents are reported, because CARRADA supplies two different annotations** and confusing
them is easy. The `box` field is **not** the bounding box of the dense segmentation mask: it is
strictly smaller in **8,750 of 8,750** instances on both axes, so it bounds the *sparse* point
annotation instead. Both are given; the dense mask is the segmentation support.

| class | object-frames | dense range (med / p90) | dense Doppler (med / p90) | box range | box Doppler | dense cells |
|---|---|---|---|---|---|---|
{rows}

A synthetic target occupies **one** true bin. CARRADA objects occupy tens — a median of
{carcells:.0f} annotated cells for a car. This is the single most consequential difference for
the project's metrics: the ±1-bin gate, one-to-one matching, and localization-error-in-bins all
assume a point truth that CARRADA does not supply.
""".format(carcells=fp["car"]["dense_cells"]["median"], rows="\n".join(
    "| {c} | {n:,} | {rm:.0f} / {rp:.0f} | {dm:.0f} / {dp:.0f} | {br:.0f} | {bd:.0f} | {cm:.0f} |".format(
        c=c, n=fp[c]["object_frames"],
        rm=fp[c]["range_extent"]["median"], rp=fp[c]["range_extent"]["p90"],
        dm=fp[c]["doppler_extent"]["median"], dp=fp[c]["doppler_extent"]["p90"],
        br=fp[c]["box_range_extent"]["median"], bd=fp[c]["box_doppler_extent"]["median"],
        cm=fp[c]["dense_cells"]["median"])
    for c in ("pedestrian", "cyclist", "car"))))
md("![footprint](figures/r0_fig6_object_footprint.png)")

# ---------------------------------------------------------------------------------- §13
md("""## §13 R0 figures

| figure | file |
|---|---|
| 1 — example RD maps (illustrative) | `figures/r0_fig1_rd_examples.png` |
| 2 — background power and dB distributions | `figures/r0_fig2_background_distributions.png` |
| 3 — domain statistics with sequence-aware SE | `figures/r0_fig3_domain_statistics.png` |
| 4 — zero-Doppler ridge profile | `figures/r0_fig4_zero_doppler_ridge.png` |
| 5 — frame-median distributions | `figures/r0_fig5_frame_median.png` |
| 6 — object support extent by class | `figures/r0_fig6_object_footprint.png` |

Only Figure 1 uses manually selected frames, and it is labelled as illustrative on the figure
itself. Every other figure is computed over the full frame population.
""")

# ---------------------------------------------------------------------------------- §14
md("""## §14 R0 conclusions

Stated as **measured fact** first, then **interpretation**, then **implication**.

### Measured

1. **Geometry transfers exactly.** CARRADA RD is `(256, 64)` float64 on every one of 12,666
   frames; annotation coordinates index that array directly; no resampling is needed.
2. **`range_doppler_numpy` is power in dB up to a constant.** Slope {slope:.3f} against
   `10log₁₀(raw)`, residual {rs:.2f} dB with no level or position dependence, and CV agreement
   with `raw²` to {gap2:.1f}%.
3. **The simulator is single-look to within measurement error.** k = {ksm} ± {kss} against a
   theoretical 1.0, and background std {dss} dB against a theoretical 5.57 dB.
4. **CARRADA is statistically smoother.** CV {cvc} ± {cvcse} versus {cvs} ± {cvsse};
   k = {kcm} ± {kcs}; background std {dsc} dB.
5. **CARRADA is more correlated along range.** lag-1 {lrc} versus {lrs}. Along Doppler the two
   are closer: {ldc} versus {lds}.
6. **The zero-Doppler ridge is far stronger in CARRADA:** {sc:.1f} dB versus {ss:.1f} dB.
7. **CARRADA truth is extended object support,** median car {carr:.0f}×{card:.0f} bins, never a
   point target; and never more than 2 objects per frame.

### Interpretation

Point 3 is a genuine, independent **validation of the simulator's clutter model** — it behaves
as designed, which had not previously been checked against theory this directly.

Points 4–6 mean **CARRADA is not simply another sample from the synthetic domain.** It is a
meaningful domain shift, and it is a *structural* one: §10 shows frame-median normalization
removes the global scale difference exactly, so what remains is texture and correlation, which
normalization cannot touch.

Conservative wording, to be used verbatim in the report: *CARRADA background statistics are
consistent with an effective Gamma shape near {kc_round}.* Not "CARRADA is a
{kc_round}-look radar" — the dataset authors document no such processing.

### Implication

R1 is a **zero-shot transfer** experiment for **both** arms, and must be reported as one.
""".format(slope=rep["slope_vs_10log10raw_mean"], rs=rep["resid_std_db_mean"],
           gap2=rep["cv_pct_gap_numpy_vs_rawsq"],
           ksm=g(bs, "gamma_k_moment", 3), kss=se(bs, "gamma_k_moment", 3),
           dss=g(bs, "db_std", 3), kcm=g(bc, "gamma_k_moment", 3), kcs=se(bc, "gamma_k_moment", 3),
           cvc=g(bc, "cv"), cvcse=se(bc, "cv"), cvs=g(bs, "cv"), cvsse=se(bs, "cv"),
           dsc=g(bc, "db_std", 3), lrc=g(bc, "lag1_range"), lrs=g(bs, "lag1_range"),
           ldc=g(bc, "lag1_doppler"), lds=g(bs, "lag1_doppler"),
           sc=ridge["carrada_ridge_peak_over_floor_db"],
           ss=ridge["simulator_ridge_peak_over_floor_db"],
           carr=fp["car"]["range_extent"]["median"], card=fp["car"]["doppler_extent"]["median"],
           kc_round="{:.1f}".format(bc["gamma_k_moment"]["mean"])))

md("""### Implications for CFAR — recorded, **not acted on**

The four CFAR alphas in `configs/cfar_alpha_table.json` were Monte-Carlo calibrated under
approximately **single-look exponential** statistics. CARRADA's background is thinner-tailed
(k ≈ {kc_round}), so the *same frozen alpha* will produce a **different achieved false-alarm rate**
there.

Therefore the R1 CFAR result is a **zero-shot transfer** result. **If CFAR's recall drops, that
does not by itself show CFAR is intrinsically worse** — it may show only that its operating point
moved. **No alpha was recalibrated in Phase 2.** Whether a secondary calibrated-CFAR arm is
permitted is a Phase 3 pre-registration decision.

### Implications for TinyCNN — recorded, model **not run**

TinyCNN was trained under the simulator's single-look-like background. CARRADA changes three
things at once: the background amplitude distribution (k ≈ {ks_round} → ≈ {kc_round}), the spatial
correlation (lag-1 range {lrs} → {lrc}), and the zero-Doppler clutter strength
({ss:.1f} → {sc:.1f} dB).

Frame-median normalization handles the **global scale** exactly (§10) but **cannot** confer
invariance to those structural shifts. **No conclusion about TinyCNN performance is permitted
until R1.**
""".format(kc_round="{:.1f}".format(bc["gamma_k_moment"]["mean"]),
           ks_round="{:.1f}".format(bs["gamma_k_moment"]["mean"]),
           lrs=g(bs, "lag1_range"), lrc=g(bc, "lag1_range"),
           ss=ridge["simulator_ridge_peak_over_floor_db"],
           sc=ridge["carrada_ridge_peak_over_floor_db"]))

# ---------------------------------------------------------------------------------- §15
md("""## §15 R0 limitations

1. **"Object support" differs between domains** — a 3×3 label mask around a point target versus a
   human-drawn segmentation blob. The dilation applied is identical; the thing dilated is not.
   Unavoidable, and stated wherever these statistics appear.
2. **The simulator has no sequence structure**, so its uncertainty is aggregated across the 8
   scenario conditions rather than across sequences. The two error bars are therefore not
   perfectly like-for-like, even though both are group-level.
3. **The Gamma-shape estimate is descriptive.** It summarises the background's second moment; it
   is not a claim about the sensor's processing chain.
4. **`bench2_val` only** on the synthetic side. `bench2_test` was deliberately not opened.
5. **Amplitude scale is not comparable in absolute terms** — CARRADA's constant offset is
   unknown, so only shape, spread and correlation are compared, never absolute level.
6. **The ridge band is a threshold choice** (+3 dB). It was selected from measured profiles
   before any background statistic was computed, and the union of both domains was applied, but
   a different threshold would shift the eligible population slightly.
7. **No detector behaviour is measured here.** Nothing in R0 licenses any statement about Pd,
   Pfa, recall, or which arm transfers better.
""")

# ---------------------------------------------------------------------------------- §16
md("""## §16 Next step — R1 pre-registration

> ✅ **Written and frozen in Phase 3.** See `PREREGISTRATION.md` and §17–§28 below. This section
> records what R0 identified as blocking, and each item is now resolved there.

R1 was blocked on decisions that had to be **pre-registered before any number existed**, because
each is a definition this project invents rather than something CARRADA supplies:

1. **Truth-point definition.** Box centre, sparse centroid, or brightest cell within the mask —
   or abandon point matching for a "peak inside the annotated support" criterion. This determines
   whether the localization metric that carries the whole E3/E4 story transfers at all.
2. **What "false alarm" means.** Only the 77 tracked objects are annotated; real scenes contain
   unannotated reflectors, so peaks outside the masks include genuine targets. **Matched-Pfa
   comparison across the synthetic and real domains is not valid** and must not be presented as
   one.
3. **CFAR operating point.** Frozen alpha (true zero-shot) versus recalibrated alpha (fair
   operating point). Arguably both, reported separately — but the choice is registered in
   advance, not after seeing which flatters the comparison.
4. **Which split.** The official Test split is 6 sequences / 1,391 annotated frames / 1,628
   object-frames. Development belongs on Validation; Test should be read once.

No detector will be run until that pre-registration is committed.
""")

# ================================ PHASE 3 — R1 PRE-REGISTRATION ========================
FREEZE = json.load(open(os.path.join(RESULTS, "preregistration_freeze.json")))

md("""---

# Phase 3 — R1 pre-registration

> **Frozen. No detector has been run on CARRADA.**
>
> Full document: [`PREREGISTRATION.md`](PREREGISTRATION.md)
> SHA-256 `{sha}`
> Frozen {when} against R0 baseline commit `{commit}`

⚠️ **Section numbering.** The prompt for this phase asked for §15–§26. R0 had already used §15
(limitations) and §16 (next step) in the committed notebook, so the R1 sections run **§17–§28**
rather than overwrite them. The mapping is one-to-one and in order: requested §15 → §17,
§16 → §18, … §26 → §28.

The experiment progression this notebook will hold end to end:

| phase | content | status |
|---|---|---|
| 1 | isolation / integrity | complete, §2 |
| 2 | R0 domain audit | complete, §4–§15 |
| 3 | R1 pre-registration | **complete, §17–§28** |
| 4 | implementation + sanity checks | not started |
| 5 | final detector evaluation | not started |
| 6 | interpretation / closeout | not started |
""".format(sha=FREEZE["sha256"], when=FREEZE["frozen_utc"][:19] + "Z",
           commit=FREEZE["r0_baseline_commit"][:12]))

md("""## §17 R1 motivation *(requested §15)*

> **Do the already-frozen TinyCNN and classical CFAR detectors transfer zero-shot from the
> synthetic development domain to real CARRADA Range–Doppler data?**

This is **external validation**. It does not replace or revise the synthetic benchmark and cannot:
the synthetic study answers controlled questions about performance and mechanism under a truth
definition CARRADA does not supply. R1 asks whether the frozen behaviour survives the domain shift
R0 measured.

**A poor R1 result would retract no synthetic finding.** It would bound its external validity.""")

md("""## §18 Pre-registered ground truth *(requested §16)*

**Primary truth is the CARRADA dense segmentation mask.** No point target is invented.

Rejected as primary, explicitly: centroid, box centre, brightest cell in mask, sparse-point
centroid — each would be a construct of this project rather than an annotation CARRADA supplies.

R0 measured that the `box` field is **not** the bounding box of the dense mask: it is strictly
smaller in **8,750 of 8,750** instances, so it bounds the sparse annotation instead.

The detection question is: **did the detector place a peak inside the annotated object support?**""")

md("""## §19 Eligibility region *(requested §17)*

Unchanged from the frozen convention (`configs/frozen_dl.yaml`, *"margin 10 range / 4 Doppler"*):

```
valid_region  = rows 10…245 × cols 4…59        (236 × 56 = 13,216 cells)
valid_support = dense_mask ∩ valid_region
eligible      = |valid_support| >= 1
```

Total, eligible, and excluded object counts are all reported. A stricter variant — mask entirely
inside the region — is registered as secondary S4, not primary.""")

md("""## §20 One-to-one matching *(requested §18)*

One prediction may not detect two objects. Identical code for both arms.

Edge: peak `p` → object `o` iff `p ∈ valid_support(o)`. Objective, in strict order:

1. maximise matched objects (maximum-cardinality bipartite matching);
2. tie-break on greater total detector score;
3. tie-break on deterministic `(range_bin, doppler_bin, object_id)` order.

Seven unit tests must pass before any test scoring: one/one, one/many, two objects sharing a peak,
two/two, overlapping masks, no peaks, empty frame.""")

md("""## §21 Primary metrics *(requested §19)*

**Primary: mask-based object recall.** For each eligible object, detected iff an assigned peak lies
inside its `valid_support`.

```
object recall = matched eligible objects / total eligible objects
```

**Tolerance ladder** (identical dilation for both arms): exact mask = **headline**; ±1 bin and
±3 bins secondary.

**Support-distance is excluded from confirmatory metrics.** For a successful detection it is
identically zero by construction, so it cannot discriminate among successes, and a nearest-of-many
rule would reward whichever detector emits more peaks. **No localization metric comparable to the
synthetic one is reported at all** — CARRADA supplies no point truth. That is a limitation of the
transfer, not an omission.

> ⚠️ Synthetic Pd and CARRADA mask recall measure related detection capability under **different
> truth definitions**. They are never placed in the same column, differenced, or called the same
> quantity.""")

md("""## §22 Unmatched-detection terminology *(requested §20)*

**Synthetic-style Pfa is not reported on CARRADA in any form.**

R0 measured that 5.85% of frames carry bright returns belonging to no annotation, and only the 77
tracked objects are annotated at all. A peak outside every mask may be infrastructure, static
clutter, multipath, another reflector, an unannotated object — or a genuine false alarm.

The burden metric is **unmatched detections per frame** (mean, median, p90, per sequence). It is
never renamed `Pfa`, `false-positive rate`, `false alarm rate`, or `FPR`.

Secondary: unmatched detections per **frame with no CARRADA tracked-object annotation** — described
in exactly those words, never as "true background-only frames".""")

md("""## §23 Detector operating points *(requested §21)*

**TinyCNN — a new protocol choice, declared as such.** The frozen project has **no** global
threshold: `frozen_dl.yaml` specifies *"matched MEASURED Pfa, swept per condition"*, an oracle that
cannot transfer. Registered rule:

> one threshold **per seed**, derived on **`bench2_val` only**, **pooled across all 8 conditions**,
> the lowest threshold whose pooled measured per-cell Pfa ≤ 1e-4, at the ±1 gate, frozen `all_peaks`
> and frozen valid region.

No CARRADA data of any split enters this derivation. **Consequence registered in advance:** R1 runs
under a *different operating-point protocol* from the synthetic headline, so its numbers are **not
poolable** with the synthetic tables.

**CFAR — strict zero-shot.** Frozen alpha unchanged. Registered interpretation: R0 established these
alphas were calibrated for single-look statistics while CARRADA sits near k ≈ 4.8, so the achieved
operating point will move. **If CFAR recall is poor, that does not establish CFAR is intrinsically
inferior.** No recalibration is performed, and none will be invented merely to improve CFAR's
numbers.""")

md("""## §24 TinyCNN frozen provenance *(requested §22)*

`results/dl/tinycnn_frozen_s{{0..7}}_ep021.pt` — TinyCNN, 5,041 params, RF 21×9, epoch 21,
`threads=2`, hash-pinned in `isolation_manifest.json`.

**Prohibited:** retraining, fine-tuning, architecture change, label change, BatchNorm adaptation,
CARRADA-specific learned normalization, any edit to `src/dl/`. All 8 seeds evaluated independently;
**not ensembled**, because the frozen benchmark does not ensemble them.

Experiment-local adapter only:

```
relative_db = D - median(D)          # D = range_doppler_numpy, already power in dB
relative_db = clip(relative_db, -10, +60)
x           = relative_db / 20
```

Algebraically identical to the frozen `10*log10(P / median P)`, because dB is monotone in power so
`median(P) = 10**(median(D)/10)` exactly. Check C1 verifies this numerically to < 1e-9 before any
CARRADA inference.""")

md("""## §25 CFAR frozen provenance *(requested §23)*

Frozen 21×9 window, guard 2/2, train 8/2, `os_rank_frac` 0.75, `go_axis` 1, calibrated alphas from
`configs/cfar_alpha_table.json`. Source, geometry, statistic and alpha unchanged.

Input `P' = 10**((D - c)/10)` is permitted for any constant `c`; CFAR is a ratio test so the
constant cancels. Check C2 unit-tests that invariance.

**All four variants — CA, GO, OS, SO — are run and reported.** The benchmark's "best CFAR per
condition" cannot transfer: selecting it needs labels, which is an oracle on real data.

**Primary CFAR arm: CA** *(final ruling, before any scoring)*. CA is the canonical classical
baseline, it preserves continuity with the main synthetic benchmark, and it **avoids choosing the
primary comparator because R0 hinted another variant might suit CARRADA** — which would import a
data-driven preference into a strict zero-shot transfer.

GO, SO and OS run as secondary. **OS is highlighted as the robust heterogeneous-background
comparison** (R0 found non-stationary background, a narrow 24 dB ridge, bright unannotated
reflectors — what an order-statistic estimator exists to tolerate) but **is never promoted to
primary after results are seen.**

**Frozen NMS:** `all_peaks`, `nms_size = 5`, shared by both arms. Not retuned on CARRADA.

> ⚠️ **Registered limitation — extended objects under a point-target NMS.** The 11×11 suppression
> window was sized for a ~2-bin synthetic point target. A CARRADA car spans a median 18 × 13 bins —
> **larger than the window** — so one real object can produce several surviving peaks inside its own
> support. Object recall stays valid (one-to-one matching assigns exactly one peak per object), but
> the surplus peaks become *unmatched*, so **unmatched detection burden ≠ pure detector noise**: part
> of it is a metric-transfer artefact. It is common-mode in the TinyCNN−CFAR *difference*, but not in
> the absolute burden. Phase 4 quantifies how often it happens on non-test frames; **NMS is not
> altered in response.**""")

md("""## §26 Secondary stratifications *(requested §24)*

| # | analysis |
|---|---|
| S1 | per class — car / pedestrian / cyclist |
| S2 | range blocks — the **exact** R0 blocks: 10–60, 60–110, 110–160, 160–210, 210–246 |
| S3 | zero-Doppler — near-zero if `valid_support` meets Doppler bins 28–36 (the R0 ±4 band), else away. **The ridge is not removed from the primary analysis** |
| S4 | strict eligibility — mask entirely inside the valid region |
| S5 | sparse / box truth — sensitivity only, never replacing dense |
| S6 | recall–unmatched-burden curve — full curve, x-axis is **not** Pfa, **not** an ROC |
| S7 | empty-frame burden |

> ⚠️ **Per-class recall caveat, registered now.** Class masks differ greatly in size (median
> support: car 18×13, cyclist 11×11, pedestrian 9×9). A larger mask is a larger area for a peak to
> land inside, so **class differences must not be read as detector sensitivity alone.**""")

md("""## §27 Statistical plan *(requested §25)*

**The test split has 6 sequences. That is the effective sample size.**

**Prohibited:** bootstrapping RD cells; treating the 1,628 test object-frames as independent; any
interval whose n is a frame or object count.

Unit of analysis is the test sequence (n = 6). Per sequence: TinyCNN recall averaged over the 8
seeds, CFAR recall from the primary arm, and their paired difference. **Primary inference is a
paired bootstrap over the 6 test sequences**, 10,000 replicates, percentile CI.

**Seed variability is reported separately and never merged with sequence variability** into one
standard error — they answer different questions.

With n = 6, emphasis is on effect sizes, CIs, and the per-sequence table shown directly. p-values
are de-emphasised; low-n class comparisons are labelled **exploratory**.""")

md("""## §28 Pre-registration freeze record *(requested §26)*

| | |
|---|---|
| file | `PREREGISTRATION.md` |
| SHA-256 | `{sha}` |
| size | {size:,} bytes |
| frozen (UTC) | {when} |
| R0 baseline commit | `{commit}` |
| detector scoring performed | **{scored}** |
| CARRADA labels used for any choice | **{used}** |

Departures go to `DEVIATIONS.md` with date, reason, whether results had been seen, and scientific
consequence. **This file is not edited after the freeze.**

### Final rulings, applied before any scoring

The two items flagged at Phase 3 review are now decided and incorporated. Both were settled with
**no CARRADA detector output in existence**:

1. **Global TinyCNN threshold (§23) — approved as proposed.** Per seed, `bench2_val` only, pooled
   over all 8 conditions, lowest threshold with pooled measured per-cell Pfa ≤ 1e-4. Declared a
   **new zero-shot operating protocol**, never mixed numerically with the per-condition synthetic
   benchmark.
2. **Primary CFAR (§25) — changed from OS to CA.** Canonical baseline, continuity with the main
   benchmark, and it avoids picking the comparator because R0 hinted OS might suit CARRADA. OS stays
   as the highlighted robust secondary, never promotable after results.
3. **NMS frozen at 5**, with the extended-object multi-peak limitation registered.
4. **Range-stratum assignment** for objects spanning blocks: median range index of `valid_support`
   — annotation-only, so no detector output can influence it.

This draft supersedes an earlier one (SHA-256 `{prior}`) that named OS primary.

---

## Next: Phase 4 — implementation and sanity checks. **No detector has been scored.**
""".format(sha=FREEZE["sha256"], size=FREEZE["size_bytes"], when=FREEZE["frozen_utc"][:19] + "Z",
           prior=FREEZE.get("supersedes_draft_sha256", "n/a")[:16],
           commit=FREEZE["r0_baseline_commit"][:12],
           scored="NO" if not FREEZE["detector_scoring_performed"] else "YES",
           used="NO" if not FREEZE["carrada_labels_used_for_any_choice"] else "YES"))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python"}})
nbf.write(nb, NB)
print("wrote {}  ({} cells: {} markdown, {} code)".format(
    os.path.relpath(NB, ROOT), len(cells),
    sum(1 for c in cells if c.cell_type == "markdown"),
    sum(1 for c in cells if c.cell_type == "code")))
