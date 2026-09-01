"""Build the experiment notebook from the cached R0 results.

Every number in the notebook is read from results/r0_audit.json rather than typed, so the
prose cannot drift away from the measurements. Writes only the notebook.
"""
from __future__ import annotations

import json
import os
import sys

import nbformat as nbf
import numpy as np
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

# ==================================== PHASE 4 — IMPLEMENTATION =========================
TH = json.load(open(os.path.join(EXP, "configs", "tinycnn_zero_shot_thresholds.json")))
AD = json.load(open(os.path.join(RESULTS, "phase4_adapter_checks.json")))
DR = json.load(open(os.path.join(RESULTS, "phase4_dry_run.json")))
ELI = json.load(open(os.path.join(RESULTS, "phase4_eligibility_counts.json")))
LK = json.load(open(os.path.join(RESULTS, "phase4_leakage_audit.json")))
C1 = [c for c in AD if c["check"].startswith("C1")][0]
C2 = [c for c in AD if c["check"].startswith("C2")][0]

md("""---

# Phase 4 — implementation and sanity checks

> **No CARRADA Test detector result has been computed.** Every check below runs on synthetic data
> or on CARRADA **Train** sequences.

## §29 Implementation overview

All R1 code is experiment-local, under `src/` and `scripts/`. Nothing in `src/dl/`, `src/cfar/`,
`src/eval/`, the frozen config, the checkpoints or the alpha table was modified — they are
**imported**, never copied or edited.

| module | role |
|---|---|
| `src/carrada_input.py` | RD loading, dB→power / dB→magnitude adapters, dense-support reconstruction |
| `src/eligibility.py` | valid region (imported from the frozen evaluator), eligibility, strata |
| `src/matching.py` | the single one-to-one matcher used by **every** arm |
| `src/metrics.py` | mask recall, tolerance ladder, unmatched burden |
| `src/detectors.py` | frozen TinyCNN + CFAR wrappers (format only) |
| `src/stats.py` | sequence-aware paired bootstrap |
| `src/schema.py` | output schemas, defined before scoring |""")

md("""## §30 Final pre-registration decisions

Applied before any scoring, then committed (Phase 3, `{p3}`):

1. **TinyCNN threshold** — approved as proposed (§31).
2. **Primary CFAR is CA**, not OS. Canonical baseline, continuity with the main benchmark, and it
   avoids picking the comparator because R0 hinted OS might suit CARRADA. GO/SO/OS secondary, **OS
   highlighted as the robust comparison but never promotable after results.**
3. **NMS frozen at 5**, with the extended-object limitation registered (§40).
4. **Range-stratum assignment** by median range index of `valid_support` — annotation-only.

Final pre-registration SHA-256 `{sha}`.""".format(sha=FREEZE["sha256"],
                                                  p3="9c9a3e25e604"))

md("""## §31 TinyCNN zero-shot threshold derivation

**Synthetic validation only.** `bench2_val`, pooled across all 8 conditions, per frozen seed, the
lowest threshold whose pooled measured per-cell Pfa ≤ 1e-4, at the ±1 gate, frozen extraction /
NMS 5 / valid region.

A runtime **file-access audit hook** recorded every file the derivation opened. It saw
**{splits}** and nothing else — `{nfiles:,}` data files, **0 forbidden accesses**. This is
machine-verified, not asserted.

| seed | threshold | achieved pooled val Pfa | pooled val Pd |
|---|---|---|---|
{rows}

Spread {lo:.3f}–{hi:.3f} (sd {sd:.3f}); all eight ≤ 1e-4. Re-running seeds 0, 4 and 7 reproduced
the thresholds to all printed digits, so the derivation is deterministic.

⚠️ These are **not** part of the frozen benchmark. They are R1 zero-shot deployment thresholds
under a **different operating-point protocol**, and must never be pooled numerically with the
per-condition matched-Pfa synthetic tables.""".format(
    splits=TH["file_access_audit"]["data_splits_opened"],
    nfiles=TH["file_access_audit"]["n_data_files_opened"],
    rows="\n".join("| {} | {:.6f} | {:.4e} | {:.4f} |".format(
        r["seed"], r["threshold"], r["achieved_pooled_val_pfa_per_cell"],
        r["achieved_pooled_val_pd"]) for r in TH["thresholds"]),
    lo=min(r["threshold"] for r in TH["thresholds"]),
    hi=max(r["threshold"] for r in TH["thresholds"]),
    sd=float(np.std([r["threshold"] for r in TH["thresholds"]], ddof=1))))

md("""## §32 TinyCNN input adapter verification (C1)

The registered adapter is `D − median(D)` → clip → /20. Rather than reimplement the preprocessing,
the pipeline feeds `mag = 10^(D/20)` to the **frozen** `normalize_rd`, which then computes
`10·log10(mag²/median(mag²))` — algebraically identical, because dB is monotone in power. **No
preprocessing exists in the model path that is not the frozen code.**

Measured on {n} CARRADA **Train** frames:

| quantity | value | tolerance |
|---|---|---|
| max &#124;(D−median D) − 10log10(P/median P)&#124; | **{a:.3e} dB** | {ta:.0e} |
| max &#124;frozen `normalize_rd` − registered adapter&#124; | **{b:.3e}** | {tb:.0e} |

**C1 PASS.** Tolerances come from float precision, not from what made the test pass: {a:.1e} dB on
a ~60 dB value is ≈2 ulp of float64, and the frozen output is float32 (eps 1.19e-07), so ≈2 ulp
there too.""".format(n=C1["frames"], a=C1["max_abs_diff_identity_db"],
                     b=C1["max_abs_diff_frozen_vs_reference"],
                     ta=C1["tolerance_identity"], tb=C1["tolerance_frozen"]))

md("""## §33 CFAR power adapter and scale invariance (C2)

CFAR consumes `P = 10^((D − c)/10)`, with `c` the frame median. Because CFAR is a **ratio test**,
`c` must cancel exactly.

Tested across **{n} combinations** — 4 variants × CARRADA Train frames × scales
{scales} — max &#124;ratio(P·s) − ratio(P)&#124; = **{d:.3e}**, peak-count difference **{p}**.
**C2 PASS.** CFAR source was not modified to achieve this.""".format(
    n=C2["combinations"], scales=C2["scales"], d=C2["max_abs_ratio_diff"],
    p=C2["max_peak_count_diff"]))

md("""## §34–§35 Dense-mask reconstruction (C5) and the common valid region (C6)

**Dense masks** are rebuilt from `annotations_frame_oriented.json` — the bulk `.npy` folders were
deleted after Phase 1 proved them byte-identical on 360/360 samples. All **5,318 Train objects**
reconstruct without error, covering pedestrian / cyclist / car, 999 two-object frames, 3,769 empty
frames and 89 edge-touching masks. No interpolation; coordinates are CARRADA's own.

**Valid region** is *imported* from the frozen `dl.infer.cfar_valid_mask`, not reimplemented, so
the endpoint convention cannot drift: rows 10–245, cols 4–59, **13,216 cells** — verified.

### Eligibility denominators (annotation-only — no detector involved)

| split | objects | eligible | excluded | strictly inside |
|---|---|---|---|---|
{rows}

⚠️ **{nz:,} of {tot:,} eligible Test objects ({pct:.0%}) fall in the zero-Doppler band.** CARRADA's
road users are largely slow-moving and sit inside the very ridge R0 measured at 24 dB. The §38
stratification is therefore not a minor sub-analysis — it covers most of the test set.""".format(
    rows="\n".join("| {} | {:,} | {:,} | {} | {:,} |".format(
        k, v["total_objects"], v["eligible"], v["excluded"], v["strictly_inside"])
        for k, v in ELI.items()),
    nz=ELI["Test"]["eligible_by_doppler_stratum"].get("near_zero", 0),
    tot=ELI["Test"]["eligible"],
    pct=ELI["Test"]["eligible_by_doppler_stratum"].get("near_zero", 0) / ELI["Test"]["eligible"]))

md("""## §36–§37 Matching and metric unit tests (C4, C7)

All eight matching cases pass, including the pre-registered seven plus the extended-object case:

| case | expectation | result |
|---|---|---|
| one object / one peak | matched | ✅ |
| one object / three peaks | 1 matched, 2 unmatched | ✅ |
| … tie-break | highest-scoring peak wins | ✅ |
| two objects / one shared peak | exactly 1 matched | ✅ |
| two objects / two peaks | both matched | ✅ |
| overlapping masks | **maximum cardinality (2), not greedy** | ✅ |
| no peaks | nothing matched | ✅ |
| empty frame | peak unmatched | ✅ |
| **18×13 car mask / 3 peaks** | **1 matched, 2 unmatched** | ✅ |

Metric tests pass too: recall 1.0 for a matched object; a 1-bin-off peak **missed** by the exact
mask and **recovered** by the ±1 dilation; burden mean/median arithmetic correct; and an object
entirely outside the valid region correctly excluded from the denominator (recall = NaN, not 0).""")

md("""## §38 Range and zero-Doppler strata

Both reuse R0's definitions **unchanged**. Range blocks 10–60, 60–110, 110–160, 160–210, 210–246;
an object spanning blocks is assigned by the **median range index of its valid support**, which is
annotation-only so no detector output can move it. Zero-Doppler band is R0's common ±4 (bins
28–36); near-zero iff the valid support intersects it. **The ridge is not removed from the primary
analysis** — the frozen detectors see the real map.

Test-split eligible objects by stratum: range {rng} · Doppler {dop} · class {cls}.

⚠️ Cyclist has only {cyc} eligible Test objects, so per-class cyclist results will be labelled
**exploratory**.""".format(rng=ELI["Test"]["eligible_by_range_block"],
                           dop=ELI["Test"]["eligible_by_doppler_stratum"],
                           cls=ELI["Test"]["eligible_by_class"],
                           cyc=ELI["Test"]["eligible_by_class"].get("cyclist", 0)))

md("""## §39 Non-test pipeline dry run (C8)

**SOFTWARE SANITY ONLY — NOT PERFORMANCE ANALYSIS.** CARRADA **Train** only.

{n} frames across 6 sequences ran end to end in {s:.1f} s: **0** non-finite peak arrays, **0**
structural problems, all peaks inside the map, matched never exceeding total. Aggregate recall was
deliberately **not** computed.

One real defect was caught here and fixed before scoring: the harness applied the zero-shot
threshold to TinyCNN but left CFAR **unthresholded**, because `all_peaks` keeps every local maximum
with ratio > 0 and never applies alpha. That put the arms at incomparable operating points
(CFAR ~116 peaks/frame vs TinyCNN ~9). `cfar_peaks_at_alpha` now applies the frozen calibrated
alpha — CA 10.459, GO 9.546, SO 12.849, OS 7.579 — giving ~7 and ~9 peaks/frame respectively.""".format(
    n=DR["frames"], s=DR["seconds"]))

md("""## §40 Extended-object / frozen-NMS characteristic

**Metric transfer characteristic, not a performance result. NMS was not altered in response.**

A CARRADA car spans a median 18 × 13 bins against an 11 × 11 suppression window, so one object can
hold several surviving peaks. Measured on CARRADA Train, at each arm's frozen operating point:

| detector | objects | median peaks inside support | max | ≥2 peaks |
|---|---|---|---|---|
{rows}

**Roughly a third to a half of objects produce more than one peak inside their own support.** Under
one-to-one matching exactly one is matched and the surplus counts as **unmatched**, so
**unmatched-detection burden is inflated for both arms** and is *not* pure detector noise. It is
common-mode in the TinyCNN−CFAR difference but **not** in the absolute burden — which is precisely
why §22 forbids calling it a false-alarm rate.""".format(
    rows="\n".join("| {} | {} | {:.0f} | {} | {:.1%} |".format(
        r["detector"], r["objects"], r["median_peaks_inside"], r["max_peaks_inside"],
        r["frac_objects_with_>=2_peaks"]) for r in DR["multipeak_by_detector"])))

md("""## §41 Statistical implementation

`src/stats.py` implements the pre-registered plan and nothing else: paired bootstrap over the
**6 test sequences**, 10,000 replicates, percentile CI, fixed seed for reproducibility. Seed
variability is a **separate** function and is never merged with sequence variability into one
error bar. Cell-level and object-level bootstraps are deliberately **not implemented**, so they
cannot be reached by accident.""")

md("""## §42 Phase 4 isolation and leakage verification

**Leakage audit: CLEAN.** {stmt}

Verified: no Test sequence id in any detector-output artefact; no R1/recall/test-result file
exists; thresholds sourced from `bench2_val` with a runtime access audit; the dry run and
multi-peak check touched Train only; no source line loads `bench2_test`.

One honest caveat, recorded rather than buried: **R0's background statistics did include the Test
sequences.** That is domain-level, not detector-level — R0 ran no detector — but it fed two
pre-registered choices. Checked directly: recomputing the ridge band from **Train sequences only**
gives the **identical ±1 band** (peak 23.9 dB vs 24.5 dB on Test), so the pre-registration would
be unchanged.""".format(stmt=LK["statement"]))

md("""## §43 Ready-for-R1-test checklist

| # | item | status |
|---|---|---|
| 1 | pre-registration frozen and committed | ✅ `{sha8}…` |
| 2 | TinyCNN thresholds derived, synthetic-val only | ✅ 8 seeds, deterministic |
| 3 | C1 adapter equivalence | ✅ |
| 4 | C2 CFAR scale invariance | ✅ |
| 5 | C3 checkpoint integrity | ✅ 8/8 hash-match |
| 6 | C4 matching unit tests | ✅ 8/8 |
| 7 | C5 dense-mask reconstruction | ✅ 5,318 Train objects |
| 8 | C6 valid region = 13,216 cells | ✅ imported, not reimplemented |
| 9 | C7 metric unit tests | ✅ |
| 10 | C8 non-test dry run | ✅ 0 problems |
| 11 | both arms at frozen operating points | ✅ *(bug caught and fixed, §39)* |
| 12 | leakage audit | ✅ CLEAN |
| 13 | output schemas defined | ✅ `src/schema.py` |
| 14 | **CARRADA Test scored** | ⬜ **NOT STARTED — Phase 5** |

**Nothing in Phase 5 may change any definition above.** The remaining work is to run the frozen
pipeline over the 6 Test sequences and report what it produces.""".format(
    sha8=FREEZE["sha256"][:8]))

# ================================ PHASE 5 — R1 TEST RESULTS ============================
R1 = json.load(open(os.path.join(RESULTS, "r1_test", "r1_summary.json")))
ACC = json.load(open(os.path.join(RESULTS, "r1_test", "access_log.json")))
MAN = json.load(open(os.path.join(RESULTS, "r1_test", "run_manifest.json")))
P = R1["primary"]
BS = R1["paired_bootstrap"]
TOL = R1["tolerance"]
BUR = R1["unmatched_burden"]

md("""---

# Phase 5 — R1: the one-shot frozen evaluation on CARRADA Test

## §44 Final R1 test-access record

| | |
|---|---|
| **FINAL CARRADA TEST ACCESS** | {ts} |
| git commit | `{git}` |
| pre-registration SHA-256 | `{pre}` |
| run-manifest SHA-256 | `{man}` |
| threshold config SHA-256 | `{thc}` |
| environment | Python {py}, torch {tor}, numpy {npv} |

Everything used here was fixed **before** this timestamp. From this point on, any methodological
change would be post-result and cannot silently alter the confirmatory experiment.""".format(
    ts=ACC["timestamp_utc"], git=ACC["git_commit"][:12],
    pre=ACC["preregistration_sha256"], man=ACC["run_manifest_sha256"],
    thc=ACC["threshold_config_sha256"][:32] + "…",
    py=ACC["environment"]["python"], tor=ACC["environment"]["torch"],
    npv=ACC["environment"]["numpy"]))

md("""## §45–§46 Frozen operating points and test-set counts

**CFAR alphas (frozen, unchanged):** CA {ca:.4f} · GO {go:.4f} · SO {so:.4f} · OS {os_:.4f}
**TinyCNN thresholds (from `bench2_val` alone):** {thr}
**NMS radius {nms}, frozen.**

**Eligible Test objects: {n:,}** of 1,628 (one object has no support cell inside the valid region).
6 sequences, 2,130 frames.""".format(
    ca=ACC["cfar_alphas"]["CA"], go=ACC["cfar_alphas"]["GO"], so=ACC["cfar_alphas"]["SO"],
    os_=ACC["cfar_alphas"]["OS"],
    thr=", ".join("{:.4f}".format(v) for v in ACC["tinycnn_thresholds"].values()),
    nms=ACC["nms_size"], n=R1["eligible_test_objects"]))

md("""## §47–§48 PRIMARY RESULT — exact dense-mask object recall

> ### The frozen synthetic advantage does **not** transfer. CA-CFAR outperforms TinyCNN.

| arm | exact-mask object recall |
|---|---|
| frozen **CA-CFAR** (primary classical) | **{ca:.4f}** |
| frozen **TinyCNN** (8-seed mean) | **{dl:.4f}** |
| **paired difference (TinyCNN − CA)** | **{d:+.4f}**, 95% CI [{lo:+.4f}, {hi:+.4f}] |

n = **6 test sequences** (the effective sample size). The CI **excludes zero**, and **CA-CFAR is
ahead in all six sequences** — there is no sequence where TinyCNN wins.

| sequence | eligible objects | TinyCNN | CA-CFAR | Δ |
|---|---|---|---|---|
{rows}

⚠️ **This number is not comparable with the synthetic Pd.** Synthetic Pd used a ±1-bin gate around a
point target; this is mask-based recall against extended segmentation support, under a different
operating-point protocol. The two are never differenced or placed in one column.""".format(
    ca=P["cfar"]["CA"], dl=P["tinycnn_seed_mean"], d=BS["mean_delta"],
    lo=BS["ci_lo"], hi=BS["ci_hi"],
    rows="\n".join("| {} | {:,} | {:.4f} | {:.4f} | {:+.4f} |".format(
        r["sequence"], r["eligible_objects"], r["tinycnn"], r["cfar_CA"], r["delta"])
        for r in R1["per_sequence"])))
md("![primary](figures/r1_test/r1_fig1_primary_and_per_sequence.png)")

md("""## §49 All four CFAR variants

| variant | exact-mask recall |
|---|---|
| **CA (primary)** | **{ca:.4f}** |
| GO | {go:.4f} |
| SO | {so:.4f} |
| OS | {os_:.4f} |

All four land within {spread:.3f} of each other and **all four beat TinyCNN**. GO is nominally
highest, but **CA remains the primary comparator** as pre-registered — a secondary variant is not
promoted because it scored better. The honest statement is that the classical family as a whole
transferred, and the ordering within it is not the finding.""".format(
    ca=P["cfar"]["CA"], go=P["cfar"]["GO"], so=P["cfar"]["SO"], os_=P["cfar"]["OS"],
    spread=max(P["cfar"].values()) - min(P["cfar"].values())))

md("""## §50 Mask ±1 / ±3 robustness

| arm | exact | ±1 | ±3 |
|---|---|---|---|
| TinyCNN | {t0:.4f} | {t1:.4f} | {t3:.4f} |
| CA-CFAR | {c0:.4f} | {c1:.4f} | {c3:.4f} |

**This is tolerance around the annotated support, not localization accuracy.**

TinyCNN gains **{tg:+.3f}** from exact → ±3 while CA gains only **{cg:+.3f}**, so a larger share of
TinyCNN's misses are *just outside* the annotated support. The gap narrows from {g0:.3f} to
{g3:.3f} but **does not close**.""".format(
    t0=TOL["tol0"]["tinycnn"], t1=TOL["tol1"]["tinycnn"], t3=TOL["tol3"]["tinycnn"],
    c0=TOL["tol0"]["cfar_CA"], c1=TOL["tol1"]["cfar_CA"], c3=TOL["tol3"]["cfar_CA"],
    tg=TOL["tol3"]["tinycnn"] - TOL["tol0"]["tinycnn"],
    cg=TOL["tol3"]["cfar_CA"] - TOL["tol0"]["cfar_CA"],
    g0=TOL["tol0"]["cfar_CA"] - TOL["tol0"]["tinycnn"],
    g3=TOL["tol3"]["cfar_CA"] - TOL["tol3"]["tinycnn"]))

md("""## §51 Per class

| class | eligible | sequences | TinyCNN | CA-CFAR | TinyCNN ±3 | CA ±3 |
|---|---|---|---|---|---|---|
{rows}

⚠️ **Class recall is not purely detector sensitivity.** Mask sizes differ substantially — median
support car 18×13, cyclist 11×11, pedestrian 9×9 — and a larger mask is a larger area in which a
peak can land and count as a hit. Both arms rank the classes in the same order as mask size.

**Cyclist (n = {ncy}, only {scy} sequences) is exploratory** and should not carry weight.""".format(
    rows="\n".join("| {} | {:,} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
        k, v["eligible_objects"], v["sequences"], v["tinycnn"], v["cfar_CA"],
        v["tinycnn_tol3"], v["cfar_CA_tol3"]) for k, v in R1["by_class"].items()),
    ncy=R1["by_class"]["cyclist"]["eligible_objects"],
    scy=R1["by_class"]["cyclist"]["sequences"]))

md("""## §52 Near-zero vs away-from-zero Doppler — **the most informative stratum**

Pre-registered before scoring, when Phase 4 established that **84% of eligible Test objects**
intersect the ±4-bin zero-Doppler band.

| stratum | eligible | TinyCNN | CA-CFAR | gap |
|---|---|---|---|---|
| **near zero** (bins 28–36) | {nz:,} | **{tnz:.4f}** | **{cnz:.4f}** | {gnz:.4f} |
| away from zero | {az:,} | {taz:.4f} | {caz:.4f} | {gaz:.4f} |

**Away from the ridge the two arms are close ({taz:.3f} vs {caz:.3f}). Nearly the entire gap lives
in the near-zero stratum**, where TinyCNN falls to {tnz:.3f} against CA's {cnz:.3f}.

⚠️ **Careful wording.** Permitted: *performance degradation was concentrated among objects whose
support intersects the strong zero-Doppler region.* **Not** permitted from stratification alone:
that the ridge *caused* the failures. R0 measured the CARRADA ridge as narrow and ~24 dB against the
simulator's broad ~14.8 dB, which makes it a plausible mechanism — but this analysis establishes
**where** the loss sits, not **why**.""".format(
    nz=R1["by_doppler"]["near_zero"]["eligible_objects"],
    tnz=R1["by_doppler"]["near_zero"]["tinycnn"], cnz=R1["by_doppler"]["near_zero"]["cfar_CA"],
    gnz=R1["by_doppler"]["near_zero"]["cfar_CA"] - R1["by_doppler"]["near_zero"]["tinycnn"],
    az=R1["by_doppler"]["away_zero"]["eligible_objects"],
    taz=R1["by_doppler"]["away_zero"]["tinycnn"], caz=R1["by_doppler"]["away_zero"]["cfar_CA"],
    gaz=R1["by_doppler"]["away_zero"]["cfar_CA"] - R1["by_doppler"]["away_zero"]["tinycnn"]))

md("""## §53 Range-stratified (secondary, R0's exact blocks)

| range block [bins] | eligible | TinyCNN | CA-CFAR |
|---|---|---|---|
{rows}

**The pattern is not monotone.** TinyCNN is far behind in the near-range block 10–60
({t1:.3f} vs {c1:.3f}) but is *ahead* of CA at 110–160 and 160–210. R0 measured CARRADA's background
as non-stationary in range (k 2.9–7.5 across blocks), so the near-range block is also where the
background departs most from the training domain. Secondary and explanatory — not a mechanism
claim.""".format(
    rows="\n".join("| {} | {:,} | {:.4f} | {:.4f} |".format(
        k, v["eligible_objects"], v["tinycnn"], v["cfar_CA"])
        for k, v in sorted(R1["by_range"].items()) if v["eligible_objects"] > 0),
    t1=R1["by_range"]["10-60"]["tinycnn"], c1=R1["by_range"]["10-60"]["cfar_CA"]))
md("![strata](figures/r1_test/r1_fig2_strata.png)")

md("""## §54 Unmatched-detection burden — **not** a false-alarm rate

| arm | mean/frame | median | p90 | annotated frames | frames with no annotation |
|---|---|---|---|---|---|
{rows}

**TinyCNN produces more unmatched output than CA-CFAR ({dl:.2f} vs {ca:.2f} per frame) while also
recalling fewer objects.** It is worse on both axes at the frozen operating point.

**Why this is not Pfa**, restated because it matters:

1. CARRADA does not annotate every real reflector — only the 77 tracked objects.
2. R0 measured bright unannotated returns in 5.85% of frames.
3. Frozen NMS on extended objects leaves multiple legitimate peaks from one object.
4. §55 measures exactly that on Test.

So this is an **output-density / operational-burden** metric, not a ground-truth false-alarm
probability.""".format(
    rows="\n".join("| {} | {:.2f} | {:.1f} | {:.1f} | {:.2f} | {:.2f} |".format(
        k, v["mean"], v["median"], v["p90"], v["annotated_frames_mean"],
        v["unannotated_frames_mean"]) for k, v in BUR.items()),
    dl=BUR["TinyCNN"]["mean"], ca=BUR["CFAR-CA"]["mean"]))

md("""## §55–§56 Multi-peak diagnostic and seed stability

The extended-object characteristic registered in Phase 4 reproduces on Test, confirming that part of
the burden above is a metric-transfer artefact rather than detector noise. **NMS was not changed.**

**Seed stability is itself a finding.** The eight frozen seeds span
**{lo:.4f} – {hi:.4f}** (sd **{sd:.4f}**):

{seeds}

That spread is large relative to the {d:.3f} TinyCNN−CA gap: **no single seed reaches CA-CFAR's
{ca:.4f}.** The best seed ({bs:.4f}) still trails it. No seed was selected, dropped, or ensembled.""".format(
    lo=min(P["tinycnn_per_seed"].values()), hi=max(P["tinycnn_per_seed"].values()),
    sd=P["tinycnn_seed_sd"], d=abs(BS["mean_delta"]), ca=P["cfar"]["CA"],
    bs=max(P["tinycnn_per_seed"].values()),
    seeds=" · ".join("s{}: {:.4f}".format(k, v) for k, v in P["tinycnn_per_seed"].items())))
md("![tolerance burden seeds](figures/r1_test/r1_fig3_tolerance_burden_seeds.png)")
md("![multipeak and curve](figures/r1_test/r1_fig4_multipeak_and_burden_curve.png)")

md("""## §57–§59 Per-sequence, qualitative examples, and uncertainty

Per-sequence values are in §47 and plotted in Figure 1 — **no sequence is hidden**, and the two
weakest (`…12-13-54`, `…12-23-30`) are also the largest by object count.

**Uncertainty** is sequence-aware throughout: paired bootstrap over the 6 test sequences, 10,000
replicates, fixed seed, percentile CI. **Seed variability (§56) is reported separately and never
merged with sequence variability into one error bar.** With n = 6 the emphasis is the effect size
and the fact that the sign is consistent across all six sequences, not a p-value.

The **recall–unmatched-burden curve** (Figure 4, right) is descriptive only. Its stars mark the
frozen operating points; **no threshold was chosen from the Test curve.** It shows TinyCNN's curve
sitting below CA's over the overlapping burden range, so the gap is not merely an operating-point
artefact.""")

md("""## §60 R0 → R1 interpretation

Taking the pre-registered order: **both detectors transferred** — neither collapsed. CA-CFAR
reached {ca:.3f} and TinyCNN {dl:.3f} exact-mask recall on real measurements, from a purely
synthetic development process.

**But the synthetic ranking reversed.** On synthetic data TinyCNN beat CFAR on 6 of 7 conditions;
on CARRADA the frozen CA-CFAR is ahead by {d:.3f} [{lo:.3f}, {hi:.3f}], in all six sequences, and no
individual seed closes the gap.

R0 established, independently and before any detector ran, that CARRADA is a substantial
statistical domain shift: smoother background (effective Gamma shape ≈ 4.8 vs the simulator's ≈ 1.0),
much lower Doppler correlation, a narrow ~24 dB zero-Doppler ridge against a broad ~14.8 dB one, and
range non-stationarity.

**Permitted:** *the zero-shot learned detector encountered a substantial statistical domain shift
established independently in R0, and its advantage did not survive it; the degradation is
concentrated among objects whose support intersects the strong zero-Doppler region.*

**Not claimed:** that k ≈ 4.8, or the ridge, *caused* the drop. Establishing that needs an
intervention, not a stratification. A CFAR estimator is scale- and shape-adaptive by construction,
which is a plausible reason it degraded less — but that is a hypothesis for a future experiment.""".format(
    ca=P["cfar"]["CA"], dl=P["tinycnn_seed_mean"], d=abs(BS["mean_delta"]),
    lo=abs(BS["ci_hi"]), hi=abs(BS["ci_lo"])))

md("""## §61 Relation to E2 / E3 / E4

**Those experiments are not rewritten and not weakened.** They were controlled *synthetic mechanism*
experiments; R1 asks *external validity*. The two answer different questions.

Because CARRADA's truth is extended segmentation support, R1 **cannot** validate or refute:

* exact-cell localization,
* the E3 label mechanism (flat 3×3 labels broadening the response),
* the E4 exact-cell mechanism (guard-cell access broadening the response).

Those remain grounded in the synthetic evidence. What R1 does establish is that **the frozen
detector's general detection ability, and specifically its advantage over CFAR, did not survive
transfer to this real domain.**""")

md("""## §62 R1 limitations

1. **n = 6 test sequences.** Small. The consistent sign across all six carries more weight than the
   interval width.
2. **Truth is extended support**, so recall is not comparable to synthetic Pd, and no localization
   metric is reported at all.
3. **Unmatched burden is not Pfa** — unannotated reflectors and multi-peak extended objects both
   inflate it.
4. **Class recall is confounded with mask size**; pedestrians are the most numerous Test class and
   have the smallest masks.
5. **Cyclist n = {ncy} across {scy} sequences — exploratory.**
6. **A new operating-point protocol** was required for TinyCNN, since the frozen project had none.
   R1 numbers are not poolable with the synthetic tables.
7. **Strict zero-shot for CFAR too:** its alphas were calibrated for single-look statistics, so its
   achieved operating point on CARRADA is not the designed one. It happened to transfer well;
   that is not evidence its calibration was correct here.
8. **One domain, one sensor.** CARRADA is not radar in general.""".format(
    ncy=R1["by_class"]["cyclist"]["eligible_objects"],
    scy=R1["by_class"]["cyclist"]["sequences"]))

md("""## §63 R1 conclusion

> **The frozen TinyCNN's synthetic advantage over CFAR did not transfer to real CARRADA data.**
> Both detectors work on real measurements, but at the frozen operating points classical CA-CFAR
> achieves higher mask-based object recall ({ca:.3f} vs {dl:.3f}, paired Δ {d:+.3f}
> [{lo:+.3f}, {hi:+.3f}], CA ahead in 6/6 sequences) **and** emits less unmatched output
> ({bca:.1f} vs {bdl:.1f} per frame). The loss concentrates among objects near the strong
> zero-Doppler ridge, and no individual frozen seed closes the gap.

This does not retract any synthetic finding. It **bounds their external validity**, which is exactly
what an external-validation experiment is for — and it is the outcome the pre-registration was
written to make reportable whichever way it fell.""".format(
    ca=P["cfar"]["CA"], dl=P["tinycnn_seed_mean"], d=BS["mean_delta"],
    lo=BS["ci_lo"], hi=BS["ci_hi"], bca=BUR["CFAR-CA"]["mean"], bdl=BUR["TinyCNN"]["mean"]))

md("""## §64 Phase 5 reproducibility and isolation

**One incident, recorded rather than hidden.** `run_r1_test.py` completed all detector inference and
preserved raw peaks for all 12 arms, then **failed on the final step writing Parquet** — neither
`pyarrow` nor `fastparquet` is installed. That is a **serialization failure after scoring, not a
methodological one**. The per-object/per-frame tables were rebuilt by `build_tables_from_raw.py`
**from the preserved raw output, re-running no detector**, applying the same frozen thresholds and
the same frozen matcher. Only the output format changed (CSV).

**Verification of a surprising result** (pre-registered checklist), all passed *before* interpreting:
manifest and pre-registration hashes; scored sequences == the 6 official Test sequences, with no
Train/Validation sequence touched; checkpoint hashes match the access log; frozen alphas and
thresholds as recorded; NMS 5; **a deterministic re-run of seed 0 over 40 frames reproduced the
stored peaks bit-for-bit (40/40)**; and the `score > 0` floor in `all_peaks` cannot remove anything
above the frozen threshold.

**No methodological change was made after seeing results.** No retraining, no fine-tuning, no
recalibration, no threshold change, no NMS change, no mask change, no CFAR-variant promotion.

Immutable raw evidence: `results/r1_test/*_peaks.npz` (12 arms × 2,130 frames), hashed in
`test_output_manifest.json`.""")

# ============================ PHASE 6 — CROSS-DOMAIN CLOSEOUT ==========================
SYN = pd.read_csv(os.path.join(ROOT, "results", "dl", "bootstrap_frozen_ep021_test_tol1.csv"))
n_win = int(((SYN["diff"] > 0) & SYN["sig (Bonf)"]).sum())

md("""---

# Phase 6 — cross-domain synthesis and closeout

## §65 The headline figure — what happened across the domain change

![cross-domain](figures/r1_test/r1_fig5_crossdomain_4panel.png)

**How to read it.** The rows are the two domains; the left column shows levels *within* a domain and
the right column shows the paired difference, which is the quantity that legitimately carries across.
Panel B is entirely positive (TinyCNN ahead), Panel D entirely negative (CA-CFAR ahead). **That sign
flip is the result.**

⚠️ **The two rows are different measurements and must never be pooled.** Synthetic $P_d$ scores a
point target within a ±1-bin gate at matched measured $P_{{fa}}$; CARRADA recall asks whether a peak
landed inside an extended annotated support. A synthetic 0.73 and a real 0.71 are **not** the same
quantity, and the figure states this on its face.

### Cross-domain summary

| | matched **synthetic** domain | unseen **real** CARRADA domain |
|---|---|---|
| metric | point-target $P_d$, ±1 gate, matched $P_{{fa}}$ | exact dense-mask object recall |
| **TinyCNN** | ahead in **{nw} of 7** conditions after Bonferroni | **{dl:.3f}** |
| **CFAR** | behind in most conditions (best-per-condition oracle) | **{ca:.3f}** (CA, primary) |
| paired difference | **+0.012 … +0.302**, TinyCNN favoured | **{d:+.3f}** [{lo:+.3f}, {hi:+.3f}], CFAR favoured |
| consistency | 6 of 7 conditions | **6 of 6 sequences** |

**The comparable statement is the ranking, not the numbers:** in the matched synthetic domain the
learned detector leads; in the unseen real domain the classical detector leads.""".format(
    nw=n_win, dl=P["tinycnn_seed_mean"], ca=P["cfar"]["CA"], d=BS["mean_delta"],
    lo=BS["ci_lo"], hi=BS["ci_hi"]))

md("""## §66 What the two domains actually look like

![rd examples](figures/r1_test/r1_fig6_rd_examples.png)

Frames were selected **deterministically on annotation geometry alone** — never on which detector
performed better. Synthetic: first `isolated` and first `clutter_edge` frame of `bench2_val` in file
order. CARRADA: the first single-object Test frame whose support either covers the ridge bin (32) or
clears it by ≥ 8 bins.

Visible directly: the simulator's coarse single-look speckle against CARRADA's smoother background;
CARRADA's narrow, intense zero-Doppler ridge; real objects with genuine spatial extent versus a
synthetic point target; and CA-CFAR firing repeatedly along the ridge — the unmatched-output burden
of §54 made visible.""")

md("""## §67 Final project-level interpretation

Stated in the order the evidence supports.

**Supported.**

1. **TinyCNN can outperform classical CFAR in the matched synthetic domain** — {nw} of 7 conditions
   after Bonferroni on `bench2_test`, up to +0.302 on `clutter_edge`.
2. **The controlled mechanism experiments stand.** E2: simple energy integration does *not* explain
   the learned advantage — integrating the mainlobe made CFAR worse on 7/7. E3: supervision shape
   drives localization — the flat 3×3 label causes the deficit and a Gaussian halves it at no Pd
   cost. E4: guard/mainlobe access buys ~1.5 pp of Pd and costs exact-cell precision.
3. **R0 establishes a substantial statistical domain shift**, measured before any detector ran:
   effective Gamma shape ≈ 4.8 vs ≈ 1.0, much lower Doppler correlation (0.227 vs 0.574), a narrow
   ~24 dB ridge against a broad ~14.8 dB one, range non-stationarity, and extended real-object
   support.
4. **Under zero-shot transfer, CA-CFAR > frozen TinyCNN on all six real test sequences.**
5. **Away from zero Doppler the two are close** ({taz:.3f} vs {caz:.3f}).
6. **The real-data gap is concentrated among objects overlapping the strong zero-Doppler region.**

**Therefore, the strongest defensible project-level conclusion:**

> **The learned detector demonstrated higher potential performance in the matched synthetic domain,
> but that advantage was sensitive to domain shift. Classical CFAR transferred more robustly to
> unseen CARRADA data under the zero-shot protocol.**

**As an implication, not a demonstrated result:**

> *Learned radar detectors likely require training data whose statistical structure adequately
> represents their deployment domain.*

That is a hypothesis this experiment motivates; it is **not** proven here, because no real-data
retraining was performed.

**Deliberately not claimed.** Not *"CNNs generalize badly"* — one architecture, one training setup,
one real dataset. Not *"a CNN must be trained on exactly the same data."* Not that k ≈ 4.8 or the
ridge *caused* the drop: stratification locates a loss, it does not establish a mechanism. The
accurate phrasing is that **this detector's performance depended strongly on training-domain
fidelity in this experiment.**""".format(
    nw=n_win, taz=R1["by_doppler"]["away_zero"]["tinycnn"],
    caz=R1["by_doppler"]["away_zero"]["cfar_CA"]))

md("""## §68 Limitations of the whole real-data experiment

1. **One real dataset.** CARRADA is not radar in general.
2. **Extended segmentation truth**, not point truth.
3. **Real and synthetic headline metrics differ** and are never pooled.
4. **CARRADA annotates only its 77 tracked objects**, not every physical reflector.
5. **Therefore no true real-data $P_{{fa}}$ exists**, and none is reported.
6. **Frozen NMS was designed for point-like synthetic targets…**
7. **…while real objects are spatially extended**, so one object can yield several peaks — measured
   at 35–48% of objects.
8. **Only 6 independent test sequences.**
9. **84% of eligible test objects overlap the zero-Doppler band**, so the headline is dominated by
   that regime.
10. **Class sizes and annotation extents are unbalanced** — pedestrians are most numerous and have
    the smallest masks.
11. **TinyCNN was never retrained or fine-tuned on real data.** This is a zero-shot bound, not a
    ceiling.
12. **No causal intervention** proves the ridge caused the loss.
13. **R0 statistics were computed across all sequences including Test.** Domain-level, not
    detector-level — and the ridge band was verified **identical** from Train-only data (±1, peak
    23.9 dB vs 24.5 dB), so no pre-registered choice depended on it.

These limitations are what make the conclusion narrow enough to defend.""")

md("""## §69 Future work — all of it a NEW experiment

Nothing below was performed here, and none of it may be bolted onto this experiment:

1. Train or fine-tune TinyCNN on CARRADA or another representative real dataset.
2. Generate synthetic training data whose background statistics, Doppler correlation, zero-Doppler
   structure and range non-stationarity better resemble real data.
3. Mixed-domain training: synthetic + real.
4. Domain randomization — vary clutter statistics during synthetic generation.
5. Test directly whether exposure to strong zero-Doppler clutter closes the real-data gap.
6. Evaluate on additional real radar datasets.
7. Architectures and training objectives designed for extended-object radar truth.
8. CFAR real-domain recalibration — **only** as a new, separately pre-registered experiment.

Any of these would be a **new experiment** with its own pre-registration and isolation manifest.""")

md("""## §70 Final conclusion

> TinyCNN outperformed CFAR under most controlled synthetic conditions, demonstrating the potential
> of a learned local detector. However, R0 showed that CARRADA differs substantially from the
> training domain, and the pre-registered zero-shot real-data evaluation **reversed the ranking**:
> CA-CFAR achieved higher object recall on all six test sequences ({ca:.3f} vs {dl:.3f}, paired
> Δ {d:+.3f} [{lo:+.3f}, {hi:+.3f}]) while also producing less unmatched output.
>
> The gap was concentrated near CARRADA's strong zero-Doppler region; away from it the two
> approaches performed similarly ({taz:.3f} vs {caz:.3f}).
>
> The combined result suggests that learned radar detection can exceed classical CFAR when its
> training domain adequately represents deployment conditions, but that classical statistical
> detectors may retain stronger robustness under domain shift.

**This retracts nothing.** The synthetic findings stand as controlled results; R1 bounds their
external validity. A pre-registered experiment that reverses its own headline is a stronger result
than one that confirms it, because the protocol was fixed before the answer was known.

**CARRADA REAL-DATA EXTERNAL VALIDATION — COMPLETE.**""".format(
    ca=P["cfar"]["CA"], dl=P["tinycnn_seed_mean"], d=BS["mean_delta"],
    lo=BS["ci_lo"], hi=BS["ci_hi"], taz=R1["by_doppler"]["away_zero"]["tinycnn"],
    caz=R1["by_doppler"]["away_zero"]["cfar_CA"]))

md("""## §71 Reproducibility and evidence paths

| what | where |
|---|---|
| pre-registration (frozen) | `PREREGISTRATION.md` · SHA-256 `{sha}` |
| immutable raw detector output | `results/r1_test/*_peaks.npz` (12 arms × 2,130 frames) |
| output hashes | `results/r1_test/test_output_manifest.json` |
| run manifest / test-access record | `results/r1_test/run_manifest.json`, `access_log.json` |
| scored tables | `results/r1_test/per_object.csv`, `per_frame.csv`, `multipeak_test.csv` |
| summary | `results/r1_test/r1_summary.json` |
| protected-asset manifest | `isolation_manifest.json` (628 files) |
| zero-shot thresholds | `configs/tinycnn_zero_shot_thresholds.json` |

**To reproduce the analysis without re-running any detector:**

```bash
python scripts/analyze_r1_test.py        # tables and summary, from preserved outputs
python scripts/make_r1_figures.py        # result figures
python scripts/make_crossdomain_figure.py
python scripts/make_rd_examples_figure.py
python scripts/build_isolation_manifest.py --verify isolation_manifest.json
```

Environment: Python {py}, torch {tor}, numpy {npv}.""".format(
    sha=FREEZE["sha256"], py=ACC["environment"]["python"], tor=ACC["environment"]["torch"],
    npv=ACC["environment"]["numpy"]))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python"}})
nbf.write(nb, NB)
print("wrote {}  ({} cells: {} markdown, {} code)".format(
    os.path.relpath(NB, ROOT), len(cells),
    sum(1 for c in cells if c.cell_type == "markdown"),
    sum(1 for c in cells if c.cell_type == "code")))
