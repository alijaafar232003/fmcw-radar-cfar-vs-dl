# Project Schedule

**Today:** Fri 24 July 2026 · **Target (everything done):** Thu 20 August · **Hard max:** 1 September

Everything = code + report + presentation + rehearsal.

## Two rules that make this achievable
1. **Write in parallel, not at the end.** A chapter is drafted the same week its work is done,
   while it's fresh. Much of the Background/Method is already in `docs/` — reuse it.
2. **De-risk deep learning first.** DL is the biggest unknown. Goal of Week 1 is a *minimal*
   DL model training end-to-end, even if mediocre. Improve later. Never leave DL debugging to
   the final week.

## Scope control (protect the deadline)
- Keep the DL model **small** (small CNN / U-Net). A working small model beats a broken big one.
- Do the **proposal-required ablations** first; extras (DL size, I/Q input, footprint shape)
  only if time remains.
- Realism upgrades: **range-dependent amplitude only**. Heavy-tailed clutter / Swerling are
  optional stretch.

---

## Week 0 — This weekend (24–26 Jul): unblock + set up
- [ ] Notebook Q&A — understand `cfar_comparison.ipynb` (you asked for this; it matters for the viva)
- [ ] Finish CFAR loose ends: calibrate OS/GO to an exact Pfa; quick guard/train tuning sweep
- [ ] Create the **report skeleton** (chapters) and paste in existing `docs/` material
- **Milestone:** CFAR fully done; report document exists with Background half-drafted

## Week 1 — 27 Jul – 2 Aug: DL foundation (the risk, tackled first)
- [ ] Add range-dependent amplitude; generate the **frozen train/val/test** datasets
- [ ] PyTorch dataset loader (RD map + label mask)
- [ ] Small CNN/U-Net → heatmap; get it **training end-to-end** (loss going down)
- [ ] Inference → points via peak-find + NMS, scored through `src/eval`
- [ ] **Writing:** Background chapters (radar, chirps, RD map, CFAR) — mostly from `docs/`
- **Milestone (2 Aug): DL pipeline runs end-to-end and produces scored detections**

## Week 2 — 3–9 Aug: DL working + first comparison
- [ ] Train DL properly (tune on validation), get a decent Pd/Pfa
- [ ] **CFAR vs DL on the frozen test set** — ROC, Pd@Pfa, localization, runtime
- [ ] **Writing:** Method chapters (simulator, dataset design, SNR, CFAR, DL, evaluation)
- **Milestone (9 Aug): headline CFAR-vs-DL comparison exists with real numbers**

## Week 3 — 10–16 Aug: experiments + results
- [ ] Ablations: Pd-vs-SNR, clutter strength, #targets, close-target separation, CFAR params
- [ ] Generate all final figures + tables (reuse the notebooks)
- [ ] **Writing:** Results + Discussion chapters (using the real plots/numbers)
- **Milestone (16 Aug): all experiments done, all figures generated, results chapter drafted**

## Week 4 — 17–20 Aug: finalize
- [ ] Report: Introduction, Conclusion, Abstract; polish, references, full proofread
- [ ] **Presentation slides** (structure from the report; reuse figures)
- [ ] Repo cleanup + final commit + push
- **TARGET (20 Aug): report done, slides done**

## Buffer — 21 Aug – 1 Sep: rehearsal + overflow
- [ ] Rehearse the presentation several times (out loud, timed)
- [ ] Absorb any overrun from earlier weeks
- **HARD DEADLINE: 1 September**

---

## Weekly checkpoint (every Sunday, 10 min)
Ask: is this week's **milestone** met? If not, cut scope (fewer ablations, simpler model) —
do NOT slip the milestone. The buffer is for emergencies, not planned overflow.

## Where the report content already exists
| Chapter | Source already written |
|---|---|
| FMCW background, parameters | `docs/project_reference_notes.md` |
| CFAR methods + references | `docs/cfar_methods_review.md` |
| SNR / metrics / fairness | `docs/benchmarking_conventions.md` |
| What was built + why | `PROGRESS.md` |
