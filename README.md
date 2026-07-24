# FMCW Radar Target Detection — CFAR vs Deep Learning

A simulation-based study comparing classical **CFAR** detectors against a **deep-learning**
detector for finding targets on FMCW **Range–Doppler (RD) maps**. Final-year project.

The radar is a medium-range **77 GHz automotive FMCW** profile
(B = 300 MHz, Tc = 32 µs, N = 512, M = 128 → 256 × 128 RD maps, 0.5 m / 0.48 m·s⁻¹ resolution).

## Status

| Component | State |
|---|---|
| Radar simulator (chirp → dechirp → RD map) | ✅ |
| Labelled dataset generator (7 scenarios, YAML-driven) | ✅ |
| RD-peak ↔ label ↔ mask validation | ✅ |
| SNR calibration (cell-SNR at the detector) | ✅ |
| Shared evaluation module (Pd, Pfa, ROC, localization) | ✅ |
| CFAR detectors (CA / OS / GO) | ✅ |
| Deep-learning detector | ⬜ planned |
| CFAR-vs-DL benchmark + ablations | ⬜ planned |

## Layout

```
src/radar/     radar config, simulator, RD processing, labels, IO, SNR calibration
src/data/      dataset generator (create_dataset.py, generate_single_frame.py)
src/cfar/      CA / OS / GO CFAR detectors + detection extraction
src/eval/      shared detection scoring (matching, Pd/Pfa/ROC) — used by CFAR AND DL
configs/       7 scenario definitions (clean, clutter, close_targets, multi_target, ...)
notebooks/     validate_rd_alignment.ipynb, cfar_comparison.ipynb
docs/          reference notes, CFAR methods review, benchmarking conventions
data/          generated datasets (git-ignored — regenerate from configs)
```

## Quick start

```bash
python -m venv venv
venv/Scripts/activate            # Windows;  source venv/bin/activate on Linux/macOS
pip install -r requirements.txt

# Generate all datasets (edit SCENARIO at the top of the file, or use --scenario)
python src/data/create_dataset.py

# Then open notebooks/validate_rd_alignment.ipynb or notebooks/cfar_comparison.ipynb
```

## Documentation

- `docs/project_reference_notes.md` — radar physics + parameter design
- `docs/cfar_methods_review.md` — CA / GO / SO / OS / VI / DL-CFAR with references
- `docs/benchmarking_conventions.md` — SNR definition + scoring rulebook
- `PROGRESS.md` — running log of what was done and why
