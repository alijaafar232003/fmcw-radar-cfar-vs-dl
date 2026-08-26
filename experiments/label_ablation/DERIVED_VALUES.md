# Derived values — label ablation

Computed by `build_label_cache.py`. Recorded here because `results/` is gitignored.

## Gaussian sigma, derived from the measured PSF

Source: `results/psf_profile.csv, off-bin (worst case), -3 dB width / 2.3548`

* sigma_range   = **0.6895 bins**
* sigma_Doppler = **0.7007 bins**

3x3 weights (peak normalised to 1):

| | -1 | 0 | +1 |
|---|---|---|---|
| **-1** | 0.1262 | 0.3493 | 0.1262 |
| **+0** | 0.3612 | 1.0000 | 0.3612 |
| **+1** | 0.1262 | 0.3493 | 0.1262 |

## pos_weight per arm

Rule: `w_arm = 100 * (sum(y_flat3x3) / sum(y_arm))`, holding total positive gradient mass
constant. Anchored to the incumbent 100 rather than re-derived per arm from class frequency,
because exact class balancing was measured worse in this project (Correction C3).

| arm | positive mass | ratio to reference | pos_weight |
|---|---|---|---|
| `point` | 16,122 | 0.1112x | **899.37** |
| `flat3x3` | 144,996 | 1.0000x | **100.00** |
| `gaussian` | 47,152 | 0.3252x | **307.51** |

`mass x pos_weight` is constant across arms to 0.000%.

Note the point arm is 0.1112x rather than exactly 1/9 = 0.1111: 3x3 blocks clip at the frame
border and overlap between close targets, so the ratio is measured rather than assumed.
