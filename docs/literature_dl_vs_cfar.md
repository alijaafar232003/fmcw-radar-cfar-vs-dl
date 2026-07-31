# Literature Review — Deep Learning vs CFAR for Radar Target Detection

Research conducted 30 July 2026 to answer one question: **when other groups compare a CNN against
CFAR, how large is the gain, and what do they do differently?** This directly addresses the concern
that our own result (DL winning in every condition) is too good to be true.

**Headline conclusion: the more carefully the comparison is controlled, the smaller the DL
advantage becomes. Our result currently resembles the *poorly-controlled* end of the literature.**

---

# 1. The four key papers

## 1.1 RadCNN — the closest match to this project ⭐
**F. Yavuz, "Radar Target Detection with CNN", EUSIPCO 2021** (ASELSAN Research Center)
[PDF](https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0001581.pdf)

Almost the same experiment as ours: a CNN on range-Doppler data, compared against CA/SOCA/GOCA/OS
CFAR, on *simulated* data with controlled clutter scenarios.

| Aspect | RadCNN | This project |
|---|---|---|
| Data | simulated pulse-Doppler, 3000×16 RD matrix | simulated FMCW, 256×128 RD map |
| Network | **1 conv layer, 4 filters of 5×3** (tiny) | 4 conv layers, 16 ch, ~5k params |
| Input | 25×5 patches of squared-magnitude RD | full map, dB above noise floor |
| CFAR baselines | CA, SOCA, GOCA, OS | CA, OS, GO |
| CFAR params | 20×2 reference, 4×2 guard, OS rank 0.75, Pfa 1e-3 | 21×21 window, OS rank 0.75, Pfa 1e-4 |
| Clutter | **Weibull (shape 1, 2) + K-distributed (shape 3, 15)** — heavy-tailed | Gaussian |
| Scenarios | S1 target at clutter border; S2 target inside clutter; S3 target in noise, window overlapping clutter; S4 noise only | isolated / masking / unresolved / clutter / multi / multipath / mixed |

### ⚠️ The decisive methodological difference
> *"The ambiguity matrix is constructed by **matched-filtering the received signal in range** and by
> subsequently taking FFT of the matched-filtered signal in Doppler space."*

**Both the CNN and the CFAR detectors operate on the matched-filtered map.** This is exactly the
fix proposed as Step B1 in `NEXT_SESSION.md`. In our pipeline, only the CNN effectively integrates
the mainlobe (via convolution); CFAR tests a single cell.

### Results

| Scenario | Outcome |
|---|---|
| **Average over all scenarios** | RadCNN better by **≈ 0.5 dB at low SNR**; **comparable at high SNR** |
| S1 — target at clutter border, inside clutter | RadCNN better, especially at low SNR |
| S2 — target covered by clutter | RadCNN better, **exceeding 1 dB** at high CNR |
| S3 — target near clutter, window mostly noise | **RadCNN WORSE by ~0.8 dB** at high CNR — *"the main challenge for future work"* |
| S4 — target in noise only | **RadCNN ON PAR with CFAR** (only ~10 % lower Pfa than OS/SOCA) |

Measured Pfa (design 1e-3): CA 1.3e-3, GOCA 1.2e-3, OS 1.6e-3, SOCA 1.7e-3, RadCNN 1.5e-3 —
**none of the CFAR variants held their preset Pfa under clutter.** (We observed the same effect.)

Cost: RadCNN ≈ **85× CA-CFAR** and ≈ 10× OS-CFAR in operation count.

**Two findings that matter most for us:**
1. **In pure noise, the CNN does NOT beat CFAR — it ties.** Our DL beat CA on `isolated`
   (0.880 vs 0.867). That is the single strongest indication that something in our protocol favours
   the network.
2. **A well-built CNN still LOSES in one scenario.** A detector that wins *everywhere* is atypical.

---

## 1.2 Tosi et al. — a dedicated CFAR-vs-CNN benchmark ⭐
**"Benchmarking CFAR and CNN-based Peak Detection Algorithms in ISAC under Hardware Impairments",
Nokia Bell Labs / KIT / TU Ilmenau, 2025** — [arXiv 2505.10969](https://arxiv.org/pdf/2505.10969)

Simulated OFDM ISAC radar images, CA-CFAR and censored/ordered-statistic CA-CFAR vs a CNN
(108 k parameters), Monte-Carlo over 1000 trials per SNR.

**Their CNN is given an explicit signal-processing advantage:** a *multi-window preprocessing stage*
feeding **5 differently-windowed versions** of the image (Rectangular, Hann, 3× DPSS) as separate
channels — so the network can compare sidelobe structure across windows.

### Results
* **Low SNR:** CNN performs *similarly* to the CFAR variants.
* **High SNR:** the CNN **deteriorates** — attributed to *"the lack of high SNRs in the training
  dataset"*. A direct demonstration that DL performance follows training coverage.
* **Closely-spaced targets:** the CNN separates them **significantly earlier** than CA-CFAR — but
  the authors attribute this explicitly to the **multi-windowing preprocessing**, not to the network
  being inherently smarter.
* **Windowing dominates false alarms:** without a window, sidelobes cause many false alarms for all
  detectors. (We use Blackman — good.)

> Conclusion: the CNN is *"generally capable of detecting targets with similar performance to the
> CFAR-based methods and even outperforms them in separating closely spaced targets. However, these
> benefits come at the cost of additional computational complexity."*

**Relevance:** a balanced, modern benchmark finds **rough parity**, with the CNN's one clear win
traceable to a preprocessing advantage — and even a *deterioration* where training data was thin.

---

## 1.3 "See Further Than CFAR" — where huge gaps come from
**Delft, IEEE 2024** — [arXiv 2402.12970](https://arxiv.org/html/2402.12970v2)

Real automotive radar, ground truth from **lidar** occupancy.

| Method | Pd | Pfa |
|---|---|---|
| Neural network | **65.32 %** | 1.99 % |
| Best of 5 CFAR configs | **1.72 %** | 0.09 % |

A ~38× gap. But **the authors themselves explain it away**:

> *"it is possible that a **calibration misalignment exists between the lidar and the radar that the
> neural network is learning to resolve**, significantly boosting the Pd compared to the conventional
> CFAR approaches."*
>
> *"CFAR results are much more sparse than lidar data; thus, the Pd is dropping."*

**This is our concern #5 in the literature, admitted in print:** the network learns the *labelling
convention*, while CFAR reports *physical peaks* and is scored against a ground truth it can never
match. A CFAR Pd of 1.72 % is not a real property of CFAR — it is an artifact of the protocol.

---

## 1.4 Unfolding State Space Model — a large claimed gain, weakly substantiated
**"Unfolding Target Detection with State Space Model", 2024** —
[arXiv 2410.22774](https://arxiv.org/html/2410.22774v1)

Real indoor FMCW data, depth-camera ground truth. Claims **≈10× higher detection rate at matched
false alarm** — 92.60 % vs CFAR's best of **11.10 %** (OS-CFAR).

**Caveats:** the tuned CFAR parameter values are **not disclosed**; no clutter model is
characterised; there is no discussion of comparison fairness. **A CFAR achieving only 11 % Pd is
almost certainly mis-configured**, not a fair baseline.

---

# 2. Synthesis — the pattern is unmistakable

| Study | Comparison rigour | DL advantage |
|---|---|---|
| **RadCNN** (2021) | matched filter for **both**; matched context; matched Pfa; simulated, controlled scenarios | **+0.5 dB avg; ~1 dB in clutter; −0.8 dB in one scenario; TIE in pure noise** |
| **Tosi et al.** (2025) | same input to both; windowing studied explicitly; Monte-Carlo | **≈ parity**; CNN better only at close-target separation (via preprocessing); **worse at high SNR** |
| **See Further Than CFAR** (2024) | real data, lidar GT, CFAR scored against sparse external truth | 65 % vs 1.7 % — **authors attribute it to sensor misalignment the NN learned** |
| **Unfolding SSM** (2024) | real data, camera GT, CFAR params undisclosed | 10× — **CFAR at 11 % Pd is implausible** |

**The better-controlled the experiment, the smaller the gain.** Every large reported gain traces to
a protocol artifact that the authors either admit or leave unexamined.

## Where our result sits

| Condition | Our DL vs CFAR | Literature (controlled) |
|---|---|---|
| **Pure noise / isolated** | DL wins (0.880 vs 0.867) | **RadCNN: TIE** ⚠️ |
| Clutter | DL wins clearly | RadCNN: DL wins ~1 dB ✅ consistent |
| Masking / close targets | DL wins (0.500 vs OS 0.333) | Tosi: CNN better at separation ✅ consistent |
| Any condition where DL loses | **none** | RadCNN loses 0.8 dB in S3 ⚠️ |

**Two red flags, both pointing the same way:** we win in pure noise where the literature reports a
tie, and we lose *nowhere*, whereas even a well-built CNN typically has a weak scenario.

The most likely cause remains the one identified independently before this review: **our CNN
integrates the target mainlobe while our CFAR tests a single cell.** RadCNN avoids this by
matched-filtering *before* both detectors.

---

# 3. What this changes in our plan

1. **Step B1 (give CFAR the matched-filter benefit) is now literature-supported, not just a hunch.**
   RadCNN does exactly this. It should be the *first* fix, not an optional one.
2. **Add a "pure noise / isolated" acceptance criterion:** after the fix, DL should be ≈ *tied* with
   CA-CFAR on isolated targets. If it still wins there, the protocol is still unfair.
3. **Expect — and report — a scenario where DL loses.** RadCNN loses in S3 (target in noise with the
   reference window overlapping clutter). We have no equivalent stratum; adding one would strengthen
   the study and test whether our DL has the same weakness.
4. **Report measured Pfa in a table** (as RadCNN does). The literature accepts that CFAR cannot hold
   its design Pfa under clutter — showing this is good practice, not a flaw to hide.
5. **Our Gaussian clutter is a genuine weakness relative to the field.** RadCNN uses Weibull
   (shape 1 and 2) and K-distributed (shape 3 and 15) clutter. This reinforces the realism-roadmap
   item already logged.
6. **Cost figures are consistent:** RadCNN ≈ 85× CA-CFAR; ours ≈ 8× (55 ms vs 6.9 ms). Fine to
   report.

---

# 4. Useful framings for the report

> "Reported gains for learned detectors over CFAR vary by more than an order of magnitude across the
> literature. Where the comparison is tightly controlled — matched pre-processing, matched reference
> window, matched false-alarm rate, and simulated data with known ground truth — the reported
> advantage is modest: approximately 0.5 dB on average, rising to about 1 dB in clutter, with the
> learned detector performing *on par* with CFAR in homogeneous noise and *worse* in at least one
> scenario [Yavuz, EUSIPCO 2021]. A recent dedicated benchmark similarly reports approximate parity,
> with the CNN's advantage in resolving closely-spaced targets attributable to a multi-window
> pre-processing stage rather than to the network itself [Tosi et al., 2025]."

> "Conversely, the largest reported gains arise where ground truth is derived from a different sensor.
> One study reporting 65 % detection probability for a network against 1.7 % for CFAR notes that the
> network may be learning to compensate a lidar-radar calibration misalignment, and that CFAR is
> penalised by the differing sparsity of the two representations [Delft, 2024]. Such results measure
> the evaluation protocol as much as the detector."

---

# 5. Papers I could not fully obtain (you could retrieve these)

1. **Wang et al.** — the original LeNet-5-style CNN radar detector that RadCNN improves on
   (RadCNN's reference [12]); and **Xie et al.** (reference [13]) who tested it on real data.
   RadCNN reports [12] beating CA-CFAR by only **0.2 dB at high SNR**, comparable at low SNR — a
   third independent "small gain" data point worth citing directly.
2. **"Artificial Neural Networks and Deep Learning Techniques Applied to Radar Target Detection:
   A Review"**, *Electronics* 11(1):156, 2022 — [doi.org/10.3390/electronics11010156](https://doi.org/10.3390/electronics11010156).
   Open access; a survey framing would strengthen the background chapter.
3. **Exact numeric values behind RadCNN's Figs. 5–7** — only the qualitative statements and the
   0.5 / 1.0 / 0.8 dB figures are in the text.
4. **"Deep learning-based lightweight radar target detection method"**, *J. Real-Time Image
   Processing*, 2023 — [Springer](https://link.springer.com/article/10.1007/s11554-023-01316-5)
   (paywalled; claims to outperform 1-D and 2-D CFAR on real ATC radar data).

---

## Sources
- Yavuz, *Radar Target Detection with CNN*, EUSIPCO 2021 — https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0001581.pdf
- Tosi, Schieler, Henninger, Semper, Mandelli, *Benchmarking CFAR and CNN-based Peak Detection Algorithms in ISAC under Hardware Impairments*, 2025 — https://arxiv.org/pdf/2505.10969
- *See Further Than CFAR: a Data-Driven Radar Detector Trained by Lidar*, 2024 — https://arxiv.org/abs/2402.12970
- *Unfolding Target Detection with State Space Model*, 2024 — https://arxiv.org/html/2410.22774v1
- *ANN and Deep Learning Techniques Applied to Radar Target Detection: A Review*, Electronics 2022 — https://doi.org/10.3390/electronics11010156
- *Deep learning-based lightweight radar target detection method*, JRTIP 2023 — https://link.springer.com/article/10.1007/s11554-023-01316-5
