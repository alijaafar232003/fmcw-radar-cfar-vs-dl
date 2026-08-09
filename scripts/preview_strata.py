"""One representative frame per stratum, as a single sheet — the visual check for step 3.

Numbers can pass while the physics is wrong, so the scenarios get looked at as well as measured.
Specifically worth checking by eye:
  masking       two blobs, one clearly weaker
  unresolved    ONE blob where the labels say two targets
  clutter       a horizontal band at zero velocity, target sitting in it
  clutter_edge  target just clear of that band
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.config import RadarConfig          # noqa: E402
from radar.simulator import make_axes         # noqa: E402

DATA = ROOT / "data" / "bench2_debug" / "frames"
OUT = ROOT / "results" / "bench2_debug_strata.png"

ORDER = ["isolated", "masking", "unresolved", "clutter",
         "clutter_edge", "multi", "mixed", "noise"]

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"


def pick_one_per_stratum() -> dict[str, Path]:
    """Prefer a frame with a mid-range target count, so the sheet is representative."""
    best: dict[str, tuple[int, Path]] = {}
    for fd in sorted(DATA.iterdir()):
        md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
        cond = md.get("condition")
        n = int(md.get("num_targets", 0))
        score = abs(n - 2)
        if cond not in best or score < best[cond][0]:
            best[cond] = (score, fd)
    return {k: v[1] for k, v in best.items()}


def main() -> None:
    cfg = RadarConfig()
    _, _, range_axis, vel_axis = make_axes(cfg)
    frames = pick_one_per_stratum()

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for ax, cond in zip(axes.ravel(), ORDER):
        fd = frames.get(cond)
        if fd is None:
            ax.axis("off")
            continue

        mag = np.load(fd / "rd_map_mag.npy")

        # dB above the frame's own noise-floor median, NOT below its maximum. Peak normalisation
        # hides everything: a target at 14 dB cell SNR sits only a few dB above the brightest
        # noise, so on a 0-to-max scale it vanishes into the background. Referencing the noise
        # floor puts noise at 0 dB and shows each target at its actual cell SNR -- which is also
        # the normalisation the detector sees.
        power = mag ** 2
        floor = np.median(power)
        db = 10 * np.log10(power / floor + 1e-12)

        ax.imshow(
            db, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=30,
            extent=[vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]],
        )

        n = 0
        if (fd / "labels.csv").stat().st_size > 5:
            lab = pd.read_csv(fd / "labels.csv")
            n = len(lab)
            ax.scatter(lab["velocity_mps"], lab["range_m"], s=140, facecolors="none",
                       edgecolors="#ffffff", linewidths=1.6)

        md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
        cl = md.get("clutter", {})
        extra = (f"  ·  clutter {cl.get('cnr_db', 0):.0f} dB, c={cl.get('shape', 0):.2f}"
                 if cl.get("enabled") else "")

        ax.set_title(f"{cond}  ·  {n} target{'s' if n != 1 else ''}{extra}",
                     color=INK, fontsize=10, loc="left", pad=6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#c3c2b7")

    for ax in axes[:, 0]:
        ax.set_ylabel("Range (m)", color=MUTED, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Velocity (m/s)", color=MUTED, fontsize=9)

    fig.suptitle("bench2_debug — one frame per stratum (white rings = labelled targets)",
                 color=INK, fontsize=14, x=0.007, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
