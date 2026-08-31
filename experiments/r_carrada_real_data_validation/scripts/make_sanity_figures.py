"""Phase 4 visual sanity checks — NON-TEST SOFTWARE SANITY CHECK.

Verifies orientation, coordinate mapping, mask reconstruction, candidate location and matching
behaviour. It is NOT a performance comparison, and frames are chosen by a deterministic seeded
draw over CARRADA **Train** sequences, independently of any detector output.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import carrada_input as ci      # noqa: E402
import detectors as det         # noqa: E402
import eligibility as el        # noqa: E402
import metrics as mt            # noqa: E402

FIGS = os.path.join(EXP, "figures", "phase4_sanity")
SEED = 20260831
V_BIN, R_BIN = 0.42013, 0.1952


def main():
    os.makedirs(FIGS, exist_ok=True)
    rng = np.random.default_rng(SEED)
    seqs = ci.sequences("Train")
    picks = []
    for s in rng.choice(seqs, size=3, replace=False):
        ann = [f for f in ci.frames(str(s)) if ci.objects(str(s), f)]
        picks.append((str(s), str(rng.choice(ann))))

    model = det.load_frozen_tinycnn(0)
    import json
    thr = {r["seed"]: r["threshold"] for r in json.load(
        open(os.path.join(EXP, "configs", "tinycnn_zero_shot_thresholds.json")))["thresholds"]}

    valid = el.valid_region()
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]

    for seq, fr in picks:
        D = ci.load_rd_db(seq, fr)
        objs = ci.objects(seq, fr)
        pk_dl = det.apply_threshold(det.tinycnn_peaks(model, D), thr[0])
        pk_ca = det.cfar_peaks_at_alpha(D, kind="CA")   # frozen operating point: ratio >= alpha

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
        for ax, pk, name in ((axes[0], pk_dl, "TinyCNN (seed 0, zero-shot threshold)"),
                             (axes[1], pk_ca, "CA-CFAR (frozen alpha)")):
            ax.imshow(D - np.median(D), aspect="auto", origin="lower", cmap="magma",
                      vmin=-5, vmax=35, extent=[-13.44, 13.44, 0, 50])
            ax.grid(False)

            # valid-region boundary
            ax.add_patch(plt.Rectangle(
                ((cols.min() - 32.5) * V_BIN, rows.min() * R_BIN),
                (cols.max() - cols.min() + 1) * V_BIN,
                (rows.max() - rows.min() + 1) * R_BIN,
                fill=False, edgecolor="#4FC3F7", lw=1.1, ls="--", label="valid region"))

            # zero-Doppler band
            lo, hi = el.ZERO_DOPPLER_BAND
            ax.axvspan((lo - 32.5) * V_BIN, (hi - 32.5) * V_BIN, color="#FFFFFF", alpha=0.12,
                       label="zero-Doppler band ±4")

            # dense support + matching outcome
            supports = [el.valid_support(o["dense"]) for o in objs]
            res = mt.score_frame(pk, objs, tolerance=0)
            for i, o in enumerate(objs):
                c = o["dense"]
                ax.scatter((c[:, 1] - 32) * V_BIN, c[:, 0] * R_BIN, s=2, marker="s",
                           facecolors="none", edgecolors="#39FF14", linewidths=0.4,
                           label="dense support")
                matched = res["per_object"][i]["matched"]
                r0, r1 = c[:, 0].min(), c[:, 0].max()
                d0, d1 = c[:, 1].min(), c[:, 1].max()
                ax.add_patch(plt.Rectangle(
                    ((d0 - 32.5) * V_BIN, r0 * R_BIN), (d1 - d0 + 1) * V_BIN,
                    (r1 - r0 + 1) * R_BIN, fill=False, lw=1.4,
                    edgecolor="#39FF14" if matched else "#FF4136",
                    label="matched" if matched else "missed"))

            if len(pk):
                inside = np.zeros(len(pk), dtype=bool)
                allsup = set()
                for s_ in supports:
                    allsup |= set(map(tuple, s_.tolist()))
                for j, p in enumerate(pk):
                    inside[j] = (int(p[0]), int(p[1])) in allsup
                ax.scatter((pk[~inside, 1] - 32) * V_BIN, pk[~inside, 0] * R_BIN, s=26,
                           marker="x", c="#FFD166", linewidths=1.0, label="peak (unmatched)")
                ax.scatter((pk[inside, 1] - 32) * V_BIN, pk[inside, 0] * R_BIN, s=44,
                           marker="o", facecolors="none", edgecolors="#00E5FF", linewidths=1.3,
                           label="peak inside support")
            ax.set_xlabel("velocity  [m/s]")
            ax.set_title("{}\npeaks {}  matched {}  unmatched {}".format(
                name, res["n_peaks"], res["n_matched_peaks"], res["n_unmatched_peaks"]),
                fontsize=9)
            h, l = ax.get_legend_handles_labels()
            uniq = dict(zip(l, h))
            ax.legend(uniq.values(), uniq.keys(), fontsize=6.5, loc="upper right",
                      facecolor="black", edgecolor="none", labelcolor="white", framealpha=0.6)
        axes[0].set_ylabel("range  [m]")
        fig.suptitle("NON-TEST SOFTWARE SANITY CHECK — CARRADA Train {} frame {}\n"
                     "verifies orientation, coordinates, mask reconstruction and matching; "
                     "NOT a performance comparison".format(seq, fr), fontsize=9.5)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = os.path.join(FIGS, "phase4_sanity_{}_{}.png".format(seq, fr))
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("wrote", os.path.relpath(p, ROOT))
    print("\nframes chosen by deterministic seeded draw over CARRADA Train, "
          "independently of detector output.")


if __name__ == "__main__":
    sys.exit(main())
