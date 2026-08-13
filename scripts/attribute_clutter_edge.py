"""Attribute the DL clutter_edge margin (+0.261 over the best CFAR) before it is reported.

Two competing explanations, from PROGRESS.md Milestone 22:

**H-window — reference-window contamination.** CFAR's training ring reaches into the clutter ridge,
inflating its noise estimate and pushing its threshold up, so it misses targets it would otherwise
find. The network has no reference window and is not handicapped this way. *Prediction:* the DL
advantage is largest for targets CLOSE to the ridge, where CFAR's window is most contaminated, and
shrinks as the target moves away.

**H-dilution — stratum composition.** The stratum is 34% diluted: the ridge spans +/-4 Doppler bins,
targets sit at offsets 6-10, and the 21x9 window reaches only +/-3..4, so a third of the frames are
ordinary noise frames wearing the label. *Prediction:* the advantage is roughly uniform across the
stratum, i.e. it is not an edge effect at all but the network's general margin appearing on frames
that are not really edges.

The discriminator is **target clearance from the ridge edge**, in Doppler bins -- a property of the
data that no window and no detector can change. Scoring each method within each clearance band
compares like with like, and the two hypotheses predict different shapes.

Run on `bench2_val` only. CFAR peaks come from the cached run; DL is evaluated across all 8 seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig                                       # noqa: E402
from cfar.detect import all_peaks                                      # noqa: E402
from dl.infer import cfar_valid_mask, predict_heatmap                  # noqa: E402
from dl.models import TinyCNN                                          # noqa: E402
from dl.stratified_eval import NMS, _at_measured_pfa                   # noqa: E402
from eval import (MatchConfig, aggregate, evaluate_frame,              # noqa: E402
                  roc_curve, roc_thresholds)
from radar.config import RadarConfig                                   # noqa: E402
from radar.simulator import make_axes                                  # noqa: E402

SPLIT = "bench2_val"
TARGET_PFA = 1e-4
MC = MatchConfig(1, 1)
SHAPE = (256, 64)
RIDGE_HALF = 4          # measured in Milestone 22: >3 dB above floor spans +/-4 bins
CV = ROOT / "results" / "cfar_val"
DLR = ROOT / "results" / "dl"

BANDS = {"2 (closest)": [2], "3": [3], "4": [4], "5-6 (farthest)": [5, 6]}


def pd_at_matched(frames) -> tuple[float, float]:
    """Pd measured at the threshold where this subset's achieved Pfa equals the target."""
    roc = roc_curve(frames, roc_thresholds(frames, MC), MC)
    _, thr = _at_measured_pfa(roc, TARGET_PFA)
    evs = [evaluate_frame(p[p[:, 2] >= thr][:, :2] if (len(p) and thr == thr) else np.zeros((0, 2)),
                          t, s, MC, valid_mask=v) for p, t, s, v in frames]
    a = aggregate(evs)
    return a["pd"], a["pfa_per_cell"]


def main() -> None:
    cfg = CfarConfig()
    valid = cfar_valid_mask(SHAPE, cfg)
    _, _, _, vax = make_axes(RadarConfig())
    v0 = int(np.argmin(np.abs(vax)))

    frames_meta = pd.read_csv(CV / "frames.csv")
    truths = pd.read_csv(CV / "truths.csv")
    ids = frames_meta.loc[frames_meta.condition == "clutter_edge", "frame"].to_numpy()

    # Clearance: how many Doppler bins the nearest target sits BEYOND the ridge edge.
    # Window-independent by construction, so every method is scored on the same frames.
    tb, clearance = {}, {}
    for i in ids:
        g = truths[truths.frame == i]
        tb[i] = g[["range_bin", "velocity_bin"]].to_numpy(float)
        clearance[i] = int(min(abs(int(v) - v0) for v in g.velocity_bin) - RIDGE_HALF)

    cl = pd.Series(clearance)
    print(f"{len(ids)} clutter_edge frames on {SPLIT}")
    print("frames per clearance band (bins beyond the ridge edge):")
    print(cl.value_counts().sort_index().to_string())
    reach = cfg.half_width_d                       # 4 bins
    print(f"\nthe 21x9 window's Doppler reach is {reach} bins, so clearance <= {reach - 1} means "
          f"the training ring touches the ridge")
    print(f"diluted (window never reaches the ridge): "
          f"{100 * (cl > reach - 1).mean():.1f}% of frames\n")

    # ---------------------------------------------------------------- CFAR, from cached peaks
    rows = []
    for m in ("CA", "GO", "OS"):
        arr = np.load(CV / f"peaks_{m}.npz")["peaks"]
        by = {int(f): arr[arr[:, 0] == f][:, 1:] for f in np.unique(arr[:, 0])}
        for band, vals in BANDS.items():
            sub = [i for i in ids if clearance[i] in vals]
            fr = [(by.get(int(i), np.zeros((0, 3))), tb[i], SHAPE, valid) for i in sub]
            p, pfa = pd_at_matched(fr)
            rows.append(dict(method=m, seed=np.nan, band=band, n=len(sub), pd=p, pfa=pfa))
        print(f"  scored CFAR {m}")

    # ---------------------------------------------------------------- DL, all 8 seeds
    for ck_path in sorted(DLR.glob("tinycnn_n*_ep050.pt")):
        ck = torch.load(ck_path, weights_only=False)
        model = TinyCNN(width=ck["config"].get("width", 16))
        model.load_state_dict(ck["state_dict"])
        model.eval()

        peaks = {}
        for i in ids:
            mag = np.load(ROOT / "data" / SPLIT / "frames" / f"frame_{int(i):06d}" /
                          "rd_map_mag.npy")
            peaks[int(i)] = all_peaks(predict_heatmap(model, mag), valid, nms_size=NMS)

        for band, vals in BANDS.items():
            sub = [i for i in ids if clearance[i] in vals]
            fr = [(peaks[int(i)], tb[i], SHAPE, valid) for i in sub]
            p, pfa = pd_at_matched(fr)
            rows.append(dict(method="DL", seed=ck["config"]["seed"], band=band,
                             n=len(sub), pd=p, pfa=pfa))
        print(f"  scored {ck_path.name}")

    df = pd.DataFrame(rows)
    out = DLR / "clutter_edge_attribution.csv"
    df.to_csv(out, index=False)

    # ---------------------------------------------------------------- the discriminating table
    order = list(BANDS)
    dl = (df[df.method == "DL"].groupby("band").pd
          .agg(["mean", "std", "count"]).reindex(order))
    dl["se"] = dl["std"] / np.sqrt(dl["count"])
    cf = df[df.method != "DL"].pivot_table(index="band", columns="method",
                                           values="pd").reindex(order)

    tab = pd.DataFrame({
        "n frames": df[df.method == "CA"].set_index("band").n.reindex(order),
        "CA": cf["CA"], "GO": cf["GO"], "OS": cf["OS"],
        "best CFAR": cf.max(axis=1),
        "DL mean": dl["mean"], "DL SE": dl["se"],
    })
    tab["DL - OS"] = tab["DL mean"] - tab["OS"]
    tab["DL - best"] = tab["DL mean"] - tab["best CFAR"]

    print("\n" + "=" * 96)
    print("Pd at matched MEASURED Pfa 1e-4, by target clearance from the ridge edge")
    print("=" * 96)
    print(tab.round(3).to_string())

    near, far = tab["DL - OS"].iloc[0], tab["DL - OS"].iloc[-1]
    print(f"\nDL - OS at the CLOSEST band  : {near:+.3f}")
    print(f"DL - OS at the FARTHEST band : {far:+.3f}")
    print(f"difference (near - far)      : {near - far:+.3f}")
    print()
    print("H-window predicts the margin CONCENTRATES near the ridge (near >> far).")
    print("H-dilution predicts it is roughly UNIFORM across bands (near ~ far).")


if __name__ == "__main__":
    main()
