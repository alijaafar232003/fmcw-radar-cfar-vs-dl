"""Step 4 — does the CFAR code still behave on the 256 x 64 map?

Not a results run. Five checks, and if the third fails we stop rather than calibrate on top of
a broken noise estimate:

  1. ratio maps compute at the new size at all
  2. the valid (non-border) region is the predicted 236 x 44
  3. on pure noise, each estimator returns the level theory says it should:
        CA  ~ 1.00x   arithmetic mean of the ring, unbiased
        GO  ~ 1.09x   max of two half-ring means, so slightly high by construction
        OS  ~ 1.39x   0.75-quantile of exponential noise = ln 4
     These three numbers are the fingerprint of a healthy implementation. GO reading ~2.1x is
     how the half-ring bug was originally caught.
  4. runtime per frame per method
  5. a picture of the threshold surface over a clutter frame
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map          # noqa: E402
from radar.config import RadarConfig                       # noqa: E402
from radar.simulator import make_axes                      # noqa: E402

DATA = ROOT / "data" / "bench2_debug" / "frames"
OUT = ROOT / "results" / "cfar_sanity_threshold.png"

GUARD, TRAIN = 2, 8
KINDS = ("CA", "GO", "OS")
EXPECTED = {"CA": 1.00, "GO": 1.09, "OS": 1.386}   # ln 4 for OS


def frames_by_condition() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for fd in sorted(DATA.iterdir()):
        md = json.loads((fd / "metadata.json").read_text(encoding="utf-8"))
        out.setdefault(md.get("condition", "?"), []).append(fd)
    return out


def main() -> None:
    cfg = RadarConfig()
    by_cond = frames_by_condition()

    # ---- 1 & 2: shapes -------------------------------------------------------------
    power = np.load(next(iter(by_cond["isolated"])) / "rd_map_mag.npy") ** 2
    cc = CfarConfig(kind="CA", guard=GUARD, train=TRAIN, pfa=1e-4)
    ratio, valid, alpha = cfar_ratio_map(power, cc)

    hw = cc.half_width
    exp_r, exp_v = power.shape[0] - 2 * hw, power.shape[1] - 2 * hw
    # Sample a line through the MIDDLE. Row/column 0 is border and therefore all-invalid, so
    # measuring there reports 0 x 0 and looks like a failure when the mask is in fact correct.
    got_r = int(valid[:, valid.shape[1] // 2].sum())
    got_v = int(valid[valid.shape[0] // 2, :].sum())

    print("1-2. SHAPES")
    print(f"   map {power.shape}   half-width {hw}   ref cells {cc.num_ref_cells}")
    print(f"   valid region {got_r} x {got_v}   expected {exp_r} x {exp_v}   "
          f"{'OK' if (got_r, got_v) == (exp_r, exp_v) else 'MISMATCH'}")
    print(f"   testable share of map: {100 * valid.mean():.1f}%   "
          f"(Doppler axis keeps {100 * got_v / power.shape[1]:.0f}%)")

    # ---- 3: noise estimate vs theory ----------------------------------------------
    print("\n3. NOISE ESTIMATE ON PURE NOISE (estimate / true mean)")
    print(f"   {'method':<8}{'measured':>10}{'theory':>9}{'error':>9}")
    ok = True
    for kind in KINDS:
        c = CfarConfig(kind=kind, guard=GUARD, train=TRAIN, pfa=1e-4)
        vals = []
        for fd in by_cond["noise"]:
            p = np.load(fd / "rd_map_mag.npy") ** 2
            r, v, _ = cfar_ratio_map(p, c)
            # ratio = CUT / estimate, so estimate = CUT / ratio; compare its mean to the true
            # mean cell power of the same frame.
            est = p[v] / r[v]
            vals.append(est.mean() / p[v].mean())
        m = float(np.mean(vals))
        err = 100 * (m - EXPECTED[kind]) / EXPECTED[kind]
        flag = "" if abs(err) < 6 else "   <-- OFF"
        ok &= abs(err) < 6
        print(f"   {kind:<8}{m:>10.3f}{EXPECTED[kind]:>9.3f}{err:>8.1f}%{flag}")
    print(f"   -> {'all three match theory' if ok else 'MISMATCH: stop and investigate'}")

    # ---- 4: runtime ----------------------------------------------------------------
    print("\n4. RUNTIME (ms/frame)")
    for kind in KINDS:
        c = CfarConfig(kind=kind, guard=GUARD, train=TRAIN, pfa=1e-4)
        t0 = time.perf_counter()
        for fd in by_cond["mixed"]:
            cfar_ratio_map(np.load(fd / "rd_map_mag.npy") ** 2, c)
        dt = 1000 * (time.perf_counter() - t0) / len(by_cond["mixed"])
        print(f"   {kind:<8}{dt:>8.1f} ms")

    # ---- 5: threshold surface ------------------------------------------------------
    fd = by_cond["clutter"][0]
    p = np.load(fd / "rd_map_mag.npy") ** 2
    _, _, range_axis, vel_axis = make_axes(cfg)
    ext = [vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    floor = np.median(p)
    axes[0].imshow(10 * np.log10(p / floor + 1e-12), aspect="auto", origin="lower",
                   cmap="viridis", vmin=0, vmax=30, extent=ext)
    axes[0].set_title("RD map (dB above noise floor)", loc="left", fontsize=10)

    for ax, kind in zip(axes[1:], KINDS):
        c = CfarConfig(kind=kind, guard=GUARD, train=TRAIN, pfa=1e-4)
        r, v, a = cfar_ratio_map(p, c)
        # Divide only inside the valid region: the ratio map is zeroed on the border, so a
        # blanket divide overflows there before np.where discards it.
        est = np.full(p.shape, np.nan)
        est[v] = p[v] / r[v]
        ax.imshow(10 * np.log10(est / floor + 1e-12), aspect="auto", origin="lower",
                  cmap="viridis", vmin=0, vmax=30, extent=ext)
        ax.set_title(f"{kind} noise estimate", loc="left", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Velocity (m/s)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Range (m)", fontsize=9)

    fig.suptitle("Step 4 — CFAR noise estimate should track the clutter ridge, "
                 "not the targets", x=0.006, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="#fcfcfb")
    print(f"\n5. wrote {OUT}")


if __name__ == "__main__":
    main()
