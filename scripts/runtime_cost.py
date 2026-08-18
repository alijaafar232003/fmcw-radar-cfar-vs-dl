"""T5 — computational cost, DL against CA/GO/OS/SO, on equal terms.

Runtime is a proposal-required metric and the one place a small network can beat a classical
detector outright. It is also the easiest number to quote misleadingly, so three things are
reported rather than one:

1. **Wall-clock ms per frame**, both arms timed in THIS process, back-to-back, at the same thread
   count -- so thermal state, CPU contention and BLAS threading are shared rather than compared
   across runs made hours apart on different machine loads.
2. **Multiply-accumulates per RD cell**, derived from the architecture, not timed. Hardware-free,
   so it survives the laptop this was measured on.
3. **Parameter count** -- 5,041 for TinyCNN, zero for CFAR.

WHAT THE ARITHMETIC ACTUALLY SAYS (pre-registered in `docs/test_preregistration.md` §4.2, before
this was run). The intuition "the network does the whole map in one pass while CFAR works per cell"
does NOT describe this project's CFAR: `src/cfar/cfar.py` is vectorised with `scipy.ndimage`, and
`uniform_filter` computes the box mean in O(1) per cell regardless of window size. So CA is not
paying 164 operations per cell, and TinyCNN's 4,896 MAC/cell is ~3 orders of magnitude MORE
arithmetic. The network's hope is not fewer operations but denser, more vectorisable ones -- and
OS-CFAR's per-cell rank selection is the thing that vectorises badly.

Predicted, before running: DL faster than OS, slower than CA, uncertain against GO/SO.

MEASUREMENT CONDITIONS decide the number and are recorded in the output: CPU only, thread count as
given, batch size 1 (per-frame latency, not throughput), `torch.no_grad()`, `eval()` mode. Disk I/O
is excluded -- frames are loaded first -- and NMS is excluded because both arms share it, so it
would cancel. A warm-up pass is discarded before timing.

    python scripts/runtime_cost.py --tag frozen --epoch 21 --threads 2 --n-frames 200
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfar.cfar import CfarConfig, cfar_ratio_map                     # noqa: E402
from dl.dataset import normalize_rd                                  # noqa: E402
from dl.models import TinyCNN, count_params                          # noqa: E402

SHAPE = (256, 64)
DLR = ROOT / "results" / "dl"


def dl_macs_per_cell(model: nn.Module) -> tuple[int, list[dict]]:
    """MACs per OUTPUT CELL. Padding preserves 256x64 at every layer, so cost is per-cell uniform.

    Dilation changes WHICH cells are read, never how many, so it does not enter the count.
    """
    rows, tot = [], 0
    for i, c in enumerate(m for m in model.modules() if isinstance(m, nn.Conv2d)):
        mac = c.kernel_size[0] * c.kernel_size[1] * c.in_channels * c.out_channels
        tot += mac
        rows.append(dict(layer=i, in_ch=c.in_channels, out_ch=c.out_channels,
                         kernel=f"{c.kernel_size[0]}x{c.kernel_size[1]}",
                         dilation=f"{c.dilation[0]},{c.dilation[1]}", mac_per_cell=mac))
    return tot, rows


def time_it(fn, frames, warmup: int = 5) -> float:
    """Median ms per frame. Median, not mean: one OS scheduler hiccup should not set the number."""
    for x in frames[:warmup]:
        fn(x)
    per = []
    for x in frames:
        t0 = time.perf_counter()
        fn(x)
        per.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(per))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--epoch", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0, help="one checkpoint; weights do not affect cost")
    ap.add_argument("--split", default="bench2_val")
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    fdirs = sorted((ROOT / "data" / args.split / "frames").iterdir())[:args.n_frames]
    mags = [np.load(fd / "rd_map_mag.npy") for fd in fdirs]          # I/O done up front, not timed
    powers = [m.astype(np.float64) ** 2 for m in mags]
    xs = [torch.from_numpy(normalize_rd(m))[None, None] for m in mags]
    print(f"{len(mags)} frames loaded from {args.split}, threads={args.threads}, batch=1")

    cp = DLR / f"tinycnn_{args.tag}_s{args.seed}_ep{args.epoch:03d}.pt"
    ck = torch.load(cp, weights_only=False)
    model = TinyCNN(width=ck["config"].get("width", 16))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    macs, layers = dl_macs_per_cell(model)
    cells = SHAPE[0] * SHAPE[1]

    @torch.no_grad()
    def dl_fn(x):
        return model(x)

    rows = [dict(method="DL (TinyCNN)", ms_per_frame=time_it(dl_fn, xs),
                 mac_per_cell=macs, mmac_per_frame=macs * cells / 1e6, params=count_params(model))]
    print(f"  DL: {rows[0]['ms_per_frame']:.2f} ms/frame")

    for kind in ("CA", "GO", "OS", "SO"):
        cfg = CfarConfig(kind=kind)
        ms = time_it(lambda p, c=cfg: cfar_ratio_map(p, c), powers)
        rows.append(dict(method=f"{kind}-CFAR", ms_per_frame=ms,
                         mac_per_cell=np.nan, mmac_per_frame=np.nan, params=0))
        print(f"  {kind}: {ms:.2f} ms/frame")

    df = pd.DataFrame(rows)
    df["vs_DL"] = df.ms_per_frame / df.loc[0, "ms_per_frame"]

    sp = "" if args.split == "bench2_val" else f"_{args.split.replace('bench2_', '')}"
    out = DLR / f"runtime_cost_{args.tag}_ep{args.epoch:03d}{sp}.csv"
    df.to_csv(out, index=False)

    meta = dict(split=args.split, n_frames=len(mags), threads=args.threads, batch_size=1,
                device="cpu", torch=torch.__version__, python=platform.python_version(),
                platform=platform.platform(), processor=platform.processor(),
                excludes="disk I/O, NMS (shared by both arms)", timing="median over frames",
                dl_layers=layers)
    (DLR / f"runtime_cost_{args.tag}_ep{args.epoch:03d}{sp}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"T5 — computational cost   [{args.split}, threads={args.threads}, batch 1, CPU]")
    print("=" * 78)
    print(df.round(3).to_string(index=False))
    print(f"\nTinyCNN: {macs:,} MAC/cell = {macs * cells / 1e6:.1f} MMAC/frame, "
          f"{count_params(model):,} params. CFAR: 0 params, O(1)/cell for CA "
          f"(separable box filter), per-cell rank selection for OS.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
