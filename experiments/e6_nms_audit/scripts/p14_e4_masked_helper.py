"""Phase 6f - compute E4 masked-arm peaks for a given seed list, alongside the main E4 run.

E4's masked arm has no dense equivalent: every valid cell under test needs its own masked patch,
~13,216 per frame, 1,600 frames, 8 seeds. That is one process on four threads while twelve cores sit
idle, so this runs the tail seeds concurrently with `e4_evaluate_e4.py`.

This is a SCHEDULING change only. It does not reimplement anything: the module under
`scripts/attribution/` is imported by path and its own `masked_score_map`, `all_peaks`, `VALID`,
`NMS` and `save_peaks` are the ones called here, so the arithmetic is the same code object the main
run uses. Nothing is retrained; the frozen checkpoints are loaded read-only.

Safe to run against the live main process:

  * output goes to a temporary name and is renamed into place atomically, so a reader never sees a
    half-written file;
  * the main run skips any seed whose file already exists, and computes any seed whose file does
    not, so every seed is present when its loop ends either way. The worst case is that both
    processes compute the same seed and write identical bytes.

    python p14_e4_masked_helper.py 5 6 7
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "attribution", "e4_evaluate_e4.py")

spec = importlib.util.spec_from_file_location("e4_patched", SRC)
E4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E4)


def main(seeds):
    torch.set_num_threads(int(os.environ.get("E4_THREADS", "3")))
    ids = __import__("pandas").read_csv(E4.ROOT / "results" / "cfar_val" / "frames.csv") \
        .frame.to_numpy()

    for s in seeds:
        out = E4.PKD / ("peaks_masked_s%d_ep%03d.npz" % (s, E4.EPOCH))
        if out.exists():
            print("  s%d already present, skipping" % s, flush=True)
            continue
        ck = torch.load(E4.HERE / "checkpoints" / ("tinycnn_masked_s%d_ep%03d.pt" % (s, E4.EPOCH)),
                        weights_only=False)
        m = E4.TinyCNN(width=16)
        m.load_state_dict(ck["state_dict"])
        m.eval()

        t0, pk = time.perf_counter(), {}
        for i in ids:
            x = E4.normalize_rd(np.load(E4.ROOT / "data" / E4.SPLIT / "frames" /
                                        ("frame_%06d" % int(i)) / "rd_map_mag.npy"))
            pk[int(i)] = E4.all_peaks(E4.masked_score_map(m, x), E4.VALID, nms_size=E4.NMS)

        tmp = out.with_suffix(".tmp.npz")
        E4.save_peaks(pk, tmp)
        os.replace(tmp, out)
        print("  masked_s%d: %.1f min" % (s, (time.perf_counter() - t0) / 60), flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
