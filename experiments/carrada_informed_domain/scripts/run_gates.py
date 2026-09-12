"""Run the pre-flight gates on the CARRADA-informed splits, as a standalone wiring check.

The frozen training protocol passes `--skip-gates`, and this experiment reproduces that protocol
exactly -- the gates draw from the RNG, so running them inside a training process would change the
weights relative to the frozen arm. They are still worth running once on a brand-new dataset,
because they are what catches a transposed map or a mis-scaled input, so they run here in a
SEPARATE process whose output is discarded.

Calls `gates()` from the patched trainer, unmodified. Writes results/gates_ci.json.

    python run_gates.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))

SRC = os.path.join(EXP, "scripts_patched", "train_tinycnn_ci.py")
spec = importlib.util.spec_from_file_location("train_ci", SRC)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def main():
    import torch
    from dl.dataset import CachedRDDataset, build_cache
    from dl.models import TinyCNN
    from dl.train import seed_everything

    torch.set_num_threads(4)
    print("building / loading the pre-normalised cache for both new splits")
    for split in T.EXPECTED:
        build_cache(split)

    train_ds = CachedRDDataset("carrada_informed_train")
    val_ds = CachedRDDataset("carrada_informed_val")
    for name, ds in (("carrada_informed_train", train_ds), ("carrada_informed_val", val_ds)):
        if len(ds) != T.EXPECTED[name]:
            raise SystemExit("%s: expected %d frames, got %d"
                             % (name, T.EXPECTED[name], len(ds)))
    print("train %d | val %d  (both complete)" % (len(train_ds), len(val_ds)))

    measured = T.measured_pos_weight(train_ds)
    print("measured positive-cell ratio on the NEW training labels: 1:%.0f" % measured)
    print("  (the frozen contract fixes pos_weight at 100 regardless; reported for the record)")

    args = argparse.Namespace(width=16, pos_weight=100.0, device="cpu")
    seed_everything(0)
    model = TinyCNN(width=args.width)
    info = T.gates(model, train_ds, val_ds, args)
    info["measured_pos_weight_new_domain"] = measured
    info["pos_weight_used"] = 100.0
    info["train_frames"] = len(train_ds)
    info["val_frames"] = len(val_ds)

    p = os.path.join(EXP, "results", "gates_ci.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print("\nwrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
