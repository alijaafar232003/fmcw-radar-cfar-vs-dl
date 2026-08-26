"""Train one (arm, seed) of the label ablation. Writes ONLY under experiments/label_ablation/.

Deliberately does NOT use `scripts/train_tinycnn.py`: that script writes checkpoints to
`results/dl/`, an existing project output directory. This driver imports the same training loop
(`dl.train.train`) and the same seeding, and replicates the script's call sequence exactly --
`torch.set_num_threads(threads)`, then `seed_everything(seed)`, then `TinyCNN(width)`, then
`train(...)` -- so a flat3x3 run at threads=2 should reproduce the frozen checkpoints bit-for-bit
at epoch 21. `verify_flat3x3_matches_frozen.py` asserts that.

The input tensors come from the project's own `nn_cache_x.npy`, opened READ-ONLY. Only the labels
differ between arms, and those live in this folder.

    python experiments/label_ablation/run_training.py --arm gaussian --seed 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from dl.dataset import cache_paths                      # noqa: E402
from dl.models import TinyCNN                           # noqa: E402
from dl.train import seed_everything, train             # noqa: E402
from labels import ARMS                                 # noqa: E402

EXPECTED = {"bench2_train": 8400, "bench2_val": 1600}
SAVE_AT = (5, 10, 15, 21)
CKPT = HERE / "checkpoints"
LOGS = HERE / "logs"


class ArmDataset(Dataset):
    """Project inputs (read-only) paired with this arm's labels."""

    def __init__(self, split: str, arm: str):
        xp, _ = cache_paths(split)
        self.x = np.load(xp, mmap_mode="r")                      # READ-ONLY, never written
        self.y = np.load(HERE / "cache" / f"y_{arm}_{split}.npy", mmap_mode="r")
        if len(self.x) != len(self.y):
            raise SystemExit(f"{split}: {len(self.x)} inputs vs {len(self.y)} labels")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (torch.from_numpy(np.asarray(self.x[i], dtype=np.float32))[None],
                torch.from_numpy(np.asarray(self.y[i], dtype=np.float32))[None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=21)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=16)
    args = ap.parse_args()

    CKPT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    pw = float(__import__("pandas").read_csv(HERE / "results" / "pos_weight_derivation.csv")
               .set_index("arm").loc[args.arm, "pos_weight"])

    torch.set_num_threads(args.threads)                  # exactly as train_tinycnn.py does

    train_ds, val_ds = ArmDataset("bench2_train", args.arm), ArmDataset("bench2_val", args.arm)
    for name, ds in (("bench2_train", train_ds), ("bench2_val", val_ds)):
        if len(ds) != EXPECTED[name]:
            raise SystemExit(f"{name}: expected {EXPECTED[name]} frames, got {len(ds)}")

    tag = f"{args.arm}_s{args.seed}"
    print(f"[{tag}] pos_weight {pw:g} | threads {args.threads} | {args.epochs} epochs | "
          f"train {len(train_ds):,} val {len(val_ds):,}")

    seed_everything(args.seed)                           # exactly as train_tinycnn.py does
    model = TinyCNN(width=args.width)

    cfg = dict(arm=args.arm, seed=args.seed, epochs=args.epochs, pos_weight=pw,
               threads=args.threads, batch_size=args.batch_size, lr=args.lr, width=args.width,
               optimizer="adam", weight_decay=0.0, patience=None,
               params=sum(p.numel() for p in model.parameters()),
               started=time.strftime("%Y-%m-%d %H:%M:%S"),
               torch=torch.__version__, python=sys.version.split()[0])
    (LOGS / f"config_{tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    log = (LOGS / f"train_{tag}.log").open("w", encoding="utf-8")

    def on_epoch(rec):
        line = (f"epoch {rec['epoch']:>2} train {rec['train_loss']:.6f} "
                f"val {rec['val_loss']:.6f} ({rec['seconds']:.1f}s)")
        print(f"[{tag}] {line}", flush=True)
        log.write(line + "\n")
        log.flush()
        if rec["epoch"] in SAVE_AT:
            torch.save({"state_dict": model.state_dict(), "epoch": rec["epoch"],
                        "val_loss": rec["val_loss"], "config": cfg},
                       CKPT / f"tinycnn_{tag}_ep{rec['epoch']:03d}.pt")

    history = train(model, train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, pos_weight=pw, device="cpu", verbose=False,
                    optimizer="adam", weight_decay=0.0, patience=None, seed=args.seed,
                    on_epoch=on_epoch)

    log.close()
    __import__("pandas").DataFrame(history).to_csv(LOGS / f"history_{tag}.csv", index=False)
    print(f"[{tag}] done -- {len(list(CKPT.glob(f'tinycnn_{tag}_ep*.pt')))} checkpoints")


if __name__ == "__main__":
    main()
