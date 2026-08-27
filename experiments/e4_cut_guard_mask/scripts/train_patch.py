"""Train one E4 patch arm (masked or unmasked) for one seed. Writes only under this experiment.

Both arms use the IDENTICAL TinyCNN architecture, imported read-only, and the identical frozen
hyperparameters. Both see the SAME sampled CUT positions, drawn from a seed-derived RNG that does
not depend on the arm -- so the only difference between arms is whether the 24 guard cells are
replaced by the neutral value.

    python experiments/e4_cut_guard_mask/scripts/train_patch.py --arm masked --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "src"))

from dl.dataset import cache_paths                   # noqa: E402
from dl.models import TinyCNN, count_params          # noqa: E402
from dl.train import seed_everything                 # noqa: E402
from patches import HW_D, HW_R, extract, sample_cuts, verify_mask   # noqa: E402

TRAIN, VAL = "bench2_train", "bench2_val"
CUTS_PER_FRAME = 256
SAVE_AT = (5, 10, 15, 21)


def load_split(split):
    xp, yp = cache_paths(split)
    assert "test" not in split, "E4 must never touch bench2_test"
    return np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")


def epoch_batches(x, y, arm, rng, batch, shuffle=True):
    order = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    buf_x, buf_y = [], []
    for i in order:
        cuts = sample_cuts(x[i].shape, CUTS_PER_FRAME, rng)
        buf_x.append(extract(np.asarray(x[i]), cuts, masked=(arm == "masked")))
        buf_y.append(np.asarray(y[i])[cuts[:, 0], cuts[:, 1]].astype(np.float32))
        while sum(len(b) for b in buf_x) >= batch:
            X = np.concatenate(buf_x)
            Y = np.concatenate(buf_y)
            yield torch.from_numpy(X[:batch]), torch.from_numpy(Y[:batch])
            buf_x, buf_y = ([X[batch:]], [Y[batch:]]) if len(X) > batch else ([], [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("masked", "unmasked"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=21)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=100.0)
    args = ap.parse_args()

    (HERE / "checkpoints").mkdir(parents=True, exist_ok=True)
    (HERE / "logs").mkdir(parents=True, exist_ok=True)
    verify_mask()

    torch.set_num_threads(args.threads)
    xt, yt = load_split(TRAIN)
    xv, yv = load_split(VAL)

    tag = f"{args.arm}_s{args.seed}"
    seed_everything(args.seed)
    model = TinyCNN(width=16)

    cfg = dict(arm=args.arm, seed=args.seed, epochs=args.epochs, threads=args.threads,
               batch_size=args.batch_size, lr=args.lr, pos_weight=args.pos_weight,
               cuts_per_frame=CUTS_PER_FRAME, params=count_params(model),
               patch=[2 * HW_R + 1, 2 * HW_D + 1], started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (HERE / "logs" / f"config_{tag}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[{tag}] {count_params(model):,} params | pos_weight {args.pos_weight:g} | "
          f"{CUTS_PER_FRAME} CUTs/frame | batch {args.batch_size}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))
    log = (HERE / "logs" / f"train_{tag}.log").open("w", encoding="utf-8")
    hist = []

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        # RNG depends on seed+epoch only -- both arms see the SAME CUT positions.
        rng = np.random.default_rng(10_000 * args.seed + ep)
        model.train()
        ts = tn = 0.0
        for X, Y in epoch_batches(xt, yt, args.arm, rng, args.batch_size):
            opt.zero_grad()
            out = model(X[:, None])[:, 0, HW_R, HW_D]
            loss = lossf(out, Y)
            loss.backward()
            opt.step()
            ts += float(loss.detach()) * len(X)
            tn += len(X)

        model.eval()
        rngv = np.random.default_rng(500_000 + ep)
        vs = vn = 0.0
        with torch.no_grad():
            for X, Y in epoch_batches(xv, yv, args.arm, rngv, args.batch_size, shuffle=False):
                out = model(X[:, None])[:, 0, HW_R, HW_D]
                vs += float(lossf(out, Y)) * len(X)
                vn += len(X)

        rec = dict(epoch=ep, train_loss=ts / max(tn, 1), val_loss=vs / max(vn, 1),
                   seconds=time.perf_counter() - t0)
        hist.append(rec)
        line = (f"epoch {ep:>2} train {rec['train_loss']:.6f} val {rec['val_loss']:.6f} "
                f"({rec['seconds']:.1f}s)")
        print(f"[{tag}] {line}", flush=True)
        log.write(line + "\n")
        log.flush()

        if ep in SAVE_AT:
            torch.save({"state_dict": model.state_dict(), "epoch": ep,
                        "val_loss": rec["val_loss"], "config": cfg},
                       HERE / "checkpoints" / f"tinycnn_{tag}_ep{ep:03d}.pt")

    log.close()
    __import__("pandas").DataFrame(hist).to_csv(HERE / "logs" / f"history_{tag}.csv", index=False)
    print(f"[{tag}] done")


if __name__ == "__main__":
    main()
