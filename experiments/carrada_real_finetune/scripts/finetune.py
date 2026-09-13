"""Phases 5-7 — fine-tune the 8 CARRADA-informed checkpoints on CARRADA TRAIN.

Start point: arm B, `carrada_informed_domain/checkpoints/tinycnn_ci_s{S}_ep021.pt`. Never random
initialisation. Seed identity is preserved: B seed s -> C seed s.

All layers train, BatchNorm included, which is the point -- a 5,041-parameter network has no
sub-network worth freezing, and the running statistics are exactly what should adapt to a new
background distribution. The architecture is untouched: width 16, four convolutions, dilations
(1,2,3,4)/(1,1,1,1), no downsampling, 21x9 receptive field.

CONFIGURATION, fixed in advance and deliberately small:
    Adam, lr 1e-4 (10x below the synthetic 1e-3), batch 32, at most 10 epochs,
    BCEWithLogitsLoss with pos_weight from configs/label_and_posweight.json,
    early stopping on INTERNAL-DEVELOPMENT BCE with patience 2,
    model selection = the epoch with the lowest internal-development BCE.

Model selection uses internal-development LOSS, never recall, and never anything from official
CARRADA Validation -- which this process cannot open: the allow-list guard is installed in TRAIN
mode, so a Validation or Test read raises rather than succeeding quietly.

One pre-declared fallback exists: if lr 1e-4 diverges or does not move the development loss at all,
lr 3e-5 may be used instead, decided on training/development behaviour only. Whether it was used is
recorded per seed.

    python finetune.py --seed 0
    python finetune.py --all
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import guard                                                             # noqa: E402
guard.install(guard.ALLOW_TRAIN)

import torch                                                             # noqa: E402
from torch.utils.data import DataLoader, TensorDataset                   # noqa: E402

from dl.models import TinyCNN, count_params                              # noqa: E402

CACHE = os.path.join(EXP, "cache")
SRC_CKPT = os.path.join(ROOT, "experiments", "carrada_informed_domain", "checkpoints")
OUT_CKPT = os.path.join(EXP, "checkpoints")

LR_PRIMARY = 1e-4
LR_FALLBACK = 3e-5
BATCH = 32
MAX_EPOCHS = 10
PATIENCE = 2
THREADS = 2


def load_arrays():
    Xf = np.load(os.path.join(CACHE, "fit_x.npy"), mmap_mode="r")
    Yf = np.load(os.path.join(CACHE, "fit_y.npy"), mmap_mode="r")
    Xd = np.load(os.path.join(CACHE, "dev_x.npy"), mmap_mode="r")
    Yd = np.load(os.path.join(CACHE, "dev_y.npy"), mmap_mode="r")
    return Xf, Yf, Xd, Yd


def dev_loss(model, Xd, Yd, crit, batch=64):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(Xd), batch):
            x = torch.from_numpy(np.ascontiguousarray(Xd[i:i + batch]))[:, None].float()
            y = torch.from_numpy(np.ascontiguousarray(Yd[i:i + batch]))[:, None].float()
            loss = crit(model(x), y)
            tot += float(loss) * len(x)
            n += len(x)
    return tot / max(1, n)


def finetune_seed(seed, lr, Xf, Yf, Xd, Yd, pos_weight, log):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ck = torch.load(os.path.join(SRC_CKPT, "tinycnn_ci_s%d_ep021.pt" % seed),
                    weights_only=False)
    model = TinyCNN(width=16)
    model.load_state_dict(ck["state_dict"])
    assert count_params(model) == 5041, "architecture changed"

    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    idx = np.arange(len(Xf))
    base_dev = dev_loss(model, Xd, Yd, crit)
    history = [{"epoch": 0, "train_loss": None, "dev_loss": base_dev,
                "note": "arm B checkpoint before any fine-tuning"}]
    log("  epoch  0  dev %.6f   (arm B start)" % base_dev)

    best = {"dev": base_dev, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    bad = 0
    rng = np.random.default_rng(seed)

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        rng.shuffle(idx)
        t0, run, nb = time.perf_counter(), 0.0, 0
        for i in range(0, len(idx), BATCH):
            b = np.sort(idx[i:i + BATCH])
            x = torch.from_numpy(np.ascontiguousarray(Xf[b]))[:, None].float()
            y = torch.from_numpy(np.ascontiguousarray(Yf[b]))[:, None].float()
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        tr = run / max(1, nb)
        dv = dev_loss(model, Xd, Yd, crit)
        history.append({"epoch": ep, "train_loss": tr, "dev_loss": dv,
                        "seconds": time.perf_counter() - t0})
        log("  epoch %2d  train %.6f  dev %.6f  (%.0fs)%s"
            % (ep, tr, dv, time.perf_counter() - t0,
               "  <- best" if dv < best["dev"] else ""))

        if dv < best["dev"]:
            best = {"dev": dv, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                log("  early stop: %d epochs without a development improvement" % PATIENCE)
                break

    return model, best, history


def run_one(seed, Xf, Yf, Xd, Yd, pos_weight):
    logf = open(os.path.join(EXP, "logs", "finetune_s%d.log" % seed), "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log("seed %d  lr %g  batch %d  max %d epochs  patience %d  pos_weight %g"
        % (seed, LR_PRIMARY, BATCH, MAX_EPOCHS, PATIENCE, pos_weight))
    model, best, hist = finetune_seed(seed, LR_PRIMARY, Xf, Yf, Xd, Yd, pos_weight, log)
    lr_used, fallback = LR_PRIMARY, False

    # The one pre-declared fallback, judged on development behaviour only.
    improved = best["epoch"] > 0
    diverged = any(h["train_loss"] is not None and
                   (not np.isfinite(h["train_loss"])) for h in hist)
    if diverged or not improved:
        log("  lr %g %s -> retrying once at the pre-declared fallback lr %g"
            % (LR_PRIMARY, "diverged" if diverged else "never improved the development loss",
               LR_FALLBACK))
        model, best, hist = finetune_seed(seed, LR_FALLBACK, Xf, Yf, Xd, Yd, pos_weight, log)
        lr_used, fallback = LR_FALLBACK, True

    model.load_state_dict(best["state"])
    out = os.path.join(OUT_CKPT, "tinycnn_ft_s%d.pt" % seed)
    torch.save({"state_dict": best["state"], "epoch": best["epoch"],
                "dev_loss": best["dev"], "seed": seed, "lr": lr_used,
                "pos_weight": pos_weight, "fallback_lr_used": fallback,
                "source_checkpoint": "carrada_informed_domain/checkpoints/"
                                     "tinycnn_ci_s%d_ep021.pt" % seed}, out)
    rec = {"seed": seed, "lr": lr_used, "fallback_lr_used": fallback,
           "pos_weight": pos_weight, "selected_epoch": best["epoch"],
           "best_dev_loss": best["dev"], "start_dev_loss": hist[0]["dev_loss"],
           "epochs_run": len(hist) - 1, "history": hist,
           "checkpoint_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
    log("  selected epoch %d  dev %.6f  (start %.6f)" % (best["epoch"], best["dev"],
                                                         hist[0]["dev_loss"]))
    logf.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(THREADS)
    os.makedirs(OUT_CKPT, exist_ok=True)

    meta = json.load(open(os.path.join(EXP, "configs", "label_and_posweight.json"),
                          encoding="utf-8"))
    pw = float(meta["pos_weight"])
    Xf, Yf, Xd, Yd = load_arrays()
    print("fit %s   dev %s   pos_weight %g" % (Xf.shape, Xd.shape, pw))

    seeds = range(8) if a.all else [a.seed]
    recs = []
    for s in seeds:
        recs.append(run_one(s, Xf, Yf, Xd, Yd, pw))
        p = os.path.join(EXP, "results", "finetune_s%d.json" % s)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({**recs[-1], "access_report": guard.access_report()}, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main()
