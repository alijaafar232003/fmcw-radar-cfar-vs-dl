"""Training: loss, the overfit-one-batch sanity gate, and the training loop.

Loss choice (docs/dl_design.md §6). Measured class imbalance is **~1 : 950** on the CARRADA-matched
sensor — 1:943–1:1027 across train/val/test, remeasured 11 Aug 2026. (It was 1:1563 on the old
256x128 map; the map halved while target counts stayed similar, so anything quoting 1:1563 is
stale.) Plain BCE collapses to "no target anywhere", which scores 99.9% accuracy and is useless —
accuracy must never be used as a metric here. We use `BCEWithLogitsLoss(pos_weight=...)`.

Why the exact pos_weight matters less than intuition suggests: the ROC is produced by SWEEPING
the detection threshold, so the network need not output a calibrated probability — it needs a
good *ranking* of cells. pos_weight mainly governs trainability and where the untuned operating
point sits. (This mirrors CFAR, whose operating point is also chosen by sweeping alpha.)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Tuned on the old 1:1563 ratio; the measured ratio is now ~1:950. Kept as a default
# only so old call sites behave as before -- pass it explicitly.
POS_WEIGHT_DEFAULT = 200.0


def make_loss(pos_weight: float = POS_WEIGHT_DEFAULT, device="cpu") -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))


def peak_offsets(logits: torch.Tensor, target: torch.Tensor, search: int = 3):
    """For each labelled blob, how far is the predicted peak from the label centre?

    Used by the overfit gate: a low loss alone does not prove the network puts its peak in the
    right place, so we check geometry explicitly.
    """
    from scipy import ndimage

    offsets = []

    for b in range(target.shape[0]):
        tgt = target[b, 0].detach().cpu().numpy()
        prob = torch.sigmoid(logits[b, 0]).detach().cpu().numpy()

        lbl, n = ndimage.label(tgt > 0.5)

        for k in range(1, n + 1):
            rs, vs = np.nonzero(lbl == k)
            r, v = int(round(rs.mean())), int(round(vs.mean()))

            r0, r1 = max(0, r - search), min(prob.shape[0], r + search + 1)
            v0, v1 = max(0, v - search), min(prob.shape[1], v + search + 1)

            patch = prob[r0:r1, v0:v1]
            pr, pv = np.unravel_index(int(patch.argmax()), patch.shape)
            offsets.append((abs((r0 + pr) - r), abs((v0 + pv) - v)))

    return offsets


def overfit_one_batch(model, dataset, n_frames: int = 8, steps: int = 300, lr: float = 1e-2,
                      pos_weight: float = POS_WEIGHT_DEFAULT, device="cpu", verbose: bool = True):
    """THE critical gate: can the model memorise a single batch?

    If loss does not fall and the peaks do not land on the labels, the pipeline is broken and no
    hyperparameter will fix it. Cheap, and it catches most wiring bugs.
    """
    xs, ys = zip(*[dataset[i] for i in range(n_frames)])
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)

    model = model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_loss(pos_weight, device)

    first = None
    history = []

    for s in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if first is None:
            first = float(loss)

        if verbose and (s % max(1, steps // 6) == 0 or s == steps - 1):
            print(f"    step {s:>4}  loss {float(loss):.5f}")

        history.append(float(loss))

    model.eval()
    with torch.no_grad():
        offsets = peak_offsets(model(x), y)

    within1 = sum(1 for dr, dv in offsets if dr <= 1 and dv <= 1)

    return {
        "loss_first": first,
        "loss_last": history[-1],
        "reduction": first / max(history[-1], 1e-12),
        "n_blobs": len(offsets),
        "peaks_within_1_bin": within1,
        "history": history,
    }


def seed_everything(seed: int) -> None:
    """Make a run repeatable: same seed, same weights, same batch order, same numbers."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # warn_only: a few ops have no deterministic CPU kernel; fail loudly only if one is hit.
    torch.use_deterministic_algorithms(True, warn_only=True)


OPTIMIZERS = {
    "adam": lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
    "adamw": lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
    "sgd": lambda p, lr, wd: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd),
    "rmsprop": lambda p, lr, wd: torch.optim.RMSprop(p, lr=lr, weight_decay=wd),
}


def train(model, train_ds, val_ds, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3,
          pos_weight: float = POS_WEIGHT_DEFAULT, device="cpu", verbose: bool = True,
          optimizer: str = "adam", weight_decay: float = 0.0, patience: int | None = None,
          seed: int = 0, on_epoch=None, resume_path=None, resume_every: int = 5):
    """Training loop. Reports train and val loss per epoch and keeps the BEST epoch's weights.

    What changed after the step-6d audit (all four defects it found):

    * `patience` gives early stopping on validation loss. `None` runs the full `epochs` — which is
      what you want when the question is "where SHOULD we stop?", since a stopped run cannot show
      you the shape of the curve past its stopping point.
    * the best epoch's weights are captured, not the last. The previous run kept whatever epoch 12
      happened to produce, with the validation loss still falling.
    * `val_ds` is used in full by the caller; nothing here subsamples it.
    * `pos_weight` is a parameter with no silent default tied to a stale class ratio.

    `optimizer` and `weight_decay` are exposed so a later run can change them without touching this
    file. `on_epoch(record)` is called after every epoch, for logging or live plots.

    NOTE: falling loss does not guarantee better detection. The meaningful signal is Pd at matched
    measured Pfa through `src/eval`; this loop only produces a model to measure.
    """
    seed_everything(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                          generator=g)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)

    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; have {sorted(OPTIMIZERS)}")
    opt = OPTIMIZERS[optimizer](model.parameters(), lr, weight_decay)
    loss_fn = make_loss(pos_weight, device)

    history, best = [], {"val_loss": float("inf"), "epoch": 0, "state": None}
    stale = 0
    start_ep = 0

    # Resume. A laptop sleeping mid-run cost a 4-hour, 3-seed job at epoch 31 because only model
    # weights were being checkpointed -- optimizer moments and RNG state were not, so the run could
    # not be continued, only restarted. This rolls a full state file every `resume_every` epochs.
    if resume_path is not None and Path(resume_path).exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        history, stale = ck["history"], ck["stale"]
        best = ck["best"]
        start_ep = ck["epoch"]
        torch.set_rng_state(ck["torch_rng"])
        g.set_state(ck["gen_rng"])
        np.random.set_state(ck["np_rng"])
        if verbose:
            print(f"  resumed from epoch {start_ep} (best so far: epoch {best['epoch']}, "
                  f"val {best['val_loss']:.5f})")

    for ep in range(start_ep, epochs):
        t0 = time.perf_counter()

        model.train()
        tr_sum = tr_n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr_sum += float(loss.detach()) * x.shape[0]
            tr_n += x.shape[0]

        model.eval()
        va_sum = va_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                va_sum += float(loss_fn(model(x), y)) * x.shape[0]
                va_n += x.shape[0]

        rec = {
            "epoch": ep + 1,
            "train_loss": tr_sum / max(tr_n, 1),
            "val_loss": va_sum / max(va_n, 1),
            "seconds": time.perf_counter() - t0,
            "lr": opt.param_groups[0]["lr"],
        }

        if rec["val_loss"] < best["val_loss"] - 1e-6:
            best = {"val_loss": rec["val_loss"], "epoch": rec["epoch"],
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1

        rec["best_so_far"] = best["epoch"]
        history.append(rec)

        if verbose:
            print(f"  epoch {rec['epoch']:>2}  train {rec['train_loss']:.5f}  "
                  f"val {rec['val_loss']:.5f}  ({rec['seconds']:.1f}s)"
                  f"{'  <- best' if stale == 0 else ''}", flush=True)

        if on_epoch is not None:
            on_epoch(rec)

        if resume_path is not None and (rec["epoch"] % resume_every == 0
                                        or rec["epoch"] == epochs):
            tmp = Path(str(resume_path) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": rec["epoch"], "history": history, "best": best, "stale": stale,
                        "torch_rng": torch.get_rng_state(), "gen_rng": g.get_state(),
                        "np_rng": np.random.get_state()}, tmp)
            tmp.replace(resume_path)      # atomic: a crash mid-write cannot corrupt the good file

        if patience is not None and stale >= patience:
            if verbose:
                print(f"  early stop: {patience} epochs without improvement "
                      f"(best was epoch {best['epoch']})")
            break

    return history, best
