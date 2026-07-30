"""Training: loss, the overfit-one-batch sanity gate, and the training loop.

Loss choice (docs/dl_design.md §6). Measured class imbalance is **1 : 1563** (8,388 positive
cells in 13,107,200). Plain BCE collapses to "no target anywhere", which scores 99.94% accuracy
and is useless — accuracy must never be used as a metric here. We use
`BCEWithLogitsLoss(pos_weight=...)`.

Why the exact pos_weight matters less than intuition suggests: the ROC is produced by SWEEPING
the detection threshold, so the network need not output a calibrated probability — it needs a
good *ranking* of cells. pos_weight mainly governs trainability and where the untuned operating
point sits. (This mirrors CFAR, whose operating point is also chosen by sweeping alpha.)
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


def train(model, train_ds, val_ds, epochs: int = 5, batch_size: int = 16, lr: float = 1e-3,
          pos_weight: float = POS_WEIGHT_DEFAULT, device="cpu", verbose: bool = True):
    """Standard loop. Reports train and val loss per epoch.

    NOTE: decreasing loss does not guarantee better detection. The meaningful signal is a
    detection metric (Pd at fixed Pfa) computed through src/eval — see the notebook, which
    evaluates the model properly after training.
    """
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_loss(pos_weight, device)

    history = []

    for ep in range(epochs):
        t0 = time.perf_counter()

        model.train()
        tr_sum = tr_n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr_sum += float(loss) * x.shape[0]
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
        }
        history.append(rec)

        if verbose:
            print(f"  epoch {rec['epoch']:>2}  train {rec['train_loss']:.5f}  "
                  f"val {rec['val_loss']:.5f}  ({rec['seconds']:.1f}s)")

    return history
