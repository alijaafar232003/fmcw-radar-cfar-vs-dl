"""Detector networks. Model 0 = TinyCNN with a receptive field matched to CFAR.

See docs/dl_design.md §3. Key design point: CA-CFAR decides each cell from a 21x21 window
(guard 2 + train 8 -> half-width 10 -> 2*10+1). TinyCNN is built with the SAME receptive field,
so the two detectors see identical context and differ only in the DECISION RULE — hand-derived
statistic vs learned function. That makes Model 0 a controlled experiment rather than scaffolding.

Receptive field for stacked 3x3 convs at stride 1:   RF = 1 + sum(2 * dilation_i)
    dilations (1,2,3,4) -> RF = 1 + 2*(1+2+3+4) = 21   <- matches CFAR exactly

Dilations increase smoothly and share no common factor, following Hybrid-Dilated-Convolution
guidance: patterns with common factors (e.g. 2,4,8) never sample some input positions in
combination, producing "gridding" artifacts where the output is computed from a sparse lattice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Small dilated CNN: per-cell logits at full input resolution.

    No downsampling — the output must stay 256x128, and down/up-sampling risks displacing the
    peak bin (a 1-bin error degrades the localization metric).
    """

    def __init__(self, in_ch: int = 1, width: int = 16, dilations=(1, 2, 3, 4)):
        super().__init__()
        self.dilations = tuple(dilations)

        layers: list[nn.Module] = []
        c_in = in_ch

        for i, d in enumerate(self.dilations):
            is_last = i == len(self.dilations) - 1
            c_out = 1 if is_last else width

            # padding = dilation keeps the spatial size identical through every layer
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=d, dilation=d))

            if not is_last:
                layers.append(nn.BatchNorm2d(c_out))
                layers.append(nn.ReLU(inplace=True))

            c_in = c_out

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, R, V] -> logits [B, 1, R, V] (no sigmoid; the loss applies it)."""
        return self.net(x)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(2 * d for d in self.dilations)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_receptive_field(model: nn.Module, size=(129, 129), in_ch: int = 1) -> tuple[int, int]:
    """Empirically measure the RF by backpropagating from one output pixel.

    Returns (rf_rows, rf_cols): the extent of input pixels with nonzero gradient.

    The model is put in EVAL mode deliberately: BatchNorm in train mode normalises over the
    batch and spatial dimensions, which would couple every output to every input and report a
    spuriously global receptive field.
    """
    was_training = model.training
    model.eval()

    x = torch.zeros(1, in_ch, *size, requires_grad=True)
    out = model(x)

    cr, cv = size[0] // 2, size[1] // 2
    out[0, 0, cr, cv].backward()

    grad = x.grad[0].abs().sum(0)
    rows = torch.nonzero(grad.sum(1)).flatten()
    cols = torch.nonzero(grad.sum(0)).flatten()

    if was_training:
        model.train()

    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
