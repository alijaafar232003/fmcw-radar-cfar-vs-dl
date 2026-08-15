"""Detector networks. Model 0 = TinyCNN with a receptive field matched to CFAR.

See docs/dl_design.md §3. Key design point: CFAR decides each cell from a fixed window, and
TinyCNN is built with the SAME receptive field, so the two detectors see identical context and
differ only in the DECISION RULE — hand-derived statistic vs learned function. That is what makes
Model 0 a controlled experiment rather than scaffolding, and it is why the receptive field is
derived from the CFAR geometry rather than chosen for the network's own convenience.

The window is **21 x 9** (guard 2/2, train 8/2), selected in step 6 on a dedicated tuning split.
It is rectangular because the two axes are not alike on this sensor: range has 256 bins and no
large-scale structure, while Doppler has only 64 and carries the clutter ridge, so a window sized
for range spans a third of the Doppler axis and drags ridge energy across a wide band of
thresholds. Matching that asymmetry costs the network nothing and keeps the comparison controlled.

Receptive field for stacked 3x3 convs at stride 1, per axis:  RF = 1 + sum(2 * dilation_i)
    range   dilations (1,2,3,4) -> RF = 1 + 2*(1+2+3+4) = 21
    Doppler dilations (1,1,1,1) -> RF = 1 + 2*(1+1+1+1) =  9

The range dilations increase smoothly and share no common factor, following
Hybrid-Dilated-Convolution guidance: patterns with common factors (e.g. 2,4,8) never sample some
input positions in combination, producing "gridding" artifacts where the output is computed from a
sparse lattice. The Doppler dilations are all 1, which is dense by construction and cannot grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Small dilated CNN: per-cell logits at full input resolution.

    No downsampling — the output must stay 256x128, and down/up-sampling risks displacing the
    peak bin (a 1-bin error degrades the localization metric).
    """

    def __init__(
        self,
        in_ch: int = 1,
        width: int = 16,
        dilations_r=(1, 2, 3, 4),   # range axis   -> RF 21
        dilations_d=(1, 1, 1, 1),   # Doppler axis -> RF 9
    ):
        super().__init__()

        if len(dilations_r) != len(dilations_d):
            raise ValueError("dilations_r and dilations_d must have the same length")

        self.dilations_r = tuple(dilations_r)
        self.dilations_d = tuple(dilations_d)

        layers: list[nn.Module] = []
        c_in = in_ch
        n = len(self.dilations_r)

        for i, (d_r, d_d) in enumerate(zip(self.dilations_r, self.dilations_d)):
            is_last = i == n - 1
            c_out = 1 if is_last else width

            # padding == dilation on each axis keeps the spatial size identical through every
            # layer, so the output stays at full RD resolution and no peak is displaced.
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3,
                                    padding=(d_r, d_d), dilation=(d_r, d_d)))

            if not is_last:
                layers.append(nn.BatchNorm2d(c_out))
                layers.append(nn.ReLU(inplace=True))

            c_in = c_out

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, R, V] -> logits [B, 1, R, V] (no sigmoid; the loss applies it)."""
        return self.net(x)

    @property
    def receptive_field(self) -> tuple[int, int]:
        """(range, Doppler) receptive field in bins — should equal the CFAR window."""
        return (1 + sum(2 * d for d in self.dilations_r),
                1 + sum(2 * d for d in self.dilations_d))


class TinyCNNPool(nn.Module):
    """Deliberately WORSE control: the same budget, but with a down/up-sampling stage.

    `TinyCNN`'s docstring asserts that downsampling "risks displacing the peak bin". That is an
    assertion, not a measurement. This model exists to test it: pool by 2, convolve at half
    resolution, upsample back, and see whether localization error rises.

    Held as close to `TinyCNN` as pooling allows:
      * same 3x3 kernels and channel width, so parameter count is comparable (~5k)
      * output restored to full 256x64 resolution, so the evaluator is unchanged
      * same per-cell logits, same loss, same post-processing

    **Confound, stated rather than hidden.** Pooling changes the receptive field, and not simply by
    enlarging it: measured, this model is **13 x 13** against `TinyCNN`'s **21 x 9**. Range shrinks
    (21 -> 13) while Doppler grows (9 -> 13), because a 3x3 kernel at half resolution spans twice
    the input on both axes and the dilations that stretched the range axis are gone.

    Two consequences:
      * a localization difference cannot be attributed to downsampling alone -- the RF differs too;
      * this model no longer matches the CFAR window, so it is NOT a valid arm of the CFAR-vs-DL
        comparison. It is a control for one question (does pooling displace peaks?) and nothing else.

    Parameter count happens to land at 5,041, identical to `TinyCNN`, so capacity is not a confound.
    """

    def __init__(self, in_ch: int = 1, width: int = 16, mode: str = "nearest"):
        super().__init__()
        self.mode = mode

        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        h = self.mid(self.pool(h))
        h = nn.functional.interpolate(h, size=x.shape[-2:], mode=self.mode)
        return self.head(h)

    @property
    def receptive_field(self) -> tuple[int, int]:
        """Analytic RF is not exact through pooling; measure it instead."""
        return measure_receptive_field(self, size=(129, 129))


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
