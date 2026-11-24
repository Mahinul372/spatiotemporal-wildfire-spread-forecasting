"""Shared building blocks for the wildfire spread segmentation models.

All models share one interface::

    logits = model(x)          # x: (B, T, C=41, H, W) float32
                               # logits: (B, H, W) float32 (pre-sigmoid)

Spatial size varies per event (H 297-331, W 208-285), so every block here
is fully convolutional / size-agnostic: no fixed-size linear layers on
spatial maps, no hard-coded H/W. Padding to a multiple of some patch size
is always undone by cropping the logits back to the input (H, W).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 41


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return {'total': ..., 'trainable': ...} parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def pad_to_multiple(x: torch.Tensor, multiple: int, dims: tuple[int, ...] = (-2, -1)):
    """Pad the given spatial dims up to a multiple of ``multiple``.

    Returns (padded, (pad_h, pad_w)) so the caller can crop back.
    Only the last two dims are supported (standard (..., H, W) tensors).
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h)), (pad_h, pad_w)


def crop_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return x[..., :h, :w]


def pad_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Zero-pad the spatial dims up to exactly (h, w); no-op if already there."""
    ph = max(0, h - x.shape[-2])
    pw = max(0, w - x.shape[-1])
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph))


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU (padding preserves spatial size)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two ConvBNReLU layers (the U-Net workhorse block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, kernel=kernel),
            ConvBNReLU(out_ch, out_ch, kernel=kernel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear 2x upsample + DoubleConv (size-agnostic, no transposed conv).

    Takes the decoder feature map and the skip feature map (which may be
    1 px larger/smaller after odd-sized downsampling); the skip is
    centre-cropped / padded to match before concatenation.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, kernel=kernel)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip.shape[-2:] != x.shape[-2:]:
            dh = x.shape[-2] - skip.shape[-2]
            dw = x.shape[-1] - skip.shape[-1]
            if dh >= 0 and dw >= 0:
                skip = F.pad(skip, (0, dw, 0, dh))
            else:
                skip = skip[..., : x.shape[-2], : x.shape[-1]]
        return self.conv(torch.cat([x, skip], dim=1))


class SegmentationHead(nn.Module):
    """1x1 convolution to a single-channel logit map."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).squeeze(1)  # (B, H, W)


def temporal_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean over the time axis of a (B, T, C, H, W) tensor -> (B, C, H, W)."""
    return x.mean(dim=1)


class TemporalAttentionPool(nn.Module):
    """Learnable weighted mean over time.

    Scores each day from its global-average-pooled descriptor (B, T, D)
    with a small MLP, softmaxes over T, and returns the weighted sum of
    the (B, T, C, H, W) features. Used by TD-FusionUNet to collapse the
    per-day encoder outputs to one feature map per scale.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        gap = x.mean(dim=(3, 4))  # (B, T, C)
        scores = self.scorer(gap).squeeze(-1)  # (B, T)
        w = torch.softmax(scores, dim=1)  # (B, T)
        return torch.einsum("bt,btchw->bchw", w, x)


def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def init_weights(module: nn.Module) -> None:
    """Sane default init (torch defaults are already fine; this is explicit)."""
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
