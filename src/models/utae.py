"""Plain UTAE for next-day wildfire spread segmentation.

Paper / code
------------
Garnot & Landrieu, "Panoptic Segmentation of Satellite Image Time
Series with Convolutional Temporal Attention", arXiv:2007.07743 (2021),
official implementation: https://github.com/VSainteuf/utae-paps
(src/backbones/utae.py, src/backbones/ltae.py).

Structure followed here (official defaults, which this file reproduces):
* encoder_widths = [64, 64, 64, 128], decoder_widths = [32, 32, 64, 128].
* in_conv: ConvBlock [input_dim, 64, 64] with GroupNorm.
* DownConvBlock x3 (strided 4x4 conv, k=4/s=2/p=1, then two 3x3
  convs with a residual on the second), GroupNorm, reflect padding.
* LTAE2d temporal encoder at the bottleneck: input projection to
  d_model=256, n_head=16 heads, d_k=4 per head, learned queries,
  channel-grouped attention (att_group), MLP [256, 128], positional
  encoding of the day-of-year (T=1000), GroupNorms.
* UpConvBlock x3 (1x1 skip conv + BatchNorm + ReLU, transposed 4x4
  conv, then two 3x3 convs with a residual), BatchNorm, reflect padding.
* out_conv: ConvBlock [32, 32, 1] (official: [32, 20] for 20 classes).

What is adapted (documented, not silent)
-----------------------------------------
* The official model is a multi-class semantic segmentation model for
  PASTIS (10 input channels, 20 classes). Here: 41 input channels per
  day, single-channel binary logit head (the target is the next-day
  active-fire mask), and NO classification head.
* The official code assumes fixed 128x128 inputs; all downsampled
  sizes are exact halves. Our frames are odd-sized (e.g. 331x270), so
  the transposed-convolution upsampling is followed by a crop/pad to
  the skip's exact size before concatenation (same helper as the old
  repo's UpBlock). The official block has no such step.
* batch_positions (day-of-year, shape (B, T)) is optional: when not
  given, positions 0..T-1 are used (the official forward requires it;
  the positional encoder itself is unchanged).
* Inputs are NOT assumed to be at full resolution: the encoder operates
  per day on (B*T, 41, H, W), the LTAE aggregates over T at the
  bottleneck, and skips are attention-aggregated per head group
  (att_group), exactly as in the official Temporal_Aggregator.

This is the PLAIN UTAE, not the ResNet18-UTAE variant in the old repo
(14,955,937 parameters). Expected parameter count with the official
defaults and a 41-channel input is ~1.5M (measured in the verification
script); the often-quoted "~1M" refers to the 10-channel PASTIS
configuration and a narrower stem, so the measured number is the
authoritative one here.

Pretrained weights: none exist for this input encoding; trains from scratch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import crop_to, pad_to

# Official defaults (utae-paps src/backbones/utae.py).
ENCODER_WIDTHS = (64, 64, 64, 128)
DECODER_WIDTHS = (32, 32, 64, 128)
LTAE_D_MODEL = 256
LTAE_N_HEAD = 16
LTAE_D_K = 4
LTAE_MLP = (256, 128)
LTAE_DROPOUT = 0.2
POSITIONAL_T = 1000


class ConvLayer(nn.Module):
    """One convolution + norm (+ optional ReLU), reflect padding.

    Official ConvLayer (utae-paps): nn.Conv2d(nk1, nk2, kernel, stride,
    padding, padding_mode="reflect"), then GroupNorm (n_groups=4) or
    BatchNorm, optional ReLU.
    """

    def __init__(self, nk1: int, nk2: int, kernel: int = 3, stride: int = 1,
                 padding: int = 1, norm: str = "batch", activation: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(nk1, nk2, kernel, stride, padding,
                              padding_mode="reflect", bias=False)
        if norm == "group":
            self.norm = nn.GroupNorm(4, nk2)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(nk2)
        else:
            self.norm = nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvBlock(nn.Module):
    """Two convolutions in sequence (official ConvBlock).

    nkernels is a list of channel widths; a convolution is created for
    each consecutive pair.
    """

    def __init__(self, nkernels: list[int], norm: str = "batch"):
        super().__init__()
        self.convs = nn.ModuleList(
            ConvLayer(nkernels[i], nkernels[i + 1], norm=norm)
            for i in range(len(nkernels) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        return x


class DownConvBlock(nn.Module):
    """Downsampling block (official DownConvBlock).

    Strided 4x4 conv (d_in -> d_in) halves the resolution, then two 3x3
    convs (d_in -> d_out -> d_out) with a residual connection on the
    second. GroupNorm throughout (encoder_norm = "group").
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.down = ConvLayer(d_in, d_in, kernel=4, stride=2, padding=1, norm="group")
        self.conv1 = ConvLayer(d_in, d_out, norm="group")
        self.conv2 = ConvLayer(d_out, d_out, norm="group")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv1(x)
        return x + self.conv2(x)


class UpConvBlock(nn.Module):
    """Upsampling block (official UpConvBlock), with a size-match step.

    1x1 skip convolution (+ BN + ReLU), transposed 4x4 conv (d_in ->
    d_out), concatenation with the skip, then two 3x3 convs
    ([d_out + d_skip, d_out] -> [d_out, d_out]) with a residual on the
    second. BatchNorm throughout (official default for decoder blocks).

    Adaptation: the official code assumes the transposed convolution
    lands exactly on the skip size (fixed 128x128 inputs). Our frames
    are odd-sized, so the upsampled tensor is cropped/padded to the
    skip's exact size before concatenation.
    """

    def __init__(self, d_in: int, d_out: int, d_skip: int):
        super().__init__()
        self.skip_conv = ConvLayer(d_skip, d_skip, kernel=1, padding=0, norm="batch")
        self.up = nn.ConvTranspose2d(d_in, d_out, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvLayer(d_out + d_skip, d_out, norm="batch")
        self.conv2 = ConvLayer(d_out, d_out, norm="batch")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        skip = self.skip_conv(skip)
        x = self.up(x)
        x = pad_to(crop_to(x, *skip.shape[-2:]), *skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return x + self.conv2(x)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding of the day-of-year (official ltae.py).

    PE(pos, 2i) = sin(pos / T^(2i/d)), PE(pos, 2i+1) = cos(pos / T^(2i/d)).
    The encoding of the *input* dimension (d_model // n_head) is
    repeated n_head times so that positions are added per head.
    """

    def __init__(self, d: int, T: int = POSITIONAL_T, repeat: int = LTAE_N_HEAD):
        super().__init__()
        self.d, self.T, self.repeat = d, T, repeat
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                        * (-math.log(T) / d))
        self.register_buffer("div", div)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (B, T) -> (B, T, d * repeat)."""
        B, T = positions.shape
        pos = positions.unsqueeze(-1).float()  # (B, T, 1)
        pe = torch.zeros(B, T, self.d, device=positions.device, dtype=torch.float32)
        pe[..., 0::2] = torch.sin(pos * self.div)
        pe[..., 1::2] = torch.cos(pos * self.div)
        return pe.repeat(1, 1, self.repeat)


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder, 2D version (official ltae.py).

    Input: (B, T, C, H, W) at the bottleneck. Projects the channel
    dimension to d_model, adds positional encodings, computes per-head
    attention masks over T with learned queries (key per head via a
    linear layer, values are the projected inputs), and aggregates
    (weighted mean over T) to (B, d_model, H, W), then an MLP + GroupNorm
    back to C channels. Returns (out, masks) with masks (B, n_head, T, H, W).
    """

    def __init__(self, in_channels: int = 128, d_model: int = LTAE_D_MODEL,
                 n_head: int = LTAE_N_HEAD, d_k: int = LTAE_D_K,
                 mlp: tuple[int, ...] = LTAE_MLP, dropout: float = LTAE_DROPOUT,
                 d_model_q: int | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.inconv = nn.Conv1d(in_channels, d_model, 1)
        self.in_norm = nn.GroupNorm(n_head, in_channels)
        self.out_norm = nn.GroupNorm(n_head, in_channels)
        self.pos_enc = PositionalEncoder(d_model // n_head, T=POSITIONAL_T, repeat=n_head)
        # Learned query parameters: one (d_k,) vector per head.
        self.Q = nn.Parameter(torch.randn(n_head, d_k) / math.sqrt(d_k))
        self.fc1_k = nn.Linear(d_model // n_head, d_k)
        # MLP: d_model -> mlp[0] -> mlp[1] -> in_channels.
        layers = [nn.Linear(d_model, mlp[0]), nn.ReLU(inplace=True),
                  nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.BatchNorm1d(mlp[1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch_positions: torch.Tensor):
        """x: (B, T, C, H, W); batch_positions: (B, T) day-of-year."""
        B, T, C, H, W = x.shape
        # GroupNorm over the channel groups, then the 1x1 projection to
        # d_model. Both operate on (B, C, T*H*W).
        out = self.in_norm(x.permute(0, 2, 1, 3, 4).reshape(B, C, T * H * W))
        out = self.inconv(out).reshape(B, self.d_model, T, H, W)
        pos = self.pos_enc(batch_positions)  # (B, T, d_model)
        out = out + pos.permute(0, 2, 1).reshape(B, self.d_model, T, 1, 1)
        out = self.dropout(out)
        # Per-pixel, per-head keys: (B*H*W, T, n_head, d_k).
        k = self.fc1_k(out.permute(0, 3, 4, 2, 1).reshape(
            B * H * W, T, self.n_head, self.d_model // self.n_head))
        # Attention masks over T: (B*H*W, n_head, T), softmax over T.
        att = torch.softmax(
            torch.einsum("nd,btnd->bnt", self.Q, k) / math.sqrt(self.d_k), dim=2)
        # Values (the projected inputs), weighted mean over T.
        v = out.permute(0, 3, 4, 2, 1).reshape(
            B * H * W, T, self.n_head, self.d_model // self.n_head)
        out = torch.einsum("bnt,btnd->bnd", att, v)
        out = out.reshape(B, H, W, self.d_model).permute(0, 3, 1, 2)
        out = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.norm(out.reshape(B, -1, H * W)).reshape(B, self.in_channels, H, W)
        out = self.out_norm(out)
        masks = att.reshape(B, H, W, self.n_head, T).permute(0, 3, 4, 1, 2)
        return out, masks


def _att_group_aggregate(masks: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Official Temporal_Aggregator, agg_mode='att_group'.

    masks: (B, n_head, T, h, w) at the bottleneck resolution; x:
    (B, T, C, H, W) at the skip's resolution. Each head's mask is
    applied to its own group of C // n_head channels. The masks are
    bilinearly interpolated to the skip's (H, W) first (the official
    aggregator does this; the bottleneck masks are at 1/8 resolution).
    """
    B, T, C, H, W = x.shape
    n_head = masks.shape[1]
    d = C // n_head
    h, w = masks.shape[-2:]
    if (h, w) != (H, W):
        masks = F.interpolate(masks.reshape(B, n_head * T, h, w), size=(H, W),
                              mode="bilinear", align_corners=False)
        masks = masks.reshape(B, n_head, T, H, W)
    xg = x.permute(0, 2, 1, 3, 4).reshape(B, n_head, d, T, H, W)
    mg = masks.reshape(B, n_head, 1, T, H, W)
    out = (xg * mg).sum(dim=3)
    return out.reshape(B, C, H, W)


class UTAE(nn.Module):
    """Plain UTAE (Garnot & Landrieu 2021), official utae-paps structure.

    Input: (B, T, in_channels, H, W), optional batch_positions (B, T)
    day-of-year. Output: logits (B, H, W).
    """

    def __init__(self, in_channels: int = 41,
                 encoder_widths: tuple[int, ...] = ENCODER_WIDTHS,
                 decoder_widths: tuple[int, ...] = DECODER_WIDTHS,
                 out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.in_conv = ConvBlock([in_channels, encoder_widths[0], encoder_widths[0]],
                                 norm="group")
        self.down_blocks = nn.ModuleList(
            DownConvBlock(encoder_widths[i], encoder_widths[i + 1])
            for i in range(len(encoder_widths) - 1))
        self.temporal_encoder = LTAE2d(in_channels=encoder_widths[-1])
        self.up_blocks = nn.ModuleList(
            UpConvBlock(decoder_widths[i], decoder_widths[i - 1], encoder_widths[i - 1])
            for i in range(len(decoder_widths) - 1, 0, -1))
        self.out_conv = ConvBlock([decoder_widths[0], decoder_widths[0], out_channels],
                                  norm="batch")

    def forward(self, x: torch.Tensor, batch_positions: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, 41, H, W) -> logits (B, H, W)."""
        B, T, _, H, W = x.shape
        if batch_positions is None:
            batch_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x_flat = x.reshape(B * T, self.in_channels, H, W)
        # Encoder per day; collect the feature map at each stage.
        feats: list[torch.Tensor] = []
        e = self.in_conv(x_flat)
        feats.append(e)
        for down in self.down_blocks:
            e = down(e)
            feats.append(e)
        # Bottleneck: (B, T, C, h, w) -> LTAE.
        bn = feats[-1].reshape(B, T, feats[-1].shape[1], feats[-1].shape[2], feats[-1].shape[3])
        bn, masks = self.temporal_encoder(bn, batch_positions)
        # Skips: attention-aggregated per head group (official att_group).
        # feats[i] is (B*T, C, h, w) -> (B, T, C, h, w).
        skips = [f.reshape(B, T, f.shape[1], f.shape[2], f.shape[3]) for f in feats[:-1]]
        agg_skips = [_att_group_aggregate(masks, s) for s in skips]
        # Decoder: up_blocks iterate from the deepest stage; the official
        # order pairs the i-th up block with the i-th encoder skip
        # (reversed). agg_skips[0] is the full-res skip, agg_skips[-1]
        # the deepest. The first up block pairs with the deepest skip.
        d = bn
        for i, up in enumerate(self.up_blocks):
            skip = agg_skips[len(agg_skips) - 1 - i]
            d = up(d, skip)
        return self.out_conv(d).squeeze(1)
