"""TD-FusionUNet for next-day wildfire spread segmentation.

Paper
-----
Luo, Rong, Watts & Cetin, "U-Net with Hadamard Transform and DCT Latent
Spaces for Next-day Wildfire Spread Prediction", arXiv:2602.11672
(submitted 12 Feb 2026). Evaluated on the Google Next-Day Wildfire
Spread dataset and on WildfireSpreadTS; TD-FusionUNet (base width 8)
reports F1 0.591 with 370k parameters on WildfireSpreadTS, against
F1 0.589 for a 14M-parameter ResNet18 U-Net baseline.

What is preserved from the paper
--------------------------------
* **Dual-branch transform-domain U-Net** (Fig. 2, SS2.3): one branch
  whose latent space is the **2D Hadamard (Walsh-Hadamard) transform**
  domain and a parallel branch in the **2D DCT domain**. The two branch
  outputs are fused by element-wise summation after learnable 1x1
  convolutions, F = phi(F_HT) + psi(F_DCT) (Eq. 7).
* **Transform perceptron** (SS2.2, Eqs. 1-5): per channel, a 2D
  orthogonal transform, element-wise multiplication by a **trainable
  scaling weight map W_c**, **soft-thresholding** with a non-negative
  learnable threshold (Eq. 3), and reconstruction by the inverse
  transform (1/N^2 H Z H for the unnormalised Hadamard transform;
  D^T Z D for the orthonormal DCT). This is the defining mechanism of
  the model and is implemented exactly, not approximated by a conv.
* **U-Net structure** (SS2.1): six convolutional blocks per branch
  (three encoder, one bottleneck, two decoder), each a single
  Conv-BN-ReLU; first encoder conv and final decoder conv use 4x4
  kernels, intermediate convs use 7x7 kernels; skip connections fuse
  encoder and decoder features at matching depths.
* Lightweight capacity: base channel width 8, total parameters in the
  same range as the paper's 370k (exact count measured in the smoke
  tests and recorded in MODEL_SPEC.md).

What is adapted
---------------
* **No official code exists.** The paper lists no repository, and a
  search (2026-11-23) found no public implementation. This file is
  therefore implemented faithfully from the paper text (equations and
  SS2.1-2.3) and says so explicitly; there is no official code to
  compare against line by line.
* **Input size**: the paper's inputs are square and powers of two
  (64x64 Google, 128x128 WildfireSpreadTS); the Hadamard transform
  requires square power-of-two maps. Our inputs are ~300x220
  non-square, non-power-of-two. The transform perceptron therefore
  zero-pads its feature map to a square power-of-two (64x64 at the
  bottleneck) and crops back after the inverse transform. Because a
  full per-channel (N, N) scale map at full resolution would need
  ~90k parameters *per channel*, transform perceptrons are placed at
  the **bottleneck only** (one per branch), where the padded size is
  64x64 -- the same "transform-domain latent space" idea, at the
  resolution where the paper's parameter budget allows it.
* **Thresholds**: the paper's threshold is a full per-channel map
  T_c in R^(N x N). Here it is a **per-channel scalar** (softplus-
  parameterised, hence non-negative), a documented simplification
  that keeps the model near the paper's 370k budget (a full map would
  add another ~130k parameters per branch).
* **Temporal handling**: the paper feeds a single multi-channel image
  (its WildfireSpreadTS run uses 40 channels). Here each day is run
  through the shared-weight branch encoders and the per-day features
  are fused with a learnable temporal-attention pool at every scale
  (softmax over days of a score from the day's pooled descriptor),
  so T in {1, 3, 5} all work and the most informative day is weighted
  most. The decoder then runs once.
* **Input channels**: 41 encoded channels (paper: 40 on WildfireSpreadTS,
  12 on Google) via the same stem convolution.
* The paper's custom preprocessing (random margin cropping, Gaussian-
  mixture softening of the fire mask, composite BCE+Dice+Focal loss)
  is a *training-pipeline* choice, not architecture; it is NOT
  implemented here. This project uses the Phase 2 decision (weighted
  BCE, pos_weight 324.7597) for all four architectures.

Pretrained weights
------------------
None exist (no official code or released checkpoint). Trains from scratch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNReLU, SegmentationHead, TemporalAttentionPool, crop_to

_DCT_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def fwht_1d(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised 1D Fast Walsh-Hadamard Transform along the last dim.

    Length must be a power of two. Differentiable (only adds/subs).
    Applying it twice gives N * x, so the inverse is fwht(x) / N.
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"FWHT needs power-of-2 length, got {n}"
    lead = x.shape[:-1]
    h = 1
    while h < n:
        x = x.reshape(*lead, n // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.cat([a + b, a - b], dim=-1).reshape(*lead, n)
        h *= 2
    return x


def fwht_2d(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised 2D FWHT: H X H for square power-of-2 maps."""
    x = fwht_1d(x)
    x = fwht_1d(x.transpose(-1, -2)).transpose(-1, -2)
    return x


def dct_matrix(n: int, device) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n, n); D @ D.T == I."""
    key = (n, device)
    if key not in _DCT_CACHE:
        k = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
        m = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0)
        d = torch.cos(math.pi * (2 * m + 1) * k / (2 * n))
        d = d * math.sqrt(2.0 / n)
        d[0] = d[0] / math.sqrt(2.0)
        _DCT_CACHE[key] = d
    return _DCT_CACHE[key]


def soft_threshold(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Eq. (3): sign(x) * max(|x| - t, 0), t >= 0 (per-channel scalar)."""
    return torch.sign(x) * torch.clamp(x.abs() - t, min=0.0)


class TransformPerceptron(nn.Module):
    """One trainable transform-domain layer (paper SS2.2, Eqs. 1-5).

    Forward: pad to (N, N) square power-of-two -> 2D transform ->
    element-wise trainable scale W_c (per channel, init 1) ->
    soft-thresholding (per-channel non-negative threshold) -> inverse
    transform -> crop back to the input size.

    ``kind`` is 'hadamard' (unnormalised H, inverse scaled by 1/N^2) or
    'dct' (orthonormal D, inverse D^T Z D).
    """

    def __init__(self, channels: int, size: int = 64, kind: str = "hadamard",
                 threshold_init: float = 0.01):
        super().__init__()
        assert kind in ("hadamard", "dct")
        self.channels = channels
        self.size = size
        self.kind = kind
        # Per-channel scale map W_c in R^(N x N), init 1 (identity).
        self.scale = nn.Parameter(torch.ones(channels, size, size))
        # Per-channel threshold, softplus-parameterised to stay >= 0.
        raw = math.log(math.exp(threshold_init) - 1.0)
        self.raw_threshold = nn.Parameter(
            torch.full((channels,), raw))

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "hadamard":
            return fwht_2d(x)
        d = dct_matrix(x.shape[-1], x.device)
        return d @ x @ d.T

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        if self.kind == "hadamard":
            return fwht_2d(x) / float(n * n)
        d = dct_matrix(n, x.device)
        return d.T @ x @ d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = self.size
        if h > n or w > n:
            raise ValueError(
                f"TransformPerceptron ({self.kind}): feature map {h}x{w} "
                f"exceeds the configured transform size {n}x{n}; increase "
                f"`transform_size` (bottleneck maps on this data are <=42x36, "
                f"so 64 covers all 9 events).")
        x_pad = F.pad(x, (0, n - w, 0, n - h))
        z = self._transform(x_pad)
        z = z * self.scale.unsqueeze(0)
        t = F.softplus(self.raw_threshold).view(1, c, 1, 1)
        z = soft_threshold(z, t)
        y = self._inverse(z)
        return crop_to(y, h, w)


class ConvPreserve4(nn.Module):
    """4x4 Conv-BN-ReLU that preserves spatial size.

    A 4x4 kernel cannot preserve size with symmetric padding (pad 1
    shrinks by 1, pad 2 grows by 1), so pad 2 and crop the extra row /
    column back off. Used for the paper's 4x4 first/last convolutions.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return crop_to(self.act(self.bn(self.conv(x))), h, w)


class _Branch(nn.Module):
    """One transform-domain U-Net branch (HT or DCT).

    Encoder: 3 single-conv blocks (widths ``widths``) with 2x2 max-pool
    between them; bottleneck single-conv block + TransformPerceptron.
    Decoder: 2 upsampling single-conv blocks with skips, then a final
    4x4 conv to ``out_channels`` at full resolution.
    """

    def __init__(self, in_channels: int, widths: tuple[int, ...] = (8, 16, 16),
                 kind: str = "hadamard", transform_size: int = 64):
        super().__init__()
        w0, w1, w2 = widths
        self.kind = kind
        self.enc1 = ConvPreserve4(in_channels, w0)  # first conv 4x4
        self.enc2 = ConvBNReLU(w0, w1, kernel=7)
        self.enc3 = ConvBNReLU(w1, w2, kernel=7)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBNReLU(w2, w2, kernel=7)
        self.transform = TransformPerceptron(w2, size=transform_size, kind=kind)
        self.dec1 = ConvBNReLU(w2 + w2, w1, kernel=7)
        self.dec2 = ConvBNReLU(w1 + w1, w1, kernel=7)
        self.final = nn.Conv2d(w1 + w0, 8, kernel_size=4, padding=2)
        # Temporal attention pools (one per scale) for fusing days.
        self.pool_t1 = TemporalAttentionPool(w0)
        self.pool_t2 = TemporalAttentionPool(w1)
        self.pool_t3 = TemporalAttentionPool(w2)
        self.pool_tb = TemporalAttentionPool(w2)

    def _encode_day(self, day: torch.Tensor):
        e1 = self.enc1(day)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        return e1, e2, e3, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> branch output (B, 8, H, W)."""
        b, t, _, h, w = x.shape
        encs = [self._encode_day(x[:, i]) for i in range(t)]
        e1 = self.pool_t1(torch.stack([e[0] for e in encs], dim=1))
        e2 = self.pool_t2(torch.stack([e[1] for e in encs], dim=1))
        e3 = self.pool_t3(torch.stack([e[2] for e in encs], dim=1))
        bn = self.pool_tb(torch.stack([e[3] for e in encs], dim=1))
        bn = self.transform(bn)  # transform-domain latent space
        # Upsample to each skip's exact size (max-pool floors odd sizes,
        # so a plain 2x upsample can be 1 px short, e.g. 37 -> 74 vs 75).
        d1 = F.interpolate(bn, size=e3.shape[-2:], mode="bilinear",
                           align_corners=False)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))
        d2 = F.interpolate(d1, size=e2.shape[-2:], mode="bilinear",
                           align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        out = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear",
                            align_corners=False)
        out = self.final(torch.cat([out, e1], dim=1))
        return crop_to(out, h, w)  # also removes the +1 from the 4x4 pad-2


class TDFusionUNet(nn.Module):
    """TD-FusionUNet: parallel HT + DCT branches fused (Eq. 7).

    Args:
        in_channels: encoded input channels per day (41).
        widths: encoder widths per branch (base width 8, as in the paper).
        transform_size: square power-of-two size of the transform
            perceptron (64 covers all bottleneck maps on this data).
    """

    def __init__(self, in_channels: int = 41,
                 widths: tuple[int, ...] = (8, 16, 16),
                 transform_size: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.ht_branch = _Branch(in_channels, widths, kind="hadamard",
                                 transform_size=transform_size)
        self.dct_branch = _Branch(in_channels, widths, kind="dct",
                                  transform_size=transform_size)
        # Fusion projections phi / psi (Eq. 7).
        self.phi = nn.Conv2d(8, 8, kernel_size=1)
        self.psi = nn.Conv2d(8, 8, kernel_size=1)
        self.head = SegmentationHead(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> logits (B, H, W)."""
        f_ht = self.ht_branch(x)
        f_dct = self.dct_branch(x)
        fused = self.phi(f_ht) + self.psi(f_dct)  # Eq. (7)
        return self.head(fused)
