"""ShearFuse-UNet for next-day wildfire spread segmentation.

Paper
-----
Meco, Luo, Hamdan, Watts & Cetin, "ShearFuse-UNet: Hadamard, DCT, and
Shearlet Transform Fusion for Next-Day Wildfire Spread Prediction",
arXiv:2606.14071 (v1 12 Jun 2026, v2 21 Sep 2026); also IEEE
Transactions on Geoscience and Remote Sensing. Reports F1 0.596 with
267k parameters on WildfireSpreadTS (128x128, 40 input channels) vs
F1 0.589 for a 14M-parameter ResNet18 U-Net baseline. The same paper
shows a 2-day input costs only ~2k extra parameters (Table V), i.e.
days are stacked as input channels.

What is preserved from the paper (Sec. II)
------------------------------------------
* U-Net topology (II-A): four encoder stages (each a MaxPool followed
  by a DoubleConvWHT block), a bottleneck DoubleConvWHT block, four
  symmetric decoder stages (bilinear upsampling, skip concatenation,
  double convolution), 1x1 output convolution. Base width c1=8,
  progression 8 -> 16 -> 32 -> 64 -> 128.
* DoubleConvWHT block: three transform-domain branches applied to the
  block input before the spatial double convolution.
* WHT branch (II-B): 2D Fast Walsh-Hadamard Transform, learnable
  spectral scaling, soft-thresholding with fixed per-coefficient
  thresholds init U(0, 0.01), inverse WHT with 1/(HW) scaling (Eqs. 1-4).
* DCT branch (II-C): orthonormal 2D DCT-II, optional spectral
  compression retaining the top-r low-frequency fraction (r=0.7),
  learnable scaling, fixed soft-threshold, inverse DCT (Eqs. 5-7).
* SpectralFusion gate (II-E, Eqs. 13-14): GAP of both branch outputs,
  concatenated, through a 2-layer MLP (ReLU, sigmoid, hidden
  max(4, C/8)) to a channel gate w; F_fused = w * Y_WHT + (1-w) * Y_DCT.
* Shearlet residual branch (II-G, Eqs. 18-24): cone-adapted digital
  shearlet filter bank, J=2 scales, K=4 directions per scale
  (n_f = J*K+1 = 9 filters), FFT-based circular convolution, per-subband
  learnable gains v_i and fixed thresholds, synthesis by summation over
  subbands, added as a residual F = F_fused + x_hat_sh. Filters are
  non-trainable (constructed lazily). Sparse activation: shearlet only
  at the down2 (H/4) and down4 (H/16) encoder stages (II-G3).
* Training-free facts used for the check: 1.35 GFLOPs, 3.03 ms/sample
  on GPU at 128x128 (Table IV); thresholds are FIXED (an ablation found
  learnable thresholds degraded validation F1, III-B).

What is adapted (no official code exists)
-----------------------------------------
* **No official code exists.** The paper lists no repository; a web
  search (2026-11-24) found no public implementation. Implemented
  faithfully from the paper text; adaptations are listed here.
* **Variable input resolution.** The paper's inputs are fixed 128x128.
  Ours are ~300x220 and vary per event, so:
  (a) the learnable spectral scaling matrices V (paper: full HxW per
      branch per stage) are stored as a learnable 32x32 field that is
      bilinearly upsampled to the stage's padded size at forward time.
      This keeps the per-coefficient scaling idea with bounded cost
      (1k params per branch per stage) instead of up to 262k for a
      512x512 stage-1 map. A per-channel scalar would lose the spatial
      selectivity entirely; the upsampled field preserves it smoothly.
  (b) the fixed per-coefficient thresholds are a single fixed scalar
      buffer (0.005, the mean of U(0, 0.01)) instead of full HxW maps,
      for the same reason (a full map cannot be shared across events).
  (c) the WHT pads each spatial dim to its next power of two (the
      paper's sizes are already powers of two); the DCT uses explicit
      rectangular DCT-II matrices (any size).
  (d) shearlet filters are built in the frequency domain per (H, W)
      shape and cached in a dict (the paper builds them lazily once,
      for one fixed size).
* **Shearlet FFT precision.** The shearlet branch computes its FFTs
  in float32 (input cast to float32, result cast back). cuFFT in half
  precision only supports power-of-two sizes, and our stage maps are
  e.g. 82x67, so under AMP autocast the original code raised
  "cuFFT only supports dimensions whose sizes are powers of two when
  computing in half precision" on the Colab T4 (all 6 ShearFuse smoke
  configs failed before this fix; the other three models were green).
  Fixed 2026-11-24; re-verified on CPU first (see VERIFICATION.md);
  a GPU re-test of the smoke script is still needed before ShearFuse
  training.
* **Shearlet window shapes.** The paper gives the filter formula
  skeleton (Eqs. 19-20: radial window psi_r, angular window psi_a,
  skeleton (Eqs. 19-20: radial window psi_r, angular window psi_a,
  Meyer window nu) but not the explicit psi_r / psi_a definitions.
  The windows used here are my construction following that skeleton:
  dyadic radial bands (s=1: |xi| in [0.125, 0.25], s=2: [0.25, 0.5]),
  Meyer-smoothstep bumps; angular wedges of half-width 0.5 centred at
  shear k in {-0.5, +0.5} in each cone (2 shears per cone per scale,
  giving J*K = 8 directional filters + 1 low-pass = 9, as stated);
  low-pass exactly per Eq. (20). This is a fidelity caveat: the
  directional selectivity follows the paper's design, but the exact
  window shapes are not the paper's (they are not specified in it).
* **Temporal handling.** The paper's Table V stacks days as input
  channels (a second day costs ~2k params = one more 40-channel
  first convolution). Here days are likewise stacked: the first
  convolution takes 41*T input channels, so the parameter count grows
  slightly with T (reported per window). The 41-channel encoding is
  ours (the paper's is 40; see NOTES.md for the reconciliation).
* **Decoder last stage.** The paper's encoder pools first, so there is
  no full-resolution skip; the fourth decoder stage upsamples without
  a skip connection. Documented, not a silent change.
* The paper's custom preprocessing (margin cropping, Gaussian-mixture
  softening) and composite BCE+Dice+Focal loss are training-pipeline
  choices and are NOT implemented here.

Pretrained weights: none exist (no official code or checkpoint).
Trains from scratch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DoubleConv, SegmentationHead, count_parameters, crop_to
from .td_fusion_unet import dct_matrix, fwht_2d, soft_threshold

# Shearlet configuration (paper III-B): J=2 scales, K=4 directions.
SHEARLET_SCALES = (1, 2)
SHEARLET_SHEARS = (-0.5, 0.5)  # per cone; 2 cones x 2 shears x 2 scales = 8
SHEARLET_N_FILTERS = len(SHEARLET_SCALES) * 2 * len(SHEARLET_SHEARS) + 1  # 9
SHEARLET_THRESHOLD = 0.005  # fixed scalar; paper: fixed thresholds ~ U(0, 0.01)
DCT_COMPRESSION_RATIO = 0.7
# Size of the learnable spectral-scaling field (upsampled to stage size).
SPECTRAL_FIELD_SIZE = 32


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _meyer(x: torch.Tensor) -> torch.Tensor:
    """Meyer-type smooth window nu(x) = x^2 (3 - 2x) on [0, 1] (Eq. 18).

    Extended as 0 for x <= 0 and 1 for x >= 1.
    """
    xc = x.clamp(0.0, 1.0)
    return xc * xc * (3.0 - 2.0 * xc)


def _smooth_bump(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Smooth plateau bump: 0 below lo, 1 on [lo+w, hi-w], 0 above hi.

    Rise/fall width is 25% of the band width on each side.
    """
    w = 0.25 * (hi - lo)
    rise = _meyer((u - lo) / max(w, 1e-9))
    fall = 1.0 - _meyer((u - hi) / max(w, 1e-9))
    return rise * fall


class SpectralBranch(nn.Module):
    """One transform-domain branch (WHT or DCT) of a DoubleConvWHT block.

    Forward: pad to power-of-two dims (WHT only) -> 2D transform ->
    learnable spectral scaling (32x32 field upsampled to the padded size)
    -> fixed scalar soft-thresholding -> inverse transform -> crop back.

    ``kind`` is 'wht' (unnormalised FWHT, inverse scaled by 1/(H*W)) or
    'dct' (orthonormal DCT-II, inverse D^T Z D; top-r low-frequency
    coefficients retained, r = 0.7).
    """

    def __init__(self, kind: str, compression_ratio: float = DCT_COMPRESSION_RATIO):
        super().__init__()
        assert kind in ("wht", "dct")
        self.kind = kind
        self.compression_ratio = compression_ratio if kind == "dct" else 1.0
        # Learnable spectral scaling field (paper: full HxW matrix V;
        # here a 32x32 field upsampled to the stage size -- see module doc).
        self.scale_field = nn.Parameter(torch.ones(SPECTRAL_FIELD_SIZE, SPECTRAL_FIELD_SIZE))
        # Fixed (non-learnable) soft-threshold, scalar (paper: per-coefficient
        # fixed maps; scalar here because maps cannot be shared across events).
        self.register_buffer("threshold", torch.tensor(SHEARLET_THRESHOLD))
        self._dct_cache: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _dct_mats(self, h: int, w: int, device):
        key = (h, w, device)
        if key not in self._dct_cache:
            self._dct_cache[key] = (dct_matrix(h, device), dct_matrix(w, device))
        return self._dct_cache[key]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "wht":
            return fwht_2d(x)
        dh, dw = self._dct_mats(x.shape[-2], x.shape[-1], x.device)
        return dh @ x @ dw.T

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.kind == "wht":
            return fwht_2d(x) / float(h * w)
        dh, dw = self._dct_mats(h, w, x.device)
        return dh.T @ x @ dw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.kind == "wht":
            hp, wp = _next_pow2(h), _next_pow2(w)
            xp = F.pad(x, (0, wp - w, 0, hp - h)) if (hp != h or wp != w) else x
        else:
            hp, wp = h, w
            xp = x
        z = self._transform(xp)
        if self.compression_ratio < 1.0:
            # Retain the top-r low-frequency coefficients: the top-left
            # rectangle with area fraction r (0.84^2 ~= 0.70).
            keep = math.sqrt(self.compression_ratio)
            kh, kw = max(1, int(math.ceil(keep * hp))), max(1, int(math.ceil(keep * wp)))
            mask = torch.zeros((hp, wp), dtype=z.dtype, device=z.device)
            mask[:kh, :kw] = 1.0
            z = z * mask
        scale = F.interpolate(
            self.scale_field.unsqueeze(0).unsqueeze(0), size=(hp, wp),
            mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        z = z * scale.unsqueeze(0).unsqueeze(0)
        z = soft_threshold(z, self.threshold)
        y = self._inverse(z)
        return crop_to(y, h, w)


class SpectralFusionGate(nn.Module):
    """Channel-wise SpectralFusion gate (paper II-E, Eqs. 13-14).

    GAP of both branch outputs -> concat -> 2-layer MLP (ReLU, sigmoid,
    hidden max(4, C/8)) -> w; F = w * Y_WHT + (1 - w) * Y_DCT.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, y_wht: torch.Tensor, y_dct: torch.Tensor) -> torch.Tensor:
        g1 = y_wht.mean(dim=(2, 3))
        g2 = y_dct.mean(dim=(2, 3))
        w = self.mlp(torch.cat([g1, g2], dim=1)).unsqueeze(-1).unsqueeze(-1)
        return w * y_wht + (1.0 - w) * y_dct


def _shearlet_filters(h: int, w: int, device) -> list[torch.Tensor]:
    """Cone-adapted digital shearlet filter bank in the frequency domain.

    Returns 9 complex (H, W) filters: 8 directional (J=2 scales x 2 cones
    x 2 shears) + 1 low-pass. Construction follows the paper's formula
    skeleton (Eqs. 19-20); the explicit radial/angular window shapes are
    this implementation's construction (see module docstring).
    """
    fy = torch.fft.fftfreq(h, d=1.0, device=device)  # rows: eta
    fx = torch.fft.fftfreq(w, d=1.0, device=device)  # cols: xi
    eta, xi = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xi ** 2 + eta ** 2)
    filters: list[torch.Tensor] = []
    for s in SHEARLET_SCALES:
        # Dyadic radial bands: s=1 -> |xi| in [0.125, 0.25]; s=2 -> [0.25, 0.5].
        lo, hi = 0.125 * (2 ** (s - 1)), 0.25 * (2 ** (s - 1))
        radial = _smooth_bump(xi.abs(), lo, hi)
        radial_v = _smooth_bump(eta.abs(), lo, hi)
        for k in SHEARLET_SHEARS:
            # Horizontal cone (|eta| <= |xi|): wedge around slope k.
            slope_h = torch.where(xi.abs() > 1e-9, eta / xi.clamp_min(1e-9), torch.zeros_like(xi))
            ang_h = 1.0 - _meyer(((slope_h - k).abs() - 0.25) / 0.25)
            cone_h = (eta.abs() <= xi.abs()).to(xi.dtype)
            filters.append((radial * ang_h * cone_h).to(torch.complex64))
            # Vertical cone (|xi| < |eta|): wedge around slope k.
            slope_v = torch.where(eta.abs() > 1e-9, xi / eta.clamp_min(1e-9), torch.zeros_like(xi))
            ang_v = 1.0 - _meyer(((slope_v - k).abs() - 0.25) / 0.25)
            cone_v = (eta.abs() > xi.abs()).to(xi.dtype)
            filters.append((radial_v * ang_v * cone_v).to(torch.complex64))
    # Low-pass residual filter (Eq. 20): nu(1 - r / 0.25).
    filters.append(_meyer(1.0 - r / 0.25).to(torch.complex64))
    assert len(filters) == SHEARLET_N_FILTERS
    return filters


class ShearletResidual(nn.Module):
    """Cone-adapted shearlet residual branch (paper II-G, Eqs. 21-24).

    Analysis by FFT-based circular convolution, per-subband learnable
    gains v_i and fixed thresholds, synthesis by summation over subbands,
    added as a residual. Filters are non-trainable, built per (H, W)
    shape and cached (the paper builds them lazily once, for one size).
    """

    def __init__(self):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(SHEARLET_N_FILTERS))
        self.register_buffer("threshold", torch.tensor(SHEARLET_THRESHOLD))
        self._filter_cache: dict[tuple[int, int, torch.device], list[torch.Tensor]] = {}

    def _filters(self, h: int, w: int, device) -> list[torch.Tensor]:
        key = (h, w, device)
        if key not in self._filter_cache:
            self._filter_cache[key] = _shearlet_filters(h, w, device)
        return self._filter_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT in float32, always. cuFFT in half precision only supports
        # dimensions whose sizes are powers of two; our stage maps are
        # e.g. 82x67, so under AMP autocast (fp16 inputs) torch.fft.fft2
        # raises "cuFFT only supports dimensions whose sizes are powers
        # of two when computing in half precision". Casting to float32
        # here fixes it (verified on the Colab T4 smoke test, which
        # failed on all 6 ShearFuse configs before this fix). The
        # result is cast back to the input dtype so the residual add
        # matches the surrounding (possibly autocast) precision.
        orig_dtype = x.dtype
        x32 = x.float()
        b, c, h, w = x32.shape
        filters = self._filters(h, w, x32.device)
        x_f = torch.fft.fft2(x32)
        n_f = len(filters)
        out_f = torch.zeros_like(x_f)
        for i, psi in enumerate(filters):
            sub = torch.fft.ifft2(x_f * psi).real
            sub = soft_threshold(self.gains[i] * sub, self.threshold)
            out_f = out_f + torch.fft.fft2(sub) * torch.conj(psi)
        return (torch.fft.ifft2(out_f) / n_f).real.to(orig_dtype)


class DoubleConvWHT(nn.Module):
    """One ShearFuse encoder block (paper II-A/II-E/II-G).

    Spectral branches (WHT + DCT fused by the SpectralFusion gate,
    plus the shearlet residual at selected stages) replace the first
    convolution's input; then a standard double 3x3 convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, use_shearlet: bool = False):
        super().__init__()
        self.wht = SpectralBranch("wht")
        self.dct = SpectralBranch("dct")
        self.gate = SpectralFusionGate(in_ch)
        self.shearlet = ShearletResidual() if use_shearlet else None
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.gate(self.wht(x), self.dct(x))
        if self.shearlet is not None:
            fused = fused + self.shearlet(x)
        return self.conv(fused)


class ShearFuseUNet(nn.Module):
    """ShearFuse-UNet: triple-branch transform U-Net (paper II-A).

    Args:
        in_channels: input channels of ONE day is 41; days are stacked
            as channels (paper Table V), so pass in_channels = 41 * T.
        widths: encoder widths (base width c1 = 8, paper II-A).
        use_shearlet_stages: encoder stages (1-based) with the shearlet
            residual: stages 2 (H/4) and 4 (H/16) per paper II-G3.
    Input: (B, T, 41, H, W) -> logits (B, H, W).
    """

    def __init__(self, in_channels: int = 41, widths: tuple[int, ...] = (8, 16, 32, 64),
                 bottleneck_width: int = 128,
                 use_shearlet_stages: tuple[int, ...] = (2, 4)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.in_channels = in_channels
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConvWHT(in_channels, w0, use_shearlet=1 in use_shearlet_stages)
        self.enc2 = DoubleConvWHT(w0, w1, use_shearlet=2 in use_shearlet_stages)
        self.enc3 = DoubleConvWHT(w1, w2, use_shearlet=3 in use_shearlet_stages)
        self.enc4 = DoubleConvWHT(w2, w3, use_shearlet=4 in use_shearlet_stages)
        self.bottleneck = DoubleConvWHT(w3, bottleneck_width)
        # Decoder mirrors the encoder (bilinear upsample, skip concat,
        # double convolution); the last stage has no full-resolution skip.
        self.dec1 = DoubleConv(bottleneck_width + w2, w2)
        self.dec2 = DoubleConv(w2 + w1, w1)
        self.dec3 = DoubleConv(w1 + w0, w0)
        self.dec4 = DoubleConv(w0, w0)
        self.head = SegmentationHead(w0)

    def _up(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 41, H, W) -> logits (B, H, W). Days are stacked as channels."""
        b, t, _, h, w = x.shape
        x0 = x.reshape(b, t * x.shape[2], h, w)
        e1 = self.enc1(self.pool(x0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(e4)
        d1 = self.dec1(torch.cat([self._up(bn, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d1, e2), e2], dim=1))
        d3 = self.dec3(torch.cat([self._up(d2, e1), e1], dim=1))
        # Last stage has no skip: upsample to the exact input size
        # (doubling would give 330 for a 331-high frame).
        d4 = self.dec4(F.interpolate(d3, size=(h, w), mode="bilinear",
                                     align_corners=False))
        return crop_to(self.head(d4), h, w)
