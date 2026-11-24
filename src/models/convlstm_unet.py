"""Compact ConvLSTM-UNet for next-day wildfire spread segmentation.

Design
------
A U-Net whose bottleneck is a ConvLSTM (Shi et al., "Convolutional
LSTM Network: A Machine Learning Approach for Precipitation
Nowcasting", NeurIPS 2015), following the RecUNet template in the
official UTAE repository (VSainteuf/utae-paps, src/backbones/recurrent.py):

* The same encoder/decoder as the plain UTAE in this directory
  (encoder_widths [64, 64, 64, 128], decoder_widths [32, 32, 64, 128],
  GroupNorm encoder, BatchNorm decoder, reflect padding).
* The bottleneck feature maps of all T days are fed as a sequence to
  a single-layer ConvLSTM (input_dim = 128, hidden_dim = 128,
  kernel 3x3). The LAST CELL STATE is taken as the temporal embedding
  (as in RecUNet), then a 3x3 convolution maps it back to the encoder
  width (out_convlstm).
* Skip connections use the temporal MEAN of each day's encoder
  feature map (as in RecUNet; the ConvLSTM's temporal aggregation
  happens only at the bottleneck).

What is adapted from RecUNet
----------------------------
* RecUNet's ConvLSTM takes a fixed ``input_size`` (the bottleneck
  spatial size) for state initialisation; here the states are
  initialised as zeros with the actual input's spatial shape, so the
  model works for any event frame size (ours vary, 297x219 to 331x270).
* RecUNet is a multi-class model for PASTIS (10 input channels,
  20 classes); here the input is 41 channels per day and the head is
  a single binary logit (the next-day active-fire mask).
* As in the UTAE file, the transposed-convolution upsampling is
  followed by a crop/pad to the skip's exact size, because our frames
  are odd-sized (the official code assumes 128x128).

Hidden width: 128 (the official RecUNet default). With a 41-channel
input this gives ~2.8M parameters (measured in the verification
script), inside the 1-3M design target. A hidden width of 96 would
give ~2.2M and 64 would give ~1.7M; 128 was chosen because it is the
official default and fits the budget.

Pretrained weights: none exist for this input encoding; trains from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import crop_to, pad_to
from .utae import ConvBlock, DownConvBlock, UpConvBlock, ENCODER_WIDTHS, DECODER_WIDTHS

# ConvLSTM configuration (official RecUNet defaults).
CONVLSTM_HIDDEN_DIM = 128
CONVLSTM_KERNEL_SIZE = 3


class ConvLSTMCell(nn.Module):
    """One ConvLSTM cell (Shi et al. 2015, Eqs. 1-5).

    i_t = sigmoid(W_xi * X_t + W_hi * H_{t-1} + b_i)
    f_t = sigmoid(W_xf * X_t + W_hf * H_{t-1} + b_f)
    o_t = sigmoid(W_xo * X_t + W_ho * H_{t-1} + b_o)
    g_t = tanh(W_xg * X_t + W_hg * H_{t-1} + b_g)
    C_t = f_t * C_{t-1} + i_t * g_t
    H_t = o_t * tanh(C_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv_x = nn.Conv2d(input_dim, 4 * hidden_dim, kernel_size, padding=pad)
        self.conv_h = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = torch.zeros(x.shape[0], self.hidden_dim, x.shape[2], x.shape[3],
                            device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state
        gates = self.conv_x(x) + self.conv_h(h)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    """Single-layer ConvLSTM over a time sequence.

    Input: (B, T, C, H, W). Returns (layer_outputs, (h_last, c_last))
    with layer_outputs the hidden state at each time step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        state = None
        outputs: list[torch.Tensor] = []
        for t in range(T):
            state = self.cell(x[:, t], state)
            outputs.append(state[0])
        return outputs, state


class ConvLSTMUNet(nn.Module):
    """Compact ConvLSTM-UNet (UTAE encoder/decoder + ConvLSTM bottleneck).

    Input: (B, T, in_channels, H, W). Output: logits (B, H, W).
    """

    def __init__(self, in_channels: int = 41,
                 encoder_widths: tuple[int, ...] = ENCODER_WIDTHS,
                 decoder_widths: tuple[int, ...] = DECODER_WIDTHS,
                 hidden_dim: int = CONVLSTM_HIDDEN_DIM,
                 out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.in_conv = ConvBlock([in_channels, encoder_widths[0], encoder_widths[0]],
                                 norm="group")
        self.down_blocks = nn.ModuleList(
            DownConvBlock(encoder_widths[i], encoder_widths[i + 1])
            for i in range(len(encoder_widths) - 1))
        self.temporal_encoder = ConvLSTM(encoder_widths[-1], hidden_dim,
                                         CONVLSTM_KERNEL_SIZE)
        # 3x3 convolution mapping the last cell state back to the
        # encoder width (official RecUNet's out_convlstm).
        self.out_convlstm = nn.Conv2d(hidden_dim, encoder_widths[-1], 3, padding=1)
        self.up_blocks = nn.ModuleList(
            UpConvBlock(decoder_widths[i], decoder_widths[i - 1], encoder_widths[i - 1])
            for i in range(len(decoder_widths) - 1, 0, -1))
        self.out_conv = ConvBlock([decoder_widths[0], decoder_widths[0], out_channels],
                                  norm="batch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 41, H, W) -> logits (B, H, W)."""
        B, T, _, H, W = x.shape
        x_flat = x.reshape(B * T, self.in_channels, H, W)
        feats: list[torch.Tensor] = []
        e = self.in_conv(x_flat)
        feats.append(e)
        for down in self.down_blocks:
            e = down(e)
            feats.append(e)
        # Bottleneck sequence (B, T, C, h, w) through the ConvLSTM; the
        # LAST CELL STATE is the temporal embedding (RecUNet).
        bn = feats[-1].reshape(B, T, feats[-1].shape[1], feats[-1].shape[2], feats[-1].shape[3])
        _, (_, c_last) = self.temporal_encoder(bn)
        d = self.out_convlstm(c_last)
        # Skips: temporal MEAN of each day's encoder features (RecUNet).
        skips = [f.reshape(B, T, f.shape[1], f.shape[2], f.shape[3]).mean(dim=1)
                 for f in feats[:-1]]
        for i, up in enumerate(self.up_blocks):
            skip = skips[len(skips) - 1 - i]
            d = up(d, skip)
        return self.out_conv(d).squeeze(1)
