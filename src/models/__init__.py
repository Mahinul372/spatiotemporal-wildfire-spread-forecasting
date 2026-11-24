"""Model registry for the wildfire-spread verification pass.

All models take (B, T, 41, H, W) and output logits (B, H, W).

Temporal handling differs by design (see each model's module docstring):
* ShearFuse-UNet stacks the T days as input channels (paper Table V):
  its first convolution takes 41*T channels, so it is constructed
  per window with in_channels = 41 * T.
* TD-FusionUNet pools the T days with a learnable TemporalAttentionPool
  at every scale (adaptation documented in td_fusion_unet.py).
* UTAE aggregates with the LTAE2d temporal attention encoder.
* ConvLSTM-UNet aggregates with a ConvLSTM at the bottleneck.
"""

from __future__ import annotations

import torch.nn as nn

from .shearfuse_unet import ShearFuseUNet
from .td_fusion_unet import TDFusionUNet
from .utae import UTAE
from .convlstm_unet import ConvLSTMUNet

MODEL_NAMES = ("shearfuse_unet", "td_fusion_unet", "utae", "convlstm_unet")
WINDOW_DAYS = {"E1": 1, "E2": 3, "E3": 5}
IN_CHANNELS_PER_DAY = 41


def build_model(name: str, window: str = "E3", in_channels: int = IN_CHANNELS_PER_DAY,
                **overrides) -> nn.Module:
    """Build one model for one window.

    ShearFuse-UNet is built with in_channels = 41 * T (days stacked as
    channels, paper Table V); the others take 41 channels per day and
    aggregate over time internally.
    """
    if name == "shearfuse_unet":
        return ShearFuseUNet(in_channels=in_channels * WINDOW_DAYS[window], **overrides)
    if name == "td_fusion_unet":
        return TDFusionUNet(in_channels=in_channels, **overrides)
    if name == "utae":
        return UTAE(in_channels=in_channels, **overrides)
    if name == "convlstm_unet":
        return ConvLSTMUNet(in_channels=in_channels, **overrides)
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
