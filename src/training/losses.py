"""Primary loss: weighted binary cross-entropy over valid pixels only.

The loss is BCEWithLogitsLoss with a single scalar ``pos_weight`` that
corrects the ~1:325 train imbalance (DATA_SPEC.md section 8). It is the
same for all four architectures (all output single-channel logits).

Focal / Dice are deliberately NOT training options here: they are reserved
for later, separate experiments (see MODEL_SPEC.md section "Phase 4 must
re-verify", item 6). Do not add them to the training config without a
separate experiment decision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """Weighted BCE averaged over valid (non-padded) pixels only.

    Parameters
    ----------
    logits: (B, H, W) raw model logits.
    target: (B, H, W) uint8 / float 0/1 target mask.
    valid: (B, H, W) bool mask; False for padded pixels (batches are
        padded to the largest H/W in the batch).
    pos_weight: scalar neg/pos ratio from TRAIN target masks only
        (324.7597). Must never be recomputed from validation/test data.
    """
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)}, "
            f"target {tuple(target.shape)}, valid {tuple(valid.shape)}"
        )
    if not bool(valid.any()):
        raise ValueError("valid mask is all-False; nothing to compute loss over")
    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    # Per-pixel BCE (no reduction), then mask to valid pixels.
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), pos_weight=pw, reduction="none"
    )
    return per_pixel[valid].mean()
