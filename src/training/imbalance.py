"""Training-only class weights.

The class imbalance is computed ONCE, in Phase 2, from TRAIN target masks
only (169 target days, 12,181,457 px; fire 0.3070%, neg/pos = 324.7597)
and stored in ``data/processed/class_balance.json`` (and mirrored in
``configs/training.yaml``). This module only *loads* those stored values
and wires them into training — it never recomputes weights from
validation or test data, and it never re-fits anything from raw files.
"""

from __future__ import annotations

import json
from pathlib import Path

# Fallback if the JSON cannot be found: the Phase 2 value, recorded in
# configs/training.yaml and DATA_SPEC.md section 8. Kept as a constant
# only so a missing file fails loudly in the caller, not silently.
POS_WEIGHT_TRAIN = 324.7597


def load_class_balance(path: str | Path) -> dict:
    """Load ``data/processed/class_balance.json`` as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"class-balance file not found: {path}. It is produced by "
            "scripts/build_dataset.py from data/raw/; do not hand-edit it."
        )
    return json.loads(path.read_text())


def pos_weight_from_balance(balance: dict) -> float:
    """Return the train neg/pos ratio (the BCE pos_weight).

    Cross-checks the stored ratio against pos/neg pixel counts when they
    are present, so a corrupted file is caught instead of silently used.
    """
    train = balance["train"]
    pw = float(train["neg_over_pos"])
    if "pos_pixels" in train and "neg_pixels" in train:
        recomputed = train["neg_pixels"] / train["pos_pixels"]
        if abs(recomputed - pw) > 1e-3:
            raise ValueError(
                f"class_balance.json is inconsistent: stored neg_over_pos "
                f"{pw} != neg/pos from pixel counts {recomputed}"
            )
    return pw


def load_pos_weight(
    class_balance_path: str | Path | None = None,
    fallback: float = POS_WEIGHT_TRAIN,
) -> float:
    """Load the train pos_weight, falling back to the recorded constant.

    The fallback is used only when no balance file is supplied (e.g. unit
    tests); real training always passes the file path so the value is
    read from the Phase 2 artefact, not from memory.
    """
    if class_balance_path is None:
        return float(fallback)
    return pos_weight_from_balance(load_class_balance(class_balance_path))
