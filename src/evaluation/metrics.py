"""Segmentation metrics for next-day active-fire prediction.

Primary metric: Average Precision (AP) — threshold-free, computed from
probabilities, so it is comparable across models without any threshold
choice. Precision / recall / F1 / IoU are thresholded metrics, reported
at 0.5 and at the validation-selected threshold.

All functions take flat (valid-pixels-only) arrays; the caller is
responsible for applying the ``valid`` mask before calling. Per-event
metrics group samples by event id.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def _binarize(y_true: np.ndarray, y_prob: np.ndarray,
              threshold: float) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    return y_true, y_pred


def precision_recall_f1_iou(y_true: np.ndarray, y_prob: np.ndarray,
                            threshold: float = 0.5) -> dict:
    """Precision / recall / F1 / IoU at one threshold.

    Returns zeros (not NaN) when the denominator is empty, and records
    the raw confusion counts so edge cases are auditable.
    """
    y_true, y_pred = _binarize(y_true, y_prob, threshold)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold-free AP. Returns 0.0 when a set has no positive pixels
    (AP is undefined there); the caller should note such events."""
    y_true = np.asarray(y_true).ravel()
    if int(y_true.sum()) == 0:
        return 0.0
    return float(average_precision_score(y_true, np.asarray(y_prob).ravel()))


def compute_pooled_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float = 0.5) -> dict:
    """Pooled metrics over all valid pixels: AP + thresholded scores."""
    out = {"average_precision": average_precision(y_true, y_prob),
           "n_pixels": int(np.asarray(y_true).size),
           "n_positive": int(np.asarray(y_true).astype(int).sum())}
    out.update(precision_recall_f1_iou(y_true, y_prob, threshold))
    return out


def compute_per_event_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, events: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Per-event metrics: {event_id: pooled metrics for that event}.

    Events with no positive pixels get AP 0.0 and are flagged
    (``has_positive: false``) — AP is undefined without positives, and
    silently averaging them in would distort the mean.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    events = np.asarray(events)
    out: dict[str, dict] = {}
    for event in sorted(set(events.tolist())):
        mask = events == event
        m = compute_pooled_metrics(y_true[mask], y_prob[mask], threshold)
        m["has_positive"] = bool(int(y_true[mask].astype(int).sum()) > 0)
        out[str(event)] = m
    return out


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                     thresholds: np.ndarray | None = None) -> dict:
    """F1-optimal threshold on a validation set (never fit on test).

    Scans a grid (default 0.05..0.95 step 0.05 plus 0.01..0.99 fine
    grid around the coarse best) and returns the threshold maximising
    F1, with the F1 at 0.5 for comparison.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best = {"threshold": 0.5, "f1": -1.0}
    for t in thresholds:
        f1 = precision_recall_f1_iou(y_true, y_prob, float(t))["f1"]
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}
    # Fine pass around the coarse winner.
    lo = max(0.01, best["threshold"] - 0.05)
    hi = min(0.99, best["threshold"] + 0.05)
    for t in np.round(np.arange(lo, hi + 1e-9, 0.01), 2):
        f1 = precision_recall_f1_iou(y_true, y_prob, float(t))["f1"]
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}
    best["f1_at_0.5"] = precision_recall_f1_iou(y_true, y_prob, 0.5)["f1"]
    return best


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def peak_gpu_memory_gb() -> float | None:
    """Peak allocated GPU memory (GB) since the last reset, or None on CPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e9


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def timed(label: str, store: dict | None = None):
    """Time a block; optionally record ``store[label] = seconds``."""
    t0 = time.time()
    yield
    dt = time.time() - t0
    if store is not None:
        store[label] = round(dt, 3)
