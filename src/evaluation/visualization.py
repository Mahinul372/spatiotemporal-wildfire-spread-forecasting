"""Plots, all re-runnable from saved logs / metrics JSON — no retraining.

Every plotter takes a path to a saved artefact (``history.json``,
``*_metrics.json``) or plain arrays, and writes to
``results/figures/<config_id>/`` (or a supplied directory). Colours are
fixed per model so figures are comparable across configs.

Training curves (the hard requirement) are written in BOTH PNG and SVG
for every config: ``<config_id>_loss_curve.{png,svg}``,
``<config_id>_ap_curve.{png,svg}``, ``<config_id>_lr_curve.{png,svg}``.
The data behind them is ``results/logs/<config_id>/history.json``
(train/val loss, val AP, val F1 per epoch).

Other filename conventions: ``<config_id>_pr_curve.png``,
``<config_id>_confusion_matrix.png``,
``<config_id>_per_event_ap.png``,
``<config_id>_prediction_<event>_<date>.png``; the cross-config
grouped bars live at ``results/figures/test_ap_f1_iou_grouped.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

MODEL_COLOURS = {
    "shearfuse_unet": "#9467bd",
    "td_fusion_unet": "#2ca02c",
    "utae": "#1f77b4",
    "convlstm_unet": "#d62728",
}

# Training-curve figures are written in both formats (hard requirement).
CURVE_FORMATS = ("png", "svg")


def _load_history(history_path: str | Path) -> list[dict]:
    payload = json.loads(Path(history_path).read_text())
    if isinstance(payload, dict):
        return payload["history"]
    return payload


def _fig_dir(config_id: str | None, out_dir: str | Path | None,
             root: str | Path = ".") -> Path:
    if out_dir is not None:
        d = Path(out_dir)
    elif config_id is not None:
        d = Path(root) / "results" / "figures" / config_id
    else:
        d = Path(root) / "results" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_fig(fig, base_path: Path, formats: tuple[str, ...] = CURVE_FORMATS) -> list[Path]:
    written = []
    for fmt in formats:
        p = base_path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=150)
        written.append(p)
    plt.close(fig)
    return written


def plot_training_curves(history_path: str | Path, config_id: str,
                         out_dir: str | Path | None = None,
                         root: str | Path = ".",
                         formats: tuple[str, ...] = CURVE_FORMATS) -> list[Path]:
    """Loss / val-AP+F1 / LR curves from a saved history.json.

    Writes ``<config_id>_loss_curve.{png,svg}``,
    ``<config_id>_ap_curve.{png,svg}`` and ``<config_id>_lr_curve.{png,svg}``.
    Returns the paths written.
    """
    history = _load_history(history_path)
    if not history:
        raise ValueError(f"empty history in {history_path}")
    d = _fig_dir(config_id, out_dir, root)
    epochs = [r["epoch"] for r in history]
    colour = MODEL_COLOURS.get(config_id.rsplit("_", 1)[0], "#333333")
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["train_loss"] for r in history], "o-",
            color=colour, label="train loss")
    ax.plot(epochs, [r["val_loss"] for r in history], "s--",
            color="#d62728", label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("weighted BCE loss")
    ax.set_title(f"{config_id}: train / val loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    written += _save_fig(fig, d / f"{config_id}_loss_curve", formats)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["val_ap"] for r in history], "o-", color=colour,
            label="val AP")
    ax.plot(epochs, [r["val_f1_at_0.5"] for r in history], "s--",
            color="#2ca02c", label="val F1 @ 0.5")
    ax.set_xlabel("epoch"); ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title(f"{config_id}: val AP / F1")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    written += _save_fig(fig, d / f"{config_id}_ap_curve", formats)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [r["lr"] for r in history], "o-", color=colour)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title(f"{config_id}: learning rate")
    ax.grid(alpha=0.3); fig.tight_layout()
    written += _save_fig(fig, d / f"{config_id}_lr_curve", formats)
    return written


def plot_grouped_bars(metrics_paths: list[str | Path],
                      out_path: str | Path,
                      metric_keys: tuple[str, ...] = (
                          "average_precision", "f1", "iou")) -> Path:
    """Grouped bar chart of pooled test AP / F1 / IoU across configs.

    ``metrics_paths``: list of ``*_metrics.json`` files (pooled dicts).
    One group per config, one bar per metric.
    """
    rows = []
    for p in metrics_paths:
        m = json.loads(Path(p).read_text())
        rows.append((m["config_id"], m["pooled"]))
    labels = [f"{k.replace('_', ' ').title()}" for k in metric_keys]
    x = np.arange(len(rows))
    width = 0.8 / len(metric_keys)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(rows)), 4.5))
    for i, k in enumerate(metric_keys):
        ax.bar(x + (i - (len(metric_keys) - 1) / 2) * width,
               [r[1][k] for r in rows], width, label=labels[i])
    ax.set_xticks(x, [r[0] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Test metrics by configuration (pooled, selected threshold)")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray,
                  out_path: str | Path, title: str = "Precision-Recall",
                  ap: float | None = None) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if ap is None else ap
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1f77b4")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(f"{title} (AP = {ap:.3f})")
    ax.grid(alpha=0.3); fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray,
                          threshold: float, out_path: str | Path,
                          title: str = "Confusion matrix") -> Path:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), y_pred,
                          labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred no-fire", "pred fire"])
    ax.set_yticks([0, 1], ["true no-fire", "true fire"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set_title(f"{title} (threshold {threshold:.2f})")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def plot_per_event(per_event: dict, metric: str, out_path: str | Path,
                   title: str | None = None) -> Path:
    """Bar chart of one metric across events (from a *_metrics.json
    ``per_event`` dict)."""
    events = sorted(per_event)
    values = [per_event[e][metric] for e in events]
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(events)), 4.5))
    ax.bar(events, values, color="#1f77b4")
    ax.set_ylabel(metric); ax.set_ylim(0, 1)
    ax.set_title(title or f"Per-event {metric}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def plot_prediction(x_day: np.ndarray, y_true: np.ndarray,
                    y_prob: np.ndarray, threshold: float,
                    out_path: str | Path, title: str = "Prediction",
                    channel_names: tuple[str, ...] = ("active_fire (t)",
                                                      "NDVI (t)")) -> Path:
    """Four-panel prediction visualisation.

    ``x_day``: (41, H, W) encoded input for the LAST input day (or any
    single day); panels show two of its channels (active-fire history
    ch 40 and NDVI ch 3 by default), the ground-truth mask, and the
    probability map with the thresholded mask overlaid as contours.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    af = x_day[40] if x_day.shape[0] > 40 else np.zeros_like(y_true)
    ndvi = x_day[3] if x_day.shape[0] > 3 else np.zeros_like(y_true)
    panels = [(af, "input: active fire (t)"), (ndvi, "input: NDVI (t)"),
              (y_true, "ground truth (t+1)"), (y_prob, "predicted prob.")]
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img, cmap="viridis" if label.startswith("input: NDVI")
                  else ("Reds" if "prob" in label else "Greys"))
        ax.set_title(label); ax.set_xticks([]); ax.set_yticks([])
    axes[3].contour((y_prob >= threshold).astype(float), levels=[0.5],
                    colors="red", linewidths=1)
    fig.suptitle(title); fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def plot_all_for_config(config_id: str, root: str | Path = ".") -> list[Path]:
    """Regenerate every per-config figure from saved artefacts.

    Reads ``results/logs/<id>/history.json`` (curves, PNG+SVG) and
    ``results/metrics/<id>/test_metrics.json`` (per-event bars) when
    they exist. Skips what is missing.
    NOTE: the PR curve and confusion matrix need the raw per-pixel
    probabilities, which are not stored in the metrics JSON; they are
    plotted from ``results/predictions/<id>/`` npz files when present
    (written by evaluate.py --save-predictions), otherwise skipped.
    """
    root = Path(root)
    written: list[Path] = []
    hist = root / "results" / "logs" / config_id / "history.json"
    if hist.exists():
        written += plot_training_curves(hist, config_id, root=root)
    mpath = root / "results" / "metrics" / config_id / "test_metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        fig_dir = _fig_dir(config_id, None, root)
        written.append(plot_per_event(
            m["per_event"], "average_precision",
            fig_dir / f"{config_id}_per_event_ap.png",
            title=f"{config_id}: per-event AP (test)"))
    pred_dir = root / "results" / "predictions" / config_id
    if pred_dir.exists():
        files = sorted(pred_dir.glob("test_*.npz"))
        if files:
            y_true = np.concatenate([np.load(f)["target"].ravel()
                                     for f in files])
            y_prob = np.concatenate([np.nan_to_num(np.load(f)["prob"]).ravel()
                                     for f in files])
            # Valid pixels only: prob is NaN outside the valid mask.
            valid = np.concatenate([np.load(f)["valid"].ravel()
                                    for f in files])
            y_true, y_prob = y_true[valid], y_prob[valid]
            thr = json.loads(mpath.read_text())["threshold"]["selected"] \
                if mpath.exists() else 0.5
            fig_dir = _fig_dir(config_id, None, root)
            written.append(plot_pr_curve(
                y_true, y_prob, fig_dir / f"{config_id}_pr_curve.png",
                title=f"{config_id}: precision-recall (test)"))
            written.append(plot_confusion_matrix(
                y_true, y_prob, thr,
                fig_dir / f"{config_id}_confusion_matrix.png",
                title=f"{config_id}: confusion matrix (test)"))
    return written
