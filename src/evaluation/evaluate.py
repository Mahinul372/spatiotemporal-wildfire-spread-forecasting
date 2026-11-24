"""Evaluate a trained checkpoint on validation or test.

Workflow:
  1. Load ``best.pt`` for a config.
  2. Run it over the split, collecting per-pixel probabilities and the
     event id of every sample (valid pixels only).
  3. Threshold selection: fit the F1-optimal threshold on VALIDATION
     only (``select_threshold``); the test run reuses that threshold
     and always reports 0.5 alongside it.
  4. Write pooled + per-event metrics JSON to
     ``results/metrics/<config_id>/<split>_metrics.json``. Optionally
     save per-event probability maps (``--save-predictions``) to
     ``results/predictions/<config_id>/<split>_<event>.npz`` for the
     prediction visualisations in ``visualization.py``.

Usage (from the repo root):
    python -m src.evaluation.evaluate --config shearfuse_unet_E1 \
        --split test --checkpoint results/checkpoints/shearfuse_unet_E1/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import WildfireDataset, get_dataloader  # noqa: E402
from src.data.dataset import load_samples  # noqa: E402
from src.data.preprocessing import load_stats  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_per_event_metrics,
    compute_pooled_metrics,
    count_parameters,
    peak_gpu_memory_gb,
    reset_peak_gpu_memory,
    select_threshold,
)
from src.models import build_model  # noqa: E402
from src.training.trainer import parse_config_id  # noqa: E402


def collect_predictions(model, loader, device, event_ids) -> dict:
    """Run a split, returning flat valid-pixel arrays + per-event ids."""
    model.eval()
    probs, targets, events = [], [], []
    reset_peak_gpu_memory()
    t0 = time.time()
    with torch.no_grad():
        for i, (xb, yb, valid) in enumerate(loader):
            xb, yb, valid = xb.to(device), yb.to(device), valid.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits)[valid].detach().cpu().numpy()
            t = yb[valid].detach().cpu().numpy()
            probs.append(p)
            targets.append(t)
            # Repeat this batch's event id once per valid pixel.
            ev = np.repeat(np.asarray(event_ids[i * xb.shape[0]:
                                                (i + 1) * xb.shape[0]]),
                           valid.cpu().numpy().sum(axis=(1, 2)))
            events.append(ev)
    return {
        "y_prob": np.concatenate(probs),
        "y_true": np.concatenate(targets),
        "events": np.concatenate(events),
        "elapsed_s": round(time.time() - t0, 1),
        "peak_gpu_gb": peak_gpu_memory_gb(),
    }


def evaluate_checkpoint(
    config_id: str,
    split: str,
    checkpoint_path: str | Path,
    root: str | Path = ".",
    threshold: float | None = None,
    val_threshold_path: str | Path | None = None,
    save_predictions: bool = False,
    device: str | torch.device | None = None,
) -> dict:
    """Evaluate one checkpoint on one split; write the metrics JSON.

    ``threshold``: if None, 0.5 is used and (for the test split) the
    validation-selected threshold is loaded from
    ``results/metrics/<id>/validation_metrics.json`` when it exists.
    ``val_threshold_path`` overrides where that threshold is read from.
    """
    root = Path(root)
    model_name, window = parse_config_id(config_id)
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    stats = load_stats(root / "data" / "processed" / "normalization_stats.json")
    samples = load_samples(root / "data" / "processed" / "samples.json")[window]
    split_samples = [s for s in samples if s.split == split]
    if not split_samples:
        raise ValueError(f"no {split} samples for window {window}")
    ds = WildfireDataset(root / "data" / "raw", split_samples, stats, window=window)
    loader = get_dataloader(ds, batch_size=4, shuffle=False, num_workers=0, seed=42)
    event_ids = [s.event for s in split_samples]

    model = build_model(model_name, window=window).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    ckpt_epoch = int(ckpt.get("epoch", -1))

    out = collect_predictions(model, loader, device, event_ids)
    y_true, y_prob, events = out["y_true"], out["y_prob"], out["events"]

    # Threshold: validation-fitted when evaluating test; 0.5 otherwise.
    val_threshold = None
    if split == "test":
        src = (Path(val_threshold_path) if val_threshold_path is not None
               else root / "results" / "metrics" / config_id
               / "validation_metrics.json")
        if src.exists():
            val_threshold = json.loads(src.read_text())["threshold"]["selected"]
    thr = threshold if threshold is not None else (val_threshold or 0.5)

    pooled = compute_pooled_metrics(y_true, y_prob, threshold=thr)
    pooled_05 = compute_pooled_metrics(y_true, y_prob, threshold=0.5)
    per_event = compute_per_event_metrics(y_true, y_prob, events, threshold=thr)
    per_event_05 = compute_per_event_metrics(y_true, y_prob, events, threshold=0.5)

    result: dict = {
        "config_id": config_id, "model": model_name, "window": window,
        "split": split, "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt_epoch,
        "threshold": {"selected": float(thr), "source":
                      "validation" if (split == "test" and val_threshold is not None
                                       and threshold is None) else "fixed_0.5"},
        "pooled": pooled,
        "pooled_at_0.5": pooled_05,
        "per_event": per_event,
        "per_event_at_0.5": per_event_05,
        "params": count_parameters(model),
        "elapsed_s": out["elapsed_s"],
        "peak_gpu_gb": out["peak_gpu_gb"],
        "n_samples": len(split_samples),
    }
    if split == "validation":
        sel = select_threshold(y_true, y_prob)
        result["threshold"]["selected"] = sel["threshold"]
        result["threshold"]["source"] = "validation_f1_optimal"
        result["threshold"]["f1_at_selected"] = sel["f1"]
        result["threshold"]["f1_at_0.5"] = sel["f1_at_0.5"]
        # Recompute the reported pooled/per-event metrics at the selected
        # threshold (the 0.5 numbers stay alongside for comparison).
        result["pooled"] = compute_pooled_metrics(y_true, y_prob,
                                                  threshold=sel["threshold"])
        result["per_event"] = compute_per_event_metrics(
            y_true, y_prob, events, threshold=sel["threshold"])

    out_dir = root / "results" / "metrics" / config_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{config_id}] {split}: AP={result['pooled']['average_precision']:.4f} "
          f"F1@{result['threshold']['selected']:.2f}="
          f"{result['pooled']['f1']:.4f} "
          f"(F1@0.5={pooled_05['f1']:.4f}) -> {out_path}")

    if save_predictions:
        pred_dir = root / "results" / "predictions" / config_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_prediction_maps(model, loader, device, split_samples, pred_dir,
                             split)
    return result


@torch.no_grad()
def save_prediction_maps(model, loader, device, samples, pred_dir: Path,
                         split: str) -> None:
    """Save per-sample probability maps (full HxW, padded pixels NaN)."""
    model.eval()
    for i, (xb, yb, valid) in enumerate(loader):
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        for j in range(xb.shape[0]):
            s = samples[i * xb.shape[0] + j]
            h, w = yb.shape[1], yb.shape[2]
            prob_map = np.full((h, w), np.nan, dtype=np.float32)
            prob_map[valid[j].cpu().numpy()] = probs[j][valid[j].cpu().numpy()]
            name = (f"{split}_{s.event.replace('/', '_')}_"
                    f"{s.target_date.isoformat()}.npz")
            np.savez_compressed(
                pred_dir / name, prob=prob_map,
                target=yb[j].cpu().numpy().astype(np.uint8),
                valid=valid[j].cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="e.g. shearfuse_unet_E1")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()
    evaluate_checkpoint(args.config, args.split, args.checkpoint,
                        root=args.root, threshold=args.threshold,
                        save_predictions=args.save_predictions)


if __name__ == "__main__":
    main()
