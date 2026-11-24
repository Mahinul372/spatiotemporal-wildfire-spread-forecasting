"""Training loop for one configuration (one architecture x one window).

Adapted from the wildfire-temporal-context Phase 4 trainer for the
four new models in this repo (ShearFuse-UNet, TD-FusionUNet, plain
UTAE, Compact ConvLSTM-UNet). The training recipe is unchanged:

  * AdamW (lr 1e-3, weight decay 0.01 for all four models — all are
    CNN / U-Net family) + linear warmup (3 epochs) + cosine decay to
    10% of peak LR.
  * Weighted BCE with pos_weight = 324.7597 (TRAIN target pixels only,
    loaded from data/processed/class_balance.json and cross-checked
    against configs/training.yaml), loss over valid (non-padded)
    pixels only.
  * Mixed precision (AMP) when CUDA is available; plain fp32 on CPU so
    the same code imports and runs a tiny step locally.
  * Gradient accumulation so the *effective* batch stays 4 even when
    the physical batch must drop (OOM fallback: halve batch, double
    accumulation).
  * Per-epoch train/val loss + AP + F1@0.5 + IoU@0.5 + LR, logged as
    JSONL (one line per epoch) and as history.json.
  * Best-validation checkpoint saved by val AP (``best.pt``); a
    ``last.pt`` resume checkpoint and a per-epoch checkpoint are saved
    every epoch and pruned to the last 2 so Drive usage stays bounded.
  * Early stopping on val AP with patience 10.
  * Train-vs-validation curves (train/val loss, val AP, val F1) are
    written as figures (PNG and SVG) at the end of every run — a hard
    requirement; the data behind them is history.json.
  * Resume restores model + optimizer + scheduler + epoch + best AP +
    history. A config whose history.json says "completed" and whose
    best.pt exists is skipped (``train_config`` returns status
    "skipped_complete") so a re-run after a Colab disconnect does not
    redo finished work.

Model-specific notes
-------------------
* UTAE is called as ``model(xb)`` with no batch_positions, so its
  positional encoding uses relative positions 0..T-1 inside the window
  rather than the absolute day-of-year. The windows are consecutive
  days ending the day before the target, so the ordering information
  is preserved; the dataloader does not carry dates, and changing its
  (x, y, valid) contract for this was not worth it. Documented here
  and in NOTES.md.
* ShearFuse-UNet is built per window (its first convolution takes
  41*T input channels, days stacked); the other three take 41 channels
  per day and aggregate over time internally. ``build_model`` handles
  this; the trainer passes the window through. NOTE: ShearFuse-UNet
  was dropped from the project by the user's decision on 2026-11-24
  (failed all 6 GPU smoke configs); its model file is kept in
  src/models/ for the record but no training config references it.

Nothing here touches GitHub. All paths are under the repo root (local)
or a Drive root (Colab); the caller passes ``root``.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_pooled_metrics
from src.models import build_model
from src.models.common import count_parameters
from src.training.imbalance import load_pos_weight
from src.training.losses import weighted_bce_loss

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# ---------------------------------------------------------------------------
# AMP helpers (version-safe: torch.amp on torch >= 2.0, else torch.cuda.amp).
# ---------------------------------------------------------------------------
try:
    from torch.amp import GradScaler as _AmpScaler
    from torch.amp import autocast as _amp_autocast

    def _autocast(device_type: str, enabled: bool):
        return _amp_autocast(device_type=device_type, enabled=enabled)

    def _scaler(enabled: bool):
        return _AmpScaler("cuda", enabled=enabled)

except ImportError:  # pragma: no cover - torch < 2.0
    from torch.cuda.amp import GradScaler as _AmpScaler  # type: ignore
    from torch.cuda.amp import autocast as _amp_autocast  # type: ignore

    def _autocast(device_type: str, enabled: bool):
        return _amp_autocast(enabled=enabled)

    def _scaler(enabled: bool):
        return _AmpScaler(enabled=enabled)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required to load configs")
    return yaml.safe_load(Path(path).read_text())


def config_order() -> list[str]:
    """Cheapest-first run order so early results exist if a session dies.

    By parameter count and by the measured Colab T4 step times:
    TD-FusionUNet (315k) -> UTAE (1.15M) -> ConvLSTM-UNet (2.34M);
    E1 -> E2 -> E3 within each model. (ShearFuse-UNet was dropped by
    the user's decision on 2026-11-24 — it failed all 6 GPU smoke
    configs — and is not in this list.)
    """
    return [
        "td_fusion_unet_E1", "td_fusion_unet_E2", "td_fusion_unet_E3",
        "utae_E1", "utae_E2", "utae_E3",
        "convlstm_unet_E1", "convlstm_unet_E2", "convlstm_unet_E3",
    ]


def parse_config_id(config_id: str) -> tuple[str, str]:
    """'shearfuse_unet_E3' -> ('shearfuse_unet', 'E3')."""
    if "_" not in config_id:
        raise ValueError(f"bad config id {config_id!r}; expected <model>_<window>")
    model, window = config_id.rsplit("_", 1)
    if window not in ("E1", "E2", "E3"):
        raise ValueError(f"bad window in config id {config_id!r}")
    return model, window


def resolve_config(
    config_id: str, training_cfg: dict, experiments_cfg: dict
) -> dict:
    """Merge per-model/per-config training settings into one flat dict."""
    model, window = parse_config_id(config_id)
    entry = next(
        (c for c in experiments_cfg["configurations"] if c["id"] == config_id),
        None,
    )
    if entry is None:
        raise ValueError(f"config {config_id!r} not in configs/experiments.yaml")
    t = training_cfg
    per_model = t.get("per_model", {}).get(model, {})
    per_config = t.get("per_config", {}).get(config_id, {})
    batch = per_config.get("batch_size", per_model.get("batch_size",
                                                       t["dataloader"]["batch_size"]))
    accum = per_config.get("grad_accum_steps",
                           per_model.get("grad_accum_steps", 1))
    return {
        "config_id": config_id,
        "model": model,
        "window": window,
        "lr": per_model.get("lr", t["optimizer"]["lr"]),
        "weight_decay": per_model.get(
            "weight_decay", t["optimizer"]["weight_decay"]),
        "batch_size": int(batch),
        "grad_accum_steps": int(accum),
        "epochs": int(t["epochs"]),
        "patience": int(t["early_stopping"]["patience"]),
        "warmup_epochs": int(t["scheduler"]["params"].get("warmup_epochs", 3)),
        "eta_min_fraction": float(
            t["scheduler"]["params"].get("eta_min_fraction", 0.1)),
        "amp": bool(t.get("amp", True)),
        "num_workers": int(t["dataloader"].get("num_workers", 0)),
        "seed": int(t.get("random_seed", 42)),
        "pos_weight": float(t["loss"]["pos_weight"]),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    best_val_ap: float, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,  # epochs completed (1-based)
        "best_val_ap": best_val_ap,
        "history": history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_loaders(root: Path, window: str, batch_size: int,
                  num_workers: int, seed: int, max_batches: int | None = None):
    """Train/val loaders for one window. ``max_batches`` caps the number
    of training batches (dry-run use only); validation is never capped."""
    from src.data.dataloader import WildfireDataset, get_dataloader
    from src.data.dataset import load_samples
    from src.data.preprocessing import load_stats

    stats = load_stats(root / "data" / "processed" / "normalization_stats.json")
    samples = load_samples(root / "data" / "processed" / "samples.json")[window]
    train_samples = [s for s in samples if s.split == "train"]
    val_samples = [s for s in samples if s.split == "validation"]
    train_ds = WildfireDataset(root / "data" / "raw", train_samples, stats,
                               window=window)
    val_ds = WildfireDataset(root / "data" / "raw", val_samples, stats,
                             window=window)
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, seed=seed)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, seed=seed)
    if max_batches is not None:
        train_loader = _CappedLoader(train_loader, max_batches)
    return train_loader, val_loader, len(train_samples), len(val_samples)


class _CappedLoader:
    """Wrap a DataLoader, yielding at most N batches (dry-run helper)."""

    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_batches)


# ---------------------------------------------------------------------------
# Epochs
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, pos_weight: float) -> dict:
    """One validation pass: pooled loss + AP + F1/IoU at 0.5 over all
    valid pixels. Returns a dict of metrics (no threshold tuning here;
    threshold selection is a separate, end-of-training step)."""
    model.eval()
    total_loss, total_px = 0.0, 0
    probs, targets = [], []
    for xb, yb, valid in loader:
        xb, yb, valid = xb.to(device), yb.to(device), valid.to(device)
        with _autocast("cuda" if device.type == "cuda" else "cpu",
                       enabled=False):
            logits = model(xb)
        loss = weighted_bce_loss(logits, yb, valid, pos_weight)
        total_loss += float(loss.detach()) * int(valid.sum())
        total_px += int(valid.sum())
        probs.append(torch.sigmoid(logits)[valid].detach().cpu().numpy())
        targets.append(yb[valid].detach().cpu().numpy())
    y_true = np.concatenate(targets)
    y_prob = np.concatenate(probs)
    m = compute_pooled_metrics(y_true, y_prob, threshold=0.5)
    m["loss"] = total_loss / max(total_px, 1)
    m["n_valid_pixels"] = int(total_px)
    return m


def train_one_epoch(model, loader, optimizer, scheduler_step, device,
                    pos_weight: float, grad_accum_steps: int, scaler,
                    use_amp: bool) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_px, n_opt = 0.0, 0, 0
    for step, (xb, yb, valid) in enumerate(loader):
        xb, yb, valid = xb.to(device), yb.to(device), valid.to(device)
        with _autocast("cuda" if device.type == "cuda" else "cpu",
                       enabled=use_amp):
            logits = model(xb)
            loss = weighted_bce_loss(logits, yb, valid, pos_weight)
            loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += float(loss.detach()) * grad_accum_steps * int(valid.sum())
        total_px += int(valid.sum())
        if (step + 1) % grad_accum_steps == 0 or step == len(loader) - 1:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_opt += 1
            if scheduler_step is not None:
                scheduler_step()
    return total_loss / max(total_px, 1)


# ---------------------------------------------------------------------------
# Full run for one config
# ---------------------------------------------------------------------------
def train_config(
    config_id: str,
    root: str | Path = ".",
    device: str | torch.device | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    max_batches: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> dict:
    """Train one config end to end. Returns a summary dict with
    ``status`` in {"completed", "skipped_complete", "early_stopped",
    "dry_run"} plus paths and the best validation AP.

    Layout (all under ``root``):
      results/checkpoints/<id>/best.pt, last.pt, checkpoint_epoch_NNN.pt
      results/logs/<id>/training_log.jsonl, history.json
      results/figures/<id>/<id>_loss_curve.{png,svg},
                         <id>_ap_curve.{png,svg}, <id>_lr_curve.{png,svg}
    """
    root = Path(root)
    training_cfg = load_yaml(root / "configs" / "training.yaml")
    experiments_cfg = load_yaml(root / "configs" / "experiments.yaml")
    cfg = resolve_config(config_id, training_cfg, experiments_cfg)
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    if batch_size is not None:
        cfg["batch_size"] = int(batch_size)
    if dry_run:
        cfg["epochs"] = min(cfg["epochs"], 2)
        max_batches = max_batches or 3

    ckpt_dir = root / "results" / "checkpoints" / config_id
    log_dir = root / "results" / "logs" / config_id
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    history_path = log_dir / "history.json"
    jsonl_path = log_dir / "training_log.jsonl"

    # Skip configs already complete (disconnect-resume contract).
    if resume and best_path.exists() and history_path.exists():
        try:
            hist = json.loads(history_path.read_text())
            if hist.get("status") == "completed":
                return {"config_id": config_id, "status": "skipped_complete",
                        "best_val_ap": hist.get("best_val_ap"),
                        "best_checkpoint": str(best_path)}
        except (json.JSONDecodeError, KeyError):
            pass

    set_seed(cfg["seed"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    scaler = _scaler(enabled=True) if use_amp else None

    train_loader, val_loader, n_train, n_val = build_loaders(
        root, cfg["window"], cfg["batch_size"], cfg["num_workers"],
        cfg["seed"], max_batches=max_batches)

    model = build_model(cfg["model"], window=cfg["window"]).to(device)
    n_params = count_parameters(model)
    pos_weight = load_pos_weight(
        root / "data" / "processed" / "class_balance.json",
        fallback=cfg["pos_weight"])
    if abs(pos_weight - cfg["pos_weight"]) > 1e-3:
        raise ValueError(
            f"pos_weight mismatch: configs/training.yaml says "
            f"{cfg['pos_weight']}, class_balance.json says {pos_weight}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg["grad_accum_steps"]))
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(1, warmup_steps))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps),
        eta_min=cfg["lr"] * cfg["eta_min_fraction"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[max(1, warmup_steps)])

    history: list[dict] = []
    start_epoch, best_val_ap, bad_epochs = 0, -1.0, 0
    resume_path = ckpt_dir / "last.pt"
    if resume and resume_path.exists():
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = int(ckpt["epoch"])
        best_val_ap = float(ckpt["best_val_ap"])
        history = list(ckpt.get("history", []))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
        print(f"[{config_id}] resumed from epoch {start_epoch} "
              f"(best val AP {best_val_ap:.4f})")

    # Re-open the JSONL in append mode on resume (do not wipe old lines).
    log_f = jsonl_path.open("a")
    n_hist_start = len(history)
    status = "completed"
    t_start = time.time()
    try:
        for epoch in range(start_epoch, cfg["epochs"]):
            train_loss = train_one_epoch(
                model, train_loader, optimizer,
                scheduler.step, device, pos_weight,
                cfg["grad_accum_steps"], scaler, use_amp)
            val = validate(model, val_loader, device, pos_weight)
            row = {
                "config_id": config_id, "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val["loss"], 6),
                "val_ap": round(val["average_precision"], 6),
                "val_f1_at_0.5": round(val["f1"], 6),
                "val_iou_at_0.5": round(val["iou"], 6),
                "val_precision_at_0.5": round(val["precision"], 6),
                "val_recall_at_0.5": round(val["recall"], 6),
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_ap": round(max(best_val_ap, val["average_precision"]), 6),
                "elapsed_s": round(time.time() - t_start, 1),
            }
            history.append(row)
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(f"[{config_id}] epoch {epoch + 1}/{cfg['epochs']} "
                  f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
                  f"val_AP={val['average_precision']:.4f} "
                  f"val_F1@0.5={val['f1']:.4f} "
                  f"lr={row['lr']:.2e}")

            improved = val["average_precision"] > best_val_ap
            if improved:
                best_val_ap = val["average_precision"]
                bad_epochs = 0
                save_checkpoint(best_path, model, optimizer, scheduler,
                                epoch + 1, best_val_ap, history)
            else:
                bad_epochs += 1
            save_checkpoint(resume_path, model, optimizer, scheduler,
                            epoch + 1, best_val_ap, history)
            # Keep only the last 2 epoch checkpoints + best.pt (Drive space).
            ep_path = ckpt_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(ep_path, model, optimizer, scheduler,
                            epoch + 1, best_val_ap, history)
            for old in sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))[:-2]:
                old.unlink(missing_ok=True)
            # Persist resume state needed across a restart.
            ckpt = torch.load(resume_path, map_location="cpu",
                              weights_only=False)
            ckpt["bad_epochs"] = bad_epochs
            torch.save(ckpt, resume_path)

            if bad_epochs >= cfg["patience"]:
                status = "early_stopped"
                print(f"[{config_id}] early stopping: val AP did not improve "
                      f"for {cfg['patience']} epochs")
                break
    finally:
        log_f.close()

    if dry_run:
        status = "dry_run"
    summary = {
        "config_id": config_id, "model": cfg["model"],
        "window": cfg["window"], "status": status,
        "epochs_run": len(history) - n_hist_start,
        "epochs_completed": history[-1]["epoch"] if history else start_epoch,
        "best_val_ap": round(best_val_ap, 6),
        "best_checkpoint": str(best_path),
        "params": n_params,
        "n_train": n_train, "n_val": n_val,
        "batch_size": cfg["batch_size"],
        "grad_accum_steps": cfg["grad_accum_steps"],
        "effective_batch": cfg["batch_size"] * cfg["grad_accum_steps"],
        "elapsed_s": round(time.time() - t_start, 1),
        "device": str(device),
    }
    history_path.write_text(json.dumps(
        {**summary, "history": history}, indent=2))

    # Hard requirement: train-vs-validation curves for every config,
    # as data (history.json, written above) + figures (PNG and SVG).
    # Written here so a completed / early-stopped / dry-run config
    # always has its curves, even if the driver is interrupted before
    # its post-training plotting step.
    try:
        from src.evaluation.visualization import plot_training_curves
        figs = plot_training_curves(history_path, config_id, root=root)
        summary["figures"] = [str(f) for f in figs]
    except Exception as e:  # noqa: BLE001 - curves are required, but a
        # plotting failure must not mask a finished training run.
        print(f"[{config_id}] WARNING: curve plotting failed: {e}")
        summary["plotting_error"] = str(e)
    return summary
