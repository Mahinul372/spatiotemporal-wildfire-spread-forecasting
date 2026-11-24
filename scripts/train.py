#!/usr/bin/env python3
"""Training driver: run the 9 configurations in cheapest-first order.

Order (so early results exist even if a Colab session dies):
  TD-FusionUNet E1->E2->E3, UTAE E1->E2->E3, ConvLSTM-UNet E1->E2->E3.

Each config is resumable from its Drive checkpoint and skipped when
already complete (see src/training/trainer.py). After each config
finishes, its training curves are plotted (PNG + SVG) and it is
evaluated on validation (threshold selection) and test.

Usage (from the repo root):
    python scripts/train.py [--config td_fusion_unet_E1]
    python scripts/train.py --config all --root /content/drive/MyDrive/wildfire
    python scripts/train.py --config utae_E3 --no-evaluate
    python scripts/train.py --config td_fusion_unet_E1 --dry-run
        # 2 epochs x 3 batches on real data (chain check only)
    python scripts/train.py --config td_fusion_unet_E1 --plot-only
        # regenerate curves + figures from saved history/metrics, no training

A run summary is written to results/logs/run_summary.json (appended per
config) so the parent agent can see status without re-reading every log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_checkpoint  # noqa: E402
from src.evaluation.visualization import plot_all_for_config  # noqa: E402
from src.training.trainer import config_order, train_config  # noqa: E402


def run_one(config_id: str, root: Path, evaluate: bool, dry_run: bool,
            **train_kwargs) -> dict:
    summary = train_config(config_id, root=root, dry_run=dry_run,
                           **train_kwargs)
    print(f"[{config_id}] train status: {summary['status']} "
          f"best val AP={summary.get('best_val_ap')}")
    if evaluate and summary["status"] in ("completed", "early_stopped"):
        best = Path(summary["best_checkpoint"])
        try:
            val = evaluate_checkpoint(config_id, "validation", best,
                                      root=root)
            summary["val_threshold"] = val["threshold"]["selected"]
            test = evaluate_checkpoint(config_id, "test", best, root=root,
                                       save_predictions=True)
            summary["test_ap"] = test["pooled"]["average_precision"]
            summary["test_f1"] = test["pooled"]["f1"]
            summary["test_iou"] = test["pooled"]["iou"]
        except Exception as e:  # noqa: BLE001 - report, don't kill the run
            print(f"[{config_id}] WARNING: evaluation failed: {e}")
            summary["evaluation_error"] = str(e)
        try:
            figs = plot_all_for_config(config_id, root=root)
            summary["figures"] = [str(f) for f in figs]
        except Exception as e:  # noqa: BLE001
            print(f"[{config_id}] WARNING: plotting failed: {e}")
            summary["plotting_error"] = str(e)
    return summary


def plot_only(config_id: str, root: Path) -> dict:
    """Regenerate curves + figures for a config from saved artefacts."""
    figs = plot_all_for_config(config_id, root=root)
    return {"config_id": config_id, "status": "plotted",
            "figures": [str(f) for f in figs]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="all",
                    help="one config id (e.g. td_fusion_unet_E1) or 'all'")
    ap.add_argument("--root", default=".",
                    help="repo root (local) or Drive copy of the repo")
    ap.add_argument("--no-evaluate", action="store_true",
                    help="train only; skip val/test evaluation + plots")
    ap.add_argument("--dry-run", action="store_true",
                    help="2 epochs x 3 batches per config (chain check)")
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate figures from saved logs; no training")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints (fresh start)")
    args = ap.parse_args()

    root = Path(args.root)
    configs = config_order() if args.config == "all" else [args.config]
    unknown = [c for c in configs if c not in config_order()]
    if unknown:
        print(f"ERROR: unknown config(s): {unknown}", file=sys.stderr)
        return 2

    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "run_summary.json"
    summaries: list[dict] = []
    if summary_path.exists():
        try:
            summaries = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summaries = []

    t0 = time.time()
    for config_id in configs:
        try:
            if args.plot_only:
                s = plot_only(config_id, root)
            else:
                s = run_one(config_id, root, evaluate=not args.no_evaluate,
                            dry_run=args.dry_run, epochs=args.epochs,
                            batch_size=args.batch_size,
                            max_batches=args.max_batches,
                            resume=not args.no_resume)
        except Exception as e:  # noqa: BLE001 - one bad config must not
            # kill the remaining configs; record and continue.
            print(f"[{config_id}] ERROR: {e}", file=sys.stderr)
            s = {"config_id": config_id, "status": "failed",
                 "error": str(e)}
        s["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        summaries = [x for x in summaries if x["config_id"] != config_id]
        summaries.append(s)
        summary_path.write_text(json.dumps(summaries, indent=2))
        print(f"[{config_id}] -> {s['status']} "
              f"(elapsed total {time.time() - t0:.0f}s)")

    print("\n=== run summary ===")
    for s in summaries:
        print(f"  {s['config_id']:22s} {s['status']:16s} "
              f"best_val_AP={s.get('best_val_ap', '-')}")
    print(f"summary -> {summary_path}")
    failed = [s for s in summaries if s["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
