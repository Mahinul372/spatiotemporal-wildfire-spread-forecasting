#!/usr/bin/env python3
"""Build the derived data artefacts for the new pipeline (run once, from raw).

Steps:
  1. Inventory data/raw/ and verify the 4/2/3 event split.
  2. Fit preprocessing statistics on TRAIN days only
     -> data/processed/normalization_stats.json
  3. Build the common eligible target-date sample lists for E1/E2/E3
     -> data/processed/samples.json
  4. Compute class-imbalance statistics on TRAIN TARGET masks only and
     derive the BCE pos_weight for the train split -> printed here;
     the value is printed for reference.

Nothing under data/raw/ is touched. The per-day .npz cache of the old
project is NOT rebuilt: the dataloader reads raw GeoTIFFs and normalises
on the fly (394 files / 1.3 GB; fast enough, and it keeps a single source
of truth in data/raw/).

Usage (from the repository root):
    .venv/bin/python scripts/build_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from data.dataset import (  # noqa: E402
    SPLITS,
    SPLIT_OF_EVENT,
    build_samples,
    inventory_events,
    save_samples,
    split_counts,
)
from data.preprocessing import (  # noqa: E402
    active_fire_mask,
    fit_preprocessing_stats,
    load_day,
    save_stats,
)


def main() -> int:
    raw_dir = ROOT / "data" / "raw"
    proc_dir = ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    events = inventory_events(raw_dir)
    print(f"inventoried {len(events)} events, "
          f"{sum(len(d) for d in events.values())} days")
    missing = [e for e in SPLIT_OF_EVENT if e not in events]
    if missing:
        print(f"ERROR: configured events not present in data/raw: {missing}")
        return 1

    # 1. Fit stats on TRAIN days only.
    train_days = [
        p for e, by_date in events.items() if SPLIT_OF_EVENT[e] == "train"
        for p in by_date.values()
    ]
    train_events = SPLITS["train"]
    print(f"fitting preprocessing stats on {len(train_days)} train days ...")
    stats = fit_preprocessing_stats(train_days, fitted_on=train_events, seed=42)
    stats_path = proc_dir / "normalization_stats.json"
    save_stats(stats, stats_path)
    print(f"  land-cover classes: {stats['landcover_classes']}")
    print(f"  stats -> {stats_path}")

    # 2. Build samples (common target-date set across windows).
    samples = build_samples(events)
    save_samples(samples, proc_dir / "samples.json")
    counts = split_counts(samples)
    print(f"sample counts per window: {counts}")
    print(f"  samples -> {proc_dir / 'samples.json'}")

    # 3. Class imbalance on TARGET masks only (the pixels the loss sees).
    print("class balance over target masks (per split):")
    report: dict = {}
    for split in ("train", "validation", "test"):
        pos = neg = 0
        per_event: dict[str, dict] = {}
        for window_samples in [samples["E1"]]:  # target dates identical per window
            for s in window_samples:
                if s.split != split:
                    continue
                mask = active_fire_mask(load_day(events[s.event][s.target_date]))
                p = int(mask.sum())
                n = int(mask.size - p)
                pos += p
                neg += n
                ev = per_event.setdefault(s.event, {"pos": 0, "neg": 0})
                ev["pos"] += p
                ev["neg"] += n
        total = pos + neg
        fire_pct = 100.0 * pos / total
        pos_weight = neg / pos if pos else float("inf")
        report[split] = {
            "pos_pixels": pos,
            "neg_pixels": neg,
            "total_pixels": total,
            "fire_pct": fire_pct,
            "neg_over_pos": pos_weight,
            "per_event": {
                e: {
                    "pos": v["pos"],
                    "neg": v["neg"],
                    "fire_pct": 100.0 * v["pos"] / (v["pos"] + v["neg"]),
                }
                for e, v in sorted(per_event.items())
            },
        }
        print(f"  {split:10s} pos={pos:>7d} neg={neg:>9d} "
              f"fire={fire_pct:.4f}%  neg/pos={pos_weight:.2f}")
    train_pw = report["train"]["neg_over_pos"]
    print(f"\nBCE pos_weight (train target masks only): {train_pw:.4f}")
    print("-> train neg/pos ratio (use as class weight when training).")
    (proc_dir / "class_balance.json").write_text(json.dumps(report, indent=2))
    print(f"  class balance -> {proc_dir / 'class_balance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
