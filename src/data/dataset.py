"""Dataset inventory, event-disjoint splits, and sample construction.

Splits are EVENT-DISJOINT and year-based (the benchmark's example split):
  train:      2018 + 2020 (4 events, 189 days)
  validation: 2019 (2 events, 62 days)
  test:       2021 (3 events, 143 days)

Temporal windows (target is always day t+1 = D; inputs end at D-1):
  E1: 1-day history  (D-1 -> D)
  E2: 3-day history  (D-3..D-1 -> D)
  E3: 5-day history  (D-5..D-1 -> D)

All three windows share ONE common eligible target-date set: a target date D
is eligible only if all five historical observations D-5..D-1 AND the target
day D itself exist as files. Eligibility is checked with calendar-date
arithmetic (not row indices), so it stays correct even if a day were missing.
With the verified data (no date gaps in any event) this gives, per event,
n_days - 5 eligible targets: train 169, validation 52, test 128 (349 total),
identical across E1/E2/E3.

No temporal leakage: every input date is strictly before its target date,
and the target day's file is never used as an input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Splits (event IDs are "<year>/fire_<id>", matching data/raw/ layout).
# ---------------------------------------------------------------------------
SPLITS: dict[str, list[str]] = {
    "train": [
        "2018/fire_21889943",
        "2018/fire_21890524",
        "2020/fire_24332764",
        "2020/fire_24333000",
    ],
    "validation": [
        "2019/fire_23159792",
        "2019/fire_22938749",
    ],
    "test": [
        "2021/fire_25295808",
        "2021/fire_25411093",
        "2021/fire_25295855",
    ],
}

SPLIT_OF_EVENT: dict[str, str] = {
    event: split for split, events in SPLITS.items() for event in events
}

# Temporal windows: name -> number of input days (history length).
WINDOWS: dict[str, int] = {
    "E1": 1,  # t -> t+1
    "E2": 3,  # t-2..t -> t+1
    "E3": 5,  # t-4..t -> t+1
}

# Longest history any window needs (days before the target, excluding it).
MAX_HISTORY_DAYS = max(WINDOWS.values())  # 5

SEED = 42


@dataclass(frozen=True)
class Sample:
    """One prediction sample: input days ending at D-1, target day D."""

    event: str
    split: str
    window: str
    target_date: date
    input_dates: tuple[date, ...]

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "split": self.split,
            "window": self.window,
            "target_date": self.target_date.isoformat(),
            "input_dates": [d.isoformat() for d in self.input_dates],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            event=d["event"],
            split=d["split"],
            window=d["window"],
            target_date=date.fromisoformat(d["target_date"]),
            input_dates=tuple(date.fromisoformat(x) for x in d["input_dates"]),
        )


# ---------------------------------------------------------------------------
# Inventory.
# ---------------------------------------------------------------------------
def inventory_events(raw_dir: str | Path) -> dict[str, dict[date, Path]]:
    """Map each event to {date: GeoTIFF path} for every file under raw_dir."""
    raw_dir = Path(raw_dir)
    events: dict[str, dict[date, Path]] = {}
    for path in sorted(raw_dir.rglob("*.tif")):
        event = f"{path.parent.parent.name}/{path.parent.name}"
        events.setdefault(event, {})[date.fromisoformat(path.stem)] = path
    return events


def event_shape(raw_dir: str | Path, event: str) -> tuple[int, int]:
    """(H, W) of an event, from its first file (verified constant per event)."""
    import tifffile

    events = inventory_events(raw_dir)
    first = sorted(events[event].values())[0]
    with tifffile.TiffFile(str(first)) as tf:
        shape = tf.pages[0].shape
    assert len(shape) == 3 and shape[2] == 23, f"{first}: unexpected shape {shape}"
    return int(shape[0]), int(shape[1])


# ---------------------------------------------------------------------------
# Sample construction.
# ---------------------------------------------------------------------------
def eligible_target_dates(dates: set[date]) -> list[date]:
    """Target dates D with all of D-5..D-1 and D itself present.

    The 5-day requirement is imposed for ALL windows so E1/E2/E3 share one
    identical target-date set (the longest history any window needs).
    """
    eligible = []
    for d in sorted(dates):
        needed = [d - timedelta(days=k) for k in range(0, MAX_HISTORY_DAYS + 1)]
        if all(x in dates for x in needed):
            eligible.append(d)
    return eligible


def build_samples(
    events: dict[str, dict[date, Path]],
    split_of: dict[str, str] | None = None,
) -> dict[str, list[Sample]]:
    """Build samples for every window.

    Returns {window: [Sample, ...]} with identical target-date sets across
    windows (same event, same target date in each window; only the number of
    input days differs).
    """
    split_of = split_of if split_of is not None else SPLIT_OF_EVENT
    out: dict[str, list[Sample]] = {w: [] for w in WINDOWS}
    for event in sorted(events):
        split = split_of[event]
        targets = eligible_target_dates(set(events[event]))
        for target in targets:
            for window, n_days in WINDOWS.items():
                inputs = tuple(
                    target - timedelta(days=k)
                    for k in range(n_days, 0, -1)
                )
                out[window].append(
                    Sample(
                        event=event,
                        split=split,
                        window=window,
                        target_date=target,
                        input_dates=inputs,
                    )
                )
    for window in out:
        out[window].sort(key=lambda s: (s.event, s.target_date))
    return out


def split_counts(samples: dict[str, list[Sample]]) -> dict[str, dict[str, int]]:
    """{window: {split: count}}."""
    counts: dict[str, dict[str, int]] = {}
    for window, slist in samples.items():
        per: dict[str, int] = {}
        for s in slist:
            per[s.split] = per.get(s.split, 0) + 1
        counts[window] = per
    return counts


def save_samples(samples: dict[str, list[Sample]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        window: [s.to_dict() for s in slist] for window, slist in samples.items()
    }
    path.write_text(json.dumps(payload, indent=2))


def load_samples(path: str | Path) -> dict[str, list[Sample]]:
    payload = json.loads(Path(path).read_text())
    return {
        window: [Sample.from_dict(d) for d in slist]
        for window, slist in payload.items()
    }
