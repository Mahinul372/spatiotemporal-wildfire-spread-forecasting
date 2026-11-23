"""PyTorch dataset / dataloader for WildfireSpreadTS next-day prediction.

Each sample returns:
  x: (T, C=41, H, W) float32 input tensor (T = 1, 3 or 5 input days)
  y: (H, W) uint8 target mask for day t+1 (0 = no fire, 1 = active fire)

H and W vary per event (297-331 x 208-285), so batches are padded to the
largest H/W in the batch and a boolean `valid` mask (B, H, W) marks real
pixels (losses/metrics must ignore padded pixels). `valid` is all-True when
a batch happens to contain a single event/shape.

Determinism: no shuffling across events is ever done by the dataset itself;
the DataLoader shuffles sample order only within the training split when
asked, using a torch.Generator seeded with 42. Sample construction is
fully deterministic (sorted events and dates).

Torch is NOT a hard dependency of this module: it is optional.
Without torch installed, `WildfireDataset` still works as a plain Python
sequence returning numpy arrays, and `collate_batch` stacks to numpy.
`get_dataloader` requires torch and raises an informative ImportError until
then. Tests use the numpy path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .dataset import Sample, inventory_events, load_samples
from .preprocessing import (
    N_ENCODED_CHANNELS,
    active_fire_mask,
    load_day,
    load_stats,
    transform_day,
)

try:  # torch is optional; the numpy path works without it.
    import torch
    from torch.utils.data import DataLoader, Dataset as _TorchDataset
except ImportError:  # pragma: no cover - exercised when torch is not installed
    torch = None
    DataLoader = None
    _TorchDataset = object


class WildfireDataset(_TorchDataset):
    """Event-disjoint, leakage-free dataset for one temporal window.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw GeoTIFFs (data/raw/).
    samples:
        List of `Sample` for one window (see src/data/dataset.py), or a
        path to a samples.json file (all windows; `window` selects one).
    stats:
        Preprocessing statistics dict (see src/data/preprocessing.py), or a
        path to a normalization_stats.json file. Must be fitted on TRAIN
        data only.
    window:
        Window name ("E1", "E2" or "E3"); used only when `samples` is a path.
    transform:
        Optional callable applied to (x, y) after loading, e.g. augmentation.
        Must preserve shapes and the (x, y) contract.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        samples: Sequence[Sample] | str | Path,
        stats: dict | str | Path,
        window: str = "E1",
        transform: Callable | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        if isinstance(samples, (str, Path)):
            samples = load_samples(samples)[window]
        self.samples: list[Sample] = list(samples)
        self.window = window
        self.stats = load_stats(stats) if isinstance(stats, (str, Path)) else stats
        self.transform = transform
        self._events = inventory_events(self.raw_dir)
        # Path lookup: (event, date) -> file.
        self._paths = {
            (event, d): p for event, by_date in self._events.items()
            for d, p in by_date.items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[idx]
        days = []
        for d in sample.input_dates:
            day = load_day(self._paths[(sample.event, d)])
            days.append(transform_day(day, self.stats))
        x = np.stack(days, axis=0).astype(np.float32)  # (T, C, H, W)
        target_day = load_day(self._paths[(sample.event, sample.target_date)])
        y = active_fire_mask(target_day)  # (H, W), uint8
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y

    def get_sample_meta(self, idx: int) -> Sample:
        return self.samples[idx]


def collate_batch(
    batch: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[Any, Any, Any]:
    """Pad a batch to the max H/W and return (x, y, valid).

    x: (B, T, C, H, W) float32, y: (B, H, W) uint8,
    valid: (B, H, W) bool — True for real pixels, False for padding.
    Returns torch tensors when torch is available, else numpy arrays.
    Padding value is 0 (the standardised mean for continuous channels).
    """
    b = len(batch)
    t, c, h_max, w_max = 0, N_ENCODED_CHANNELS, 0, 0
    for x, _ in batch:
        t = max(t, x.shape[0])
        h_max = max(h_max, x.shape[2])
        w_max = max(w_max, x.shape[3])
    x_pad = np.zeros((b, t, c, h_max, w_max), dtype=np.float32)
    y_pad = np.zeros((b, h_max, w_max), dtype=np.uint8)
    valid = np.zeros((b, h_max, w_max), dtype=bool)
    for i, (x, y) in enumerate(batch):
        h, w = x.shape[2], x.shape[3]
        x_pad[i, : x.shape[0], :, :h, :w] = x
        y_pad[i, :h, :w] = y
        valid[i, :h, :w] = True
    if torch is not None:
        return (
            torch.from_numpy(x_pad),
            torch.from_numpy(y_pad),
            torch.from_numpy(valid),
        )
    return x_pad, y_pad, valid


def get_dataloader(
    dataset: WildfireDataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> Any:
    """Build a torch DataLoader (requires torch; optional dependency).

    Shuffling uses a torch.Generator seeded with `seed` so training order
    is reproducible. No shuffling across events is imposed by the dataset;
    `shuffle=True` only reorders samples within the given split.
    """
    if torch is None or DataLoader is None:
        raise ImportError(
            "torch is not installed in this environment; it is optional. "
            "Use WildfireDataset directly (numpy path) until then."
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        generator=generator,
    )
