"""Preprocessing for WildfireSpreadTS daily GeoTIFFs (tensor-model pipeline).

Verified facts about the raw data (read from all 394 files, 2026-11-23):
  * Each file is one day: shape (H, W, 23), float32, 375 m pixels, UTM.
    H and W are constant within an event and vary across events (297-331 x 208-285).
  * Channel order is the dataset's own order (see configs/data.yaml).
  * Active fire (channel 22) is NaN wherever no fire was detected and a value
    in [806, 2212] where fire was detected (NOT 0/1). Binarisation is
    "finite and > 0 -> 1, else 0"; it is unambiguous because no finite value
    is below 806. The dataset documentation prescribes treating missing
    active-fire values as zero.
  * Other channels have NaNs (clouds / sensor gaps): ch 0-2 5.1%, ch 3-4 12.7%,
    ch 5-11 4.5%, ch 12-13 4.4%, ch 14 4.3%, ch 15 5.9%, ch 16-21 0%.
    Some whole days have an entire channel all-NaN (e.g. ch 0-2 on cloudy days,
    ch 3-4 on 2021-08-13..20 in all three 2021 events). No day is entirely missing.
  * NoData is NaN (no GDAL NoData tag is set on the files).
  * Wind direction, aspect and forecast wind direction are angular and use
    MIXED conventions (wind_direction 0-360 deg, fcst_wind_direction
    -90..+90 deg, aspect 0-360 deg). They are encoded as (sin, cos) so the
    encoding is invariant to the convention and preserves circularity.
  * Land cover (channel 16) is categorical (MODIS IGBP classes). It is
    one-hot encoded over the 16 classes observed in TRAIN events
    [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]; this set covers
    every class observed in validation and test events. An unseen class (or a
    NaN, never observed) encodes as an all-zero row.
  * Reflectance channels (0-2) contain finite sentinel-like minima of -100 and
    NDVI (ch 3) reaches -9687; these finite values are passed through unchanged
    (only NaN is filled). Flagged as a known quirk in DATA_SPEC.md.

Encoded tensor layout (per input day, channel-first):
  18 standardised continuous channels (dataset order, NaN -> train median,
    then (x - train_mean) / train_std),
   6 angular sin/cos channels (wind_direction, aspect, fcst_wind_direction;
    NaN -> train median angle first; values in [-1, 1], not standardised),
  16 land-cover one-hot channels (ascending class order),
   1 active-fire history channel (binarised 0/1, not standardised).
  Total: 41 channels. The same 23 raw channels feed every configuration;
  the 41-channel encoded layout is identical across all windows and models.

All fitted statistics (per-channel medians for NaN filling, means/stds for
standardisation, observed land-cover classes) are computed on TRAIN events
only and stored as JSON, so no validation/test information leaks into
preprocessing. `fit_preprocessing_stats` records exactly which events/days
it was fitted on, so this can be audited from the saved file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

# ---------------------------------------------------------------------------
# Raw channel definitions (dataset's own order, 0-based).
# ---------------------------------------------------------------------------
N_RAW_CHANNELS = 23
TARGET_CHANNEL = 22
TARGET_NAME = "active_fire"

CONTINUOUS_CHANNELS: tuple[tuple[int, str], ...] = (
    (0, "viirs_m11"),
    (1, "viirs_i2"),
    (2, "viirs_i1"),
    (3, "ndvi"),
    (4, "evi2"),
    (5, "total_precipitation"),
    (6, "wind_speed"),
    (8, "min_temperature"),
    (9, "max_temperature"),
    (10, "energy_release"),
    (11, "specific_humidity"),
    (12, "slope"),
    (14, "elevation"),
    (15, "pdsi"),
    (17, "fcst_precipitation"),
    (18, "fcst_wind_speed"),
    (20, "fcst_temperature"),
    (21, "fcst_humidity"),
)

ANGULAR_CHANNELS: tuple[tuple[int, str], ...] = (
    (7, "wind_direction"),
    (13, "aspect"),
    (19, "fcst_wind_direction"),
)

LANDCOVER_CHANNEL = 16
LANDCOVER_NAME = "landcover"

# Classes observed in TRAIN events (verified against all 394 files: this set
# covers every class observed in validation and test events as well).
LANDCOVER_CLASSES: tuple[int, ...] = (
    1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
)

# Encoded (model-facing) channel names, in tensor order.
ENCODED_CHANNEL_NAMES: tuple[str, ...] = tuple(
    [name for _, name in CONTINUOUS_CHANNELS]
    + [f"{name}_{part}" for _, name in ANGULAR_CHANNELS for part in ("sin", "cos")]
    + [f"landcover_{c}" for c in LANDCOVER_CLASSES]
    + [TARGET_NAME]
)
N_ENCODED_CHANNELS = len(ENCODED_CHANNEL_NAMES)  # 41

# Index slices into the encoded channel axis (for tests / model code).
CONT_SLICE = slice(0, len(CONTINUOUS_CHANNELS))
ANG_SLICE = slice(len(CONTINUOUS_CHANNELS),
                  len(CONTINUOUS_CHANNELS) + 2 * len(ANGULAR_CHANNELS))
LC_SLICE = slice(ANG_SLICE.stop, ANG_SLICE.stop + len(LANDCOVER_CLASSES))
AF_INDEX = N_ENCODED_CHANNELS - 1


# ---------------------------------------------------------------------------
# Loading and target mask.
# ---------------------------------------------------------------------------
def load_day(path: str | Path) -> np.ndarray:
    """Load one daily GeoTIFF as a float32 (H, W, 23) array."""
    arr = tifffile.imread(str(path))
    if arr.ndim != 3 or arr.shape[2] != N_RAW_CHANNELS:
        raise ValueError(f"{path}: expected (H, W, 23), got {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def active_fire_mask(day: np.ndarray) -> np.ndarray:
    """Binarised active-fire map for one day.

    NaN (no detection) -> 0; any finite positive value -> 1. All finite
    values in the dataset are >= 806, so this is unambiguous.
    """
    af = day[:, :, TARGET_CHANNEL]
    with np.errstate(invalid="ignore"):
        mask = np.where(np.isfinite(af) & (af > 0), 1, 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Fitting statistics (TRAIN data only).
# ---------------------------------------------------------------------------
def fit_preprocessing_stats(
    day_paths: list[Path],
    fitted_on: list[str] | None = None,
    seed: int = 42,
    max_pixels_per_day_for_median: int = 50_000,
) -> dict:
    """Fit NaN-fill medians and standardisation stats.

    Parameters
    ----------
    day_paths:
        Paths of the days to fit on. Callers MUST pass train days only;
        `fitted_on` records the provenance (event names) in the returned dict.
    fitted_on:
        Human-readable labels (e.g. event names) recorded in the stats file.
    seed:
        RNG seed for the per-day pixel subsample used for medians.
    max_pixels_per_day_for_median:
        Pixels sampled per day for median estimation. Means/stds are computed
        exactly over ALL pixels of ALL given days (streaming accumulation).

    Returns a JSON-serialisable dict with per-channel median/mean/std
    (continuous + angular channels), the observed land-cover classes, and
    provenance metadata.
    """
    rng = np.random.default_rng(seed)
    median_channels = tuple(sorted(
        {c for c, _ in CONTINUOUS_CHANNELS} | {c for c, _ in ANGULAR_CHANNELS}
    ))
    medians: dict[int, list[float]] = {c: [] for c in median_channels}
    sums = {c: 0.0 for c, _ in CONTINUOUS_CHANNELS}
    sqsums = {c: 0.0 for c, _ in CONTINUOUS_CHANNELS}
    counts = {c: 0 for c, _ in CONTINUOUS_CHANNELS}
    landcover_classes: set[int] = set()
    n_days = 0
    for path in day_paths:
        day = load_day(path)
        h, w, _ = day.shape
        n = h * w
        n_days += 1
        idx = np.arange(n)
        if n > max_pixels_per_day_for_median:
            idx = rng.choice(idx, size=max_pixels_per_day_for_median, replace=False)
        flat = day.reshape(n, N_RAW_CHANNELS)[idx]
        lc = flat[:, LANDCOVER_CHANNEL]
        lc = lc[np.isfinite(lc)]
        landcover_classes.update(int(v) for v in np.unique(lc))
        for c in median_channels:
            v = flat[:, c]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            medians[c].append(float(np.median(v)))
        # Exact mean/std over ALL pixels (second pass over the full day).
        full = day.reshape(n, N_RAW_CHANNELS)
        for c, _ in CONTINUOUS_CHANNELS:
            v = full[:, c]
            v = v[np.isfinite(v)].astype(np.float64)
            if v.size == 0:
                continue
            sums[c] += float(v.sum())
            sqsums[c] += float((v ** 2).sum())
            counts[c] += int(v.size)

    stats: dict = {
        "medians": {},
        "means": {},
        "stds": {},
        "fitted_on": fitted_on if fitted_on is not None else [],
        "n_fit_days": n_days,
        "seed": seed,
        "max_pixels_per_day_for_median": max_pixels_per_day_for_median,
    }
    for c in median_channels:
        med = float(np.median(medians[c])) if medians[c] else 0.0
        stats["medians"][str(c)] = med
    for c, _ in CONTINUOUS_CHANNELS:
        if counts[c] > 0:
            mean = sums[c] / counts[c]
            var = max(sqsums[c] / counts[c] - mean * mean, 0.0)
            std = float(np.sqrt(var)) or 1.0
        else:
            mean, std = 0.0, 1.0
        stats["means"][str(c)] = float(mean)
        stats["stds"][str(c)] = float(std)
    stats["landcover_classes"] = sorted(landcover_classes)
    return stats


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Transforming one day.
# ---------------------------------------------------------------------------
def transform_day(day: np.ndarray, stats: dict) -> np.ndarray:
    """Transform one raw (H, W, 23) day into a (C=41, H, W) float32 tensor.

    The target day's active fire is NEVER included as an input feature by
    this function alone: it always appends the day's own active-fire mask as
    the last channel. Callers must only pass INPUT days here; the target
    day's mask comes from `active_fire_mask` separately.
    """
    h, w, _ = day.shape
    medians = stats["medians"]
    means = stats["means"]
    stds = stats["stds"]
    classes = stats["landcover_classes"]

    out = np.empty((N_ENCODED_CHANNELS, h, w), dtype=np.float32)

    for j, (c, _) in enumerate(CONTINUOUS_CHANNELS):
        v = day[:, :, c].astype(np.float64)
        v = np.where(np.isfinite(v), v, float(medians[str(c)]))
        out[j] = ((v - float(means[str(c)])) / float(stds[str(c)])).astype(np.float32)

    base = len(CONTINUOUS_CHANNELS)
    for j, (c, _) in enumerate(ANGULAR_CHANNELS):
        v = day[:, :, c].astype(np.float64)
        v = np.where(np.isfinite(v), v, float(medians.get(str(c), 0.0)))
        rad = np.deg2rad(v)
        out[base + 2 * j] = np.sin(rad).astype(np.float32)
        out[base + 2 * j + 1] = np.cos(rad).astype(np.float32)

    lc_base = base + 2 * len(ANGULAR_CHANNELS)
    lc = day[:, :, LANDCOVER_CHANNEL]
    for j, cls in enumerate(classes):
        out[lc_base + j] = (lc == cls).astype(np.float32)

    out[AF_INDEX] = active_fire_mask(day).astype(np.float32)
    return out


def transform_day_with_target(day: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Transform an input day and return (features, active-fire mask)."""
    feats = transform_day(day, stats)
    return feats, active_fire_mask(day)
