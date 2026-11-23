# Dataset specification

Everything below was verified from the actual files on 2026-11-23.
Do not re-derive these numbers from memory.

## 1. Dataset at a glance

- Source: WildfireSpreadTS, Zenodo record 8006177, CC BY 4.0.
- Local copy: `data/raw/` — **394 GeoTIFFs, 1,300,244,769 bytes**, 9 events.
  Layout: `data/raw/{2018,2019,2020,2021}/fire_<id>/YYYY-MM-DD.tif`.
  One file per event-day; **no missing calendar days, no date gaps**.
- Each file: `(H, W, 23)`, float32, **375 m** pixels, UTM. H/W constant
  within an event, varies across events (see §4).
- NoData is **NaN** (no GDAL NoData tag). Never delete/modify `data/raw/`.

## 2. Splits (event-disjoint, year-based)

| Split | Events | Days | Samples (each window) |
|---|---|---|---|
| train | 2018/fire_21889943, 2018/fire_21890524, 2020/fire_24332764, 2020/fire_24333000 | 189 | 169 |
| validation | 2019/fire_23159792, 2019/fire_22938749 | 62 | 52 |
| test | 2021/fire_25295808, 2021/fire_25411093, 2021/fire_25295855 | 143 | 128 |
| **total** | **9 events** | **394** | **349** |

## 3. Temporal windows and sample definition

Target is always day **t+1 = D**; inputs end at D−1. All three windows share
**one identical eligible target-date set**: D is eligible only if D−5..D−1
and D all exist as files (the 5-day restriction). Counts are therefore
identical across windows (train 169 / val 52 / test 128).

| Window | History | Input dates for target D | Input tensor shape |
|---|---|---|---|
| E1 | 1 day | D−1 | (1, 41, H, W) |
| E2 | 3 days | D−3, D−2, D−1 | (3, 41, H, W) |
| E3 | 5 days | D−5, D−4, D−3, D−2, D−1 | (5, 41, H, W) |

- Input tensor: `(T, C=41, H, W)`, float32. Target: `(H, W)`, uint8, 0/1.
- Sample lists: `data/processed/samples.json` (keys E1/E2/E3; each entry has
  `event`, `split`, `window`, `target_date`, `input_dates`).
- No temporal leakage: every input date < target date; the target day's file
  is never used as an input (verified: every input date is strictly before its target date).

## 4. Spatial dimensions (H × W per event)

| Event | H × W | Pixels |
|---|---|---|
| 2018/fire_21889943 | 303 × 211 | 63,933 |
| 2018/fire_21890524 | 304 × 208 | 63,232 |
| 2019/fire_22938749 | 328 × 285 | 93,480 |
| 2019/fire_23159792 | 311 × 231 | 71,841 |
| 2020/fire_24332764 | 300 × 245 | 73,500 |
| 2020/fire_24333000 | 331 × 270 | 89,370 |
| 2021/fire_25295808 | 305 × 209 | 63,745 |
| 2021/fire_25295855 | 316 × 228 | 72,048 |
| 2021/fire_25411093 | 297 × 219 | 65,043 |

Batches are padded to the largest H/W in the batch; the collate function
returns a boolean `valid` mask `(B, H, W)` — compute loss/metrics over valid
pixels only. Padding value is 0.

## 5. Channels — 23 raw → 41 encoded

The same 23 raw channels feed every configuration. Encoding expands them
to 41 model channels (identical across all windows and models).

Raw channel order (dataset's own, 0-based):

| Idx | Name | Group | Idx | Name | Group |
|---|---|---|---|---|---|
| 0 | viirs_m11 | reflectance | 12 | slope | topography |
| 1 | viirs_i2 | reflectance | 13 | aspect | topography, angular |
| 2 | viirs_i1 | reflectance | 14 | elevation | topography |
| 3 | ndvi | vegetation | 15 | pdsi | weather |
| 4 | evi2 | vegetation | 16 | landcover | landcover, categorical |
| 5 | total_precipitation | weather | 17 | fcst_precipitation | forecast |
| 6 | wind_speed | weather | 18 | fcst_wind_speed | forecast |
| 7 | wind_direction | weather, angular | 19 | fcst_wind_direction | forecast, angular |
| 8 | min_temperature | weather | 20 | fcst_temperature | forecast |
| 9 | max_temperature | weather | 21 | fcst_humidity | forecast |
| 10 | energy_release | fuel | 22 | **active_fire (target)** | target |
| 11 | specific_humidity | weather | | | |

Encoded tensor channel order (C = 41), axis 0 of each input day:

- 0–17: continuous, standardised, in raw order:
  `viirs_m11, viirs_i2, viirs_i1, ndvi, evi2, total_precipitation,
  wind_speed, min_temperature, max_temperature, energy_release,
  specific_humidity, slope, elevation, pdsi, fcst_precipitation,
  fcst_wind_speed, fcst_temperature, fcst_humidity`
- 18–23: angular sin/cos pairs, in raw order (7, 13, 19):
  `wind_direction_sin, wind_direction_cos, aspect_sin, aspect_cos,
  fcst_wind_direction_sin, fcst_wind_direction_cos`
- 24–39: land-cover one-hot, ascending class:
  `landcover_1, landcover_2, landcover_4, landcover_5, landcover_6,
  landcover_7, landcover_8, landcover_9, landcover_10, landcover_11,
  landcover_12, landcover_13, landcover_14, landcover_15, landcover_16,
  landcover_17`
- 40: `active_fire` history (binarised 0/1, NOT standardised)

Constants in `src/data/preprocessing.py`: `ENCODED_CHANNEL_NAMES`,
`N_ENCODED_CHANNELS = 41`, `CONT_SLICE`, `ANG_SLICE`, `LC_SLICE`, `AF_INDEX = 40`.

## 6. Target encoding

- Target = active-fire mask of day D (t+1), `(H, W)` uint8, **0 = no fire,
  1 = active fire**.
- Raw channel 22 is NaN where no fire was detected and a value in
  **[806, 2212]** where fire was detected (NOT 0/1 — the old project's
  docstring was wrong about this). Binarisation: finite and > 0 → 1,
  else 0. Unambiguous: no finite value is below 806.
- 340 of 394 days have ≥ 1 fire pixel; 54 days have zero fire pixels
  (all-NaN channel 22 → all-zero target). Those days are valid samples.

## 7. Normalisation (fitted on TRAIN only)

Stored in `data/processed/normalization_stats.json` (fitted on all 189
train days, seed 42; `fitted_on` lists the 4 train events; `n_fit_days: 189`).

- **Continuous (18 ch):** NaN → train median, then `(x − train mean) / train std`.
  Means/stds are exact over ALL pixels of all 189 train days; medians from a
  50,000-pixel/day subsample (seed 42).
- **Angular (3 ch):** NaN → train median angle, then (sin, cos). NOT
  standardised (values already in [−1, 1]).
- **Landcover:** one-hot over the 16 train-observed classes (see §5).
  Unseen class or NaN (never observed) → all-zero row.
- **Active fire:** binarised 0/1, NOT standardised.

Fitted values (load the JSON in code — do not hard-code):

| Raw ch | Median (NaN fill) | Mean | Std |
|---|---|---|---|
| 0 viirs_m11 | 1321.0 | 1760.957561 | 1192.691179 |
| 1 viirs_i2 | 2392.0 | 2848.077173 | 1736.970289 |
| 2 viirs_i1 | 908.0 | 1807.369678 | 2013.922138 |
| 3 ndvi | 4916.0 | 4988.995964 | 1996.808660 |
| 4 evi2 | 2385.0 | 2403.166998 | 956.762552 |
| 5 total_precipitation | 0.0 | 0.267421 | 1.271371 |
| 6 wind_speed | 3.3 | 3.521163 | 1.467626 |
| 7 wind_direction (angular) | 265.0 | — | — |
| 8 min_temperature | 283.6 | 282.414440 | 6.006588 |
| 9 max_temperature | 298.4 | 296.861288 | 7.405492 |
| 10 energy_release | 68.0 | 66.745466 | 12.927633 |
| 11 specific_humidity | 0.00601 | 0.005667 | 0.001769 |
| 12 slope | 13.982887 | 12.088209 | 9.059422 |
| 13 aspect (angular) | 173.888931 | — | — |
| 14 elevation | 1470.0 | 1559.083673 | 904.666377 |
| 15 pdsi | −0.97 | −0.879126 | 1.926243 |
| 17 fcst_precipitation | 1.0 | 7.503387 | 21.111383 |
| 18 fcst_wind_speed | 1.589788 | 2.023199 | 1.745222 |
| 19 fcst_wind_direction (angular) | −16.997732 | — | — |
| 20 fcst_temperature | 17.649179 | 16.610421 | 5.396918 |
| 21 fcst_humidity | 0.006132 | 0.005822 | 0.001824 |

## 8. Class imbalance and loss

Computed over **train target masks only** (169 target days, 12,181,457 px):

| Split | Fire px | Non-fire px | Fire % | neg/pos |
|---|---|---|---|---|
| train | 37,394 | 12,144,063 | **0.3070%** | **324.7597** |
| validation | 4,977 | 4,315,008 | 0.1152% | 866.9898 |
| test | 32,621 | 8,536,986 | 0.3807% | 261.7022 |
| overall (all days, all events) | 77,009 | 28,275,000 | 0.2716% | 367.19 |

Per-event fire % (target masks): train 21889943: 0.3131%, 21890524: 0.4539%,
24332764: 0.3344%, 24333000: 0.1573%; val 22938749: 0.1784%,
23159792: 0.0263% (fewest positives); test 25295808: 0.5316%,
25295855: 0.0398%, 25411093: 0.6179% (most positives).
Full breakdown: `data/processed/class_balance.json`.

The train neg/pos ratio (**324.7597**) is the reference value for
class-weighting when training on this split (e.g. a weighted loss's
positive weight). Loss and metrics should be computed over valid
(non-padded) pixels only.

## 9. How to use the pipeline

```python
import sys; sys.path.insert(0, "src")
from pathlib import Path
from data.dataset import load_samples
from data.preprocessing import load_stats
from data.dataloader import WildfireDataset, collate_batch, get_dataloader

stats = load_stats("data/processed/normalization_stats.json")
samples = load_samples("data/processed/samples.json")   # keys E1/E2/E3

ds = WildfireDataset("data/raw", samples["E3"], stats, window="E3")
x, y = ds[0]                 # x: (5, 41, H, W) float32; y: (H, W) uint8
loader = get_dataloader(ds, batch_size=4, shuffle=True, seed=42)
xb, yb, valid = next(iter(loader))  # (B,5,41,H,W), (B,H,W), (B,H,W) bool
```

- `WildfireDataset(raw_dir, samples, stats, window=...)` — `samples` may be
  a list of `Sample` or a path to `samples.json`.
- `get_dataloader(...)` requires torch (optional dependency). Without torch,
  the dataset still works as a plain sequence returning numpy arrays and
  `collate_batch` stacks to numpy.
- Regenerate artefacts from raw: `.venv/bin/python scripts/build_dataset.py`
  (fits stats, rebuilds samples, recomputes class balance; ~25 s).
- Seed: **42** everywhere (stats fitting, sample construction, dataloader).
- Requirements: `requirements.txt` (numpy, tifffile, imagecodecs, PyYAML,
  remotezip). Torch is optional and only needed for `get_dataloader`.

## 10. Quirks and gotchas

1. **Mixed angular conventions** (verified): wind_direction 0–360°,
   fcst_wind_direction −90..+90°, aspect 0–360°. Sin/cos encoding handles
   this; do NOT standardise raw degrees.
2. **Whole-channel all-NaN days** (cloud gaps): ch 0–2 all-NaN on ~15 days,
   ch 3–4 all-NaN on ~20 days (incl. 2021-08-13..20 in all three 2021
   events). Median fill handles them; the transformed tensor is always
   finite (tested). Do NOT drop these samples — they are valid.
3. **No-fire days**: 54 days have an all-NaN channel 22 → all-zero target.
   Valid samples, important negatives.
4. **Reflectance sentinel values**: ch 0–2 have finite minima of −100 and
   NDVI (ch 3) reaches −9687 — likely fill/sentinel values, passed through
   unchanged (only NaN is filled). Flagged for future work; do not
   silently change the fill rule without noting it.
5. **Landcover**: 16 classes observed overall; train union covers every
   class in val/test. Unseen class → all-zero one-hot row (never occurs in
   this data, but the code handles it).
6. **NaN fractions** (all days): ch 0–2: 5.13%, ch 3–4: 12.67%,
   ch 5–11: 4.49%, ch 12–13: 4.36%, ch 14: 4.30%, ch 15: 5.85%,
   ch 16–21: 0%, ch 22: 99.73% (mostly no fire).
7. **Variable H/W**: batches pad; always use the `valid` mask in loss/metrics.
8. **No per-day `.npz` cache**: the old 4.3 GB cache was deleted (it baked in
   a 40-channel transformed layout and subsampled stats). The dataloader
   reads raw GeoTIFFs and normalises on the fly; a cache can be added later
   if I/O demands it.
9. **Validation is the low-fire split** (0.1152% fire; fire_23159792 only
   0.0263%): expect noisy validation metrics on that event; threshold
   selection uses validation only, never test.
10. **Value ranges (raw, all files)**: ch 0: [−100, 15987], ch 1: [−100, 15752],
    ch 2: [−100, 15534], ch 3: [−9687, 9992], ch 4: [−859, 9067],
    ch 5: [0, 60.7], ch 6: [0.4, 11.4], ch 7: [0, 360], ch 8: [261.1, 301.8],
    ch 9: [266.2, 319.9], ch 10: [16, 105], ch 11: [0.0006, 0.0127],
    ch 12: [0, 54.39], ch 13: [0, 359.8], ch 14: [−7, 4051], ch 15: [−7.81, 6.56],
    ch 16: [1, 17], ch 17: [0, 237.7], ch 18: [0.025, 14.3],
    ch 19: [−89.9, 90.0], ch 20: [0.45, 35.3], ch 21: [0.0021, 0.0104],
    ch 22: finite values only in [806, 2212].

## 11. File map

```
spatiotemporal-wildfire-spread-forecasting/
├── CITATION.cff  LICENSE  README.md  DATA_SPEC.md  requirements.txt
├── configs/
│   └── data.yaml          # full data spec (splits, windows, channels, balance)
├── data/
│   ├── raw/               # manifest.json only — the 394 GeoTIFFs are
│   │                      # fetched by scripts/download_colab_data.py
│   └── processed/         # normalization_stats.json, samples.json, class_balance.json
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── dataset.py        # splits, sample construction
│       ├── preprocessing.py  # 23 -> 41 encoding, train-only stats
│       └── dataloader.py     # dataset + collate (torch optional)
└── scripts/
    ├── download_colab_data.py  # fetch the 394 GeoTIFFs from Zenodo
    └── build_dataset.py        # regenerate processed/ artefacts from raw
```
