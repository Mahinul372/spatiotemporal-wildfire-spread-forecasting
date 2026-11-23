# Spatiotemporal Wildfire Spread Forecasting

**Question:** does temporal context — 1, 3 or 5 days of history — help forecast where a wildfire will spread tomorrow?

**Data:** a 9-event subset of [WildfireSpreadTS](https://zenodo.org/records/8006177) (Zenodo 8006177, CC BY 4.0) — 394 daily GeoTIFFs, 2018–2021, each (H, W, 23) at 375 m resolution. The task is next-day active-fire prediction: given the past days, predict tomorrow's fire mask.

**Status:** the dataset is ready. This repository contains the dataset and its definitions only — the file manifest, the 1/3/5-day sample lists, the normalisation statistics and the downloader. Models have not been chosen yet and nothing has been trained.

## The dataset at a glance

- 9 fire events, 394 event-days, no missing calendar days.
- Event-disjoint splits: train 2018 + 2020 (4 events, 189 days), validation 2019 (2 events, 62 days), test 2021 (3 events, 143 days).
- Three temporal windows share one eligible target-date set — 349 targets (train 169, validation 52, test 128). E1 uses 1 day of history, E2 uses 3 days, E3 uses 5 days; the target is always the next day.
- 23 raw channels per day; channel 22 is active fire (NaN = no fire detected, 806–2212 = fire). Encoded inputs are 41 channels (angular sin/cos, land-cover one-hot).
- Fire is rare: 0.27% of pixels overall; train 0.31% (neg/pos 324.76).

## Get the data

The raw GeoTIFFs (1.3 GB) are not stored in this repo — they are fetched from Zenodo on demand:

```bash
pip install -r requirements.txt
python scripts/download_colab_data.py --dest data/raw
python scripts/download_colab_data.py --dest data/raw --verify-only   # check: 394/394, sizes match
```

The script carries the exact list of 394 files (name + byte size) inside it, pulls only those out of the 48 GB Zenodo archive over HTTP Range requests, and verifies each file's size on download. Files already present with the right size are skipped.

## What's in here

- `data/raw/manifest.json` — the 394 files with exact byte sizes.
- `data/processed/samples.json` — the E1/E2/E3 target-date lists (349 each, identical across windows).
- `data/processed/normalization_stats.json` — normalisation fitted on train days only.
- `data/processed/class_balance.json` — fire prevalence per split.
- `scripts/download_colab_data.py` — fetch the raw files from Zenodo.
- `scripts/build_dataset.py` — rebuild the processed artefacts from raw (provenance).
- `src/data/` — dataset loader: reads raw GeoTIFFs, encodes and normalises on the fly (torch optional).
- `configs/data.yaml` — splits, windows, channels, normalisation.
- `DATA_SPEC.md` — full dataset specification.

## Citation

- Dataset: Gerard, S., Zhao, Y., Sullivan, J. *WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction.* Zenodo, 2023. doi:10.5281/zenodo.8006177
- Related work: Li, W., Hsu, C.-Y., Wang, S. *Adapting Video Foundation Models for Spatiotemporal Wildfire Forecasting via Cross-Modal Progressive Fine-Tuning.* IEEE Transactions on Geoscience and Remote Sensing, 2026. doi:10.1109/TGRS.2026.3652453

## License

GPLv3 — see LICENSE.
