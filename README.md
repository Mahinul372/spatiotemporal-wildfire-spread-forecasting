# Spatiotemporal Wildfire Spread Forecasting

**Question:** does temporal context — 1, 3 or 5 days of history — help forecast where a wildfire will spread tomorrow?

**Data:** a 9-event subset of [WildfireSpreadTS](https://zenodo.org/records/8006177) (Zenodo 8006177, CC BY 4.0) — 394 daily GeoTIFFs, 2018–2021, each (H, W, 23) at 375 m resolution. The task is next-day active-fire prediction: given the past days, predict tomorrow's fire mask.

**Status:** 1 of 9 training configs is complete. TD-FusionUNet on 1-day history (E1) finished with test AP 0.4330 — well above the persistence baseline (test AP 0.288). The remaining 8 configs are running on Kaggle GPU.

## First result: TD-FusionUNet, 1-day history (E1)

| | Average precision | F1 | IoU |
|---|---|---|---|
| Validation (2019, threshold 0.83) | 0.1561 | 0.2783 | 0.1616 |
| Test (2021, threshold 0.83) | **0.4330** | 0.2959 | 0.1736 |

- Early stopped at epoch 22 (best checkpoint epoch 11, best validation AP 0.1561). Train loss fell 1.11 → 0.09; validation AP did not improve for 10 epochs.
- 315,441 trainable parameters. Full logs, metrics and curves: `results/logs/td_fusion_unet_E1/`, `results/metrics/td_fusion_unet_E1/`, `results/figures/td_fusion_unet_E1/`.
- For context, the trivial persistence baseline (tomorrow's fire = today's fire) scores AP 0.266 on train, **0.057 on validation and 0.288 on test**. The model beats persistence on both held-out splits — validation by ~2.7×, test by ~1.5×.
- Per-event test AP: fire_25295808: 0.463, fire_25295855: 0.181, fire_25411093: 0.442. The small event (fire_25295855, only 1,232 positive pixels) is the hardest.

## The plan: 9 configs

Three models × three temporal windows (E1 = 1 day, E2 = 3 days, E3 = 5 days), all on identical eligible target dates (349 targets: train 169, validation 52, test 128) and event-disjoint splits (train 2018 + 2020, validation 2019, test 2021):

1. **TD-FusionUNet** (2026) — transform-domain UNet, 315K parameters.
2. **UTAE** (Garnot & Landrieu 2021) — temporal attention encoder, ~1.1M parameters.
3. **Compact ConvLSTM-UNet** — ~2.3M parameters.

E1 of TD-FusionUNet is done (above). The other 8 configs are running on Kaggle GPU; results will be added here as they complete.

## How training handles class imbalance

Fire is extremely rare, so a naive loss would learn to predict "no fire" everywhere:

- **Weighted loss.** Training uses binary cross-entropy with `pos_weight = 324.7597`, computed from training target pixels only: 37,394 positive vs 12,144,063 negative pixels (0.3070% fire). No validation or test data enters this number.
- **Valid pixels only.** The loss is computed over non-padded pixels only; padded borders never contribute.
- **Threshold from validation only.** The decision threshold is chosen as the F1-optimal threshold on the validation split (0.83 for E1) and then reused unchanged on test. The 0.5 threshold is always reported alongside it for comparison.
- **Identical samples.** All three windows share the same 349 eligible target dates, so window comparisons are apples-to-apples.

## Training recipe

- Optimiser: AdamW, learning rate 1e-3, weight decay 0.01; 3-epoch warmup, cosine decay to 10%.
- Up to 50 epochs, early stopping with patience 10 on validation AP.
- Effective batch 8 (TD-FusionUNet batch 8; UTAE/ConvLSTM batch 2 with 4 gradient-accumulation steps).
- **fp32, no mixed precision.** An earlier Colab run with AMP/fp16 failed to learn at all (validation AP 0.0085, far below persistence) while the identical code in fp32 on CPU reached test AP 0.4330 — the fp16 path is the prime suspect, so all GPU runs use fp32.
- Curves per config (train/validation loss, validation AP and F1 per epoch) are saved as `results/logs/<config>/history.json` and plotted in `results/figures/<config>/`.

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

## Run the training

```bash
pip install -r requirements.txt   # torch is preinstalled on Colab/Kaggle — do not reinstall it there
python scripts/train.py --config all --root .
```

`scripts/train.py` runs the 9 configs cheapest-first, resumes interrupted configs from their last checkpoint, and skips completed ones. Configs are defined in `configs/experiments.yaml`; the recipe in `configs/training.yaml`.

## What's in here

- `data/raw/manifest.json` — the 394 files with exact byte sizes.
- `data/processed/samples.json` — the E1/E2/E3 target-date lists (349 each, identical across windows).
- `data/processed/normalization_stats.json` — normalisation fitted on train days only.
- `data/processed/class_balance.json` — fire prevalence per split.
- `scripts/download_colab_data.py` — fetch the raw files from Zenodo.
- `scripts/build_dataset.py` — rebuild the processed artefacts from raw (provenance).
- `scripts/train.py` — train and evaluate the 9 configs.
- `src/data/` — dataset loader: reads raw GeoTIFFs, encodes and normalises on the fly.
- `src/models/` — TD-FusionUNet, UTAE, ConvLSTM-UNet (plus the removed ShearFuse-UNet, inert).
- `src/training/` — trainer, losses, class-imbalance handling.
- `src/evaluation/` — metrics, evaluation, curve/figure generation.
- `configs/data.yaml` — splits, windows, channels, normalisation.
- `configs/training.yaml`, `configs/experiments.yaml` — training recipe and the 9 configs.
- `results/` — per-config logs, metrics and figures (E1 complete so far).
- `DATA_SPEC.md` — full dataset specification.

Model checkpoints (`.pth` files) are **not** stored in this repo — they are binary and too large for this upload path. Metrics, logs and figures are the record of each run.

## Citation

- Dataset: Gerard, S., Zhao, Y., Sullivan, J. *WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction.* Zenodo, 2023. doi:10.5281/zenodo.8006177
- Related work: Li, W., Hsu, C.-Y., Wang, S. *Adapting Video Foundation Models for Spatiotemporal Wildfire Forecasting via Cross-Modal Progressive Fine-Tuning.* IEEE Transactions on Geoscience and Remote Sensing, 2026. doi:10.1109/TGRS.2026.3652453

## License

GPLv3 — see LICENSE.
