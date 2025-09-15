# Earthquake ML Portfolio Project

**Title:** Next-event magnitude prediction & seismicity analysis (demo pipeline)

**Author:** prasag23


This repository contains a reproducible pipeline to clean earthquake catalog data, perform feature engineering, train a baseline model (RandomForest) to predict event magnitude, evaluate results, and produce a short report. Designed for use as a portfolio piece for geoscience graduates.


## Files
- `data_loader.py` - utilities to load and preprocess the TSV catalog.
- `train_baseline.py` - demo script that trains a RandomForest baseline and saves results.
- `requirements.txt` - python packages required.
- `README.md` - this file.
- `report.md` - auto-generated sample report from demo run.
- `models/xgb_baseline.pkl` - saved model (created after running demo).

## How to use
1. Put `katalog_gempa_v2.tsv` in the repository root (or update path in `train_baseline.py`).
2. Create a virtualenv and install dependencies: `pip install -r requirements.txt`.
3. Run demo: `python train_baseline.py`.

## Notes
- This is a demonstration pipeline. For a production-ready project you would:
  - Expand feature engineering (spatio-temporal grids, focal mechanism angles, magnitude of completeness),
  - Add time-series models (LSTM/transformer on binned-time sequences),
  - Use domain-aware evaluation and uncertainty quantification.
