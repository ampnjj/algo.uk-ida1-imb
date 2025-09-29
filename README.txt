# Time Series Forecasting with Lag Features (Ridge, Lasso, LightGBM)

This project trains and evaluates **three machine learning models** (Ridge, Lasso, LightGBM) on lagged time-series features to forecast the target variable `premium`.  
It provides:

- `train_forecast.py` → Train, evaluate, and save the best model  
- `predict.py` → Load the trained model and predict the next `12`, `24`, or `96` steps  

---

## Overview

- **Target column:** `premium`  
- **Timestamp column:** `valueDateTimeOffset` (parsed to UTC, DST-safe)  
- **Regressors:** All other columns (used only as **lagged** features → no future leakage)  
- **Horizons supported:** `12`, `24`, `96` steps ahead  
- **Models compared:** Ridge, Lasso, LightGBM  
- **Metric:** MAE (primary), RMSE (secondary)  
- **Split:** 80% train, 20% test (chronological order preserved)  

---

## Training Workflow (`train_forecast.py`)

### 1. Load data
- Reads a CSV with `valueDateTimeOffset` and `premium` (plus regressors).  
- Parses timestamps to **UTC** for DST safety.  

### 2. Initial cleaning
- Sort by time.  
- Drop rows with missing timestamp or target.  
- Drop duplicate timestamps (keep last).  

### 3. (Optional) regularize frequency
- If `--freq` (e.g., `H`) is given → reindex to a regular UTC grid.  
- If not → keep the original spacing.  

### 4. Outlier removal
- Removes rows where `premium` is more than **3 standard deviations** away from the mean.  

### 5. Feature engineering
- Creates **lag features** for:  
  - `premium` (lags from `{1,2,3,6,12,24,48,96}` up to `--max-lag`)  
  - Every regressor column (same lag set)  
- Adds **calendar features**: `hour`, `dow` (day of week), `dom` (day of month), `month`.  
- Drops rows with NaNs caused by shifting.  

### 6. Time-based split
- 80% → training  
- 20% → testing  
- Chronological order is preserved.  

### 7. Model training
- **Ridge** & **Lasso**: median imputation + scaling.  
- **LightGBM**: median imputation only (no scaling).  
- Each trained on train set, evaluated on test set.  

### 8. Evaluation & selection
- Compute **MAE** and **RMSE**.  
- Pick the model with **lowest MAE**.  
- Refit that best model on **all rows** after feature engineering.  

### 9. Save artifacts
- `model.pkl` → trained model + metadata (feature list, lag set, freq, etc.)  
- `forecast.csv` → `timestamp, y_true, y_pred` for the test window  
- `metrics.json` → evaluation scores  

---

## Prediction Workflow (`predict.py`)

### 1. Load model bundle
- Reads `model.pkl`.  
- Retrieves model, feature list, lag set, and frequency.  

### 2. Load history
- Reads latest CSV with `valueDateTimeOffset`, `premium`, and regressors.  
- Parses timestamps to UTC, sorts, deduplicates.  

### 3. Determine frequency
- Uses stored `freq` from training.  
- If missing, infers frequency from history.  
- Can be overridden with `--freq`.  

### 4. Build future timeline
- Generates the next **N timestamps** (where N = `12`, `24`, or `96`).  
- Timeline is built in **UTC**.  

### 5. Recursive prediction loop
For each future step:
1. Build lag features from current history.  
2. Add calendar features (`hour`, `dow`, `dom`, `month`).  
3. Predict `premium` using the model.  
4. Append prediction to history (so deeper lags are available for subsequent steps).  

### 6. Save predictions
- Output CSV with `[valueDateTimeOffset, y_pred]`.  

---

## Why this avoids leakage

- Future regressors are **never used directly**. Only their lagged versions are included.  
- Calendar features are derived from the timestamp (safe).  
- Multi-step forecasts are produced **recursively**, feeding each prediction back into the history.  

---

## DST Handling

- All timestamps are parsed to **UTC**.  
- Avoids ambiguity/absence of local times during DST transitions.  
- If you need local time, convert predictions after saving results.  

---

## Metrics

- **MAE** (mean absolute error)  
- **RMSE** (root mean squared error)  
- Best model chosen by lowest **MAE**.  

---

## Quick Start

### Train
```bash
python train_forecast.py \
  --input-csv data/history.csv \
  --output-dir artifacts/run1 \
  --horizon 24 \
  --max-lag 96
# (Optional) add --freq H for hourly data
