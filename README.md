# ☀️ Solar PV Time Series Prediction & Anomaly Detection

A machine learning pipeline for forecasting hourly solar photovoltaic (PV) energy output using **Facebook Prophet**, with built-in anomaly detection to flag days where actual generation significantly deviates from predictions.

---

## 📌 Overview

This project combines energy generation data with weather observations spanning **2014–2022** to train a time series forecasting model. The trained Prophet model is persisted as a `.pkl` file for reuse, enabling both retrospective evaluation and real-time anomaly detection on recent data windows.

---

## 📁 Project Structure

```
.
├── SolarTimeSeriesPred.ipynb   # Main notebook: preprocessing, prediction, anomaly detection
├── model.pkl                   # Trained Facebook Prophet model (pickle)
├── Total_energy.csv            # Hourly energy data (PV, electricity, cooling, heat, emissions)
├── Total_weather.csv           # Hourly weather data (temperature, humidity, wind, pressure, etc.)
├── trial.csv                   # Extracted window of recent data for anomaly testing
├── output.csv                  # Prophet predictions for the trial window
└── trial_daily_anomalies.csv   # Daily-aggregated results with anomaly flags
```

---

## 📊 Dataset

Two CSV files are merged on `Year`, `Month`, `Day`, `Hour`:

| Source | Key Columns |
|---|---|
| `Total_energy.csv` | `PV`, `Electricity`, `Cooling`, `Heat`, `Emission`, `Total Energy` |
| `Total_weather.csv` | `Temperature`, `Dew Point`, `Humidity`, `Wind Speed`, `Wind Gust`, `Pressure`, `Precip` |

- **Time range:** 2014–2022 (hourly granularity → ~78,888 rows)
- **Target variable:** `PV` — solar photovoltaic energy output
- Features are merged into a unified DataFrame with a `Datetime` index

---

## 🧠 Model

**[Facebook Prophet](https://facebook.github.io/prophet/)** — a decomposable time series model designed for daily/weekly/yearly seasonality with strong robustness to missing data and outliers.

- Model is trained on historical PV data and serialized with `pickle`
- Loaded at inference time from `model.pkl`
- Predictions include `yhat`, `yhat_lower`, and `yhat_upper` (uncertainty intervals)

### Train / Test Split

| Split | Rows | Period |
|---|---|---|
| Train | First ~48,888 rows | 2014 – early 2019 |
| Test | Next 25,000 rows | ~2019 – 2021 |
| Anomaly window | Last 20 days | Dec 12–31, 2022 |

---

## 📈 Evaluation

Predictions are resampled to **daily averages** for evaluation.

| Metric | Value |
|---|---|
| MAPE (Mean Absolute Percentage Error) | **~19.3%** |

Visualizations plot train, test, and predicted daily PV output with Prophet confidence intervals.

---

## 🚨 Anomaly Detection

The notebook includes a configurable anomaly detection module over a rolling window of recent data:

```python
days = 20        # Number of recent days to evaluate
threshold = 100  # Absolute difference (PV units) to trigger anomaly flag
```

**How it works:**
1. Extract the last `days × 24` hourly rows from the dataset
2. Run Prophet prediction on this window
3. Resample both actuals and predictions to daily averages
4. Flag days where `|yhat - y| > threshold` as `"Anomaly Detected"`

Results are saved to `trial_daily_anomalies.csv`.

### Sample Anomaly Output (Dec 2022)

| Date | Actual PV | Diff | Anomaly |
|---|---|---|---|
| 2022-12-12 | 766.2 | 110.8 | ✅ Anomaly Detected |
| 2022-12-13 | 681.5 | 10.1 | — |
| 2022-12-14 | 1090.1 | 379.2 | ✅ Anomaly Detected |
| 2022-12-28 | 118.9 | 499.3 | ✅ Anomaly Detected |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install prophet pandas matplotlib scikit-learn
```

### Run the Notebook

```bash
jupyter notebook SolarTimeSeriesPred.ipynb
```

### Load the Pre-trained Model

```python
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new data (must have 'ds' column as datetime)
forecast = model.predict(new_df)
```

---

## 🔧 Configuration

To adjust anomaly detection sensitivity, modify these two parameters in the notebook:

```python
days = 20        # Rolling window size (in days)
threshold = 100  # Sensitivity — lower = more anomalies flagged
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `prophet` | Time series forecasting |
| `pandas` | Data manipulation |
| `matplotlib` | Visualization |
| `scikit-learn` | Evaluation metrics (MAPE, MAE) |
| `pickle` | Model serialization |

---

## 📄 License

This project is open source. Feel free to use and adapt.
