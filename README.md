# Predictive-Maintanence-Using-Facebook-Prophet
Solar anomaly detection with Prophet, PowerBI alerts pipeline

# ☀️ Solar Energy Time Series Forecasting & Anomaly Detection

## 📌 Overview
This project focuses on **time series forecasting and anomaly detection** for solar energy generation using historical energy and weather data.

It uses **Facebook Prophet** for forecasting and a **threshold-based anomaly detection pipeline** to identify abnormal solar panel behavior.

---

## 🚀 Features
- 📊 Time series forecasting using Prophet  
- ⚡ Integration of solar + weather data  
- 📉 Daily aggregation and visualization  
- 🚨 Automated anomaly detection pipeline  
- 📈 Evaluation using MAPE  
- 📤 Export results for dashboard tools (Power BI)

---

## 🗂️ Dataset
- `Total_energy.csv` → Solar energy output  
- `Total_weather.csv` → Weather data  

### Key Features
- GHI, DNI, DHI  
- Temperature  
- Relative Humidity  
- Wind Speed  
- Solar Zenith Angle  

---

## ⚙️ Installation

```bash
pip install prophet pandas matplotlib scikit-learn
