# 📈 Forecasting Tool

A modular, user-friendly time series forecasting app built with Streamlit. Supports multiple models and Excel/CSV uploads.

## 🔧 Features

- Upload `.csv` or `.xlsx` files with a `Date` column and one numeric column
- Choose from 5 forecasting models:
  - Simple Moving Average
  - ARIMA
  - SARIMA
  - ETS
  - XGBoost
- Interactive Plotly charts
- Evaluation metrics: RMSE, MAE, MAPE
- Sample data download for quick testing

## 📁 File Format

Your file should have:
- Column A: `Date` (e.g., 2023-01-01)
- Column B: Numeric values (e.g., sales, revenue)

## 📊 Sample Data

Click the “Download sample data” button in the app to get a ready-to-use CSV.  
It includes weekly sales data with a `Date` column and numeric values.

## 🚀 Deployment

This app is ready for Streamlit Cloud. Just upload the repo and go live.

## 📦 Requirements

See `requirements.txt` for all dependencies.

## 🧠 Next Steps

- Add Prophet support once deployment is stable
- Enable model tuning and export options
- Add dashboard-style layout and user onboarding
