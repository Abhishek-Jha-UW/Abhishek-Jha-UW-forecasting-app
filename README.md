# 📈 Time Series Forecasting App

This interactive Streamlit app empowers users to forecast time series data using five different models—ranging from simple averages to advanced machine learning. Designed for clarity, usability, and impact, the app helps users discover better forecasting methods beyond basic techniques like Simple Moving Average.

---

## 🚀 Features

- **Upload your own data** (CSV or Excel)
- **Forecast using 5 models**:
  - Simple Moving Average
  - ARIMA
  - SARIMA
  - ETS (Exponential Smoothing)
  - XGBoost
- **Compare all models** with RMSE, MAE, MAPE
- **Visualize forecasts** with interactive charts
- **Download forecast results** as CSV
- **Synthetic sample data** included to demonstrate model differences
- **Best model recommendation** based on RMSE
- **Clean UI** with tooltips, instructions, and expandable model descriptions

---

## 📊 Sample Data Format

| Date       | Sales |
|------------|-------|
| 2022-01-01 | 100   |
| 2022-01-08 | 115   |
| ...        | ...   |

- Column A: Date (weekly or monthly)
- Column B: Numeric values (e.g., sales, demand)

---

## 🧠 Why This App?

Many forecasting tools default to simple methods like moving averages. This app shows how more sophisticated models—like SARIMA or XGBoost—can outperform them, especially on complex, noisy, or seasonal data.

---

## 🛠️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/forecasting-app.git
cd forecasting-app

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
