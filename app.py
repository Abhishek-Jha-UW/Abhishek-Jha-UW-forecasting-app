import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

from forecasting_models import (
    ARIMAForecaster,
    SARIMAForecaster,
    ETSForecaster,
    XGBoostForecaster
)

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="Time Series Forecasting Studio",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time Series Forecasting Studio")
st.caption("Multi-Model Forecasting Engine with Automated Model Selection")

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    st.header("⚙ Forecast Configuration")

    forecast_horizon = st.slider(
        "Forecast Horizon (Periods)",
        min_value=3,
        max_value=24,   # reduced max for speed
        value=12
    )

    compare_all = st.checkbox("Compare All Models (Slower)", value=False)

    model_choice = st.selectbox(
        "Select Model",
        ["ETS (Fast)", "ARIMA", "SARIMA (Slow)", "XGBoost"]
    )

    fast_mode = st.checkbox("⚡ Fast Mode (Recommended)", value=True)

# =====================================================
# Sample Data
# =====================================================

np.random.seed(42)
dates = pd.date_range(start="2019-01-01", periods=260, freq="W")
trend = np.linspace(200, 500, 260)
seasonality = 40 * np.sin(np.linspace(0, 20 * np.pi, 260))
noise = np.random.normal(0, 25, 260)
values = trend + seasonality + noise

sample_df = pd.DataFrame({
    "Date": dates,
    "Sales": values
})

use_sample = st.checkbox("Use Demo Dataset", value=True)

if use_sample:
    df = sample_df.copy()
else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)

# =====================================================
# Data Prep
# =====================================================

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

series = df.select_dtypes(include="number").iloc[:, 0]

if len(series) < 20:
    st.warning("Dataset too small for reliable forecasting.")

# =====================================================
# MODEL CACHING (CRITICAL FIX)
# =====================================================

@st.cache_data(show_spinner=False)
def run_model(model_name, series_values, horizon):
    series_local = pd.Series(series_values)

    model_map = {
        "ETS (Fast)": ETSForecaster,
        "ARIMA": ARIMAForecaster,
        "SARIMA (Slow)": SARIMAForecaster,
        "XGBoost": XGBoostForecaster,
    }

    model = model_map[model_name](series_local, steps=horizon)
    metrics, forecast = model.fit_forecast()

    return metrics, forecast


# =====================================================
# Forecasting
# =====================================================

st.markdown("---")
st.subheader("📊 Model Performance")

results = []
forecasts = {}

if compare_all:
    progress = st.progress(0)

    model_list = ["ETS (Fast)", "ARIMA"]

    if not fast_mode:
        model_list += ["SARIMA (Slow)", "XGBoost"]

    for i, name in enumerate(model_list):
        metrics, forecast = run_model(name, series.values, forecast_horizon)

        results.append({
            "Model": name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"]
        })

        forecasts[name] = forecast

        progress.progress((i + 1) / len(model_list))

    comp_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    best_model_name = comp_df.iloc[0]["Model"]
    best_forecast = forecasts[best_model_name]
    best_metrics = comp_df.iloc[0]

    st.success(f"🏆 Best Model: {best_model_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
    col2.metric("MAE", f"{best_metrics['MAE']:.2f}")
    col3.metric("MAPE (%)", f"{best_metrics['MAPE']:.2f}")

    st.dataframe(comp_df, use_container_width=True)

else:
    metrics, best_forecast = run_model(model_choice, series.values, forecast_horizon)
    best_model_name = model_choice

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col2.metric("MAE", f"{metrics['MAE']:.2f}")
    col3.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")

# =====================================================
# Visualization
# =====================================================

st.markdown("---")
st.subheader("📈 Forecast")

freq = pd.infer_freq(series.index)
if freq is None:
    freq = "W"

future_dates = pd.date_range(
    start=series.index[-1] + pd.tseries.frequencies.to_offset(freq),
    periods=forecast_horizon,
    freq=freq
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=series.index,
    y=series.values,
    name="Historical"
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=best_forecast.values,
    name="Forecast",
    line=dict(dash="dash")
))

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Download
# =====================================================

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": best_forecast.values
})

st.download_button(
    "Download Forecast CSV",
    forecast_df.to_csv(index=False).encode("utf-8"),
    f"{best_model_name}_forecast.csv"
)
