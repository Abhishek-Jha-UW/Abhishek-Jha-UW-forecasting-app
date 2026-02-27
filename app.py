import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import time

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

# =====================================================
# Optimized Model Logic (CACHED)
# =====================================================

@st.cache_data(show_spinner=False)
def run_single_forecast(model_name, series, horizon):
    """Caches individual model runs to prevent unnecessary retraining."""
    model_map = {
        "ARIMA": ARIMAForecaster,
        "SARIMA": SARIMAForecaster,
        "ETS": ETSForecaster,
        "XGBoost": XGBoostForecaster,
    }
    model = model_map[model_name](series, steps=horizon)
    metrics, forecast = model.fit_forecast()
    return metrics, forecast

@st.cache_data(show_spinner=False)
def run_all_models(series, horizon):
    """Caches the entire comparison leaderboard."""
    model_names = ["ARIMA", "SARIMA", "ETS", "XGBoost"]
    results = []
    forecasts = {}
    
    for name in model_names:
        metrics, forecast = run_single_forecast(name, series, horizon)
        results.append({
            "Model": name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"]
        })
        forecasts[name] = forecast
        
    comp_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return comp_df, forecasts

# =====================================================
# Sidebar & Configuration
# =====================================================
st.title("📈 Time Series Forecasting Studio")

with st.sidebar:
    st.header("⚙ Configuration")
    forecast_horizon = st.slider("Forecast Horizon", 3, 52, 12)
    compare_all = st.checkbox("Compare All Models", value=True)
    
    model_choice = st.selectbox(
        "Select Model (if not comparing)",
        ["ARIMA", "SARIMA", "ETS", "XGBoost"],
        disabled=compare_all
    )
    
    # Performance Tip
    if compare_all:
        st.info("⚡ Caching is active. First run takes time, subsequent changes are instant.")

# =====================================================
# Data Loading & Preparation (Minimal changes for speed)
# =====================================================

@st.cache_data
def get_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2019-01-01", periods=260, freq="W")
    trend = np.linspace(200, 500, 260)
    seasonality = 40 * np.sin(np.linspace(0, 20 * np.pi, 260))
    values = trend + seasonality + np.random.normal(0, 25, 260)
    return pd.DataFrame({"Date": dates, "Sales": values, "Revenue": values*8, "Orders": values/5})

use_sample = st.checkbox("Use Built-in Demo Dataset", value=True)

if use_sample:
    df = get_sample_data()
else:
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not uploaded_file: st.stop()
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

# Date processing
date_col = df.select_dtypes(include=['datetime']).columns.tolist()
if not date_col:
    # Fallback to your auto-detect logic
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            date_col = [col]
            break
        except: continue

if not date_col:
    st.error("No date column found!")
    st.stop()

df = df.set_index(date_col[0]).sort_index()
target_column = st.selectbox("Select Target Column", df.select_dtypes(include="number").columns)
series = df[target_column].dropna()

# =====================================================
# Execution Logic (The Speed Improvement)
# =====================================================
st.markdown("---")
st.subheader("📊 Model Results")

start_time = time.time()

with st.spinner("Calculating..."):
    if compare_all:
        comp_df, all_forecasts = run_all_models(series, forecast_horizon)
        best_model_name = comp_df.iloc[0]["Model"]
        best_forecast = all_forecasts[best_model_name]
        best_metrics = comp_df.iloc[0]
        
        # Display Leaderboard
        st.success(f"🏆 Best Model: {best_model_name}")
        cols = st.columns(3)
        cols[0].metric("RMSE", f"{best_metrics['RMSE']:.2f}")
        cols[1].metric("MAE", f"{best_metrics['MAE']:.2f}")
        cols[2].metric("MAPE (%)", f"{best_metrics['MAPE']:.2f}")
        st.table(comp_df)
    else:
        metrics, best_forecast = run_single_forecast(model_choice, series, forecast_horizon)
        best_model_name = model_choice
        cols = st.columns(3)
        cols[0].metric("RMSE", f"{metrics['RMSE']:.2f}")
        cols[1].metric("MAE", f"{metrics['MAE']:.2f}")
        cols[2].metric("MAPE (%)", f"{metrics['MAPE']:.2f}")

end_time = time.time()
st.caption(f"Computation time: {end_time - start_time:.2f} seconds")

# =====================================================
# Visualization (Remains standard as it's fast)
# =====================================================
freq = pd.infer_freq(series.index) or "W"
future_dates = pd.date_range(start=series.index[-1], periods=forecast_horizon + 1, freq=freq)[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical"))
fig.add_trace(go.Scatter(x=future_dates, y=best_forecast.values, name="Forecast", line=dict(dash="dash")))
fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# Download Section
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": best_forecast.values})
st.download_button("Download Forecast", forecast_df.to_csv(index=False), "forecast.csv")
