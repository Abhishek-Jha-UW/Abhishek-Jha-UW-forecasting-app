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
st.set_page_config(page_title="Forecasting Studio Pro", page_icon="📈", layout="wide")

# --- SPEED OPTIMIZATION: CACHED MODEL ENGINE ---
@st.cache_data(show_spinner=False)
def run_forecasting(series, horizon, model_choice, compare_all):
    model_map = {
        "ARIMA": ARIMAForecaster,
        "SARIMA": SARIMAForecaster,
        "ETS": ETSForecaster,
        "XGBoost": XGBoostForecaster,
    }
    
    results = []
    forecasts = {}

    if compare_all:
        for name, model_class in model_map.items():
            model = model_class(series, steps=horizon)
            metrics, forecast = model.fit_forecast()
            results.append({"Model": name, **metrics})
            forecasts[name] = forecast
        
        comp_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
        best_name = comp_df.iloc[0]["Model"]
        return comp_df, forecasts[best_name], best_name
    else:
        model = model_map[model_choice](series, steps=horizon)
        metrics, forecast = model.fit_forecast()
        comp_df = pd.DataFrame([{"Model": model_choice, **metrics}])
        return comp_df, forecast, model_choice

# =====================================================
# Sidebar & Configuration
# =====================================================
with st.sidebar:
    st.header("⚙ Configuration")
    forecast_horizon = st.slider("Forecast Horizon", 3, 52, 12)
    compare_all = st.checkbox("Compare All Models", value=True)
    model_choice = st.selectbox("Single Model", ["ARIMA", "SARIMA", "ETS", "XGBoost"], disabled=compare_all)
    
    st.divider()
    st.info("💡 **Pro Tip:** Caching is enabled. Changing the horizon or model will trigger a re-train, but changing UI layout won't.")

# =====================================================
# Data Loading (Optimized for Speed)
# =====================================================
@st.cache_data
def get_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2019-01-01", periods=260, freq="W")
    values = np.linspace(200, 500, 260) + 40 * np.sin(np.linspace(0, 20 * np.pi, 260)) + np.random.normal(0, 25, 260)
    return pd.DataFrame({"Date": dates, "Sales": values, "Revenue": values * 8, "Orders": values / 5})

st.title("📈 Time Series Forecasting Studio")

use_sample = st.checkbox("Use Demo Dataset", value=True)

if use_sample:
    df = get_sample_data()
else:
    uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"])
    if not uploaded_file: st.stop()
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

# --- Automatic Data Prep ---
date_col = next((c for c in df.columns if pd.to_datetime(df[c], errors='coerce').notna().mean() > 0.7), None)
if not date_col:
    st.error("No Date column found!")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col])
df = df.set_index(date_col).sort_index()
target_column = st.selectbox("Target Column", df.select_dtypes("number").columns)
series = df[target_column].dropna()

# =====================================================
# Execution Logic
# =====================================================
st.divider()

with st.spinner("Calculating forecasts..."):
    # This call is cached!
    comp_df, best_forecast, best_model_name = run_forecasting(
        series, forecast_horizon, model_choice, compare_all
    )

# --- Display Metrics ---
st.success(f"🏆 Champion Model: {best_model_name}")
m_col1, m_col2, m_col3 = st.columns(3)
best_m = comp_df[comp_df["Model"] == best_model_name].iloc[0]
m_col1.metric("RMSE", f"{best_m['RMSE']:.2f}")
m_col2.metric("MAE", f"{best_m['MAE']:.2f}")
m_col3.metric("MAPE", f"{best_m['MAPE']:.2f}%")

# =====================================================
# Visualization (Always Fast)
# =====================================================
freq = pd.infer_freq(series.index) or "D"
future_dates = pd.date_range(start=series.index[-1], periods=forecast_horizon + 1, freq=freq)[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual"))
fig.add_trace(go.Scatter(x=future_dates, y=best_forecast, name="Forecast", line=dict(dash='dash', color='orange')))
fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Download Section
# =====================================================
forecast_export = pd.DataFrame({"Date": future_dates, "Forecast": best_forecast})
st.download_button("Export Forecast", forecast_export.to_csv(index=False), "forecast.csv", "text/csv")
