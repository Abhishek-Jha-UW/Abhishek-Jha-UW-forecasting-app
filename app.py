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
# Sidebar Configuration
# =====================================================

with st.sidebar:
    st.header("⚙ Forecast Configuration")

    forecast_horizon = st.slider(
        "Forecast Horizon (Periods)",
        min_value=3,
        max_value=36,
        value=6
    )

    compare_all = st.checkbox("Compare All Models (Recommended)", value=True)

    model_choice = st.selectbox(
        "Select Model (if not comparing)",
        ["ARIMA", "SARIMA", "ETS", "XGBoost"]
    )

    st.markdown("---")
    st.markdown("### 📂 Data Instructions")
    st.markdown("""
    Upload a file containing:
    - A **Date** column (or any datetime-like column)
    - At least one **numeric column**
    """)

# =====================================================
# Better Sample Dataset (5 Years Weekly Data)
# =====================================================

np.random.seed(42)

dates = pd.date_range(start="2019-01-01", periods=260, freq="W")

trend = np.linspace(200, 500, 260)
seasonality = 40 * np.sin(np.linspace(0, 20 * np.pi, 260))
noise = np.random.normal(0, 25, 260)

# occasional demand spikes
spikes = np.random.choice([0, 120], size=260, p=[0.95, 0.05])

values = trend + seasonality + noise + spikes

sample_df = pd.DataFrame({
    "Date": dates,
    "Sales": values
})

# =====================================================
# Data Selection
# =====================================================

st.subheader("📂 Data Source")

# =====================================================
# Download Template
# =====================================================

template_df = pd.DataFrame({
    "Date": pd.date_range(start="2023-01-01", periods=10, freq="W"),
    "Value": np.random.randint(100, 200, 10)
})

template_csv = template_df.to_csv(index=False).encode("utf-8")

st.markdown("### 📥 Download Data Template")
st.download_button(
    "Download Forecasting Template (CSV)",
    template_csv,
    "forecast_template.csv",
    "text/csv"
)

st.caption("Use this template format. Replace the sample values with your own data.")

use_sample = st.checkbox("Use Built-in Sample Dataset (Demo Mode)", value=True)

if use_sample:
    df = sample_df.copy()
    st.info("Using synthetic weekly sales dataset with trend, seasonality, and occasional spikes.")
else:
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

# =====================================================
# Data Preparation
# =====================================================

# Auto detect first datetime column
date_col = None
for col in df.columns:
    try:
        parsed = pd.to_datetime(df[col])
        if parsed.notna().sum() > len(df) * 0.7:
            date_col = col
            break
    except:
        continue

if date_col is None:
    st.error("No valid date column detected. Please use the template.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

numeric_cols = df.select_dtypes(include="number").columns

if len(numeric_cols) == 0:
    st.error("No numeric columns found for forecasting.")
    st.stop()

target_column = st.selectbox("Select Target Column", numeric_cols)

series = df[target_column].dropna()

st.markdown("### 🧾 Data Preview")
st.dataframe(df.head(), use_container_width=True)

# =====================================================
# Model Mapping
# =====================================================

model_map = {
    "ARIMA": ARIMAForecaster,
    "SARIMA": SARIMAForecaster,
    "ETS": ETSForecaster,
    "XGBoost": XGBoostForecaster,
}

# =====================================================
# Run Forecasting
# =====================================================

st.markdown("---")
st.subheader("📊 Model Performance")

results = []
forecasts = {}

with st.spinner("Running forecasting models..."):

    if compare_all:
        for name, model_class in model_map.items():
            model = model_class(series, steps=forecast_horizon)
            metrics, forecast = model.fit_forecast()

            results.append({
                "Model": name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"]
            })

            forecasts[name] = forecast

        comp_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

        best_model_name = comp_df.iloc[0]["Model"]
        best_metrics = comp_df.iloc[0]
        best_forecast = forecasts[best_model_name]

        # --- Best Model Card ---
        st.success(f"🏆 Best Performing Model: {best_model_name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
        col2.metric("MAE", f"{best_metrics['MAE']:.2f}")
        col3.metric("MAPE (%)", f"{best_metrics['MAPE']:.2f}")

        st.markdown("### 📊 Model Leaderboard")
        st.dataframe(
            comp_df.style.format({
                "RMSE": "{:.2f}",
                "MAE": "{:.2f}",
                "MAPE": "{:.2f}"
            }),
            use_container_width=True
        )

    else:
        model = model_map[model_choice](series, steps=forecast_horizon)
        metrics, best_forecast = model.fit_forecast()
        best_model_name = model_choice

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
        col2.metric("MAE", f"{metrics['MAE']:.2f}")
        col3.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")

# =====================================================
# Forecast Visualization
# =====================================================

st.markdown("---")
st.subheader("📈 Forecast Visualization")

freq = pd.infer_freq(series.index)
if freq is None:
    freq = "D"

future_dates = pd.date_range(
    start=series.index[-1],
    periods=forecast_horizon + 1,
    freq=freq
)[1:]

fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=series.index,
    y=series.values,
    name="Historical",
    line=dict(width=2)
))

# Forecast
fig.add_trace(go.Scatter(
    x=future_dates,
    y=best_forecast.values,
    name="Forecast",
    line=dict(dash="dash", width=3)
))

# Vertical separation line
fig.add_vline(x=series.index[-1], line=dict(dash="dot"))

# Shaded forecast region
fig.add_vrect(
    x0=series.index[-1],
    x1=future_dates[-1],
    fillcolor="lightgray",
    opacity=0.2,
    layer="below",
    line_width=0
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title=target_column,
    legend_title="Legend"
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Forecast generated using out-of-sample backtesting and best-performing model.")

# =====================================================
# Download Forecast
# =====================================================

st.markdown("---")
st.subheader("📥 Download Forecast")

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": best_forecast.values
})

st.dataframe(forecast_df, use_container_width=True)

csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label=f"Download {best_model_name} Forecast (CSV)",
    data=csv,
    file_name=f"{best_model_name}_forecast.csv",
    mime="text/csv"
)

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.caption("Built by Abhishek Jha | MSBA | Multi-Model Time Series Forecasting Engine")
