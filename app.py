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

# =====================================================
# Core Forecasting Engine (Optimized for Speed)
# =====================================================
@st.cache_data(show_spinner=False)
def run_forecasting(series, horizon, compare_all, single_model_choice):
    """
    Cached forecasting function. Only re-runs if the dataset, horizon, 
    or model choices change. Saves massive amounts of time.
    """
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
        
        return comp_df, best_forecast, best_model_name, best_metrics
    else:
        model = model_map[single_model_choice](series, steps=horizon)
        metrics, best_forecast = model.fit_forecast()
        
        comp_df = pd.DataFrame([{
            "Model": single_model_choice,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"]
        }])
        
        return comp_df, best_forecast, single_model_choice, comp_df.iloc[0]

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header("⚙ Forecast Configuration")

    forecast_horizon = st.slider(
        "Forecast Horizon (Periods)",
        min_value=3,
        max_value=52,
        value=12
    )

    compare_all = st.checkbox("Compare All Models (Recommended)", value=True)

    model_choice = st.selectbox(
        "Select Model (if not comparing)",
        ["ARIMA", "SARIMA", "ETS", "XGBoost"],
        disabled=compare_all
    )

    st.markdown("---")
    st.markdown("### 📂 Data Requirements")
    st.markdown("""
    ✔ One Date column  
    ✔ One or more numeric columns  
    ✔ At least 20 rows recommended  
    """)
    st.info("⚡ **Speed Mode Active:** Model results are cached to prevent unnecessary re-runs.")

# =====================================================
# App Header
# =====================================================
st.title("📈 Time Series Forecasting Studio")
st.caption("Multi-Model Forecasting Engine with Automated Model Selection")

# =====================================================
# Built-In Sample Dataset & Templates
# =====================================================
@st.cache_data
def get_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2019-01-01", periods=260, freq="W")
    trend = np.linspace(200, 500, 260)
    seasonality = 40 * np.sin(np.linspace(0, 20 * np.pi, 260))
    noise = np.random.normal(0, 25, 260)
    spikes = np.random.choice([0, 120], size=260, p=[0.95, 0.05])
    values = trend + seasonality + noise + spikes
    
    return pd.DataFrame({
        "Date": dates,
        "Sales": values,
        "Revenue": values * 8 + np.random.normal(0, 200, 260),
        "Orders": values / 5 + np.random.normal(0, 5, 260)
    })

sample_df = get_sample_data()

# Template Generation
template_df = pd.DataFrame({
    "Date": pd.date_range(start="2020-01-01", periods=52, freq="W"),
    "Sales": np.random.randint(100, 200, 52),
    "Revenue": np.random.randint(1000, 5000, 52),
    "Orders": np.random.randint(20, 80, 52)
})

csv_template = template_df.to_csv(index=False).encode("utf-8")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    template_df.to_excel(writer, index=False)
excel_template = excel_buffer.getvalue()

# =====================================================
# Data Section
# =====================================================
st.subheader("📂 Data Source")
st.markdown("### 📥 Download Forecasting Template")

col1, col2 = st.columns(2)
col1.download_button("Download Template (CSV)", csv_template, "forecast_template.csv", "text/csv")
col2.download_button("Download Template (Excel)", excel_template, "forecast_template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("""
**Template Rules:** Keep ONE Date column | Add numeric columns side-by-side | Replace sample values | Do not change the Date format
""")

use_sample = st.checkbox("Use Built-in Demo Dataset", value=True)

if use_sample:
    df = sample_df.copy()
    st.info("Using synthetic weekly dataset with trend, seasonality, and demand spikes.")
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
# Data Preparation (Fast Date Inference)
# =====================================================
date_col = next((col for col in df.columns if pd.to_datetime(df[col], errors='coerce').notna().mean() > 0.7), None)

if date_col is None:
    st.error("No valid date column detected. Please use the provided template.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

numeric_cols = df.select_dtypes(include="number").columns
if len(numeric_cols) == 0:
    st.error("No numeric columns found.")
    st.stop()

target_column = st.selectbox("Select Target Column to Forecast", numeric_cols)
series = df[target_column].dropna()

if len(series) < 20:
    st.warning("Dataset is very small. Model accuracy may be unreliable.")

with st.expander("🧾 View Data Preview"):
    st.dataframe(df.head(), use_container_width=True)

# =====================================================
# Forecasting Execution
# =====================================================
st.markdown("---")
st.subheader("📊 Model Performance")

with st.spinner("Running forecasting models..."):
    # Calls the high-speed cached function
    comp_df, best_forecast, best_model_name, best_metrics = run_forecasting(
        series, forecast_horizon, compare_all, model_choice
    )

st.success(f"🏆 Best Performing Model: {best_model_name}")

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
col2.metric("MAE", f"{best_metrics['MAE']:.2f}")
col3.metric("MAPE (%)", f"{best_metrics['MAPE']:.2f}")

if compare_all:
    st.markdown("### 📊 Model Leaderboard")
    st.dataframe(
        comp_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE": "{:.2f}"}),
        use_container_width=True
    )

# =====================================================
# Visualization
# =====================================================
st.markdown("---")
st.subheader("📈 Forecast Visualization")

# Fail-safe frequency inference
freq = pd.infer_freq(series.index)
if not freq:
    freq = "D" # Default to daily if Pandas can't figure it out

# Generate future dates safely
future_dates = pd.date_range(
    start=series.index[-1] + pd.Timedelta(days=1) if freq == "D" else series.index[-1] + pd.tseries.frequencies.to_offset(freq),
    periods=forecast_horizon,
    freq=freq
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=series.index, y=series.values, name="Historical", line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=future_dates, y=best_forecast.values, name="Forecast", line=dict(dash="dash", width=3, color="orange")
))

fig.add_vline(x=series.index[-1], line=dict(dash="dot"))

# Shaded forecast region
fig.add_vrect(
    x0=series.index[-1], x1=future_dates[-1],
    fillcolor="lightgray", opacity=0.3, layer="below", line_width=0
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title=target_column,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Download Forecast
# =====================================================
st.markdown("---")
st.subheader("📥 Download Forecast")

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": best_forecast.values
})

st.dataframe(forecast_df.head(), use_container_width=True)

csv_forecast = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label=f"Download {best_model_name} Forecast (CSV)",
    data=csv_forecast,
    file_name=f"{best_model_name}_forecast.csv",
    mime="text/csv"
)

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.caption("Built by Abhishek Jha | MSBA | Multi-Model Time Series Forecasting Engine")
