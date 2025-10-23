import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from forecasting_models import (
    SimpleMA, ARIMAForecaster, SARIMAForecaster,
    ETSForecaster, XGBoostForecaster, evaluate_forecast
)

st.set_page_config(page_title="Forecasting Tool", layout="wide")
st.title("📈 Time Series Forecasting Tool")

# --- Sidebar Instructions ---
with st.sidebar:
    st.header("📘 How to Use This App")
    st.markdown("""
    1. Upload a time series file with:
       - A **Date** column in **A1**
       - At least one **numeric column**
    2. Choose a forecasting model
    3. Set the forecast horizon
    4. View predictions, metrics, and download results
    """)
    st.markdown("ℹ️ Models differ in how they handle trend, seasonality, and complexity.")

# --- Sample Data ---
sample_csv = """Date,Sales
2022-01-01,100
2022-01-08,120
2022-01-15,130
2022-01-22,125
2022-01-29,140
2022-02-05,150
2022-02-12,160
2022-02-19,170
2022-02-26,180
2022-03-05,190
2022-03-12,200
2022-03-19,210
2022-03-26,220
2022-04-02,230
2022-04-09,240
2022-04-16,250
2022-04-23,260
2022-04-30,270
2022-05-07,280
2022-05-14,290
"""
sample_bytes = io.BytesIO(sample_csv.encode("utf-8"))
st.download_button("📥 Download sample data (CSV)", sample_bytes, "sample_data.csv", "text/csv")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your time series file", type=["csv", "xlsx"])

# --- Forecast Settings ---
st.markdown("### 🔧 Forecast Settings")
col1, col2 = st.columns(2)
with col1:
    forecast_horizon = st.slider("Number of periods to forecast", 4, 24, 6)
with col2:
    model_choice = st.selectbox(
        "Select a forecasting model (hover to learn more)",
        ["Simple MA", "ARIMA", "SARIMA", "ETS", "XGBoost"],
        help="Simple MA: Moving average\n\nARIMA: Autoregressive Integrated Moving Average\n\nSARIMA: Seasonal ARIMA\n\nETS: Exponential Smoothing\n\nXGBoost: Tree-based regression"
    )

# --- Main Logic ---
if uploaded_file:
    try:
        # Load and clean data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "Date" not in df.columns:
            st.error("Missing 'Date' column. Please ensure your file has a 'Date' column in A1.")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        df = df.sort_index()
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            st.error("No numeric column found for forecasting.")
            st.stop()

        st.write("### 🧾 Preview of Uploaded Data")
        st.dataframe(df.head())

        target_column = st.selectbox("Select column to forecast", numeric_cols)

        # Initialize model
        model_map = {
            "Simple MA": SimpleMA,
            "ARIMA": ARIMAForecaster,
            "SARIMA": SARIMAForecaster,
            "ETS": ETSForecaster,
            "XGBoost": XGBoostForecaster,
        }
        model = model_map[model_choice](df, target_column, steps=forecast_horizon)
        forecast = model.forecast()

        # Plot forecast
        st.markdown("### 📊 Forecast Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(df))), y=df[target_column].values, name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(
            x=list(range(len(df), len(df) + forecast_horizon)),
            y=forecast.values,
            name="Forecast",
            line=dict(color="orange", dash="dash"),
            mode="lines+markers",
            text=[f"Step {i+1}: {v:.2f}" for i, v in enumerate(forecast.values)],
            hoverinfo="text"
        ))
        fig.add_vline(x=len(df), line=dict(color="gray", dash="dot"),
                      annotation_text="Forecast Start", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.subheader("📅 Forecasted Values")
        forecast_df = pd.DataFrame({"Step": list(range(1, forecast_horizon + 1)), "Forecast": forecast.values})
        st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}"}))

        # Download
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download forecast as CSV", csv, "forecast.csv", "text/csv")

        # Evaluation
        if len(df) >= forecast_horizon:
            true = df[target_column].iloc[-forecast_horizon:].values
            pred = forecast.reset_index(drop=True).iloc[:forecast_horizon].values
            rmse, mae, mape = evaluate_forecast(true, pred)
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{rmse:.2f}")
            c2.metric("MAE", f"{mae:.2f}")
            c3.metric("MAPE", f"{mape:.2f}%")
        else:
            st.warning("Not enough historical data to evaluate forecast accuracy.")

    except Exception as e:
        st.error(f"Error processing file or forecast: {e}")
