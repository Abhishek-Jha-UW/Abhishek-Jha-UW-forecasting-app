import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from forecasting_models import (
    SimpleMA, ARIMAForecaster, SARIMAForecaster,
    ETSForecaster, XGBoostForecaster, evaluate_forecast
)

st.set_page_config(page_title="Forecasting Tool", layout="wide")
st.title("📈 Forecasting Tool")

st.markdown("""
Upload a time series file with a `Date` column and one numeric column.  
Supported formats: `.csv`, `.xlsx`  

**Required format:**
- Column A: `Date` (e.g., 2023-01-01)
- Column B: Numeric values (e.g., sales, revenue)
""")

# 🔹 Embedded sample data
sample_csv = """Date,Sales
2023-01-01,100
2023-01-08,120
2023-01-15,130
2023-01-22,125
2023-01-29,140
2023-02-05,150
"""
sample_bytes = io.BytesIO(sample_csv.encode("utf-8"))
st.download_button("Download sample data", sample_bytes, "sample_data.csv", "text/csv")

# 🔹 Upload and model selection
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
forecast_horizon = st.slider("Number of weeks to forecast", 4, 24, 6)
model_choice = st.selectbox("Choose forecasting model", ["Simple MA", "ARIMA", "SARIMA", "ETS", "XGBoost"])

# 🔹 Forecasting logic
if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, parse_dates=['Date'])
        else:
            st.error("Unsupported file format.")
            st.stop()

        df.set_index('Date', inplace=True)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        target_column = st.selectbox("Select column to forecast", df.select_dtypes(include='number').columns)

        # Initialize model
        if model_choice == "Simple MA":
            model = SimpleMA(df, target_column, steps=forecast_horizon)
        elif model_choice == "ARIMA":
            model = ARIMAForecaster(df, target_column, steps=forecast_horizon)
        elif model_choice == "SARIMA":
            model = SARIMAForecaster(df, target_column, steps=forecast_horizon)
        elif model_choice == "ETS":
            model = ETSForecaster(df, target_column, steps=forecast_horizon)
        elif model_choice == "XGBoost":
            model = XGBoostForecaster(df, target_column, steps=forecast_horizon)

        forecast = model.forecast()

        # 🔹 Plot chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[target_column], name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast", line=dict(color="orange", dash="dash"), mode="lines+markers"))

        for date, value in zip(forecast.index, forecast.values):
            fig.add_annotation(x=date, y=value, text=f"{value:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-20)

        fig.add_vline(x=forecast.index[0], line=dict(color="gray", dash="dot"), annotation_text="Forecast Start", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

        # 🔹 Forecast table
        st.subheader("📅 Forecasted Values")
        forecast_df = pd.DataFrame({"Date": forecast.index, "Forecast": forecast.values})
        st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}"}))

        # 🔹 Evaluation metrics
        if len(df) >= forecast_horizon:
            true = df[target_column][-forecast_horizon:]
            pred = forecast[:forecast_horizon]
            rmse, mae, mape = evaluate_forecast(true.values, pred.values)
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("MAPE", f"{mape:.2f}%")
        else:
            st.warning("Not enough historical data to evaluate forecast accuracy.")

    except Exception as e:
        st.error(f"Error processing file or forecast: {e}")
