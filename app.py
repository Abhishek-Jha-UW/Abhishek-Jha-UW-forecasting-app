import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from forecasting_models import (
    SimpleMA, ARIMAForecaster, SARIMAForecaster,
    ETSForecaster, XGBoostForecaster, evaluate_forecast
)

st.set_page_config(page_title="Forecasting Tool", layout="wide")
st.title("📈 Forecasting Tool")

st.markdown("""
Upload a time series file with a `Date` column and one numeric column.  
Supported formats: `.csv`, `.xlsx`  
**Note:** Your file should have:
- Column A: `Date` (e.g., 2023-01-01)
- Column B: Numeric values (e.g., sales, revenue)
""")

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
forecast_horizon = st.slider("Number of weeks to forecast", 4, 24, 6)
model_choice = st.selectbox("Choose forecasting model", ["Simple MA", "ARIMA", "SARIMA", "ETS", "XGBoost"])

with open("sample_data.csv", "rb") as file:
    st.download_button("Download sample data", file, file_name="sample_data.csv", mime="text/csv")

if uploaded_file:
    try:
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

        forecast = None
        if model_choice == "Simple MA":
            model = SimpleMA(df, target_column)
            result = model.apply()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.index, y=result[target_column], name="Actual"))
            fig.add_trace(go.Scatter(x=result.index, y=result['SMA'], name="SMA"))
            st.plotly_chart(fig, use_container_width=True)

        else:
            if model_choice == "ARIMA":
                model = ARIMAForecaster(df, target_column, steps=forecast_horizon)
            elif model_choice == "SARIMA":
                model = SARIMAForecaster(df, target_column, steps=forecast_horizon)
            elif model_choice == "ETS":
                model = ETSForecaster(df, target_column, steps=forecast_horizon)
            elif model_choice == "XGBoost":
                model = XGBoostForecaster(df, target_column, steps=forecast_horizon)

            forecast = model.forecast()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[target_column], name="Actual"))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)

            true = df[target_column][-forecast_horizon:]
            pred = forecast[:forecast_horizon]
            rmse, mae, mape = evaluate_forecast(true.values, pred.values)
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("MAPE", f"{mape:.2f}%")

    except Exception as e:
        st.error(f"Error processing file: {e}")
