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

st.markdown("""
Upload a time series file with a `Date` column and one numeric column.  
Supported formats: `.csv`, `.xlsx`  
""")

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

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
forecast_horizon = st.slider("Number of periods to forecast", 4, 24, 6)
model_choice = st.selectbox("Choose forecasting model", ["Simple MA", "ARIMA", "SARIMA", "ETS", "XGBoost"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "Date" not in df.columns:
            st.error("Missing 'Date' column.")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        df = df.sort_index()
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            st.error("No numeric column found for forecasting.")
            st.stop()

        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        target_column = st.selectbox("Select column to forecast", numeric_cols)

        model_map = {
            "Simple MA": SimpleMA,
            "ARIMA": ARIMAForecaster,
            "SARIMA": SARIMAForecaster,
            "ETS": ETSForecaster,
            "XGBoost": XGBoostForecaster,
        }
        model = model_map[model_choice](df, target_column, steps=forecast_horizon)
        forecast = model.forecast()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[target_column], name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast
