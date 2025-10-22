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

# Sample data
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
        # Load data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Validate and clean
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[target_column], name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast",
                                 line=dict(color="orange", dash="dash"), mode="lines+markers"))

        fig.add_vline(x=forecast.index[0], line=dict(color="gray", dash="dot"),
                      annotation_text="Forecast Start", annotation_position="top left")

        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.subheader("📅 Forecasted Values")
        forecast_df = pd.DataFrame({"Date": forecast.index, "Forecast": forecast.values})
        st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}"}))

        # Optional download
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast as CSV", csv, "forecast.csv", "text/csv")

        # Evaluation
        if len(df) >= forecast_horizon:
            true = df[target_column][-forecast_horizon:]
            pred = forecast[:forecast_horizon]
            rmse, mae, mape = evaluate_forecast(true.values, pred.values)
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{rmse:.2f}")
            c2.metric("MAE", f"{mae:.2f}")
            c3.metric("MAPE", f"{mape:.2f}%")
        else:
            st.warning("Not enough historical data to evaluate forecast accuracy.")

    except Exception as e:
        st.error(f"Error processing file or forecast: {e}")
