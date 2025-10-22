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
st.download_button("📥 Download sample data", sample_bytes, "sample_data.csv", "text/csv")

uploaded_file = st.file_uploader("📤 Upload your file", type=["csv", "xlsx"])
forecast_horizon = st.slider("Number of periods to forecast", 4, 24, 6)
model_choice = st.selectbox("Choose forecasting model", ["Simple MA", "ARIMA", "SARIMA", "ETS", "XGBoost"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        else:
            df = pd.read_excel(uploaded_file, parse_dates=['Date'])

        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df.dropna(subset=[df.columns[0]], inplace=True)
        df = df.sort_index()
        df.index = df.index.tz_localize(None)

        freq = pd.infer_freq(df.index)
        if freq is None:
            freq = "7D"

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        target_column = st.selectbox("Select column to forecast", df.select_dtypes(include='number').columns)

        # Parameter inputs
        if model_choice == "ARIMA":
            p = st.number_input("AR order (p)", 0, 5, 1)
            d = st.number_input("Differencing (d)", 0, 2, 1)
            q = st.number_input("MA order (q)", 0, 5, 1)
            model = ARIMAForecaster(df, target_column, order=(p, d, q), steps=forecast_horizon, freq=freq)

        elif model_choice == "SARIMA":
            p = st.number_input("AR order (p)", 0, 5, 1)
            d = st.number_input("Differencing (d)", 0, 2, 1)
            q = st.number_input("MA order (q)", 0, 5, 1)
            sp = st.number_input("Seasonal AR (P)", 0, 3, 1)
            sd = st.number_input("Seasonal Diff (D)", 0, 2, 1)
            sq = st.number_input("Seasonal MA (Q)", 0, 3, 1)
            s = st.number_input("Seasonal Periods (s)", 0, 52, 12)
            model = SARIMAForecaster(df, target_column, order=(p, d, q), seasonal_order=(sp, sd, sq, s), steps=forecast_horizon, freq=freq)

        elif model_choice == "ETS":
            trend = st.selectbox("Trend Type", ["add", "mul", None])
            seasonal = st.selectbox("Seasonal Type", [None, "add", "mul"])
            model = ETSForecaster(df, target_column, trend=trend, seasonal=seasonal, steps=forecast_horizon, freq=freq)

        elif model_choice == "XGBoost":
            lags = st.slider("Number of lags", 1, 12, 6)
            model = XGBoostForecaster(df, target_column, lags=lags, steps=forecast_horizon, freq=freq)

        else:
            model = SimpleMA(df, target_column, steps=forecast_horizon, freq=freq)

        forecast, conf_int = model.forecast()

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[target_column], name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast", line=dict(color="orange", dash="dash"), mode="lines+markers"))

        # Confidence interval
        if conf_int is not None:
            fig.add_trace(go.Scatter(
                x=conf_int.index, y=conf_int.iloc[:, 1],
                line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=conf_int.index, y=conf_int.iloc[:, 0],
                fill='tonexty', fillcolor='rgba(255,165,0,0.2)',
                line=dict(width=0), name='Confidence Interval'
            ))

        fig.add_vline(x=forecast.index[0], line=dict(color="gray", dash="dot"),
                      annotation_text="Forecast Start", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

        # Results
        st.subheader("📅 Forecasted Values")
        forecast_df = pd.DataFrame({"Date": forecast.index, "Forecast": forecast.values})
        st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}"}))

        # Download option
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("📤 Download Forecast Results", csv, "forecast_output.csv", "text/csv")

        # Evaluation metrics
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
