import streamlit as st
import pandas as pd
from forecasting_models import (
    SimpleMA, ARIMAForecaster, ProphetForecaster,
    SARIMAForecaster, ETSForecaster, XGBoostForecaster,
    evaluate_forecast
)

st.set_page_config(page_title="Forecasting Tool", layout="wide")
st.title("📈 Forecasting Tool")

st.markdown("""
Upload a time series file with a `Date` column and one numeric column.  
Supported formats: `.csv`, `.xlsx`
""")

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
forecast_horizon = st.slider("Forecast horizon (weeks)", 4, 24, 6)
model_choice = st.selectbox("Choose forecasting model", ["Simple MA", "ARIMA", "Prophet", "SARIMA", "ETS", "XGBoost"])

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
            st.line_chart(result[[target_column, 'SMA']])

        elif model_choice == "ARIMA":
            model = ARIMAForecaster(df, target_column, steps=forecast_horizon)
            forecast = model.forecast()

        elif model_choice == "Prophet":
            model = ProphetForecaster(df, target_column, periods=forecast_horizon)
            forecast_df = model.forecast()
            forecast = forecast_df.set_index('ds')['yhat']

        elif model_choice == "SARIMA":
            model = SARIMAForecaster(df, target_column, steps=forecast_horizon)
            forecast = model.forecast()

        elif model_choice == "ETS":
            model = ETSForecaster(df, target_column, steps=forecast_horizon)
            forecast = model.forecast()

        elif model_choice == "XGBoost":
            model = XGBoostForecaster(df, target_column, steps=forecast_horizon)
            forecast = model.forecast()

        if forecast is not None:
            combined = pd.concat([df[target_column], forecast], axis=1)
            combined.columns = [target_column, 'Forecast']
            st.line_chart(combined)

            try:
                true = df[target_column][-forecast_horizon:]
                pred = forecast[:forecast_horizon]
                rmse, mae, mape = evaluate_forecast(true.values, pred.values)
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("MAE", f"{mae:.2f}")
                st.metric("MAPE", f"{mape:.2f}%")
            except Exception as e:
                st.warning("Evaluation metrics could not be computed. Check data alignment.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
