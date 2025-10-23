import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("📈 Forecasting Tool")

st.write("Upload a CSV or Excel file with a Date column and one numeric column.")

uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip()
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if 'Date' not in df.columns:
            st.error("The file must have a 'Date' column.")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            if not numeric_cols:
                st.error("No numeric column found for forecasting.")
            else:
                target_col = st.selectbox("Select the target column to forecast:", numeric_cols)
                df = df[['Date', target_col]].dropna()

                model_choice = st.selectbox(
                    "Choose Forecasting Model",
                    ["Simple Moving Average", "ARIMA", "ETS", "XGBoost"]
                )

                periods = st.number_input("Forecast periods (e.g. weeks, months)", min_value=1, value=10)

                if st.button("Run Forecast"):
                    try:
                        y = df[target_col].values
                        train = y[:-periods] if len(y) > periods else y
                        history = list(train)

                        if model_choice == "Simple Moving Average":
                            forecast = [np.mean(history[-3:])] * periods

                        elif model_choice == "ARIMA":
                            model = ARIMA(y, order=(1,1,1))
                            model_fit = model.fit()
                            forecast = model_fit.forecast(steps=periods)

                        elif model_choice == "ETS":
                            model = ExponentialSmoothing(y, trend='add', seasonal=None)
                            model_fit = model.fit()
                            forecast = model_fit.forecast(periods)

                        elif model_choice == "XGBoost":
                            X = np.arange(len(y)).reshape(-1, 1)
                            model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
                            model.fit(X, y)
                            X_future = np.arange(len(y), len(y) + periods).reshape(-1, 1)
                            forecast = model.predict(X_future)

                        # Create forecast dates properly
                        freq = pd.infer_freq(df['Date']) or 'D'
                        last_date = df['Date'].iloc[-1]
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit=freq[0]), periods=periods, freq=freq)

                        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

                        fig = px.line(df, x='Date', y=target_col, title=f"{model_choice} Forecast")
                        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast')
                        st.plotly_chart(fig)

                        st.write("### Forecasted Values")
                        st.dataframe(forecast_df)

                    except Exception as e:
                        st.error(f"Error processing forecast: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin.")
