import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from forecasting_models import (
    SimpleMA, ARIMAForecaster, SARIMAForecaster,
    ETSForecaster, XGBoostForecaster, evaluate_forecast
)

st.set_page_config(page_title="Forecasting Tool", layout="wide")

# --- Tabs: Project Overview + Forecasting ---
tab2, tab1 = st.tabs(["📈 Forecasting Tool", "📘 Project Overview"])

with tab1:
    st.title("📘 About the Project")
    st.markdown("""
    This app compares five forecasting models and recommends the best one based on RMSE.  
    Upload your time series data, choose a model, and download the forecast.  
    Built for analysts, planners, and decision-makers who need fast, reliable forecasts.
    """)

with tab2:
    st.title("📈 Time Series Forecasting Tool")

    # --- Sidebar Instructions ---
    with st.sidebar:
        st.header("📘 How to Use This App")
        st.markdown("""
        1. Upload a time series file with:
           - A **Date** column in **A1**
           - A **numeric value** column in **B**
        2. Choose a forecasting model
        3. Forecast the time series
        4. View and download results
        """)

    # --- Sample Data ---
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=60, freq="W")
    trend = np.linspace(100, 300, 60)
    seasonality = 20 * np.sin(np.linspace(0, 12 * np.pi, 60))
    noise = np.random.normal(0, 15, 60)
    spikes = np.random.choice([0, 50], size=60, p=[0.9, 0.1])
    values = trend + seasonality + noise + spikes
    sample_df = pd.DataFrame({"Date": dates, "Sales": values})
    sample_csv = sample_df.to_csv(index=False)
    sample_bytes = io.BytesIO(sample_csv.encode("utf-8"))
    st.download_button("📥 Download sample data (CSV)", sample_bytes, "sample_data.csv", "text/csv")

    # --- File Upload ---
    uploaded_file = st.file_uploader("📂 Upload your data file", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                import openpyxl
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            elif uploaded_file.name.endswith(".xls"):
                import xlrd
                df = pd.read_excel(uploaded_file, engine="xlrd")
            else:
                st.error("Unsupported file format.")
                st.stop()

            # --- Forecast Settings ---
            st.markdown("### 🔧 Forecast Settings")
            col1, col2 = st.columns(2)
            with col1:
                forecast_horizon = st.slider("Number of periods to forecast", 4, 24, 6)
            with col2:
                model_choice = st.selectbox(
                    "Select a forecasting model",
                    ["Simple MA", "ARIMA", "SARIMA", "ETS", "XGBoost"]
                )

            compare_all = st.checkbox("📊 Compare all models")

            # --- Data Prep ---
            if "Date" not in df.columns:
                st.error("Missing 'Date' column.")
                st.stop()

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                st.error("No numeric column found.")
                st.stop()

            st.write("### 🧾 Preview of Uploaded Data")
            st.dataframe(df.head())

            target_column = st.selectbox("Select column to forecast", numeric_cols)

            # --- Future Date Generator ---
            def generate_future_dates(index, horizon):
                recent = index[-5:]
                deltas = recent.to_series().diff().dropna()
                avg_gap = deltas.mean()
                future_dates = [index[-1] + (i + 1) * avg_gap for i in range(horizon)]
                return future_dates, avg_gap

            future_dates, avg_gap = generate_future_dates(df.index, forecast_horizon)
            st.info(f"🕒 Average step size: {avg_gap}")

            model_map = {
                "Simple MA": SimpleMA,
                "ARIMA": ARIMAForecaster,
                "SARIMA": SARIMAForecaster,
                "ETS": ETSForecaster,
                "XGBoost": XGBoostForecaster,
            }

            if compare_all:
                st.markdown("### 📊 Model Comparison")
                st.info("⏳ Running all models and comparing results…")
                results, all_forecasts = [], {}
                for name, cls in model_map.items():
                    model = cls(df, target_column, steps=forecast_horizon)
                    forecast = model.forecast()
                    all_forecasts[name] = forecast
                    if len(df) >= forecast_horizon:
                        true = df[target_column].iloc[-forecast_horizon:].values
                        pred = forecast.reset_index(drop=True).iloc[:forecast_horizon].values
                        rmse, mae, mape = evaluate_forecast(true, pred)
                        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape})

                if results:
                    comp_df = pd.DataFrame(results).sort_values("RMSE")
                    st.dataframe(comp_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE": "{:.2f}"}))
                    best_model = comp_df.iloc[0]["Model"]
                    st.success(f"✅ Best model based on RMSE: {best_model}")

                    best_forecast = all_forecasts[best_model]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df[target_column], name="Actual", line=dict(color="blue")))
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=best_forecast.values,
                        name="Forecast",
                        line=dict(color="green", dash="dash"),
                        mode="lines+markers"
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    forecast_df = pd.DataFrame({
                        "Date": future_dates,
                        "Forecast": best_forecast.values
                    })
                    st.dataframe(forecast_df)
                    csv_best = forecast_df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"📥 Download {best_model} forecast", csv_best, f"{best_model}_forecast.csv", "text/csv")

            else:
                model = model_map[model_choice](df, target_column, steps=forecast_horizon)
                forecast = model.forecast()

                st.markdown("### 📊 Forecast Visualization")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df[target_column], name="Actual", line=dict(color="blue")))
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast.values,
                    name="Forecast",
                    line=dict(color="orange", dash="dash"),
                    mode="lines+markers"
                ))
                st.plotly_chart(fig, use_container_width=True)

                forecast_df = pd.DataFrame({
                    "Date": future_dates,
                    "Forecast": forecast.values
                })
                st.dataframe(forecast_df)
                csv = forecast_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download forecast as CSV", csv, "forecast.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file or forecast: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by Mithilesh", unsafe_allow_html=True)
