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

st.set_page_config(
    page_title="Time Series Forecasting Studio",
    layout="wide",
    page_icon="📈"
)

# =====================================================
# Header
# =====================================================

st.title("📈 Time Series Forecasting Studio")
st.caption("Multi-Model Forecasting Engine with Automated Model Selection")

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    st.header("⚙ Forecast Configuration")

    forecast_horizon = st.slider("Forecast Horizon", 3, 36, 6)

    model_choice = st.selectbox(
        "Select Model",
        ["ARIMA", "SARIMA", "ETS", "XGBoost"]
    )

    compare_all = st.checkbox("Compare All Models", value=True)

    st.markdown("---")
    st.markdown("Upload a CSV/XLSX file with:")
    st.markdown("- A **Date** column")
    st.markdown("- A **numeric value** column")

# =====================================================
# File Upload
# =====================================================

uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"])

if uploaded_file:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "Date" not in df.columns:
            st.error("File must contain a 'Date' column.")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) == 0:
            st.error("No numeric column found.")
            st.stop()

        target_column = st.selectbox("Select Target Column", numeric_cols)

        series = df[target_column].dropna()

        # =====================================================
        # Model Mapping
        # =====================================================

        model_map = {
            "ARIMA": ARIMAForecaster,
            "SARIMA": SARIMAForecaster,
            "ETS": ETSForecaster,
            "XGBoost": XGBoostForecaster,
        }

        # =====================================================
        # Run Models
        # =====================================================

        if compare_all:

            st.subheader("📊 Model Performance Comparison")

            results = []
            forecasts = {}

            with st.spinner("Running forecasting models..."):

                for name, model_class in model_map.items():

                    model = model_class(series, steps=forecast_horizon)
                    metrics, forecast = model.fit_forecast()

                    results.append({
                        "Model": name,
                        "RMSE": metrics["RMSE"],
                        "MAE": metrics["MAE"],
                        "MAPE": metrics["MAPE"]
                    })

                    forecasts[name] = forecast

            comp_df = pd.DataFrame(results).sort_values("RMSE")
            st.dataframe(
                comp_df.style.format({
                    "RMSE": "{:.2f}",
                    "MAE": "{:.2f}",
                    "MAPE": "{:.2f}"
                }),
                use_container_width=True
            )

            best_model = comp_df.iloc[0]["Model"]
            st.success(f"🏆 Best Performing Model: {best_model}")

            best_forecast = forecasts[best_model]

        else:
            model = model_map[model_choice](series, steps=forecast_horizon)
            metrics, best_forecast = model.fit_forecast()
            best_model = model_choice

            st.subheader("📊 Model Performance")

            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col2.metric("MAE", f"{metrics['MAE']:.2f}")
            col3.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")

        # =====================================================
        # Generate Future Dates
        # =====================================================

        freq = pd.infer_freq(series.index)

        if freq is None:
            freq = "D"

        future_dates = pd.date_range(
            start=series.index[-1],
            periods=forecast_horizon + 1,
            freq=freq
        )[1:]

        # =====================================================
        # Visualization
        # =====================================================

        st.subheader("📈 Forecast Visualization")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name="Historical",
            line=dict(width=2)
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=best_forecast.values,
            name="Forecast",
            line=dict(dash="dash", width=3)
        ))

        fig.add_vline(x=series.index[-1], line=dict(dash="dot"))

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title=target_column,
            legend_title="Legend"
        )

        st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # Download Section
        # =====================================================

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": best_forecast.values
        })

        st.subheader("📥 Download Forecast")

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            csv,
            f"{best_model}_forecast.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.caption("Built by Abhishek Jha | MSBA | Multi-Model Time Series Forecasting Engine")
