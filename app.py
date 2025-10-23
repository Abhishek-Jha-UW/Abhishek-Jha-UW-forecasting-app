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
tab1, tab2 = st.tabs(["📘 Project Overview", "📈 Forecasting Tool"])

with tab1:
    st.title("📘 About the Project")
    st.markdown("""
    When I started as a cost engineer, I noticed most forecasting was done in Excel using Simple Moving Averages. From sales estimates to raw material tracking, it gave rough trends but wasn’t reliable for decisions.

    That’s what led me to build this app.

    I’ve worked with real company data and seen how better forecasts can save costs—like timing steel purchases based on predicted price drops. But not everyone knows Python, and building ML models for every case isn’t practical.

    So, I built a no-code tool where you can:
    - Upload an Excel file  
    - Run five models (MA, ARIMA, SARIMA, ETS, XGBoost)  
    - Get the best-fit recommendation  
    - View and download results instantly  

    Even Power BI and Tableau can’t do this without scripts.

    **Why not one ML model?**  
    Because ML works best later, once you have large, clean data. Most teams need quick, reliable forecasts for sales, inventory, or price trends without heavy setup.

    **Impact:**  
    Using this approach, we saved 3–5% on raw material costs by timing purchases better.

    **Use cases:**  
    - Sales and demand forecasting  
    - Inventory and price prediction  
    - Inflation or trend estimation  

    More models coming soon, including Prophet and ensembles. Would love your feedback!
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
           - No non-numeric columns
        2. Choose a forecasting model
        3. Forecast the time series
        4. View predictions, metrics, and download results
        """)
        with st.expander("📘 About This App"):
            st.markdown("""
            This app was born from a real-world challenge: relying on Simple Moving Average for critical forecasts—from RM pricing to inventory planning.

            I’ve seen how better models can drive smarter decisions, like timing steel purchases to match market dips—saving 3–5% in costs.

            But building ML models for every small prediction isn’t scalable. This tool helps you find the best-fit model first—no coding required.

            Use it for:
            - Sales volume forecasting
            - Inventory analysis
            - Inflation trend prediction
            - Any time series over weeks, months, or quarters

            More models coming soon—including Meta’s Prophet and bootstrapped ensembles.
            """)

    # --- Synthetic Sample Data ---
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
    uploaded_file = st.file_uploader("📂 Upload your private files", type=["csv", "xlsx"])

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

    # --- Main Logic ---
    if uploaded_file:
        try:
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

            model_map = {
                "Simple MA": SimpleMA,
                "ARIMA": ARIMAForecaster,
                "SARIMA": SARIMAForecaster,
                "ETS": ETSForecaster,
                "XGBoost": XGBoostForecaster,
            }

            if compare_all:
                st.markdown("### 📊 Model Comparison")
                results = []
                all_forecasts = {}
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
                    fig.add_trace(go.Scatter(x=list(range(len(df))), y=df[target_column].values, name="Actual", line=dict(color="blue")))
                    fig.add_trace(go.Scatter(
                        x=list(range(len(df), len(df) + forecast_horizon)),
                        y=best_forecast.values,
                        name="Forecast",
                        line=dict(color="green", dash="dash"),
                        mode="lines+markers",
                        text=[f"Step {i+1}: {v:.2f}" for i, v in enumerate(best_forecast.values)],
                        hoverinfo="text"
                    ))
                    fig.add_vline(x=len(df), line=dict(color="gray", dash="dot"),
                                  annotation_text="Forecast Start", annotation_position="top left")
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📅 Forecasted Values (Best Model)")
                    forecast_df = pd.DataFrame({"Step": list(range(1, forecast_horizon + 1)), "Forecast": best_forecast.values})
                    st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}"}))
                    csv_best = forecast_df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"📥 Download {best_model} forecast", csv_best, f"{best_model}_forecast.csv", "text/csv")

            else:
                model = model_map[model_choice](df, target_column, steps=forecast_horizon)
                forecast = model.forecast()

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

                st.subheader("📅 Forecasted Values")
                forecast_df = pd.DataFrame({"Step": list(range(1, forecast_horizon + 1)), "Forecast": forecast.values})
                st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}"}))
                csv = forecast_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download forecast as CSV", csv, "forecast.csv", "text/csv")

                if len(df) >= forecast_horizon:
                    true = df[target_column].iloc[-forecast_horizon:].values
                    pred = forecast.reset_index(drop=True).iloc[:forecast_horizon].values
                    rmse, mae, mape = evaluate_forecast(true, pred)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE", f"{rmse:.2f}")
                    c2.metric("MAE", f"{mae:.2f}")
                    c3.metric("MAPE", f"{mape:.2f}%")

                    if model_choice in ["ARIMA", "SARIMA"]:
                        st.caption(f"Model parameters: {model_choice} with default seasonal and trend settings")
                else:
                    st.warning("Not enough historical data to evaluate forecast accuracy.")

        except Exception as e:
            st.error(f"Error processing file or forecast: {e}")

    # --- Footer ---
    st.markdown("---")
    st.markdown("Made with ❤️ by Abhishek Jha", unsafe_allow_html=True)
