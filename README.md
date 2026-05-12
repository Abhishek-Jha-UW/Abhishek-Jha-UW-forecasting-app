# Time Series Forecasting Studio

Streamlit app for **portfolio-grade** time series work: baselines, classical models (ARIMA / SARIMA / ETS), gradient boosting (XGBoost), **frequency-aware seasonality**, optional **walk-forward** back-testing, lightweight **diagnostics**, approximate **forecast intervals**, CSV export, and an optional **OpenAI** narrative that only receives **aggregated metrics** (no raw series rows by default).

---

## Features

- **Data**: CSV / Excel upload or built-in weekly demo data; templates for download.
- **Models**: Naive, Seasonal naive, Moving average, ARIMA, SARIMA, ETS, XGBoost (lag features + recursive horizon).
- **Validation**: Single chronological hold-out, or **walk-forward** averaging over three windows when the series is long enough.
- **Metrics**: RMSE, MAE, MAPE on back-test predictions; leaderboard with automatic “best by RMSE”.
- **Charts**: Plotly history + forecast; optional illustrative interval fan (residual-based).
- **Diagnostics**: Missing counts, duplicate timestamps, simple ACF bar chart.
- **AI (optional)**: One API call per click; uses `OPENAI_API_KEY` (and optional `OPENAI_MODEL`) from **Streamlit secrets**.

---

## Sample data format

| Date       | Sales |
|------------|-------|
| 2022-01-01 | 100   |
| 2022-01-08 | 115   |

One date column plus numeric columns. Sorting is applied in the app.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit secrets (OpenAI)

In Streamlit Community Cloud, add in **App secrets**:

```toml
OPENAI_API_KEY = "sk-..."
# optional — defaults to gpt-4o-mini
OPENAI_MODEL = "gpt-4o-mini"
```

Locally, use `.streamlit/secrets.toml` with the same keys (do not commit secrets).

---

## Snapshot of original code

A copy of the pre-refactor files is in the `intial_data/` folder (same spelling as requested).

---

## Limitations (explicit)

- Intervals are **illustrative**, not calibrated prediction intervals.
- Fixed ARIMA/SARIMA orders are a pragmatic default; production systems would tune orders or use automated selection.
- Walk-forward requires sufficient history; short series fall back to a single hold-out split.
